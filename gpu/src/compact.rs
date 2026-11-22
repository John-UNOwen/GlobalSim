//! The compact record fold the app's compact lane consumes. It is the fold
//! of `globalsim-wasm::compact_record` (app repo globalsim-wasm/src/lib.rs,
//! native-engine branch, at the commit the lane gates pin) fed the launch's
//! kept frame rows instead of the GSIMRP01 packet, so a compact launch ships
//! neither. The frame walk, key set and serde_json output mirror the app
//! fold line by line: the packet's per-frame f64 reads become reads of the
//! kept rows, and the times are the engine's own `loop_end(tick)`, exactly
//! what the packet carried.
//!
//! `kept` lists the runner index behind each frame row, in row order. An
//! index outside it reads as absent, matching the packet fold's
//! `index >= horse_count` walk breaks.

use serde_json::{json, Value};

const TAIL_STEP: f64 = 0.25;
const TAIL_WINDOW: f64 = 10.;
const TAIL_SAMPLES: usize = 41; // floor(TAIL_WINDOW / TAIL_STEP) + 1

/// The kept-row view of one race's frames.
struct Rows<'a> {
    kept: &'a [usize],
    times: &'a [f64],
    frames: &'a [Vec<[f64; 6]>],
}

impl Rows<'_> {
    fn cell(&self, frame: usize, index: usize, col: usize) -> Option<f64> {
        let slot = self.kept.iter().position(|&k| k == index)?;
        Some(*self.frames.get(frame)?.get(slot)?.get(col)?)
    }
    fn time(&self, frame: usize) -> Option<f64> {
        self.times.get(frame).copied()
    }
    fn len(&self) -> usize {
        self.frames.len()
    }
}

/// metrics.ts samplePlayerTail, the app fold's 41-point walk over the last
/// ten seconds, over the kept rows.
fn sample_tail(frames: &Rows, index: usize, finish: f64) -> Vec<[f64; 3]> {
    let from = (finish - TAIL_WINDOW).max(0.);
    let mut slice: Vec<[f64; 3]> = Vec::new();
    for i in 0..frames.len() {
        let Some(t) = frames.time(i) else { break };
        if t < from {
            continue;
        }
        if t > finish {
            break;
        }
        let (Some(p), Some(s)) = (frames.cell(i, index, 0), frames.cell(i, index, 1)) else {
            break;
        };
        slice.push([t, p, s]);
    }
    let n = slice.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(TAIL_SAMPLES);
    if n == 1 {
        for k in 0..TAIL_SAMPLES {
            out.push([(k as f64) * TAIL_STEP, slice[0][1], slice[0][2]]);
        }
        return out;
    }
    let mut lo: isize = n as isize - 2;
    for k in 0..TAIL_SAMPLES {
        let t = (finish - (k as f64) * TAIL_STEP).max(0.);
        while lo > 0 && slice[lo as usize][0] > t {
            lo -= 1;
        }
        let a = slice[lo as usize];
        let b = slice[lo as usize + 1];
        let w = if b[0] > a[0] { (t - a[0]) / (b[0] - a[0]) } else { 0. };
        out.push([(k as f64) * TAIL_STEP, a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])]);
    }
    out
}

/// The app fold's spurtPlanPoint over the kept rows.
fn spurt_plan_point(frames: &Rows, index: usize, distance: f64) -> Option<([f64; 2], usize)> {
    if !(distance > 0.) {
        return None;
    }
    let boundary = ((distance as f32 / 24.) * 16.) as f64;
    let mut crossed = false;
    for i in 0..frames.len() {
        let pos = frames.cell(i, index, 0)?;
        if crossed {
            return Some(([pos, frames.cell(i, index, 2)?], i));
        }
        if pos >= boundary {
            crossed = true;
        }
    }
    None
}

/// The app fold's spurtHeals over the kept rows.
fn spurt_heals(frames: &Rows, index: usize, plan_frame: usize, distance: f64) -> Vec<[f64; 4]> {
    let mut out = Vec::new();
    let at = |i: usize| -> Option<(f64, f64)> {
        Some((frames.cell(i, index, 0)?, frames.cell(i, index, 2)?))
    };
    for j in plan_frame + 1..frames.len() {
        let (Some((pos, hp)), Some((_, next)), Some((_, before))) = (at(j - 1), at(j), at(j - 2))
        else {
            break;
        };
        if pos >= distance {
            break;
        }
        if next > hp {
            out.push([pos, hp, next, before - hp]);
        }
    }
    out
}

/// A runner's HP on the first frame: the race starts at maximum HP.
fn first_hp(frames: &Rows, index: usize) -> Option<f64> {
    frames.cell(0, index, 2)
}

/// serde_json Number to the string JS `String(num)` would produce (skill-id
/// matching only; ids are always positive integers in practice).
fn num_to_js_string(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        format!("{}", n.as_f64().unwrap_or(0.))
    }
}

/// `x ?? fallback` at JSON level: null/absent gets the fallback value, else
/// passthrough.
fn or(v: Option<&Value>, fallback: Value) -> Value {
    match v {
        Some(Value::Null) | None => fallback,
        Some(x) => x.clone(),
    }
}

fn finish_block(
    meta: &Value,
    index: usize,
    second: bool,
) -> Result<serde_json::Map<String, Value>, String> {
    let results = meta.get("results").and_then(|v| v.as_array());
    let res = results.and_then(|a| a.get(index)).and_then(|v| v.as_object());
    let res = match res {
        Some(r) => r,
        None => return Err(if second { "second".into() } else { String::new() }),
    };
    let finish = match res.get("finishTimeRaw").and_then(|v| v.as_f64()) {
        Some(f) => f,
        None => return Err(if second { "second".into() } else { String::new() }),
    };
    let mut out = serde_json::Map::new();
    out.insert("finishTimeRaw".into(), json!(finish));
    out.insert("finishOrder".into(), res.get("finishOrder").cloned().unwrap_or(Value::Null));
    out.insert("hpEmptyOnRace".into(), or(res.get("hpEmptyOnRace"), Value::Null));
    out.insert("rushingStartCount".into(), or(res.get("rushingStartCount"), json!(0)));
    out.insert("rushingPenaltyCount".into(), or(res.get("rushingPenaltyCount"), json!(0)));
    out.insert("lastSpurtStartDistance".into(), or(res.get("lastSpurtStartDistance"), Value::Null));
    out.insert("startDelayTime".into(), or(res.get("startDelayTime"), json!(0)));
    Ok(out)
}

/// The error record strings the app fold emits, mapped from a sentinel Err.
fn record_error(seed: f64, second: bool) -> String {
    let msg = if second {
        "reply has no second player finishTimeRaw"
    } else {
        "reply has no player finishTimeRaw"
    };
    format!(r#"{{"seed":{},"error":"{}"}}"#, seed, msg)
}

/// The per-seed compact record of one launched race, built from the frameless
/// meta reply and the launch's kept rows. Same output as
/// `globalsim-wasm::compact_record` over the same race's meta and packet.
pub fn record(
    meta: &str,
    kept: &[usize],
    times: &[f64],
    frames: &[Vec<[f64; 6]>],
    seed: f64,
    track_skill_id: &str,
    second_player_index: i32,
    distance: f64,
) -> Result<String, String> {
    let v: Value = match serde_json::from_str(meta) {
        Ok(v) => v,
        Err(e) => {
            return Ok(format!(r#"{{"seed":{},"error":"compact: metadata parse failed: {}"}}"#, seed, e))
        }
    };
    let rows = Rows { kept, times, frames };
    let pi = v.get("playerHorseIndex").and_then(|x| x.as_f64()).map(|x| x as usize);
    let pi = match pi {
        Some(p) => p,
        None => return Ok(record_error(seed, false)),
    };

    let own = match finish_block(&v, pi, false) {
        Ok(b) => b,
        Err(_) => return Ok(record_error(seed, false)),
    };
    let finish = own.get("finishTimeRaw").and_then(|x| x.as_f64()).unwrap_or(0.);
    let mut out = serde_json::Map::new();
    out.insert("seed".into(), json!(seed));
    for (k, val) in own {
        if k == "finishTimeRaw"
            || k == "finishOrder"
            || k == "hpEmptyOnRace"
            || k == "rushingStartCount"
            || k == "rushingPenaltyCount"
            || k == "lastSpurtStartDistance"
            || k == "startDelayTime"
        {
            out.insert(k, val);
        }
    }
    let tail = sample_tail(&rows, pi, finish);
    out.insert("tail".into(), if tail.is_empty() { Value::Null } else { json!(tail) });
    if let Some(([pos, hp], plan_frame)) = spurt_plan_point(&rows, pi, distance) {
        out.insert("spurtPlanPos".into(), json!(pos));
        out.insert("spurtPlanHp".into(), json!(hp));
        let heals = spurt_heals(&rows, pi, plan_frame, distance);
        if !heals.is_empty() {
            out.insert("spurtHeals".into(), json!(heals));
            out.insert("spurtMaxHp".into(), json!(first_hp(&rows, pi)));
        }
    }

    if let Some(results) = v.get("results").and_then(|x| x.as_array()) {
        let mut best_rival: Option<f64> = None;
        for (i, r) in results.iter().enumerate() {
            if i == pi {
                continue;
            }
            if let Some(t) = r.get("finishTimeRaw").and_then(|x| x.as_f64()) {
                if t.is_finite() && best_rival.map_or(true, |b| t < b) {
                    best_rival = Some(t);
                }
            }
        }
        if let Some(t) = best_rival {
            out.insert("bestRivalFinishTimeRaw".into(), json!(t));
        }
    }

    if second_player_index >= 0 {
        let si = second_player_index as usize;
        let mut opp = match finish_block(&v, si, true) {
            Ok(b) => b,
            Err(_) => return Ok(record_error(seed, true)),
        };
        let opp_finish = opp.get("finishTimeRaw").and_then(|x| x.as_f64()).unwrap_or(0.);
        let opp_tail = sample_tail(&rows, si, opp_finish);
        opp.insert("tail".into(), if opp_tail.is_empty() { Value::Null } else { json!(opp_tail) });
        out.insert("opp".into(), Value::Object(opp));
    }

    if let Some(events) = v.get("skillEvents").and_then(|x| x.as_array()) {
        let mut per_skill: Vec<(String, usize)> = Vec::new();
        let mut n = 0usize;
        for ev in events {
            let horse_index_matches =
                ev.get("horseIndex").and_then(|x| x.as_f64()).map(|x| x as usize) == Some(pi);
            if !horse_index_matches {
                continue;
            }
            let id = |key: &str| -> Option<String> {
                let v = ev.get(key)?;
                if v.is_null() {
                    return None;
                }
                match v.as_number() {
                    Some(num) => Some(num_to_js_string(num)),
                    None => v.as_str().map(|s| s.to_string()),
                }
            };
            let id = match id("skillId") {
                Some(s) => s,
                None => match id("id") {
                    Some(s) => s,
                    None => continue,
                },
            };
            if !track_skill_id.is_empty() && id == track_skill_id {
                n += 1;
            }
            match per_skill.iter_mut().find(|(k, _)| *k == id) {
                Some((_, c)) => *c += 1,
                None => per_skill.push((id, 1)),
            }
        }
        per_skill.sort();
        let mut map = serde_json::Map::new();
        for (k, c) in per_skill {
            map.insert(k, json!(c));
        }
        out.insert("skillActivations".into(), Value::Object(map));
        if !track_skill_id.is_empty() {
            out.insert("activations".into(), json!(n));
        }
    }

    Ok(serde_json::to_string(&Value::Object(out))
        .unwrap_or_else(|_| format!(r#"{{"seed":{},"error":"compact: serialize failed"}}"#, seed)))
}
