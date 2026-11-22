//! The compact launch gate (handoff Phase B, route B1): `run_compact_batch`
//! builds each compact record straight from the race's meta and its kept
//! frame rows. Those records must be byte-identical to
//! `globalsim-wasm::compact_record` run over `run_batch`'s full meta and
//! packet, for seeds 1..=64 on the v77 request. The packet fold below is that
//! app function vendored (app repo globalsim-wasm/src/lib.rs, native-engine
//! branch, verbatim), so the test pins the lane against the fold the app
//! actually runs. Solo, second-player, same-row (player == second) and
//! tracked-skill configurations all run.

#![cfg(feature = "race")]

use globalsim_gpu::facade::{CompactOptions, GpuRaces};

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_line() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

/// The course length pd carries, straight from data/course_data.json like
/// `pack_course_scalars` reads it.
fn course_distance(course_id: i64) -> f64 {
    let text = std::fs::read_to_string(repo_file("data/course_data.json")).expect("course data");
    let root = globalsim_gpu::json::parse(&text).expect("course data json");
    root.get(&course_id.to_string())
        .and_then(|c| c.get("distance"))
        .and_then(|d| d.as_f64())
        .expect("course distance")
}

// ---------------------------------------------------------------------------
// Vendored globalsim-wasm::compact_record (app repo globalsim-wasm/src/lib.rs,
// native-engine branch). Kept verbatim so byte equality below means exactly
// what the app sees.
// ---------------------------------------------------------------------------

const PACKET_MAGIC: &[u8; 8] = b"GSIMRP01";
const PACKET_HEADER: usize = 8 + 8 + 4 + 4 + 8;
const TAIL_STEP: f64 = 0.25;
const TAIL_WINDOW: f64 = 10.;
const TAIL_SAMPLES: usize = 41; // floor(TAIL_WINDOW / TAIL_STEP) + 1

fn packet_field(packet: &[u8], frame_count: usize, horse_count: usize, meta_len: usize, i: usize) -> Option<(f64, usize)> {
    let base = PACKET_HEADER + meta_len + i * (8 + horse_count * 48);
    let end = base + 8 + horse_count * 48;
    if end > packet.len() { return None; }
    let t = f64::from_le_bytes(packet[base..base + 8].try_into().ok()?);
    Some((t, base + 8))
}
fn packet_f64(packet: &[u8], off: usize, col: usize) -> Option<f64> {
    let at = off + col * 8;
    if at + 8 > packet.len() { return None; }
    Some(f64::from_le_bytes(packet[at..at + 8].try_into().ok()?))
}

fn sample_tail(packet: &[u8], frame_count: usize, horse_count: usize, meta_len: usize, player_index: usize, finish: f64) -> Vec<[f64; 3]> {
    let from = (finish - TAIL_WINDOW).max(0.);
    let mut slice: Vec<[f64; 3]> = Vec::new();
    for i in 0..frame_count {
        let Some((t, row)) = packet_field(packet, frame_count, horse_count, meta_len, i) else { break };
        if t < from { continue; }
        if t > finish { break; }
        if player_index >= horse_count { break; } // JS: missing horse tuple → break
        let col = player_index * 6;
        let pos = packet_f64(packet, row, col);
        let speed = packet_f64(packet, row, col + 1);
        match (pos, speed) { (Some(p), Some(s)) => slice.push([t, p, s]), _ => break }
    }
    let n = slice.len();
    if n == 0 { return Vec::new(); }
    let mut out = Vec::with_capacity(TAIL_SAMPLES);
    if n == 1 {
        for k in 0..TAIL_SAMPLES { out.push([(k as f64) * TAIL_STEP, slice[0][1], slice[0][2]]); }
        return out;
    }
    let mut lo: isize = n as isize - 2;
    for k in 0..TAIL_SAMPLES {
        let t = (finish - (k as f64) * TAIL_STEP).max(0.);
        while lo > 0 && slice[lo as usize][0] > t { lo -= 1; }
        let a = slice[lo as usize];
        let b = slice[lo as usize + 1];
        let w = if b[0] > a[0] { (t - a[0]) / (b[0] - a[0]) } else { 0. };
        out.push([(k as f64) * TAIL_STEP, a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])]);
    }
    out
}

fn spurt_plan_point(packet: &[u8], frame_count: usize, horse_count: usize, meta_len: usize, index: usize, distance: f64) -> Option<([f64; 2], usize)> {
    if !(distance > 0.) || index >= horse_count { return None; }
    let boundary = ((distance as f32 / 24.) * 16.) as f64;
    let mut crossed = false;
    for i in 0..frame_count {
        let (_, row) = packet_field(packet, frame_count, horse_count, meta_len, i)?;
        let col = index * 6;
        let pos = packet_f64(packet, row, col)?;
        if crossed { return Some(([pos, packet_f64(packet, row, col + 2)?], i)); }
        if pos >= boundary { crossed = true; }
    }
    None
}

fn spurt_heals(packet: &[u8], frame_count: usize, horse_count: usize, meta_len: usize, index: usize, plan_frame: usize, distance: f64) -> Vec<[f64; 4]> {
    let mut out = Vec::new();
    let col = index * 6;
    let at = |i: usize| -> Option<(f64, f64)> {
        let (_, row) = packet_field(packet, frame_count, horse_count, meta_len, i)?;
        Some((packet_f64(packet, row, col)?, packet_f64(packet, row, col + 2)?))
    };
    for j in plan_frame + 1..frame_count {
        let (Some((pos, hp)), Some((_, next)), Some((_, before))) = (at(j - 1), at(j), at(j - 2)) else { break };
        if pos >= distance { break; }
        if next > hp { out.push([pos, hp, next, before - hp]); }
    }
    out
}

fn first_hp(packet: &[u8], frame_count: usize, horse_count: usize, meta_len: usize, index: usize) -> Option<f64> {
    let (_, row) = packet_field(packet, frame_count, horse_count, meta_len, 0)?;
    packet_f64(packet, row, index * 6 + 2)
}

fn num_to_js_string(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() { i.to_string() }
    else if let Some(u) = n.as_u64() { u.to_string() }
    else { format!("{}", n.as_f64().unwrap_or(0.)) }
}

fn or(v: Option<&serde_json::Value>, fallback: serde_json::Value) -> serde_json::Value {
    match v { Some(serde_json::Value::Null) | None => fallback, Some(x) => x.clone() }
}

fn finish_block(meta: &serde_json::Value, index: usize, second: bool) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let results = meta.get("results").and_then(|v| v.as_array());
    let res = results.and_then(|a| a.get(index)).and_then(|v| v.as_object());
    let res = match res { Some(r) => r, None => return Err(if second { "second".into() } else { String::new() }) };
    let finish = match res.get("finishTimeRaw").and_then(|v| v.as_f64()) {
        Some(f) => f,
        None => return Err(if second { "second".into() } else { String::new() }),
    };
    let mut out = serde_json::Map::new();
    out.insert("finishTimeRaw".into(), serde_json::json!(finish));
    out.insert("finishOrder".into(), res.get("finishOrder").cloned().unwrap_or(serde_json::Value::Null));
    out.insert("hpEmptyOnRace".into(), or(res.get("hpEmptyOnRace"), serde_json::Value::Null));
    out.insert("rushingStartCount".into(), or(res.get("rushingStartCount"), serde_json::json!(0)));
    out.insert("rushingPenaltyCount".into(), or(res.get("rushingPenaltyCount"), serde_json::json!(0)));
    out.insert("lastSpurtStartDistance".into(), or(res.get("lastSpurtStartDistance"), serde_json::Value::Null));
    out.insert("startDelayTime".into(), or(res.get("startDelayTime"), serde_json::json!(0)));
    Ok(out)
}

fn record_error(seed: f64, second: bool) -> String {
    let msg = if second { "reply has no second player finishTimeRaw" } else { "reply has no player finishTimeRaw" };
    format!(r#"{{"seed":{},"error":"{}"}}"#, seed, msg)
}

fn compact_record(meta: &str, packet: &[u8], seed: f64, track_skill_id: &str, second_player_index: i32, distance: f64) -> Result<String, String> {
    if packet.get(..8).map_or(true, |m| m != PACKET_MAGIC) {
        return Ok(format!(r#"{{"seed":{},"error":"compact: unexpected packet magic"}}"#, seed));
    }
    let u = |o: usize, n: usize| -> u64 {
        let mut v = 0u64;
        for i in 0..n { v |= (packet[o + i] as u64) << (8 * i); }
        v
    };
    let meta_len = u(8, 8) as usize;
    let frame_count = u(16, 4) as usize;
    let horse_count = u(20, 4) as usize;

    let v: serde_json::Value = match serde_json::from_str(meta) {
        Ok(v) => v,
        Err(e) => return Ok(format!(r#"{{"seed":{},"error":"compact: metadata parse failed: {}"}}"#, seed, e)),
    };
    let pi = v.get("playerHorseIndex").and_then(|x| x.as_f64()).map(|x| x as usize);
    let pi = match pi { Some(p) => p, None => return Ok(record_error(seed, false)) };

    let own = match finish_block(&v, pi, false) {
        Ok(b) => b,
        Err(_) => return Ok(record_error(seed, false)),
    };
    let finish = own.get("finishTimeRaw").and_then(|x| x.as_f64()).unwrap_or(0.);
    let mut out = serde_json::Map::new();
    out.insert("seed".into(), serde_json::json!(seed));
    for (k, val) in own {
        if k == "finishTimeRaw" || k == "finishOrder" || k == "hpEmptyOnRace" || k == "rushingStartCount"
            || k == "rushingPenaltyCount" || k == "lastSpurtStartDistance" || k == "startDelayTime"
        { out.insert(k, val); }
    }
    let tail = sample_tail(packet, frame_count, horse_count, meta_len, pi, finish);
    out.insert("tail".into(), if tail.is_empty() { serde_json::Value::Null } else { serde_json::json!(tail) });
    if let Some(([pos, hp], plan_frame)) = spurt_plan_point(packet, frame_count, horse_count, meta_len, pi, distance) {
        out.insert("spurtPlanPos".into(), serde_json::json!(pos));
        out.insert("spurtPlanHp".into(), serde_json::json!(hp));
        let heals = spurt_heals(packet, frame_count, horse_count, meta_len, pi, plan_frame, distance);
        if !heals.is_empty() {
            out.insert("spurtHeals".into(), serde_json::json!(heals));
            out.insert("spurtMaxHp".into(), serde_json::json!(first_hp(packet, frame_count, horse_count, meta_len, pi)));
        }
    }

    if let Some(results) = v.get("results").and_then(|x| x.as_array()) {
        let mut best_rival: Option<f64> = None;
        for (i, r) in results.iter().enumerate() {
            if i == pi { continue; }
            if let Some(t) = r.get("finishTimeRaw").and_then(|x| x.as_f64()) {
                if t.is_finite() && best_rival.map_or(true, |b| t < b) { best_rival = Some(t); }
            }
        }
        if let Some(t) = best_rival {
            out.insert("bestRivalFinishTimeRaw".into(), serde_json::json!(t));
        }
    }

    if second_player_index >= 0 {
        let si = second_player_index as usize;
        let mut opp = match finish_block(&v, si, true) {
            Ok(b) => b,
            Err(_) => return Ok(record_error(seed, true)),
        };
        let opp_finish = opp.get("finishTimeRaw").and_then(|x| x.as_f64()).unwrap_or(0.);
        let opp_tail = sample_tail(packet, frame_count, horse_count, meta_len, si, opp_finish);
        opp.insert("tail".into(), if opp_tail.is_empty() { serde_json::Value::Null } else { serde_json::json!(opp_tail) });
        out.insert("opp".into(), serde_json::Value::Object(opp));
    }

    if let Some(events) = v.get("skillEvents").and_then(|x| x.as_array()) {
        let mut per_skill: Vec<(String, usize)> = Vec::new();
        let mut n = 0usize;
        for ev in events {
            let horse_index_matches = ev.get("horseIndex").and_then(|x| x.as_f64()).map(|x| x as usize) == Some(pi);
            if !horse_index_matches { continue; }
            let id = |key: &str| -> Option<String> {
                let v = ev.get(key)?;
                if v.is_null() { return None; }
                match v.as_number() { Some(num) => Some(num_to_js_string(num)), None => v.as_str().map(|s| s.to_string()) }
            };
            let id = match id("skillId") { Some(s) => s, None => match id("id") { Some(s) => s, None => continue } };
            if !track_skill_id.is_empty() && id == track_skill_id { n += 1; }
            match per_skill.iter_mut().find(|(k, _)| *k == id) {
                Some((_, c)) => *c += 1,
                None => per_skill.push((id, 1)),
            }
        }
        per_skill.sort();
        let mut map = serde_json::Map::new();
        for (k, c) in per_skill { map.insert(k, serde_json::json!(c)); }
        out.insert("skillActivations".into(), serde_json::Value::Object(map));
        if !track_skill_id.is_empty() { out.insert("activations".into(), serde_json::json!(n)); }
    }

    Ok(serde_json::to_string(&serde_json::Value::Object(out)).unwrap_or_else(|_| format!(r#"{{"seed":{},"error":"compact: serialize failed"}}"#, seed)))
}

// ---------------------------------------------------------------------------
// The comparison.
// ---------------------------------------------------------------------------

fn check(label: &str, id: &str, seeds: &[i32], full: &[(String, Vec<u8>)], records: &[Option<String>], second: i32, distance: f64, problems: &mut Vec<String>, rich: &mut usize) {
    for (slot, seed) in seeds.iter().enumerate() {
        let want = compact_record(&full[slot].0, &full[slot].1, *seed as f64, id, second, distance).unwrap();
        match &records[slot] {
            None => problems.push(format!("{label} seed {seed}: no record")),
            Some(got) => {
                if got != &want {
                    let point = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                    problems.push(format!(
                        "{label} seed {seed}: differs at byte {point}\n  got:  {}\n  want: {}",
                        got, want
                    ));
                }
                if got.contains("spurtPlanPos") && got.contains("\"tail\":[") {
                    *rich += 1;
                }
            }
        }
    }
}

#[test]
fn compact_launch_records_match_the_packet_fold() {
    let request = v77_line();
    let root = globalsim_gpu::json::parse(&request).expect("request json");
    let course_id = root.get("courseId").and_then(|c| c.as_f64()).expect("courseId") as i64;
    let distance = course_distance(course_id);
    assert!(distance > 0.);
    let seeds: Vec<i32> = (1..=64).collect();
    let requests: Vec<String> = seeds.iter().map(|_| request.clone()).collect();

    let mut gpu = GpuRaces::new().expect("gated device opens");
    let full: Vec<(String, Vec<u8>)> = gpu
        .run_batch(&requests, &seeds)
        .expect("full batch")
        .into_iter()
        .map(|outcome| match (outcome.meta, outcome.packet, outcome.fallback) {
            (Some(meta), Some(packet), None) => (meta, packet),
            (_, _, Some(reason)) => panic!("full path fell back: {reason}"),
            _ => panic!("full path answered without meta and packet"),
        })
        .collect();

    // A skill id the player's seed-1 events carry, so the tracked-skill
    // configuration counts real activations.
    let events = serde_json::from_str::<serde_json::Value>(&full[0].0)
        .expect("meta parses")
        .get("skillEvents")
        .and_then(|e| e.as_array())
        .cloned()
        .unwrap_or_default();
    let tracked = events
        .iter()
        .find(|ev| ev.get("horseIndex").and_then(|x| x.as_f64()).map(|x| x as usize) == Some(0))
        .and_then(|ev| {
            ev.get("skillId")
                .map(|id| id.as_str().map(str::to_string).unwrap_or_else(|| id.to_string()))
        })
        .unwrap_or_else(|| "200361".into());

    let mut problems: Vec<String> = Vec::new();
    let mut rich = 0usize;
    let configs: [(&str, i32, &str); 4] = [
        ("solo", -1, ""),
        ("h2h", 4, ""),
        ("same row", 0, ""),
        ("tracked", -1, &tracked),
    ];
    for (label, second, id) in configs {
        let opts = CompactOptions { track_skill_id: id.into(), second_player_index: second };
        let outcomes = gpu.run_compact_batch(&requests, &seeds, &opts).expect("compact batch");
        let records: Vec<Option<String>> = outcomes
            .into_iter()
            .map(|outcome| match (outcome.record, outcome.fallback) {
                (Some(record), None) => Some(record),
                (_, Some(reason)) => panic!("{label} compact path fell back: {reason}"),
                _ => panic!("{label} compact path answered without a record"),
            })
            .collect();
        check(label, id, &seeds, &full, &records, second, distance, &mut problems, &mut rich);
    }
    // The fold walked real frames: plan points and tails must show up.
    assert!(rich > 0, "no record carried a plan point and a tail, the fold ran vacuously");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
