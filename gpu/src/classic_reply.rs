//! Full (non-compact) replies for the classic lane, assembled from the
//! classic kernel's status exports and record channels with the vendored
//! engine json module, so key set (BTreeMap sort) and number formatting are
//! byte-equal to the engine's reply by construction.
//!
//! Engine anchors: the per-runner results at src/lib.rs:984-1004, the reply
//! object at src/lib.rs:1038-1114, the competition pushes at
//! src/lib.rs:898-918, the player spurt capture at src/lib.rs:920-922, the
//! always-inserted keys at src/lib.rs:1549-1557, and the frame encoder at
//! src/lib.rs:1181-1250.

use crate::classic_exec::ClassicGpu;
use crate::json::{self, Value};
use crate::replay_frames::ReplayFrames;

const DENSE_LEAD_FRAMES: usize = 17;
const SPARSE_STRIDE: usize = 16;

/// The classic loop clock after `ticks` additions of dt. The classic path
/// accumulates `1.0 / 15.0` in plain f64 (src/lib.rs:581, :939).
pub fn classic_loop_end(ticks: usize) -> f64 {
    let mut t = 0f64;
    for _ in 0..ticks {
        t += 1.0 / 15.0;
    }
    t
}

fn base_style(strategy: u32) -> u32 {
    if strategy == 5 {
        1
    } else {
        strategy
    }
}

fn f64w(lo: u32, hi: u32) -> f64 {
    f64::from_bits(u64::from(lo) | (u64::from(hi) << 32))
}

// The unified skillEvents stream in engine push order (src/lib.rs:661-720,
// :1009-1016; the classic path never sorts skillEvents, src/lib.rs:1021-1031).
// Tag 0 is a numeric-skill activation, tag 1 the kakari deactivate entry.
fn skill_events(skev: &[u32]) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < skev.len() {
        match skev[i] {
            0 => {
                if i + 9 > skev.len() {
                    return Err("skill event stream is ragged".to_string());
                }
                let rec = &skev[i..i + 9];
                out.push(json::obj(vec![
                    ("duration", json::num(f64w(rec[7], rec[8]))),
                    ("horseIndex", json::num(rec[2] as f64)),
                    ("isUnique", json::num(rec[4] as f64)),
                    ("skillId", json::num(rec[3] as f64)),
                    ("t", json::num(f64w(rec[5], rec[6]))),
                ]));
                i += 9;
            }
            1 => {
                if i + 6 > skev.len() {
                    return Err("kakari event stream is ragged".to_string());
                }
                let rec = &skev[i..i + 6];
                out.push(json::obj(vec![
                    ("duration", json::num(f64w(rec[4], rec[5]))),
                    ("horseIndex", json::num(rec[1] as f64)),
                    ("isUnique", json::num(0.0)),
                    ("skillId", Value::String("kakari".into())),
                    ("t", json::num(f64w(rec[2], rec[3]))),
                ]));
                i += 6;
            }
            _ => return Err("unknown skill event tag".to_string()),
        }
    }
    Ok(out)
}

// skillDetailEvents, runner-major like the engine's per-solver collection
// (src/lib.rs:1078-1083). The kernel pushes in activation order, so the
// decode groups by runner and keeps each runner's push order.
fn detail_events(det: &[u32]) -> Result<Vec<Value>, String> {
    if det.len() % 5 != 0 {
        return Err("skill detail stream is ragged".to_string());
    }
    let mut recs: Vec<&[u32]> = det.chunks(5).collect();
    recs.sort_by_key(|rec| rec[0]);
    Ok(recs.into_iter().map(|rec| {
        json::obj(vec![
            ("detailIndex", json::num(rec[2] as f64)),
            ("horseIndex", json::num(rec[0] as f64)),
            ("skillId", json::num(rec[1] as f64)),
            ("t", json::num(f64w(rec[3], rec[4]))),
        ])
    }).collect())
}

// skillTargetEvents, runner-major (src/lib.rs:1085-1094). A count word of
// 0xFFFFFFFF is the null selection, k is the list of k runner indices.
fn target_events(tgt: &[u32]) -> Result<Vec<Value>, String> {
    let mut recs: Vec<([u32; 6], Vec<Value>)> = Vec::new();
    let mut i = 0;
    while i < tgt.len() {
        if i + 6 > tgt.len() {
            return Err("skill target stream is ragged".to_string());
        }
        let head: [u32; 6] = [tgt[i], tgt[i + 1], tgt[i + 2], tgt[i + 3],
            tgt[i + 4], tgt[i + 5]];
        i += 6;
        let mut ability = Vec::new();
        for _ in 0..head[5] {
            if i >= tgt.len() {
                return Err("skill target stream is ragged".to_string());
            }
            let count = tgt[i];
            i += 1;
            if count == 0xFFFF_FFFF {
                ability.push(Value::Null);
                continue;
            }
            if i + count as usize > tgt.len() {
                return Err("skill target stream is ragged".to_string());
            }
            ability.push(json::arr(
                tgt[i..i + count as usize].iter()
                    .map(|v| json::num(*v as f64)).collect()));
            i += count as usize;
        }
        recs.push((head, ability));
    }
    recs.sort_by_key(|rec| rec.0[0]);
    Ok(recs.into_iter().map(|(head, ability)| {
        json::obj(vec![
            ("abilityTargets", json::arr(ability)),
            ("detailIndex", json::num(head[2] as f64)),
            ("horseIndex", json::num(head[0] as f64)),
            ("skillId", json::num(head[1] as f64)),
            ("t", json::num(f64w(head[3], head[4]))),
        ])
    }).collect())
}

/// The engine's frame encoder (src/lib.rs:1181-1250) over the classic
/// replay. Classic frames start at t = dt, so the gate frame is always
/// needed once the list is thinned.
fn encode_frames(mut result: Value, mut frames: ReplayFrames, retain_all: bool) -> Value {
    let map = match &mut result {
        Value::Object(m) => m,
        _ => return result,
    };
    if retain_all {
        map.insert("frames".into(), Value::ReplayFrames(Box::new(frames)));
        return result;
    }
    let gate_needed = frames.raw_len() > 0 && frames.raw_frame(0).0 > 0.0;
    let off = if gate_needed { 1 } else { 0 };
    let eff_len = frames.len() + off;
    let mut keep = vec![false; eff_len];
    if eff_len <= 2 {
        for k in keep.iter_mut() {
            *k = true;
        }
    } else {
        for k in keep.iter_mut().take(DENSE_LEAD_FRAMES.min(eff_len)) {
            *k = true;
        }
        let mut i = 0;
        while i < eff_len {
            keep[i] = true;
            i += SPARSE_STRIDE;
        }
        keep[eff_len - 1] = true;
    }
    let mut finish_frames: Vec<Value> = Vec::new();
    let results = match map.get("results").and_then(Value::as_array) {
        Some(rows) => rows.clone(),
        None => Vec::new(),
    };
    for (hi, r) in results.iter().enumerate() {
        let finish_t = r
            .get("finishTimeRaw")
            .and_then(Value::as_f64)
            .unwrap_or(f64::INFINITY);
        let mut best = None;
        for index in 0..frames.raw_len() {
            let (t, horses) = frames.raw_frame(index);
            if t <= finish_t {
                best = Some((t, horses));
            } else {
                break;
            }
        }
        let entry = match best.and_then(|(t, horses)| horses.get(hi).map(|h| (t, h))) {
            Some((t, h)) => json::obj(vec![
                ("pos", json::num(h[0])),
                ("speed", json::num(h[1])),
                ("t", json::num(t)),
            ]),
            None => Value::Null,
        };
        finish_frames.push(entry);
    }
    let raw_frame_count = frames.len();
    frames.select((0..raw_frame_count).filter(|&i| keep[i + off]).collect());
    map.insert("frames".to_string(), Value::ReplayFrames(Box::new(frames)));
    map.insert("finishFrames".to_string(), Value::Array(finish_frames));
    map.insert("rawFrameCount".to_string(), json::num(raw_frame_count as f64));
    map.insert("framesDownsampled".to_string(), Value::Bool(true));
    map.insert("gateFrameNeeded".to_string(), Value::Bool(gate_needed));
    result
}

/// Assemble the engine's full classic reply for one kernel race. The keys a
/// classic reply never carries: `postFinishMovementVersion` is retail-only
/// (src/lib.rs:1130-1134), `gutsOrder` and `wizOrder` are 0 outside the
/// retail runtime (src/lib.rs:992-993), `hpEmptyOnRace` is Null because
/// `solver.retail` is None (src/lib.rs:994), and `additionalTargetEvents`
/// stays empty because the classic lane refuses additional activation
/// (src/lib.rs:1096-1106). `skillEvents` keeps the hook push order: the
/// classic path skips the retail sort (src/lib.rs:1021-1031), and the
/// detail and target arrays are collected runner-major (src/lib.rs:1078-1094).
/// `conservePowerDiagnosticVersion` belongs to a refused flag
/// (src/lib.rs:1115-1123). Grid mode is ported: the reply carries the
/// engine's `experimentalGridVersion` tag (src/lib.rs:1135-1139). singleMode
/// is ported: it tags the reply with `singleModeVersion` (src/lib.rs:1125-1129)
/// and its +400 lands in the adjusted stats (src/race_solver_builder.rs:304-321).
pub fn build(request: &str, seed: i64, race: &ClassicGpu) -> Result<String, String> {
    let root = json::parse(request)?;
    let n = race.frames.first().map_or(0, |f| f.len());
    if n == 0 {
        return Err("the race has no runners".to_string());
    }
    if race.frames.iter().any(|f| f.len() != n) {
        return Err("frame rows changed length".to_string());
    }
    if let Some(reason) = &race.fallback {
        return Err(format!("the race fell back: {reason}"));
    }
    // prepare refuses all of these. The reply re-checks the shape it needs.
    if ["retailRuntime", "experimentalRetailRuntime", "experimentalRetailInitialization",
        "experimentalRetailSections", "experimentalRetailGeometry"]
        .iter()
        .any(|key| root.get(*key).and_then(Value::as_bool).unwrap_or(false))
    {
        return Err("retail runtime belongs to the retail lane".to_string());
    }
    // A classic trace is an engine error (lib.rs:1504-1506), quoted verbatim.
    if root.get("traceConservePower").and_then(Value::as_bool) == Some(true) {
        return Err("traceConservePower requires experimentalRetailRuntime".to_string());
    }
    let horses = root.get_array("horses")?;
    if horses.len() != n {
        return Err("roster length changed".to_string());
    }

    let t_end = race.loop_end;
    let mut finish: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        finish.push(race.finish_raw_all[i].unwrap_or(t_end));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| finish[a].partial_cmp(&finish[b]).unwrap());
    let mut finish_order = vec![0usize; n];
    for (rank, &i) in order.iter().enumerate() {
        finish_order[i] = rank;
    }

    if race.phr.len() % 2 != 0 {
        return Err("positioning stream is ragged".to_string());
    }
    let mut phases = vec![Vec::new(); n];
    for rec in race.phr.chunks(2) {
        if (rec[0] as usize) < n {
            phases[rec[0] as usize].push(rec[1] as i64);
        }
    }

    if race.comp.len() % 3 != 0 {
        return Err("competition stream is ragged".to_string());
    }
    let competition_events: Vec<Value> = race
        .comp
        .chunks(3)
        .map(|rec| {
            json::obj(vec![
                ("horseIndex", json::num(rec[2] as f64)),
                ("t", json::num(classic_loop_end(rec[0] as usize + 1))),
                ("type", json::num(rec[1] as f64)),
            ])
        })
        .collect();

    let player = root
        .get("playerHorseIndex")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as usize;
    let player_last_spurt = match race.spurt_tick_all.get(player).copied().unwrap_or(0) {
        0 => race
            .finish_step_all
            .get(player)
            .copied()
            .flatten()
            .unwrap_or(t_end),
        tick => classic_loop_end(tick as usize),
    };

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let last_spurt = if race.lsttrans_all[i] > 0.0 {
            race.lsttrans_all[i]
        } else {
            -1.0
        };
        results.push(json::obj(vec![
            ("finishOrder", json::num(finish_order[i] as f64)),
            (
                "finishTime",
                json::num(race.finish_step_all[i].unwrap_or(t_end)),
            ),
            ("finishTimeRaw", json::num(finish[i])),
            ("gutsOrder", json::num(0.0)),
            ("wizOrder", json::num(0.0)),
            ("hpEmptyOnRace", Value::Null),
            ("startDelayTime", json::num(race.start_delay_all[i])),
            ("lastSpurtStartDistance", json::num(last_spurt)),
            (
                "actualLastSpurtStartTime",
                race.spurt_time_all[i].map(json::num).unwrap_or(Value::Null),
            ),
            (
                "actualLastSpurtStartDistance",
                race.spurt_dist_all[i].map(json::num).unwrap_or(Value::Null),
            ),
            ("runningStyle", json::num(base_style(race.strategies[i]) as f64)),
            ("rushingStartCount", json::num(race.tempt_all[i] as f64)),
            ("rushingPenaltyCount", json::num(race.rscore_all[i] as f64)),
            (
                "positioningPhases",
                json::arr(phases[i].iter().map(|&p| json::num(p as f64)).collect()),
            ),
        ]));
    }

    let skill_events = skill_events(&race.skev)?;
    let detail_events = detail_events(&race.skdet)?;
    let target_events = target_events(&race.sktgt)?;
    let conserve_events: Vec<Value> = race.skcns.chunks(3).map(|rec| {
        json::obj(vec![
            ("horseIndex", json::num(rec[0] as f64)),
            ("t", json::num(f64w(rec[1], rec[2]))),
        ])
    }).collect();

    let mut result = json::obj(vec![
        ("seed", json::num(seed as f64)),
        ("scoreTelemetryVersion", json::num(3.0)),
        ("runningStyleVersion", json::num(2.0)),
        ("raceInitializationVersion", json::num(1.0)),
        ("positionKeepVersion", json::num(1.0)),
        ("skillWindowVersion", json::num(1.0)),
        ("postNumberVersion", json::num(1.0)),
        ("skillOrderVersion", json::num(1.0)),
        ("lateralPhysicsVersion", json::num(2.0)),
        ("overtakeTimersVersion", json::num(2.0)),
        ("hpStateRatesVersion", json::num(1.0)),
        ("adjustedStatsVersion", json::num(1.0)),
        ("targetSpeedVersion", json::num(1.0)),
        ("laneSkillConditionsVersion", json::num(4.0)),
        ("courseDataVersion", json::num(3.0)),
        ("statAdjustmentVersion", json::num(1.0)),
        ("rosterPassiveVersion", json::num(2.0)),
        ("spurtConditionVersion", json::num(1.0)),
        ("elapsedTimeConditionVersion", json::num(1.0)),
        ("skillDurationVersion", json::num(1.0)),
        ("conservePowerEventVersion", json::num(2.0)),
        ("thresholdStatsVersion", json::num(1.0)),
        ("staminaRulesVersion", json::num(1.0)),
        ("startingGridVersion", json::num(1.0)),
        ("fieldTriggerVersion", json::num(6.0)),
        ("rushingModeVersion", json::num(1.0)),
        ("skillObservationVersion", json::num(1.0)),
        ("randomActivationVersion", json::num(6.0)),
        ("raceContextVersion", json::num(2.0)),
        ("skillReplayDurationVersion", json::num(1.0)),
        ("pacingModeVersion", json::num(2.0)),
        ("currentSpeedReceiverVersion", json::num(1.0)),
        ("experimentalLaterStaminaMechanics",
            Value::Bool(root.get("experimentalLaterStaminaMechanics")
                .and_then(Value::as_bool).unwrap_or(false))),
        ("movementVersion", json::num(1.0)),
        ("laneSelectionVersion", json::num(4.0)),
        ("competitionMechanicsVersion", json::num(1.0)),
        ("skillEventVersion", json::num(1.0)),
        ("skillValueVersion", json::num(4.0)),
        ("rosterConditionVersion", json::num(2.0)),
        ("results", json::arr(results)),
        ("skillEvents", json::arr(skill_events)),
        ("skillDetailObservationVersion", json::num(1.0)),
        ("skillDetailEvents", json::arr(detail_events)),
        ("skillTargetObservationVersion", json::num(1.0)),
        ("skillTargetEvents", json::arr(target_events)),
        ("additionalTargetObservationVersion", json::num(1.0)),
        ("additionalTargetEvents", json::arr(Vec::new())),
        ("competitionEvents", json::arr(competition_events)),
        ("conservePowerEvents", json::arr(conserve_events)),
        ("playerLastSpurtTime", json::num(player_last_spurt)),
        ("playerHorseIndex", json::num(player as f64)),
    ]);
    if let Value::Object(map) = &mut result {
        map.insert("experimentalRetailInitialization".into(), Value::Bool(false));
        // singleMode tags its reply (lib.rs:1125-1129); a non-boolean flag
        // was ignored by the spec parse, so no tag.
        if root.get("singleMode").and_then(Value::as_bool) == Some(true) {
            map.insert("singleModeVersion".into(), json::num(1.0));
        }
        if root.get("pairedRandomStreams").and_then(Value::as_bool) == Some(true) {
            map.insert("pairedRandomStreams".into(), Value::Bool(true));
        }
        // The engine tags grid replies on every lane (lib.rs:1135-1139); a
        // valid shape already passed prepare, so presence is the test.
        if root.get("experimentalGrid").is_some() {
            map.insert("experimentalGridVersion".into(), json::num(1.0));
        }
        if let Some(signature) = root.get("dataSignature") {
            map.insert("dataSignature".to_string(), signature.clone());
        }
    }

    let mut frames = ReplayFrames::new(n);
    for (k, frame) in race.frames.iter().enumerate() {
        frames.capture(classic_loop_end(k + 1), |i| {
            let row = frame[i];
            [row[0], row[1], row[2], row[3], row[4], row[5]]
        });
    }
    let retain_all = root.get("retainAllFrames").and_then(Value::as_bool) == Some(true);
    let result = encode_frames(result, frames, retain_all);
    let mut out = String::new();
    json::to_string_into(&result, &mut out);
    Ok(out)
}
