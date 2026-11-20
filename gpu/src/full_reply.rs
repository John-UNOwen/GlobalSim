//! Full (non-compact) race replies assembled from the kernel's reply
//! channels. The result JSON is built with the vendored engine's own json
//! module, so key order (BTreeMap sort) and number formatting are byte-equal
//! to the engine's reply by construction. `build` covers the exact key set
//! the engine's retail lane emits; the facade refuses every shape it cannot
//! answer.
//!
//! Engine anchors: the reply object at src/lib.rs:1038-1149, the activate
//! and deactivate hooks at src/lib.rs:635-724, the pending finalizer and
//! sort at src/lib.rs:1008-1036, the frame encoder at src/lib.rs:1181-1250,
//! the competition reorder at src/race_field.rs:181-201 and the order-key
//! snapshots at src/lib.rs:811-827.

use crate::json::{self, Value};
use crate::race_exec::{self, RaceGpu};
use crate::replay_frames::ReplayFrames;
use crate::retail_sort::comparison_sort;

const DENSE_LEAD_FRAMES: usize = 17;
const SPARSE_STRIDE: usize = 16;
const DT: f32 = 0.0666;

/// The engine's loop clock after `ticks` retail steps, bit for bit.
pub fn loop_end(ticks: usize) -> f64 {
    let mut t = 0f64;
    for _ in 0..ticks {
        t = ((t as f32) + DT) as f64;
    }
    t
}

fn dt_f64() -> f64 {
    DT as f64
}

fn f64w(lo: u32, hi: u32) -> f64 {
    f64::from_bits(u64::from(lo) | (u64::from(hi) << 32))
}

/// The strategy int a request carries for one horse, widened like the
/// engine's parse.
fn strategy_of(horse: &Value) -> Result<i64, String> {
    let text = horse
        .get("strategy")
        .and_then(Value::as_str)
        .ok_or("horse has no strategy")?;
    match text.to_uppercase().as_str() {
        "NIGE" => Ok(1),
        "SENKOU" => Ok(2),
        "SASI" | "SASHI" => Ok(3),
        "OIKOMI" => Ok(4),
        "OONIGE" => Ok(5),
        _ => Err("invalid running strategy".to_string()),
    }
}

/// The order keys the engine snapshots after the constructor fires, keyed
/// on the gate identity with tolerance ties (src/lib.rs:811-827).
fn parameter_orders(snap: &[f64], n: usize) -> Vec<usize> {
    let value: Vec<f32> = snap
        .iter()
        .take(n)
        .map(|v| crate::retail_physics::bounded_stat(*v as f32))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    comparison_sort(&mut order, |&a, &b| {
        let (left, right) = (value[a], value[b]);
        let tolerance = (1e-6_f32 * left.abs().max(right.abs())).max(8.0 * f32::from_bits(1));
        if (left - right).abs() < tolerance {
            a.cmp(&b)
        } else {
            right.total_cmp(&left)
        }
    });
    let mut ranks = vec![0usize; n];
    for (rank, i) in order.into_iter().enumerate() {
        ranks[i] = rank;
    }
    ranks
}

/// One skillEvents record before the final sort. Kakari entries carry no
/// ordinal, which the sort reads as +inf (src/lib.rs:1030-1033).
struct Event {
    t: f64,
    ordinal: Option<f64>,
    repeat: bool,
    fields: Vec<(&'static str, Value)>,
}

fn skill_event(
    t: f64,
    ordinal: Option<f64>,
    repeat: bool,
    horse: usize,
    skill: Value,
    duration: f64,
    is_unique: bool,
) -> Event {
    Event {
        t,
        ordinal,
        repeat,
        fields: vec![
            ("duration", json::num(duration)),
            ("horseIndex", json::num(horse as f64)),
            ("isUnique", json::num(if is_unique { 1.0 } else { 0.0 })),
            ("skillId", skill),
            ("t", json::num(t)),
        ],
    }
}

/// The per-horse isUnique sets the activate hook reads (src/lib.rs:623-626).
fn unique_ids(root: &Value, n: usize) -> Result<Vec<Vec<String>>, String> {
    let horses = root.get_array("horses")?;
    if horses.len() != n {
        return Err("roster length changed".to_string());
    }
    horses
        .iter()
        .map(|horse| {
            Ok(match horse.get("skills").and_then(Value::as_array) {
                None => Vec::new(),
                Some(list) => list
                    .iter()
                    .filter_map(|entry| entry.as_object())
                    .filter(|o| o.get("isUnique").and_then(Value::as_bool) == Some(true))
                    .filter_map(|o| {
                        o.get("skillId")
                            .and_then(Value::as_str)
                            .map(|s| s.to_string())
                            .or_else(|| {
                                o.get("skillId")
                                    .and_then(Value::as_f64)
                                    .map(|n| (n as i64).to_string())
                            })
                    })
                    .collect(),
            })
        })
        .collect()
}

/// The activation stream turned into skillEvents: a pending activation parks
/// its tuple in the engine's single (horse, id) slot, a finalize pops and
/// pushes the parked tuple, and race end flushes what stays parked in
/// ordinal order (src/lib.rs:1008-1020), before the global sort.
fn skill_events(act: &[u32], unique: &[Vec<String>]) -> Result<Vec<Event>, String> {
    struct Slot {
        start: f64,
        duration: f64,
        ordinal: u64,
        unique: bool,
    }
    if act.len() % 10 != 0 {
        return Err("activation stream is ragged".to_string());
    }
    let mut slots: std::collections::BTreeMap<(usize, i64), Slot> = std::collections::BTreeMap::new();
    let mut repeats: Vec<u64> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    for rec in act.chunks(10) {
        let (r, sid) = (rec[0] as usize, rec[1] as i64);
        let clock = f64w(rec[3], rec[4]);
        let duration = f64w(rec[5], rec[6]);
        let ordinal = (rec[7] - 1) as u64;
        let flags = rec[8];
        let unique_flag = unique.get(r).is_some_and(|ids| ids.contains(&sid.to_string()));
        if flags & 2 != 0 {
            repeats.push(ordinal);
        }
        if flags & 4 != 0 {
            if let Some(slot) = slots.remove(&(r, sid)) {
                events.push(skill_event(
                    slot.start,
                    Some(slot.ordinal as f64),
                    repeats.contains(&slot.ordinal),
                    r,
                    json::num(sid as f64),
                    slot.duration,
                    slot.unique,
                ));
            }
            continue;
        }
        if flags & 1 != 0 {
            slots.insert(
                (r, sid),
                Slot {
                    start: clock,
                    duration,
                    ordinal,
                    unique: unique_flag,
                },
            );
            continue;
        }
        events.push(skill_event(
            clock,
            Some(ordinal as f64),
            repeats.contains(&ordinal),
            r,
            json::num(sid as f64),
            0.0,
            unique_flag,
        ));
    }
    let mut parked: Vec<_> = slots
        .into_iter()
        .map(|((r, sid), slot)| (r, sid, slot))
        .collect();
    parked.sort_by_key(|(_, _, slot)| slot.ordinal);
    for (r, sid, slot) in parked {
        events.push(skill_event(
            slot.start,
            Some(slot.ordinal as f64),
            repeats.contains(&slot.ordinal),
            r,
            json::num(sid as f64),
            slot.duration,
            slot.unique,
        ));
    }
    Ok(events)
}

/// Rush episodes paired from the kernel's edge stream (src/lib.rs:646-702).
fn kakari_events(kak: &[u32]) -> Vec<Event> {
    if kak.len() % 4 != 0 {
        return Vec::new();
    }
    let mut started: Vec<Option<f64>> = Vec::new();
    let mut events = Vec::new();
    for rec in kak.chunks(4) {
        let r = rec[1] as usize;
        let clock = f64w(rec[2], rec[3]);
        if started.len() <= r {
            started.resize(r + 1, None);
        }
        if rec[0] == 0 {
            started[r] = Some(clock);
        } else if let Some(start) = started[r].take() {
            events.push(skill_event(
                start,
                None,
                false,
                r,
                Value::String("kakari".to_string()),
                clock - start,
                false,
            ));
        }
    }
    events
}

/// The engine's final sort: time, then startup repeats last, then the
/// activation ordinal, with kakari ordered after every ordinal event.
fn sort_events(events: &mut [Event]) {
    events.sort_by(|a, b| {
        a.t.total_cmp(&b.t)
            .then_with(|| a.repeat.cmp(&b.repeat))
            .then_with(|| {
                a.ordinal
                    .unwrap_or(f64::MAX)
                    .total_cmp(&b.ordinal.unwrap_or(f64::MAX))
            })
    });
}

/// One decoded record from a target or fan stream.
type Selected = (usize, i64, usize, f64, Vec<Option<Vec<usize>>>);

/// Decode a target stream: runner, id, detail, clock, effect count, then one
/// length-prefixed list per effect where a length of 0 encodes None.
fn parse_targets(words: &[u32]) -> Result<Vec<Selected>, String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < words.len() {
        if words.len() - at < 6 {
            return Err("target stream is ragged".to_string());
        }
        let (r, sid, detail) = (words[at] as usize, words[at + 1] as i64, words[at + 2] as usize);
        let t = f64w(words[at + 3], words[at + 4]);
        let count = words[at + 5] as usize;
        let mut at2 = at + 6;
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            if at2 >= words.len() {
                return Err("target stream is ragged".to_string());
            }
            let enc = words[at2] as usize;
            at2 += 1;
            if enc == 0 {
                list.push(None);
                continue;
            }
            let len = enc - 1;
            if words.len() - at2 < len {
                return Err("target stream is ragged".to_string());
            }
            list.push(Some(words[at2..at2 + len].iter().map(|w| *w as usize).collect()));
            at2 += len;
        }
        out.push((r, sid, detail, t, list));
        at = at2;
    }
    Ok(out)
}

/// Decode a fan stream: one single-slot delivery per record, the encoded
/// list living directly in header word 5.
fn parse_fans(words: &[u32]) -> Result<Vec<Selected>, String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < words.len() {
        if words.len() - at < 6 {
            return Err("fan stream is ragged".to_string());
        }
        let (r, sid, detail) = (words[at] as usize, words[at + 1] as i64, words[at + 2] as usize);
        let t = f64w(words[at + 3], words[at + 4]);
        let enc = words[at + 5] as usize;
        let mut at2 = at + 6;
        let nids;
        let list = if enc == 0 {
            nids = 0;
            vec![None]
        } else {
            let len = enc - 1;
            if words.len() - at2 < len {
                return Err("fan stream is ragged".to_string());
            }
            nids = len;
            vec![Some(words[at2..at2 + len].iter().map(|w| *w as usize).collect())]
        };
        out.push((r, sid, detail, t, list));
        at = at2 + nids;
    }
    Ok(out)
}

/// Group per-solver vectors the way the engine emits them: by horse, then
/// each horse's push order, which is the kernel's stream order per horse.
fn group_by_runner(records: Vec<Selected>) -> Vec<Selected> {
    let mut keyed: Vec<(usize, usize, Selected)> =
        records.into_iter().enumerate().map(|(k, rec)| (rec.0, k, rec)).collect();
    keyed.sort_by_key(|(r, k, _)| (*r, *k));
    keyed.into_iter().map(|(_, _, rec)| rec).collect()
}

fn ability_targets(list: &[Option<Vec<usize>>]) -> Value {
    json::arr(
        list.iter()
            .map(|selection| match selection {
                Some(indices) => json::arr(indices.iter().map(|i| json::num(*i as f64)).collect()),
                None => Value::Null,
            })
            .collect(),
    )
}

/// Type-4 slots reordered like RaceField::order_competition_records
/// (src/race_field.rs:181-201) within one drain batch. The kernel writes
/// startup-pass records with the pre-tick marker word, and the engine drains
/// them together with tick 0, so the marker joins the tick 0 batch.
fn order_competition(comp: &[u32], strategies: &[i64], n: usize) -> Result<Vec<(i64, f64, usize)>, String> {
    if comp.len() % 3 != 0 {
        return Err("competition stream is ragged".to_string());
    }
    // Field-construction positions are all zero outside grid mode, which
    // never reaches this lane. Every pair compares Equal, so the engine's
    // introsort leaves the roster filter order untouched.
    let roster = {
        let mut recording: Vec<usize> =
            (0..n).filter(|&i| strategies[i] == 1 || strategies[i] == 5).collect();
        let mut ranks = vec![0usize; n];
        for (rank, i) in recording.iter().copied().enumerate() {
            ranks[i] = rank;
        }
        ranks
    };
    let mut out: Vec<(i64, f64, usize)> = Vec::with_capacity(comp.len() / 3);
    let mut at = 0usize;
    while at < comp.len() {
        let tick = if comp[at] == u32::MAX { 0 } else { comp[at] };
        let mut end = at;
        while end < comp.len()
            && if comp[end] == u32::MAX { 0 } else { comp[end] } == tick
        {
            end += 3;
        }
        let batch = &comp[at..end];
        let mut spots: Vec<usize> = Vec::new();
        let mut leaders: [Option<usize>; 6] = [None; 6];
        for slot in (0..batch.len()).step_by(3) {
            if batch[slot + 1] != 4 {
                continue;
            }
            let i = batch[slot + 2] as usize;
            let strategy = strategies[i] as usize;
            if leaders[strategy].is_none() {
                leaders[strategy] = Some(i);
            }
            spots.push(slot);
        }
        spots.sort_by_key(|&slot| {
            let i = batch[slot + 2] as usize;
            (
                strategies[i] != 5,
                leaders[strategies[i] as usize] != Some(i),
                roster[i],
            )
        });
        let t = loop_end(tick as usize) + dt_f64();
        let mut ordered = spots.into_iter();
        for slot in (0..batch.len()).step_by(3) {
            let kind = batch[slot + 1] as i64;
            let i = if kind == 4 {
                let source = ordered.next().ok_or("competition stream is ragged")?;
                batch[source + 2] as usize
            } else {
                batch[slot + 2] as usize
            };
            out.push((kind, t, i));
        }
        at = end;
    }
    Ok(out)
}

fn positioning_phrases(phr: &[u32], n: usize) -> Vec<Vec<i64>> {
    let mut phases = vec![Vec::new(); n];
    for rec in phr.chunks(4) {
        if rec.len() == 4 && (rec[0] as usize) < n && rec[3] != 0 {
            phases[rec[0] as usize].push(rec[1] as i64);
        }
    }
    phases
}

/// The engine's frame encoder (src/lib.rs:1181-1250), thinning lane.
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

/// Assemble the engine's full reply for one kernel race. `request` is the
/// original request text; the retainAllFrames key, when set, keeps every
/// frame exactly like the engine's binary lane.
pub fn build(request: &str, seed: i64, race: &RaceGpu) -> Result<String, String> {
    let root = json::parse(request)?;
    let raw = race
        .reply
        .as_ref()
        .ok_or("the race carries no reply channels".to_string())?;
    let n = race.frames.first().map_or(0, |f| f.len());
    if n == 0 {
        return Err("the race has no runners".to_string());
    }
    if race.grid {
        return Err("grid mode needs the engine's staggered roster ranks".to_string());
    }
    for key in ["traceConservePower", "singleMode"] {
        if root.get(key).and_then(Value::as_bool) == Some(true) {
            return Err(format!("{key} is outside the full GPU lane"));
        }
    }
    if root.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool) == Some(true) {
        return Err("experimentalLaterStaminaMechanics is outside the full GPU lane".to_string());
    }
    let horses = root.get_array("horses")?;
    if horses.len() != n {
        return Err("roster length changed".to_string());
    }
    let strategies: Vec<i64> = horses.iter().map(strategy_of).collect::<Result<_, _>>()?;
    let unique = unique_ids(&root, n)?;

    // The loop end is the engine's finish value for runners that never crossed.
    let t_end = race_exec::loop_end(race.ticks);
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
    let guts = parameter_orders(&raw.snap_guts, n);
    let wiz = parameter_orders(&raw.snap_wis, n);
    let phases = positioning_phrases(&raw.phr, n);

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let last_spurt = if race.lss_all[i] > 0.0 { race.lss_all[i] } else { -1.0 };
        results.push(json::obj(vec![
            ("finishOrder", json::num(finish_order[i] as f64)),
            ("finishTime", json::num(race.finish_step_all[i].unwrap_or(t_end))),
            ("finishTimeRaw", json::num(finish[i])),
            ("gutsOrder", json::num(guts[i] as f64)),
            ("wizOrder", json::num(wiz[i] as f64)),
            ("hpEmptyOnRace", Value::Bool(race.hp_empty_all[i])),
            ("startDelayTime", json::num(race.delay_all[i])),
            ("lastSpurtStartDistance", json::num(last_spurt)),
            (
                "actualLastSpurtStartTime",
                raw.spurt_time[i].map(json::num).unwrap_or(Value::Null),
            ),
            (
                "actualLastSpurtStartDistance",
                raw.spurt_dist[i].map(json::num).unwrap_or(Value::Null),
            ),
            (
                "runningStyle",
                json::num(if strategies[i] == 5 { 1.0 } else { strategies[i] as f64 }),
            ),
            ("rushingStartCount", json::num(race.tempt_all[i] as f64)),
            ("rushingPenaltyCount", json::num(race.rscore_all[i] as f64)),
            (
                "positioningPhases",
                json::arr(phases[i].iter().map(|&p| json::num(p as f64)).collect()),
            ),
        ]));
    }

    let mut events = skill_events(&raw.act, &unique)?;
    events.extend(kakari_events(&raw.kak));
    sort_events(&mut events);
    let skill_events: Vec<Value> = events.iter().map(|e| json::obj(e.fields.clone())).collect();

    // Detail and target events group by horse, then kernel stream order,
    // which is the engine's per-solver push order (src/lib.rs:1048-1076).
    let targets = group_by_runner(parse_targets(&raw.tgt)?);
    let mut detail_events = Vec::new();
    let mut target_events = Vec::new();
    for (r, sid, detail, t, list) in &targets {
        detail_events.push(json::obj(vec![
            ("detailIndex", json::num(*detail as f64)),
            ("horseIndex", json::num(*r as f64)),
            ("skillId", json::num(*sid as f64)),
            ("t", json::num(*t)),
        ]));
        target_events.push(json::obj(vec![
            ("abilityTargets", ability_targets(list)),
            ("detailIndex", json::num(*detail as f64)),
            ("horseIndex", json::num(*r as f64)),
            ("skillId", json::num(*sid as f64)),
            ("t", json::num(*t)),
        ]));
    }
    let mut additional_events = Vec::new();
    for (r, sid, detail, t, list) in group_by_runner(parse_fans(&raw.fan)?) {
        additional_events.push(json::obj(vec![
            ("abilityIndexScope", Value::String("runtime_subset".to_string())),
            ("abilityTargets", ability_targets(&list)),
            ("detailIndex", json::num(detail as f64)),
            ("horseIndex", json::num(r as f64)),
            ("skillId", json::num(sid as f64)),
            ("t", json::num(t)),
        ]));
    }

    let competition_events: Vec<Value> = order_competition(&raw.comp, &strategies, n)?
        .into_iter()
        .map(|(kind, t, i)| {
            json::obj(vec![
                ("horseIndex", json::num(i as f64)),
                ("t", json::num(t)),
                ("type", json::num(kind as f64)),
            ])
        })
        .collect();
    if raw.cons.len() % 3 != 0 {
        return Err("conserve stream is ragged".to_string());
    }
    let conserve_events: Vec<Value> = raw
        .cons
        .chunks(3)
        .map(|rec| {
            json::obj(vec![
                ("horseIndex", json::num(rec[0] as f64)),
                ("t", json::num(f64w(rec[1], rec[2]))),
            ])
        })
        .collect();

    let player = root
        .get("playerHorseIndex")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as usize;
    let player_last_spurt = match raw.spfirst.get(player).copied().unwrap_or(0) {
        0 => race.finish_step_all.get(player).copied().flatten().unwrap_or(t_end),
        frame => loop_end(frame - 1) + dt_f64(),
    };

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
        (
            "experimentalLaterStaminaMechanics",
            Value::Bool(
                root.get("experimentalLaterStaminaMechanics")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
        ),
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
        ("additionalTargetEvents", json::arr(additional_events)),
        ("competitionEvents", json::arr(competition_events)),
        ("conservePowerEvents", json::arr(conserve_events)),
        ("playerLastSpurtTime", json::num(player_last_spurt)),
        ("playerHorseIndex", json::num(player as f64)),
    ]);
    if let Value::Object(map) = &mut result {
        map.insert("experimentalRetailInitialization".into(), Value::Bool(true));
        map.insert("postFinishMovementVersion".to_string(), json::num(1.0));
        if root.get("pairedRandomStreams").and_then(Value::as_bool) == Some(true) {
            map.insert("pairedRandomStreams".into(), Value::Bool(true));
        }
        if let Some(signature) = root.get("dataSignature") {
            map.insert("dataSignature".to_string(), signature.clone());
        }
    }

    // Frames: the kernel columns replay straight into the engine's own
    // ReplayFrames, then through the engine's own thinning.
    let mut frames = ReplayFrames::new(n);
    for (k, frame) in race.frames.iter().enumerate() {
        let t = loop_end(k);
        frames.capture(t, |i| frame[i]);
    }
    let retain_all = root.get("retainAllFrames").and_then(Value::as_bool) == Some(true);
    let result = encode_frames(result, frames, retain_all);
    let mut out = String::new();
    json::to_string_into(&result, &mut out);
    Ok(out)
}
