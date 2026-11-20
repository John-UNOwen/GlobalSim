//! Wave 12b: the classic lane's skill runtime. A 6696xx synthetic battery
//! files every accepted effect kind, target selection, the vision cone, the
//! cooldown re-fire, both window directions, the immunity skip, the lane
//! readers, the constructor delay and kakari folds, and the natural-decel
//! expiry. Real merged ids 200111 and 200112 (a StaminaUp pair gated by
//! track_id==10003) run on course 10307 of the track 10003 family. Every
//! scene replays frame for frame bit-exact against the CPU engine, answers
//! without a fallback, and must be seen activating in the kernel's own event
//! stream.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];

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

fn def(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

fn effect(kind: i64, target: i64, target_value: i64, debuff: bool, modifier: i64) -> String {
    format!(
        "{{\"type\":{kind},\"target\":{target},\"targetValue\":{target_value},\"additionalActivate\":0,\"valueUsage\":1,\"durationUsage\":1,\"targetIsDebuff\":{debuff},\"modifier\":{modifier}}}"
    )
}

/// The v77 request with the four retail flags cleared, an optional skillData
/// overlay, chosen horses re-rostered onto given skill ids (and an optional
/// teamId patch), on one course and seed.
fn classic_request(
    base: &str,
    course: i32,
    seed: i64,
    battery: Option<&json::Value>,
    hosts: &[(usize, &[i64], Option<i64>)],
) -> String {
    let mut value = json::parse(base).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
        if let Some(battery) = battery {
            map.insert("skillData".into(), battery.clone());
        }
        if let Some(json::Value::Array(horses)) = map.get_mut("horses") {
            for (index, ids, team) in hosts {
                if let Some(json::Value::Object(horse)) = horses.get_mut(*index) {
                    horse.insert(
                        "skills".into(),
                        json::arr(
                            ids.iter()
                                .map(|&id| {
                                    json::obj(vec![
                                        ("skillId", json::Value::Number(id as f64)),
                                        ("level", json::Value::Number(1.)),
                                    ])
                                })
                                .collect(),
                        ),
                    );
                    if let Some(team_id) = team {
                        horse.insert("teamId".into(), json::Value::Number((*team_id) as f64));
                    }
                }
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::multi_request(&text, course, seed).expect("field request")
}

struct Fired {
    ordinals: Vec<(i64, i64, i64, f64)>,
}

/// Walks the kernel's unified skillEvents stream. Tag 0 rows are numeric
/// activations (ordinal, runner, sid, t), tag 1 rows are kakari entries.
fn decode_skill_events(words: &[u32]) -> Fired {
    let mut ordinals = Vec::new();
    let mut i = 0;
    while i < words.len() {
        match words[i] {
            0 if i + 9 <= words.len() => {
                let t = f64::from_bits(
                    u64::from(words[i + 6]) << 32 | u64::from(words[i + 5]),
                );
                ordinals.push((
                    words[i + 1] as i64,
                    words[i + 2] as i64,
                    words[i + 3] as i64,
                    t,
                ));
                i += 9;
            }
            1 if i + 6 <= words.len() => i += 6,
            _ => break,
        }
    }
    Fired { ordinals }
}

fn fired_sid(fired: &Fired, sid: i64) -> bool {
    fired.ordinals.iter().any(|&(_, _, s, _)| s == sid)
}

fn sid_count(fired: &Fired, sid: i64) -> usize {
    fired.ordinals.iter().filter(|&&(_, _, s, _)| s == sid).count()
}

// The kernel's detail stream: [runner, sid, detail, tlo, thi] per activation.
fn decode_details(words: &[u32], sid: i64) -> usize {
    words.chunks_exact(5).filter(|w| w[1] as i64 == sid).count()
}

struct Scene {
    sid: i64,
    cond: &'static str,
    cooldown: i64,
    duration: i64,
    kind: i64,
    target: i64,
    target_value: i64,
    debuff: bool,
    modifier: i64,
    host: usize,
    course: i32,
    constructor: bool,
    min_fires: usize,
    min_details: usize,
}

fn battery_defs(scenes: &[Scene]) -> json::Value {
    let base = json::parse(&v77_line()).expect("parse v77");
    let mut skill_data = base
        .get("skillData")
        .cloned()
        .expect("v77 skillData");
    if let json::Value::Object(map) = &mut skill_data {
        for scene in scenes {
            let def = def(
                scene.cond,
                scene.cooldown,
                scene.duration,
                &effect(
                    scene.kind,
                    scene.target,
                    scene.target_value,
                    scene.debuff,
                    scene.modifier,
                ),
            );
            map.insert(scene.sid.to_string(), json::parse(&def).expect("battery def parses"));
        }
    }
    skill_data
}

const SCENES: [Scene; 20] = [
    // SpeedUp on self, mid-race: the stat write moves target speed.
    Scene { sid: 669601, cond: "distance_rate>=30", cooldown: 5000000, duration: 60000, kind: 1, target: 1, target_value: 0, debuff: false, modifier: 30000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // StaminaUp on self: hp capacity is NOT re-derived after construction.
    Scene { sid: 669602, cond: "distance_rate>=35", cooldown: 5000000, duration: 40000, kind: 2, target: 1, target_value: 0, debuff: false, modifier: 50000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // GutsUp on self: moves the min-speed floor and the guts drain modifier.
    Scene { sid: 669604, cond: "distance_rate>=80", cooldown: 5000000, duration: 40000, kind: 4, target: 1, target_value: 0, debuff: false, modifier: 50000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // WisdomUp on self: the subpar threshold term follows the live field.
    Scene { sid: 669605, cond: "distance_rate>=55", cooldown: 5000000, duration: 40000, kind: 5, target: 1, target_value: 0, debuff: false, modifier: 50000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // Recovery on self: hp heals on the max fraction.
    Scene { sid: 669606, cond: "distance_rate>=55", cooldown: 5000000, duration: 30000, kind: 9, target: 1, target_value: 0, debuff: false, modifier: 80000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // Accel on self: the accel receiver fold moves every later frame.
    Scene { sid: 669607, cond: "distance_rate>=40", cooldown: 5000000, duration: 30000, kind: 31, target: 1, target_value: 0, debuff: false, modifier: 20000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // Vision on self: a negative stack narrows the target-4 corridor later.
    Scene { sid: 669608, cond: "distance_rate>=45", cooldown: 5000000, duration: 60000, kind: 8, target: 1, target_value: 0, debuff: false, modifier: -30000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // TargetSpeed into the vision cone: cross deliveries queue for the flush.
    Scene { sid: 669609, cond: "distance_rate>=60", cooldown: 5000000, duration: 40000, kind: 27, target: 4, target_value: 0, debuff: false, modifier: 20000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // CurrentSpeed onto the cone: the current-speed receiver, self included.
    Scene { sid: 669610, cond: "distance_rate>=65", cooldown: 5000000, duration: 40000, kind: 21, target: 4, target_value: 0, debuff: false, modifier: 15000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // LaneMoveSpeedUp on self: feeds the lateral lane-change speed sum.
    Scene { sid: 669611, cond: "distance_rate>=50", cooldown: 5000000, duration: 40000, kind: 28, target: 1, target_value: 0, debuff: false, modifier: 20000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // PreferOutside on self: the passing score multiplies outside gaps.
    Scene { sid: 669612, cond: "distance_rate>=25", cooldown: 5000000, duration: 60000, kind: 11, target: 1, target_value: 0, debuff: false, modifier: 1, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // DebuffImmunity on self: the immune receiver check for combo scenes.
    Scene { sid: 669614, cond: "distance_rate>=20", cooldown: 5000000, duration: 90000, kind: 38, target: 1, target_value: 0, debuff: false, modifier: 1, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // MultiplyStartDelay at the gate: the constructor pass scales the delay.
    Scene { sid: 669616, cond: "distance_rate>=0", cooldown: 5000000, duration: 0, kind: 10, target: 1, target_value: 0, debuff: false, modifier: 15000, host: 0, course: 10504, constructor: true, min_fires: 1, min_details: 0 },
    // SetStartDelay at the gate: the absolute delay write.
    Scene { sid: 669617, cond: "distance_rate>=0", cooldown: 5000000, duration: 0, kind: 14, target: 1, target_value: 0, debuff: false, modifier: 25000, host: 0, course: 10504, constructor: true, min_fires: 1, min_details: 0 },
    // ModifyKakariChance at the gate: the host folds it into the rush roll.
    Scene { sid: 669618, cond: "distance_rate>=0", cooldown: 5000000, duration: 0, kind: 29, target: 1, target_value: 0, debuff: false, modifier: 50000, host: 0, course: 10504, constructor: true, min_fires: 1, min_details: 0 },
    // Cooldown re-fire. The unified stream collapses the re-activations into
    // one pending entry (lib.rs:673-678), so the non-vacuity signal is the
    // per-activation detail stream (race_solver.rs:2449).
    Scene { sid: 669619, cond: "distance_rate>=20&distance_rate<=80", cooldown: 1000, duration: 30000, kind: 31, target: 1, target_value: 0, debuff: false, modifier: 8000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 20 },
    // Both window directions: fires only between the two distance marks.
    Scene { sid: 669620, cond: "distance_rate>=65&distance_rate<=75", cooldown: 5000000, duration: 40000, kind: 22, target: 1, target_value: 0, debuff: false, modifier: 12000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // StaminaUp into the vision cone: the receiver's stat write is followed
    // by the refresh_stat_effects recompute (race_solver.rs:2644-2646).
    Scene { sid: 669621, cond: "distance_rate>=60", cooldown: 5000000, duration: 40000, kind: 2, target: 4, target_value: 0, debuff: false, modifier: 50000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // PowerUp on self, mid-race: the engine writes horse.power (race_solver.rs:2559)
    // and every delivery of a stat kind triggers the refresh recompute.
    Scene { sid: 669622, cond: "distance_rate>=45", cooldown: 5000000, duration: 40000, kind: 3, target: 1, target_value: 0, debuff: false, modifier: 20000, host: 0, course: 10504, constructor: false, min_fires: 1, min_details: 1 },
    // SetTargetLane on the emptyInMove course: the EIM branch reads the aux.
    Scene { sid: 669613, cond: "distance_rate>=55", cooldown: 5000000, duration: 40000, kind: 35, target: 1, target_value: 0, debuff: false, modifier: 10000, host: 0, course: 10508, constructor: false, min_fires: 1, min_details: 1 },
];

#[test]
fn battery_replays_bit_exact_and_activated() {
    let base = v77_line();
    let skill_data = battery_defs(&SCENES);
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut index: Vec<(i64, usize)> = Vec::new();
    for (slot, scene) in SCENES.iter().enumerate() {
        for &seed in &SEEDS {
            let request = classic_request(
                &base,
                scene.course,
                seed,
                Some(&skill_data),
                &[(scene.host, &[scene.sid], None)],
            );
            requests.push((request, seed));
            index.push((scene.sid, slot));
        }
    }
    let out = classic_exec::run(&requests).expect("classic battery run");
    let mut problems: Vec<String> = Vec::new();
    for (i, request_seed) in requests.iter().enumerate() {
        let (request, seed) = (&request_seed.0, request_seed.1);
        let (sid, slot) = index[i];
        let scene = &SCENES[slot];
        let tag = format!("sid {sid} course {} seed {seed}", scene.course);
        let race = &out.races[i];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!(
                "{tag}: tick count cpu {} gpu {}",
                cpu.frames.len(),
                race.frames.len()
            ));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
            {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "{tag}: frame mismatch tick {tick} runner {runner} column {column}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
        let fired = decode_skill_events(&race.skev);
        let count = sid_count(&fired, sid);
        if count < scene.min_fires {
            problems.push(format!("{tag}: fired {count} times, expected {}", scene.min_fires));
        }
        // skillDetailEvents count per activation (race_solver.rs:2449), so a
        // small cooldown shows up here while the unified stream collapses the
        // re-activations into one pending entry (lib.rs:673-678).
        let details = decode_details(&race.skdet, sid);
        if details < scene.min_details {
            problems.push(format!("{tag}: {details} detail events, expected {}", scene.min_details));
        }
        if scene.constructor && !fired.ordinals.iter().any(|&(_, _, s, t)| s == sid && t == 0.0) {
            problems.push(format!("{tag}: the gate skill never activated at t=0"));
        }
    }
    if !problems.is_empty() {
        panic!("{} battery problems:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn cross_team_debuff_and_immunity_replay_bit_exact() {
    let base = v77_line();
    let mut skill_data = json::parse(&base).expect("parse v77")
        .get("skillData")
        .cloned()
        .expect("v77 skillData");
    if let json::Value::Object(map) = &mut skill_data {
        let immune = def("distance_rate>=20", 5000000, 90000, &effect(38, 1, 0, false, 1));
        map.insert("669614".into(), json::parse(&immune).expect("immune def"));
        let debuff =
            def("distance_rate>=70", 5000000, 40000, &effect(27, 9, 0, true, -20000));
        map.insert("669615".into(), json::parse(&debuff).expect("debuff def"));
    }
    let mut requests: Vec<(String, i64)> = Vec::new();
    for &seed in &SEEDS {
        requests.push((
            classic_request(
                &base,
                10504,
                seed,
                Some(&skill_data),
                &[(0, &[669614i64], None), (3, &[669615i64], Some(1))],
            ),
            seed,
        ));
    }
    let out = classic_exec::run(&requests).expect("immunity run");
    let mut problems: Vec<String> = Vec::new();
    for (i, (request, seed)) in requests.iter().enumerate() {
        let race = &out.races[i];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("seed {seed}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("seed {seed}: tick count mismatch"));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
            {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "seed {seed}: frame mismatch tick {tick} runner {runner} column {column}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
        let fired = decode_skill_events(&race.skev);
        if !fired_sid(&fired, 669614) || !fired_sid(&fired, 669615) {
            problems.push(format!("seed {seed}: the immunity pair never activated"));
        }
    }
    if !problems.is_empty() {
        panic!("{} immunity problems:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn real_merged_ids_activate_and_replay_bit_exact() {
    let base = v77_line();
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut sids: Vec<i64> = Vec::new();
    for &sid in &[200111i64, 200112, 0] {
        for &seed in &SEEDS {
            let hosts: &[(usize, &[i64], Option<i64>)] =
                if sid == 0 { &[] } else { &[(0, &[sid], None)] };
            requests.push((classic_request(&base, 10307, seed, None, hosts), seed));
            sids.push(sid);
        }
    }
    let out = classic_exec::run(&requests).expect("real id run");
    let mut problems: Vec<String> = Vec::new();
    for (i, ((request, seed), sid)) in requests.iter().zip(sids.iter()).enumerate() {
        let race = &out.races[i];
        let tag = format!("sid {sid} seed {seed}");
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count mismatch"));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
            {
                for (column, (cpu_val, gpu_val)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu_val.to_bits() != gpu_val.to_bits() {
                        problems.push(format!(
                            "{tag}: frame mismatch tick {tick} runner {runner} column {column}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
        let fired = decode_skill_events(&race.skev);
        if *sid != 0 && !fired.ordinals.iter().any(|&(_, _, s, t)| s == *sid && t == 0.0) {
            problems.push(format!("{tag}: the track-gated id never activated at t=0"));
        }
    }
    if !problems.is_empty() {
        panic!("{} real id problems:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn eight_distinct_ids_on_one_runner_are_not_refused() {
    let base = v77_line();
    let skill_data = battery_defs(&SCENES);
    let ids = [
        669601i64, 669602, 669604, 669605, 669606, 669607, 669608, 669612,
    ];
    let mut requests: Vec<(String, i64)> = Vec::new();
    for &seed in &SEEDS {
        requests.push((
            classic_request(&base, 10504, seed, Some(&skill_data), &[(0, &ids, None)]),
            seed,
        ));
    }
    let out = classic_exec::run(&requests).expect("eight id run");
    let mut problems: Vec<String> = Vec::new();
    for (i, (request, seed)) in requests.iter().enumerate() {
        let race = &out.races[i];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("seed {seed}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("seed {seed}: tick count mismatch"));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
            {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "seed {seed}: frame mismatch tick {tick} runner {runner} column {column}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
        let fired = decode_skill_events(&race.skev);
        for id in &ids {
            if !fired_sid(&fired, *id) {
                problems.push(format!("seed {seed}: sid {id} never activated"));
            }
        }
    }
    if !problems.is_empty() {
        panic!("{} eight id problems:\n{}", problems.len(), problems.join("\n"));
    }
}
