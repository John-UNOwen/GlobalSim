//! Wave-6b identity gates: the activation-time value usages, the
//! additionalActivate turn machinery and the random-skill reservations. The
//! real merged ids 910071 (kind 37), 910211/910351 (aa 3/2), 202031/202032
//! (value_usage 8), 910341 (11) and 100981 (14) run beside a 667xxx battery
//! for kind 36, value_usage 9 and the deeper aa shapes. Every row replays
//! frame-for-frame bit-exact against the CPU engine on seeds 1..3, and each
//! construct must be seen activating there.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn merged_skill_data() -> json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn v77_request() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

fn compare_race(
    tag: &str,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
    problems: &mut Vec<String>,
) {
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{tag}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in
        cpu.frames.iter().zip(race.frames.iter()).enumerate()
    {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
            for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cv.to_bits() != gv.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                        COLUMNS[column],
                        cv.to_bits(),
                        gv.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
    for (runner, cpu) in cpu.finish_times.iter().enumerate() {
        let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
        if cpu.to_bits() != gpu.to_bits() {
            problems.push(format!(
                "{tag} runner {runner} finishTimeRaw: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                cpu.to_bits(),
                gpu.to_bits()
            ));
        }
    }
}

type HorsePatch<'a> = (usize, Option<&'a [i64]>, Option<&'a str>, Option<i64>, Option<i64>);

fn patched_requests(
    skill_data: &Value,
    horses: &[HorsePatch],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("skillData".into(), skill_data.clone());
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids, strategy, pop, team) in horses {
            let json::Value::Object(horse) = &mut entries[*index] else { unreachable!() };
            if let Some(ids) = ids {
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
            }
            if let Some(style) = strategy {
                horse.insert("strategy".into(), json::Value::String((*style).into()));
            }
            if let Some(popularity) = pop {
                horse.insert(
                    "popularity".into(),
                    json::Value::Number((*popularity) as f64),
                );
            }
            if let Some(team_id) = team {
                horse.insert("teamId".into(), json::Value::Number((*team_id) as f64));
            }
        }
    }
    let mut requests = Vec::new();
    for &seed in seeds {
        let mut request = value.clone();
        let json::Value::Object(map) = &mut request else { unreachable!() };
        map.insert("seed".into(), json::Value::Number(seed as f64));
        let mut text = String::new();
        json::to_string_into(&request, &mut text);
        requests.push((text, seed));
    }
    requests
}

fn fired_ids(results: &[(String, i64)], out: &race_exec::RaceRun) -> (Vec<i64>, Vec<String>) {
    let mut fired: Vec<i64> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in results.iter().enumerate() {
        if let Some(reason) = &out.races[index].fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => {
                for (_, sid, _, _) in &cpu.detail_events {
                    if !fired.contains(sid) {
                        fired.push(*sid);
                    }
                }
            }
            Err(reason) => problems.push(format!("seed {seed}: cpu race failed: {reason}")),
        }
    }
    fired.sort_unstable();
    (fired, problems)
}

fn run_gate(
    label: &str,
    skill_data: &Value,
    horses: &[HorsePatch],
    expect_fired: &[i64],
) {
    let results = patched_requests(skill_data, horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");
    let (fired, mut problems) = fired_ids(&results, &out);
    for sid in expect_fired {
        if !fired.contains(sid) {
            problems.push(format!("{sid} never activated on seeds 1..3"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(&format!("{label} seed {seed}"), request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} {label} problems, fired {fired:?}, problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// Adds battery definitions to the merged table.
fn with_battery(skill_data: &mut Value, defs: &[(i64, &str)]) {
    let json::Value::Object(sd) = skill_data else { unreachable!() };
    for (sid, def) in defs {
        sd.insert(sid.to_string(), json::parse(def).expect("battery def parses"));
    }
}

fn def(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

/// 910071 files a gold request at the half-distance point. The gold generals
/// 200331/200361 ride along as the pickable pool, and a wisdom-failed one
/// stays in the pool because a reservation overrides the failed lottery.
#[test]
fn wave6b_real_random_gold_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let hosts: [(usize, &[i64]); 3] = [
        (0, &[910071i64, 200331, 200361]),
        (4, &[910071, 200331, 200361]),
        (8, &[910071, 200331, 200361]),
    ];
    let horses: Vec<HorsePatch> =
        hosts.iter().map(|(i, ids)| (*i, Some(*ids), None, None, None)).collect();
    run_gate("910071 random gold", &skill_data, &horses, &[910071]);
}

/// Kind 36 has no merged id. 667020 files the any-request at phase 1, and
/// 667021 can only ever activate through the reservation its pick leaves
/// behind, so the pair proves the white-inclusive pool and its draw.
#[test]
fn wave6b_battery_random_any_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (667020, &def("phase>=1", 5000000, 30000, r#"{"type":36,"target":1,"modifier":10000}"#)
                .as_str()),
            (
                667021,
                &def("order>=99", 60000, 30000, r#"{"type":27,"target":1,"modifier":800}"#).as_str(),
            ),
            (
                667022,
                &def("order>=99", 60000, 30000, r#"{"type":31,"target":1,"modifier":600}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[667020i64, 667021, 667022][..]), None, None, None),
        (3, Some(&[667020, 667021][..]), None, None, None),
        (6, Some(&[667020, 667022][..]), None, None, None),
    ];
    run_gate("667020 random any", &skill_data, &horses, &[667020, 667021, 667022]);
}

/// 910211 defers one acceleration per turn behind the aa 3 cap, and 910351
/// defers behind the aa 2 cap after the final-corner window. Inert carriers
/// lift activate_count_middle past 910211's arm.
#[test]
fn wave6b_real_additional_activate_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                667120,
                &def("always>=0", 8000, 0, r#"{"type":10,"target":1,"modifier":0}"#).as_str(),
            ),
            (
                667121,
                &def("always>=0", 9000, 0, r#"{"type":10,"target":1,"modifier":0}"#).as_str(),
            ),
        ],
    );
    let hosts: [(usize, &[i64]); 4] = [
        (0, &[910211i64, 667120, 667121]),
        (5, &[910211, 667120, 667121]),
        (2, &[910351i64]),
        (7, &[910351]),
    ];
    let horses: Vec<HorsePatch> =
        hosts.iter().map(|(i, ids)| (*i, Some(*ids), None, None, None)).collect();
    run_gate("aa 3/2 real ids", &skill_data, &horses, &[910211, 910351]);
}

/// 667001 queues two aa-3 follow-ups, one turn delivers both. 667002 queues
/// four aa-2 follow-ups, so the cap strands one and the running entry never
/// retires. 667003 queues per overtake event behind the aa-1 cap.
#[test]
fn wave6b_battery_aa_turns_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                667001,
                &def(
                    "phase>=2",
                    60000,
                    30000,
                    concat!(
                        r#"{"type":27,"target":1,"modifier":1000},"#,
                        r#"{"type":27,"target":1,"modifier":500,"additionalActivate":3},"#,
                        r#"{"type":31,"target":1,"modifier":500,"additionalActivate":3}"#
                    ),
                )
                .as_str(),
            ),
            (
                667002,
                &def(
                    "phase>=2",
                    60000,
                    30000,
                    concat!(
                        r#"{"type":27,"target":1,"modifier":1000},"#,
                        r#"{"type":27,"target":1,"modifier":300,"additionalActivate":2},"#,
                        r#"{"type":31,"target":1,"modifier":300,"additionalActivate":2},"#,
                        r#"{"type":27,"target":1,"modifier":300,"additionalActivate":2},"#,
                        r#"{"type":31,"target":1,"modifier":300,"additionalActivate":2}"#
                    ),
                )
                .as_str(),
            ),
            (
                667003,
                &def(
                    "phase>=1",
                    60000,
                    30000,
                    concat!(
                        r#"{"type":27,"target":1,"modifier":1000},"#,
                        r#"{"type":31,"target":1,"modifier":400,"additionalActivate":1},"#,
                        r#"{"type":27,"target":1,"modifier":400,"additionalActivate":1}"#
                    ),
                )
                .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (1, Some(&[667001i64][..]), None, None, None),
        (4, Some(&[667002i64][..]), None, None, None),
        (7, Some(&[667003i64][..]), None, None, None),
    ];
    run_gate("aa battery turns", &skill_data, &horses, &[667001, 667002, 667003]);
}

/// The merged runtime readers: 202031 and 202032 roll the usage-8 coefficient
/// at activation, and 910341 multiplies by the late-corner overtake count.
#[test]
fn wave6b_real_runtime_value_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let hosts: [(usize, &[i64]); 6] = [
        (0, &[202031i64]),
        (1, &[202032i64]),
        (2, &[202032]),
        (3, &[910341i64]),
        (5, &[910341]),
        (7, &[910341]),
    ];
    let horses: Vec<HorsePatch> =
        hosts.iter().map(|(i, ids)| (*i, Some(*ids), None, None, None)).collect();
    run_gate("vu 8/11 real ids", &skill_data, &horses, &[202031, 202032, 910341]);
}

/// 100981 scales its usage-14 accelerations by the count of activated
/// green-tag skill ids, so its host also carries the battery trio 667012 to
/// 667014, three tagged ids that each activate once. 667010 pins the usage-9
/// roll, which has no real id behind it.
#[test]
fn wave6b_real_green_count_and_battery_roll_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    let tagged = |cond: &str| format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"tags\":[601],\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":60000,\"baseDuration\":30000,\"effects\":[{{\"type\":27,\"target\":1,\"modifier\":600}}]}}]}}"
    );
    with_battery(
        &mut skill_data,
        &[
            (667012, &tagged("phase>=1").as_str()),
            (667013, &tagged("phase>=2").as_str()),
            (667014, &tagged("phase>=3").as_str()),
            (
                667010,
                &def("phase>=1", 15000, 30000,
                    r#"{"type":9,"target":1,"modifier":100000,"valueUsage":9}"#).as_str(),
            ),
        ],
    );
    let hosts: [(usize, &[i64]); 4] = [
        (1, &[100981i64, 667012, 667013, 667014, 667010]),
        (3, &[100981, 667012, 667013, 667014, 667010]),
        (5, &[100981, 667012, 667013, 667014]),
        (7, &[667010]),
    ];
    let horses: Vec<HorsePatch> =
        hosts.iter().map(|(i, ids)| (*i, Some(*ids), None, None, None)).collect();
    run_gate("vu 9/14", &skill_data, &horses, &[100981, 667010, 667012, 667013, 667014]);
}


