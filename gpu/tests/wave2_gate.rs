//! Wave-2 identity gates: the dynamic condition fields ported in this wave
//! must compile from the real merged skill set and replay frame-for-frame
//! bit-exact against the CPU engine, and each construct must fire through
//! synthetic defs so the gate is not vacuous.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

/// One real merged skill per ported construct, all confirmed to compile by
/// the scratch sweep. The otherself family has no compiling real skill (all
/// twelve users carry an unported target-another-runner effect), so the
/// firing gate covers that fold instead. distance_rate_after_random keeps
/// only 202122: the other users hit an unported effect type 37 or a rescaled
/// value_usage (8, 10, 12).
const REAL_SKILLS: [(i64, &str); 10] = [
    (910561, "overtake_target_time"),
    (910521, "distance_diff_top"),
    (202122, "distance_rate_after_random"),
    (910061, "activate_count_heal"),
    (900191, "change_order_up_end_after"),
    (900361, "lane_type"),
    (900341, "compete_fight_count"),
    (900121, "is_behind_in"),
    (201142, "is_move_lane"),
    (200442, "blocked_front_continuetime"),
];

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

/// The v77 base with merged skillData, horse 0's skill list replaced.
fn patched_requests(
    skill_data: &Value,
    horse_skills: &[(usize, Vec<i64>)],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("skillData".into(), skill_data.clone());
        let json::Value::Array(horses) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids) in horse_skills {
            let json::Value::Object(horse) = &mut horses[*index] else { unreachable!() };
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

/// One real merged skill per ported construct on horse 0 of the v77 roster:
/// zero CPU fallback and frame-for-frame identity over the seeds.
#[test]
fn wave2_real_skills_replay_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry(10504);

    let mut problems: Vec<String> = Vec::new();
    let mut fired_report: Vec<String> = Vec::new();
    for (sid, what) in REAL_SKILLS {
        let results = patched_requests(&skill_data, &[(0, vec![sid])], &SEEDS);
        let out = race_exec::run(&results).expect("kernel run");
        let (fired, mut fires) = fired_ids(&results, &out);
        problems.append(&mut fires);
        fired_report.push(format!("{sid} ({what}) fired {fired:?}"));
        for (index, (request, seed)) in results.iter().enumerate() {
            let tag = format!("{sid} ({what}) seed {seed}");
            compare_race(&tag, request, *seed, &out.races[index], &mut problems);
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} wave2 real-skill problems, firing report:\n{}\nproblems:\n{}",
            problems.len(),
            fired_report.join("\n"),
            problems.join("\n")
        );
    }
}

/// Each ported construct must actually gate an activation. The synthetic
/// defs (662xxx) sit on v77 horses with thresholds the race crosses, and the
/// whole field stays bit-exact against the CPU engine.
#[test]
fn wave2_constructs_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();

    let template_alt = {
        let merged = merged_skill_data();
        merged
            .get("200361")
            .expect("merged 200361")
            .get("alternatives")
            .and_then(json::Value::as_array)
            .expect("alternatives")[0]
            .clone()
    };
    let make_def = |cond: &str, recovery: bool| -> Value {
        let mut alt = template_alt.clone();
        if recovery {
            let json::Value::Object(map) = &mut alt else { unreachable!() };
            let json::Value::Array(effects) = map.get_mut("effects").expect("effects") else {
                unreachable!()
            };
            let json::Value::Object(effect) = &mut effects[0] else { unreachable!() };
            effect.insert("type".into(), json::Value::Number(9.));
            effect.insert("modifier".into(), json::Value::Number(0.05));
        }
        let json::Value::Object(map) = &mut alt else { unreachable!() };
        map.insert("precondition".into(), json::Value::String(String::new()));
        map.insert("condition".into(), json::Value::String(cond.into()));
        json::obj(vec![
            ("rarity", json::Value::Number(1.)),
            ("wisdomCheck", json::Value::Number(0.)),
            ("priority", json::Value::Number(0.)),
            ("alternatives", json::arr(vec![alt])),
        ])
    };
    let battery: [(i64, &str, bool); 13] = [
        (662001, "distance_diff_top<=1", false),
        (662002, "lane_type==0", false),
        (662003, "is_behind_in==1", false),
        (662004, "is_move_lane==1", false),
        (662005, "blocked_front_continuetime>=1", false),
        (662006, "overtake_target_time>=1", false),
        (662007, "change_order_up_end_after>=1", false),
        (662008, "compete_fight_count>0", false),
        (662009, "activate_count_heal>=1", false),
        (662010, "order_rate>=0", true),
        (662011, "distance_rate_after_random==50", false),
        (662012, "running_style_count_senko_otherself>=1", false),
        (662013, "running_style_count_senko_otherself>=9", false),
    ];
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, cond, recovery) in battery {
            sd.insert(sid.to_string(), make_def(cond, recovery));
        }
    }
    let probes = [662001i64, 662002, 662003, 662004, 662005, 662006, 662007, 662011, 662012];
    let carriers = [662008i64, 662009, 662010, 662013];
    let mut horse0: Vec<i64> =
        probes.iter().filter(|s| **s != 662006).copied().chain(carriers).collect();
    horse0.sort_unstable();
    let results = patched_requests(
        &skill_data,
        &[(0, horse0.clone()), (4, horse0), (8, vec![662006])],
        &SEEDS,
    );
    let out = race_exec::run(&results).expect("kernel run");

    let (fired, mut problems) = fired_ids(&results, &out);
    // The negative control must stay unfired on both engines.
    if fired.contains(&662013) {
        problems.push("662013 fired: the otherself count is wrong".into());
    }
    let required: [(&str, &[i64]); 11] = [
        ("overtake_target_time", &[662006]),
        ("distance_diff_top", &[662001]),
        ("distance_rate_after_random", &[662011]),
        ("activate_count_heal", &[662009]),
        ("change_order_up_end_after", &[662007]),
        ("lane_type", &[662002]),
        ("compete_fight_count", &[662008]),
        ("is_behind_in", &[662003]),
        ("is_move_lane", &[662004]),
        ("blocked_front_continuetime", &[662005]),
        ("running_style_count_otherself", &[662012]),
    ];
    for (what, ids) in required {
        if !ids.iter().any(|id| fired.contains(id)) {
            problems.push(format!("{what}: no synthetic firing (ids {ids:?})"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        let tag = format!("battery seed {seed}");
        compare_race(&tag, request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} battery problems, cpu fired {:?}:\n{}",
            problems.len(),
            fired,
            problems.join("\n")
        );
    }
}
