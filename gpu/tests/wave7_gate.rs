//! Wave-7 identity gates: the mid-race stat fires. Self WisdomUp behind a
//! phase gate, a static bump chained into a loop fire, cross SpeedUp and
//! WisdomUp delivery with the per-receiver refresh, the debuff filters and a
//! repeat-fire cooldown cycle, all frame-for-frame bit-exact on seeds 1..3.

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

/// 668001 lifts its own wisdom at phase 1 and every 15 seconds after, 668002
/// drops it at phase 2 with a positive instance duration. The live field
/// feeds the pace and downhill draws each tick, while the spurt plan keeps
/// its capture until a refresh.
#[test]
fn wave7_battery_wisdom_self_mid_refire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                668001,
                &def("phase>=1", 15000, -1, r#"{"type":5,"target":1,"modifier":50000}"#).as_str(),
            ),
            (
                668002,
                &def("phase>=2", 60000, 30000,
                    r#"{"type":5,"target":1,"modifier":-100000}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[668001i64][..]), None, None, None),
        (5, Some(&[668002i64][..]), None, None, None),
    ];
    run_gate("wisdom self mid", &skill_data, &horses, &[668001, 668002]);
}

/// 668030 is a construction-time bump, 668031 fires at phase 1 on the same
/// runner. The loop fire must continue from the unclamped f32 accumulator
/// the bump left behind, not from the clamped field.
#[test]
fn wave7_battery_static_chain_into_loop_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                668030,
                &def("always>=0", 0, -1, r#"{"type":5,"target":1,"modifier":30000}"#).as_str(),
            ),
            (
                668031,
                &def("phase>=1", 40000, -1, r#"{"type":5,"target":1,"modifier":20000}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[668030i64, 668031][..]), None, None, None),
        (6, Some(&[668031, 668030][..]), None, None, None),
    ];
    run_gate("static chain into loop fire", &skill_data, &horses, &[668030, 668031]);
}

/// Cross wisdom fires: 668010 hits the runner ahead, 668011 the field behind
/// through the self-inclusive fan, 668012 the visible fan. Each receiver
/// carries the stat write, then the receiver refresh that flips the target
/// base to the legacy form and re-rounds HP.
#[test]
fn wave7_battery_cross_wisdom_fans_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                668010,
                &def("phase>=1", 20000, 30000, r#"{"type":5,"target":9,"modifier":40000}"#)
                    .as_str(),
            ),
            (
                668011,
                &def("phase>=1", 30000, -1, r#"{"type":5,"target":10,"modifier":20000}"#)
                    .as_str(),
            ),
            (
                668012,
                &def("phase>=2", 60000, 30000, r#"{"type":5,"target":4,"modifier":30000}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[668010i64, 668011, 668012][..]), None, None, None),
        (5, Some(&[668011][..]), None, None, None),
        (8, Some(&[668010][..]), None, None, None),
    ];
    run_gate("cross wisdom fans", &skill_data, &horses, &[668010, 668011, 668012]);
}

/// Cross SpeedUp flips the receiver's target base from the retail form to
/// the legacy one and repatches its spurt maximum from the bumped speed.
/// 668021 fans behind including the owner, so the owner refreshes itself.
#[test]
fn wave7_battery_cross_speedup_flip_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                668020,
                &def("phase>=1", 20000, -1, r#"{"type":1,"target":9,"modifier":40000}"#).as_str(),
            ),
            (
                668021,
                &def("phase>=1", 30000, 30000, r#"{"type":1,"target":10,"modifier":20000}"#)
                    .as_str(),
            ),
            (
                668022,
                &def("phase>=2", 60000, -1, r#"{"type":1,"target":4,"modifier":-20000}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (1, Some(&[668020i64, 668021, 668022][..]), None, None, None),
        (7, Some(&[668021][..]), None, None, None),
        (3, Some(&[668022, 668020][..]), None, None, None),
    ];
    run_gate("cross speedup flip", &skill_data, &horses, &[668020, 668021, 668022]);
}

/// 668050 re-fires every 10 seconds at the runner ahead. The target base and
/// spurt maximum rewrite are deterministic, the HP re-round churns on every
/// (effect, receiver) pair, so the repeat cycle pins both.
#[test]
fn wave7_battery_speedup_refire_cycle_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[(
            668050,
            &def("phase>=1", 10000, -1, r#"{"type":1,"target":9,"modifier":10000}"#).as_str(),
        )],
    );
    let horses: Vec<HorsePatch> = vec![
        (2, Some(&[668050i64][..]), None, None, None),
        (8, Some(&[668050][..]), None, None, None),
    ];
    run_gate("speedup refire cycle", &skill_data, &horses, &[668050]);
}

/// The debuff filters on a negative cross fire: 668042 aims the visible fan,
/// the immune runner 668041 drops the hit at selection and the same-team
/// runner behind team 3 drops with it, while the rest of the field takes the
/// write and the refresh.
#[test]
fn wave7_battery_negative_debuff_filters_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                668042,
                &def("phase>=1", 20000, 30000, r#"{"type":5,"target":4,"modifier":-30000}"#)
                    .as_str(),
            ),
            (
                668041,
                &def("always>=0", 900000, 120000, r#"{"type":38,"target":1,"modifier":1}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (2, Some(&[668042i64][..]), None, None, Some(3)),
        (3, Some(&[668042][..]), None, None, Some(3)),
        (4, Some(&[668041, 668042][..]), None, None, Some(3)),
    ];
    run_gate("negative debuff filters", &skill_data, &horses, &[668041, 668042]);
}

/// The real kind-1 loop fire 201641 (same_skill_horse_count, self bump) and
/// 668060, a construction-pass SpeedUp probe whose fire the event stream
/// cannot see. Frame identity pins the constructor ordering either way.
#[test]
fn wave7_real_kind1_loop_and_constructor_probe_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[(
            668060,
            &def("always>=0", 0, -1, r#"{"type":1,"target":1,"modifier":400000}"#).as_str(),
        )],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[201641i64][..]), None, None, None),
        (4, Some(&[668060][..]), None, None, None),
    ];
    run_gate("real kind 1 and constructor probe", &skill_data, &horses, &[201641]);
}
