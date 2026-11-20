//! Wave-9 identity gates: the duration_usage 4 running instance extended on
//! every overtake, the additionalActivate fan that delivers to other runners,
//! and the value_usage tiers the engine has no arm for. The merged table's two
//! real duration_usage 4 ids 100531 and 900531 ride with the 6691xx battery,
//! the cross-target and self turn shapes with 6692xx, and the pass-through
//! usages with 6693xx. Every row replays frame-for-frame bit-exact against the
//! CPU engine on seeds 1..3, and each construct must activate there.

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

/// The same alternative with the wire durationUsage the battery needs.
fn def_du(
    cond: &str,
    cooldown: i64,
    base_duration: i64,
    duration_usage: i64,
    effects: &str,
) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"durationUsage\":{duration_usage},\"effects\":[{effects}]}}]}}"
    )
}

/// 100531 (gold, wisdomCheck 0, baseDuration 20000) and 900531 (its white
/// twin, baseDuration 12000) are the merged table's only duration_usage 4 ids:
/// one alternative, two self kind-31 effects, the second carrying
/// additionalActivate 1, armed only in the final corner from the 50% order
/// mark. Each host rides 100541, a self acceleration that lifts the field
/// around the host so the final-corner order window and the rank changes the
/// extension needs both arrive. 669101 is the same duration_usage 4 with no
/// delayed effect, 669102 extends an additionalActivate 1 sibling, 669103
/// combines it with a kind-2 turn and a short cooldown, and 669148 extends a
/// cross-target effect delivered to the other runners. The cross-target
/// delayed delivery of 669104 rides in `fan_churn_tracks_the_engine`, which is
/// ignored: a duration_usage 4 field and a refanning cross-target delivery
/// together drift off the engine inside 30 s of race time.
#[test]
fn wave9_delayed_instance_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669101,
                &def_du(
                    "phase>=2",
                    5000000,
                    20000,
                    4,
                    r#"{"type":31,"target":1,"modifier":900}"#,
                )
                .as_str(),
            ),
            (
                669102,
                &def_du(
                    "phase>=1",
                    5000000,
                    30000,
                    4,
                    concat!(
                        r#"{"type":31,"target":1,"modifier":700},"#,
                        r#"{"type":27,"target":1,"modifier":500,"additionalActivate":1}"#
                    ),
                )
                .as_str(),
            ),
            (
                669103,
                &def_du(
                    "phase>=2",
                    8000,
                    20000,
                    4,
                    concat!(
                        r#"{"type":31,"target":1,"modifier":400},"#,
                        r#"{"type":31,"target":1,"modifier":300,"additionalActivate":2}"#
                    ),
                )
                .as_str(),
            ),
            (
                669104,
                &def_du(
                    "phase>=2",
                    5000000,
                    20000,
                    4,
                    r#"{"type":31,"target":9,"targetValue":2,"modifier":-600,"additionalActivate":1}"#,
                )
                .as_str(),
            ),
            (
                669148,
                &def_du(
                    "phase>=2",
                    5000000,
                    20000,
                    4,
                    r#"{"type":31,"target":9,"targetValue":2,"modifier":-600}"#,
                )
                .as_str(),
            ),
            (
                669141,
                &def("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#)
                    .as_str(),
            ),
        ],
    );
    let hosts: [(usize, &[i64]); 8] = [
        (0, &[100531i64, 669141]),
        (2, &[100531, 669141]),
        (4, &[900531i64, 669141]),
        (6, &[900531, 669141]),
        (1, &[669101i64, 669141]),
        (3, &[669102i64, 669141]),
        (5, &[669103i64, 669141]),
        (7, &[669148i64, 669141]),
    ];
    let horses: Vec<HorsePatch> =
        hosts.iter().map(|(i, ids)| (*i, Some(*ids), None, None, None)).collect();
    run_gate(
        "duration_usage 4",
        &skill_data,
        &horses,
        &[100531, 900531, 669101, 669102, 669103, 669148],
    );
}

/// No merged id carries additionalActivate outside its own runner: every aa row
/// in the table is a self effect. The battery covers the fan the engine allows,
/// the ahead, behind and visible windows, the natural-deceleration kind whose
/// expiry impulse drains in reverse order, and the same-team drop a debuff
/// delivery takes. 669205 is the self shape on a kind the earlier wave refused.
#[test]
fn wave9_additional_activate_fan_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669201,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":27,"target":9,"targetValue":3,"modifier":-1000,"additionalActivate":1}"#,
                )
                .as_str(),
            ),
            (
                669202,
                &def(
                    "phase>=1",
                    5000000,
                    30000,
                    r#"{"type":31,"target":10,"targetValue":4,"modifier":800,"additionalActivate":2}"#,
                )
                .as_str(),
            ),
            (
                669203,
                &def(
                    "phase>=2",
                    5000000,
                    25000,
                    r#"{"type":28,"target":4,"targetValue":2,"modifier":-500,"additionalActivate":3}"#,
                )
                .as_str(),
            ),
            (
                669204,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":22,"target":9,"targetValue":2,"modifier":-300,"additionalActivate":1}"#,
                )
                .as_str(),
            ),
            (
                669205,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":21,"target":1,"modifier":500,"additionalActivate":2}"#,
                )
                .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669201i64][..]), None, None, Some(7)),
        (1, Some(&[669205i64][..]), None, None, Some(7)),
        (2, Some(&[669202i64][..]), None, None, Some(7)),
        (3, Some(&[669203i64][..]), None, None, None),
        (4, Some(&[669204i64][..]), None, None, None),
        (5, Some(&[669201i64][..]), None, None, None),
        (6, Some(&[669202i64][..]), None, None, None),
        (7, Some(&[669203i64][..]), None, None, None),
    ];
    run_gate(
        "additionalActivate fan",
        &skill_data,
        &horses,
        &[669201, 669202, 669203, 669204, 669205],
    );
}

/// value_usage outside the tiers the engine reads: the builder folds
/// 3..=7, 10, 12 and 13, the activation reads 8, 9 and 11, and
/// activation_modifier 14, so every other usage reaches no arm and the wire
/// value lands as it is, self target or delivered. 669305 rides i32::MAX, the
/// value the engine's saturating cast lands on for a usage past i32.
#[test]
fn wave9_value_usage_passthrough_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669301,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":31,"target":1,"valueUsage":15,"modifier":1200}"#,
                )
                .as_str(),
            ),
            (
                669302,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":27,"target":9,"targetValue":2,"valueUsage":0,"modifier":-1000}"#,
                )
                .as_str(),
            ),
            (
                669303,
                &def(
                    "phase>=1",
                    5000000,
                    30000,
                    r#"{"type":22,"target":1,"valueUsage":16,"modifier":600}"#,
                )
                .as_str(),
            ),
            (
                669304,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":31,"target":1,"valueUsage":-2,"modifier":900}"#,
                )
                .as_str(),
            ),
            (
                669305,
                &def(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":28,"target":1,"valueUsage":2147483647,"modifier":700}"#,
                )
                .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669301i64][..]), None, None, None),
        (2, Some(&[669302i64][..]), None, None, None),
        (3, Some(&[669303i64][..]), None, None, None),
        (5, Some(&[669304i64][..]), None, None, None),
        (7, Some(&[669305i64][..]), None, None, None),
    ];
    run_gate(
        "value_usage pass-through",
        &skill_data,
        &horses,
        &[669301, 669302, 669303, 669304, 669305],
    );
}

/// Six hosts refanning one cross-target additionalActivate 1 effect on short
/// cooldowns. The pool holds one entry per receiver per delivery, so a
/// receiver carries dozens at once. The scene pins two pool semantics the
/// single-fire shapes cannot reach: retail delivery reaches a receiver with a
/// higher roster index than the host before that receiver's prepare pass
/// steps its timers, so the copy takes its first timer step on the delivery
/// tick and expires a tick before a copy to a lower-indexed receiver, and the
/// three-delivery cap resets with every new running instance. The ten-seed
/// sweep of this scene lives in `wave9_fan_trace::fan_churn_ten_seeds_bit_exact`.
#[test]
fn fan_churn_tracks_the_engine() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669147,
                &def(
                    "phase>=1",
                    9000,
                    9000,
                    r#"{"type":31,"target":9,"targetValue":2,"modifier":-350,"additionalActivate":1}"#,
                )
                .as_str(),
            ),
            (
                669141,
                &def("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669147i64, 669141][..]), None, None, None),
        (1, Some(&[669147, 669141][..]), None, None, None),
        (2, Some(&[669147, 669141][..]), None, None, None),
        (3, Some(&[669141][..]), None, None, None),
        (4, Some(&[669147, 669141][..]), None, None, None),
        (5, Some(&[669147, 669141][..]), None, None, None),
        (6, Some(&[669141][..]), None, None, None),
        (7, Some(&[669147, 669141][..]), None, None, None),
    ];
    run_gate("fan churn", &skill_data, &horses, &[669147]);
}
