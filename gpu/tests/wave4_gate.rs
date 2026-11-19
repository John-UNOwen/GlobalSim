//! Wave-4 identity gates: the team-filtered debuff delivery and the inert
//! cross-target start-delay must replay frame-for-frame bit-exact against the
//! CPU engine on a roster carrying non-zero teams. The real merged cross-debuff
//! 910581 rides a teamed roster, and synthetic 665xxx defs force the same-team
//! drop, the different-team delivery and the inert fire, so the gate is not
//! vacuous.

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

/// One horse patch: skill list (or None to keep the request's), strategy,
/// popularity and team id overrides (or None to keep the request's).
type HorsePatch<'a> = (usize, Option<&'a [i64]>, Option<&'a str>, Option<i64>, Option<i64>);

/// The v77 base with merged skillData and the given horse patches applied.
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

/// Teams spread so the owner shares a non-zero team with some runners and not
/// others: the debuff drop removes the same-team receivers and the different
/// team and team-free runners stay delivered.
fn team_patches(owner: usize) -> Vec<HorsePatch<'static>> {
    let mut patches = Vec::new();
    for i in 0..9usize {
        let team = if i == owner || i == 1 || i == 3 {
            5
        } else if i == 2 || i == 4 {
            7
        } else {
            0
        };
        patches.push((i, None, None, None, Some(team as i64)));
    }
    patches
}

/// The real merged cross-debuff 910581 on a teamed roster: the record fires and
/// its ahead and behind speed debuffs reach a receiver set the CPU engine
/// filters by team, and the kernel must match frame for frame.
#[test]
fn wave4_real_cross_debuff_teamed_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry(10504);

    let mut horses = team_patches(0);
    horses[0].1 = Some(&[910581i64]);
    let results = patched_requests(&skill_data, &horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");

    let (fired, mut problems) = fired_ids(&results, &out);
    if !fired.contains(&910581) {
        problems.push("910581 never activated on seeds 1..3 with teams".into());
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        let tag = format!("910581 teamed seed {seed}");
        compare_race(&tag, request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} wave4 real cross-debuff problems, fired {fired:?}, problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// The synthetic battery: an ahead speed debuff, a behind speed debuff that
/// keeps self and drops same-team behind runners, a positive control delivered
/// to all candidates, and the inert cross start-delay. Every one fires on the
/// teamed back seat and matches the CPU engine frame for frame.
#[test]
fn wave4_battery_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();

    let make_def = |cond: &str, cooldown: i64, effects: &str| -> Value {
        json::parse(&format!(
            r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":{cooldown},"baseDuration":30000,"effects":[{effects}]}}]}}"#
        ))
        .expect("def parses")
    };
    let battery: [(i64, &str, &str); 4] = [
        (
            665001,
            "order>=2",
            r#"{"type":21,"target":9,"modifier":-1000,"targetValue":3}"#,
        ),
        (
            665002,
            "order>=2",
            r#"{"type":21,"target":10,"modifier":-1000,"targetValue":3}"#,
        ),
        (
            665003,
            "order>=2",
            r#"{"type":21,"target":9,"modifier":1000,"targetValue":3}"#,
        ),
        (
            665004,
            "order>=2",
            r#"{"type":10,"target":9,"modifier":0}"#,
        ),
    ];
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, cond, effects) in battery {
            sd.insert(sid.to_string(), make_def(cond, 60000, effects));
        }
    }

    let owner = 5usize;
    let mut horses = team_patches(owner);
    horses[owner].1 = Some(&[665001i64, 665002, 665003, 665004]);
    let results = patched_requests(&skill_data, &horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");

    let (fired, mut problems) = fired_ids(&results, &out);
    for sid in [665001i64, 665002, 665003, 665004] {
        if !fired.contains(&sid) {
            problems.push(format!("{sid} never activated on the teamed back seat"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(&format!("battery seed {seed}"), request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} wave4 battery problems, fired {fired:?}, problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}
