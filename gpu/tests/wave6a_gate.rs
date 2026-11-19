//! Wave-6a identity gates on the merged table: the shared-cooldown pair
//! 910151, the SetTargetLane pair 201261, the ExtendKakari 200801, plus a
//! 666xxx battery forcing every direction of the five new effect kinds and
//! the immune survival of a same-team drop. Every row replays frame-for-frame
//! bit-exact against the CPU engine on seeds 1..3, and each construct must be
//! seen activating there.

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

/// 910151's two live alternatives share one cooldown key. Inert self-firing
/// carriers lift activate_count_all past the twin split before the 400 m
/// window, and three seats cover the order_rate arms and the wisdom lottery.
#[test]
fn wave6a_real_shared_cooldown_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for sid in [666100i64, 666101, 666102] {
            sd.insert(
                sid.to_string(),
                json::parse(
                    r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{"precondition":"","condition":"always>=0","cooldown":8000,"baseDuration":30000,"effects":[{"type":10,"target":1,"modifier":0}]}]}"#,
                )
                .expect("carrier parses"),
            );
        }
    }
    let hosts: [(usize, &[i64]); 3] = [
        (0, &[910151i64, 666100, 666101, 666102]),
        (3, &[910151, 666100, 666101, 666102]),
        (6, &[910151, 666100, 666101, 666102]),
    ];
    let horses: Vec<HorsePatch> = hosts
        .iter()
        .map(|(i, ids)| (*i, Some(*ids), None, None, None))
        .collect();
    run_gate("910151 shared key", &skill_data, &horses, &[910151]);
}

/// 201261 arms a fixed 0.5 lane goal through its blocked early-phase
/// condition; SetTargetLane presence moves the lane target, so the lane
/// column tracks it. Three nige seats give the blocked_front/blocked_side
/// arms a shot at holding.
#[test]
fn wave6a_real_set_lane_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let horses: Vec<HorsePatch> = (0..3)
        .map(|i| (i, Some(&[201261i64][..]), Some("Nige"), None, None))
        .collect();
    run_gate("201261 set lane", &skill_data, &horses, &[201261]);
}

/// 200801 extends the rush of senko-style runners already rushing. Nine
/// senko seats make the fire's condition (one rushing senko opponent) and
/// its target group (rushing senko runners) non-empty on every seed.
#[test]
fn wave6a_real_kakari_extend_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let horses: Vec<HorsePatch> = (0..9)
        .map(|i| (i, Some(&[200801i64][..]), Some("Senkou"), None, None))
        .collect();
    run_gate("200801 kakari extend", &skill_data, &horses, &[200801]);
}

/// The synthetic battery, both directions: self lane preferences (outside
/// 666002 vs inside 666003), a self SetTargetLane goal (666004) and a
/// delivered one (666005), a self kakari extend (666001), and the immunity
/// pair: an immune ally survives the same-team drop of 666007 while the plain
/// ally drops, and the immune host's own debuff 666009 skips inside the armed
/// window and lands outside it. 666008's two live alternatives share one key.
#[test]
fn wave6a_battery_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();

    let make_def = |cond: &str, cooldown: i64, effects: &str| -> Value {
        json::parse(&format!(
            r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":{cooldown},"baseDuration":30000,"effects":[{effects}]}}]}}"#
        ))
        .expect("def parses")
    };
    let battery: [(i64, &str, i64, &str); 9] = [
        (666001, "phase>=2", 60000, r#"{"type":13,"target":1,"modifier":50000}"#),
        (666002, "phase>=2", 60000, r#"{"type":11,"target":1,"modifier":1000}"#),
        (666003, "phase>=2", 60000, r#"{"type":12,"target":1,"modifier":1000}"#),
        (666004, "phase>=2", 60000, r#"{"type":35,"target":1,"modifier":8000}"#),
        (
            666005,
            "phase>=2",
            60000,
            r#"{"type":35,"target":9,"targetValue":3,"modifier":8000}"#,
        ),
        (666006, "phase>=2", 60000, r#"{"type":38,"target":1,"modifier":1000}"#),
        (
            666007,
            "phase>=2",
            60000,
            r#"{"type":21,"target":9,"targetValue":3,"modifier":-1000}"#,
        ),
        (666009, "phase>=2", 60000, r#"{"type":27,"target":1,"modifier":-1000}"#),
        (666010, "phase>=2", 60000, r#"{"type":13,"target":9,"targetValue":3,"modifier":50000}"#),
    ];
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, cond, cooldown, effects) in battery {
            sd.insert(sid.to_string(), make_def(cond, cooldown, effects));
        }
        sd.insert(
            "666008".into(),
            json::parse(concat!(
                r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":["#,
                r#"{"precondition":"","condition":"order>=2","cooldown":50000,"baseDuration":30000,"effects":[{"type":27,"target":1,"modifier":2500}]},"#,
                r#"{"precondition":"","condition":"phase==2","cooldown":50000,"baseDuration":30000,"effects":[{"type":27,"target":1,"modifier":1500}]}]}"#,
            ))
            .expect("pair parses"),
        );
    }

    // Horse 4 casts the ahead debuff; horse 2 is its immune ally, horse 0 the
    // plain same-team ally that drops. Horse 2 also carries the self debuff
    // and the self kakari extend inside the immunity windows.
    let mut horses: Vec<HorsePatch> = Vec::new();
    for i in 0..9usize {
        let team = if i == 0 || i == 2 || i == 4 {
            5
        } else if i == 1 || i == 3 {
            7
        } else {
            0
        };
        horses.push((i, None, None, None, Some(team as i64)));
    }
    horses[0].1 = Some(&[666002i64, 666004]);
    horses[2].1 = Some(&[666006i64, 666009, 666001]);
    horses[4].1 = Some(&[666007i64, 666010]);
    horses[6].1 = Some(&[666003i64, 666005]);
    horses[8].1 = Some(&[666008i64]);

    let expect: Vec<i64> = (666001..=666010).collect();
    run_gate("wave6a battery", &skill_data, &horses, &expect);
}
