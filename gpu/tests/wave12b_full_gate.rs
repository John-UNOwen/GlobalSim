//! Wave 12b: the classic lane's skill replies. The 6696xx battery scenes
//! with heavy event traffic (self stats, the vision cone with abilityTargets,
//! the cooldown re-fire's ordinal order, the immunity and cross-debuff pair,
//! and the real track-gated id 200111) answer the engine's whole non-compact
//! reply, byte for byte, through `classic_reply::build` against
//! `dep_cpu::run_cpu_binary`, which forces `retainAllFrames`. A mismatch
//! reports the first differing byte offset with 80 characters of context.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, json, race_exec};

const SEEDS: [i64; 2] = [1, 2];

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

/// The first byte where two reply texts differ, with 80 characters of
/// context from each side.
fn first_difference(gpu: &str, engine: &str) -> String {
    let gb = gpu.as_bytes();
    let eb = engine.as_bytes();
    let at = gb.iter().zip(eb.iter()).take_while(|(a, b)| a == b).count();
    let window = |bytes: &[u8]| {
        let start = at.saturating_sub(40);
        String::from_utf8_lossy(&bytes[start..bytes.len().min(at + 40)]).into_owned()
    };
    format!(
        "first differing byte {at} (gpu {} bytes, engine {} bytes)\ngpu:    {}\nengine: {}",
        gb.len(),
        eb.len(),
        window(gb),
        window(eb)
    )
}

#[test]
fn wave12b_full_bit_exact() {
    let base = v77_line();
    let mut skill_data = json::parse(&base).expect("parse v77")
        .get("skillData")
        .cloned()
        .expect("v77 skillData");
    if let json::Value::Object(map) = &mut skill_data {
        let defs = [
            ("669601", def("distance_rate>=30", 5000000, 60000, &effect(1, 1, 0, false, 30000))),
            ("669609", def("distance_rate>=60", 5000000, 40000, &effect(27, 4, 0, false, 20000))),
            ("669610", def("distance_rate>=65", 5000000, 40000, &effect(21, 4, 0, false, 15000))),
            ("669614", def("distance_rate>=20", 5000000, 90000, &effect(38, 1, 0, false, 1))),
            ("669615", def("distance_rate>=70", 5000000, 40000, &effect(27, 9, 0, true, -20000))),
            ("669618", def("distance_rate>=0", 5000000, 0, &effect(29, 1, 0, false, 50000))),
            ("669619", def("distance_rate>=20&distance_rate<=80", 1000, 30000, &effect(31, 1, 0, false, 8000))),
        ];
        for (sid, text) in defs {
            map.insert(sid.into(), json::parse(&text).expect("battery def parses"));
        }
    }
    let mut scenes: Vec<(String, String, i64)> = Vec::new();
    for &seed in &SEEDS {
        let mut scenes_for_seed: Vec<(&str, Vec<(usize, Vec<i64>, Option<i64>)>)> = vec![
            ("speedup", vec![(0, vec![669601i64], None)]),
            ("cones", vec![(0, vec![669609i64, 669610i64], None)]),
            ("immunity", vec![(0, vec![669614i64], None), (3, vec![669615i64], Some(1))]),
            ("rush", vec![(0, vec![669618i64], None)]),
            ("refire", vec![(0, vec![669619i64], None)]),
            ("real", vec![(0, vec![200111i64], None)]),
        ];
        for (tag, hosts) in scenes_for_seed.drain(..) {
            let hosts: Vec<(usize, &[i64], Option<i64>)> = hosts
                .iter()
                .map(|(i, ids, t)| (*i, ids.as_slice(), *t))
                .collect();
            let course = if tag == "real" { 10307 } else { 10504 };
            scenes.push((
                format!("{tag} seed {seed}"),
                classic_request(
                    &base,
                    course,
                    seed,
                    if tag == "real" { None } else { Some(&skill_data) },
                    &hosts,
                ),
                seed,
            ));
        }
    }

    let requests: Vec<(String, i64)> = scenes.iter().map(|s| (s.1.clone(), s.2)).collect();
    let out = classic_exec::run(&requests).expect("classic kernel run");
    println!(
        "device {} build {:.1} ms, {} races",
        out.device.name,
        out.build_ms,
        out.races.len()
    );

    let mut problems: Vec<String> = Vec::new();
    for (index, scene) in scenes.iter().enumerate() {
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{}: fallback {reason}", scene.0));
            continue;
        }
        let full = match dep_cpu::with_retain_all_frames(&scene.1) {
            Ok(text) => text,
            Err(reason) => {
                problems.push(format!("{}: retain request failed: {reason}", scene.0));
                continue;
            }
        };
        let gpu = match classic_reply::build(&full, scene.2, race) {
            Ok(text) => text,
            Err(reason) => {
                problems.push(format!("{}: full reply build failed: {reason}", scene.0));
                continue;
            }
        };
        let engine = match dep_cpu::run_cpu_binary(&full, scene.2) {
            Ok(race_cpu) => race_cpu.reply,
            Err(reason) => {
                problems.push(format!("{}: engine run failed: {reason}", scene.0));
                continue;
            }
        };
        if gpu != engine {
            problems.push(format!("{}: {}", scene.0, first_difference(&gpu, &engine)));
        }
    }
    if !problems.is_empty() {
        panic!("{} full reply problems:\n{}", problems.len(), problems.join("\n"));
    }
}
