//! Target 11, the same-team fan, on the retail lane (handoff X14).
//!
//! skill_targets code 11 collects every runner whose team equals the owner's
//! non-zero team, in rank order, owner included, uncapped
//! (race_field.rs:454). The debuff filter then drops immune receivers and
//! same-team others (race_field.rs:457-462), so a debuff fan lands on the
//! owner only. The six CM skills that hid the fan behind condition-dead arms
//! (105102111, 109302211, 111902211 on cm-10808 seed 1817412797 runner 4,
//! 105102221, 108302111, 110502111 on cm-10506 seed 382661117 runner 5)
//! race here: each fires from the random pool, and the delivery must match
//! the CPU engine frame for frame and reply byte for byte.
//!
//! The crafted arm pins the fan shape itself on cm-10701 (teams 1,2,1,3,3,2,
//! 1,3,2): a Recovery to code 11 heals the owner's team 1, owner included
//! (runners 0, 2, 6), the negative-modifier variant lands on the owner only,
//! and on the teamless v77 roster the same fan selects nobody, as the engine
//! requires a non-zero owner team.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const SEEDS: [i64; 2] = [1, 2];
const HP_COL: usize = 2;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value(file: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(file)).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("base parses")
}

/// The exact wire definition of one skill id from the shipped table.
fn table_skill(id: &str) -> Value {
    let text = std::fs::read_to_string(repo_file("data/skill_data.json")).expect("table");
    let table: Value = json::parse(&text).expect("table parses");
    table.get(id).cloned().unwrap_or_else(|| panic!("skill_data.json has no {id}"))
}

/// Horse 0 carries the crafted skill; horse `carrier` carries the real id.
fn inject(value: &mut Value, id: &str, def: &Value, runner: usize) {
    let Value::Object(map) = value else { unreachable!() };
    {
        let Value::Object(sd) = map
            .entry("skillData".to_string())
            .or_insert_with(|| Value::Object(Default::default()))
        else {
            unreachable!()
        };
        sd.entry(id.into()).or_insert_with(|| def.clone());
    }
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    let Value::Object(h) = &mut horses[runner] else { unreachable!() };
    let mut held: Vec<Value> = h
        .get("skills")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let present = held
        .iter()
        .any(|e| e.get("skillId").and_then(Value::as_str) == Some(id) || e.as_str() == Some(id));
    if !present {
        held.push(Value::String(id.into()));
    }
    h.insert("skills".into(), Value::Array(held));
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// One skill id's detail-event stream, horse and time, from a full reply.
fn events(reply: &str, id: &str) -> Vec<String> {
    let parsed = json::parse(reply).expect("reply parses");
    let Some(Value::Array(rows)) = parsed.get("skillDetailEvents") else {
        unreachable!()
    };
    rows.iter()
        .filter(|r| {
            r.get("skillId").and_then(Value::as_i64).map(|v| v.to_string()) == Some(id.into())
                || r.get("skillId").and_then(Value::as_str) == Some(id)
        })
        .map(|r| {
            format!(
                "h {} d {} t {:.5}",
                r.get("horseIndex").and_then(Value::as_i64).unwrap_or(-1),
                r.get("detailIndex").and_then(Value::as_i64).unwrap_or(-1),
                r.get("t").and_then(Value::as_f64).unwrap_or(-1.0)
            )
        })
        .collect()
}

fn first_change(a: &race_exec::RaceGpu, b: &race_exec::RaceGpu) -> Option<(usize, usize, usize)> {
    for (tick, (fa, fb)) in a.frames.iter().zip(&b.frames).enumerate() {
        for (runner, (ra, rb)) in fa.iter().zip(fb).enumerate() {
            for (column, (p, q)) in ra.iter().zip(rb).enumerate() {
                if p.to_bits() != q.to_bits() {
                    return Some((tick, runner, column));
                }
            }
        }
    }
    None
}

/// Runs one scene through the kernel and the CPU oracle: frames through
/// to_bits(), the full reply byte-equal, and the id's detail-event streams
/// equal on both lanes. Returns the race and the engine's event stream.
fn parity(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
    id: &str,
) -> Option<(race_exec::RaceGpu, Vec<String>)> {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(why) => {
            problems.push(format!("{tag}: kernel run failed: {why}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return None;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return None;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{tag}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
    } else {
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(&race.frames).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame).enumerate() {
                for (column, (x, y)) in cpu_row.iter().zip(gpu_row).enumerate() {
                    if x.to_bits() != y.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} runner {runner} column {column}: cpu {x:?} gpu {y:?}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
    }
    let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
    let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
    let engine = dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply;
    if gpu_reply != engine {
        problems.push(format!("{tag}: full reply differs from the engine"));
    }
    let engine_events = events(&engine, id);
    let gpu_events = events(&gpu_reply, id);
    if engine_events != gpu_events {
        problems.push(format!("{tag}: events differ, engine {engine_events:?}, kernel {gpu_events:?}"));
    }
    Some(((*race).clone(), engine_events))
}

/// The six skills the fan hid behind: each fires from the random pool on its
/// sweep seed, and the team delivery must now match the engine exactly.
#[test]
fn retail_team_fan_skills_fire_frame_for_frame() {
    let scenes = [
        ("examples/cm-10808-race.jsonl", 10808, 1817412797i64, 4usize, "105102111"),
        ("examples/cm-10808-race.jsonl", 10808, 1817412797, 4, "109302211"),
        ("examples/cm-10808-race.jsonl", 10808, 1817412797, 4, "111902211"),
        ("examples/cm-10506-race.jsonl", 10506, 382661117, 5, "105102221"),
        ("examples/cm-10506-race.jsonl", 10506, 382661117, 5, "108302111"),
        ("examples/cm-10506-race.jsonl", 10506, 382661117, 5, "110502111"),
    ];
    for (file, course, seed, runner, id) in scenes {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry");
        globalsim::register_course_geometry(course, geometry);
        let mut problems: Vec<String> = Vec::new();
        let mut with = base_value(file);
        inject(&mut with, id, &table_skill(id), runner);
        let without = base_value(file);
        let tag = format!("{id} course {course} seed {seed} runner {runner}");
        let (fired, fired_events) = match parity(&tag, &mut problems, &request_text(&with), seed, id) {
            Some(pair) => pair,
            None => {
                problems.push(format!("{tag}: no parity run"));
                continue;
            }
        };
        let plain = match parity(&format!("{tag} plain"), &mut problems, &request_text(&without), seed, id) {
            Some(pair) => pair.0,
            None => continue,
        };
        if fired_events.is_empty() {
            problems.push(format!("{tag}: the skill never fired on the kernel, the scene is vacuous"));
        }
        if first_change(&fired, &plain).is_none() {
            problems.push(format!("{tag}: the fan moved nothing against the plain field"));
        }
        println!("{tag}: fired {fired_events:?}");
        if !problems.is_empty() {
            panic!("{tag}: {} problems:\n{}", problems.len(), problems.join("\n"));
        }
    }
}

/// The crafted fan arm: cm-10701 team 1 is runners 0, 2 and 6, so a code-11
/// Recovery from runner 0 heals exactly that group, the negative modifier
/// keeps only the owner, and the teamless v77 field delivers to nobody.
#[test]
fn crafted_same_team_fan_lands_on_the_team() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, geometry);
    let v77_geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, v77_geometry);
    let def = |modifier: &str| {
        json::parse(&format!(
            r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"remain_distance>=400&remain_distance<=450","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{{"type":9,"target":11,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":{modifier},"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}]}}],
"isGeneralSkill":true}}"#
        ))
        .expect("crafted skill json")
    };
    let mut problems: Vec<String> = Vec::new();
    let mut healed = 0usize;
    let mut owner_only = 0usize;
    let mut teamless = 0usize;
    for &seed in &SEEDS {
        let mut field = base_value("examples/cm-10701-race.jsonl");
        inject(&mut field, "920411", &def("550.0"), 0);
        let plain = base_value("examples/cm-10701-race.jsonl");
        let tag = format!("cm fan seed {seed}");
        if let Some((race, fan_events)) = parity(&tag, &mut problems, &request_text(&field), seed, "920411") {
            if fan_events.is_empty() {
                problems.push(format!("{tag}: the crafted fan arm never fired"));
            }
            let plain_race =
                parity(&format!("{tag} plain"), &mut problems, &request_text(&plain), seed, "920411")
                    .map(|pair| pair.0);
            if let Some(plain_race) = plain_race {
                match first_change(&race, &plain_race) {
                    Some((tick, runner, HP_COL)) if matches!(runner, 0 | 2 | 6) => {
                        println!("{tag}: first hp change tick {tick} runner {runner}");
                        healed += 1;
                    }
                    Some((tick, runner, column)) => problems.push(format!(
                        "{tag}: first change is runner {runner} column {column} at tick {tick}, \
                         expected a team hp column"
                    )),
                    None => problems.push(format!("{tag}: the team fan healed nobody")),
                }
            }
        }

        let mut debuff_field = base_value("examples/cm-10701-race.jsonl");
        inject(&mut debuff_field, "920412", &def("-550.0"), 0);
        let tag = format!("cm fan debuff seed {seed}");
        if let Some((race, fan_events)) = parity(&tag, &mut problems, &request_text(&debuff_field), seed, "920412") {
            if fan_events.is_empty() {
                problems.push(format!("{tag}: the crafted debuff arm never fired"));
            }
            let plain_race =
                parity(&format!("{tag} plain"), &mut problems, &request_text(&plain), seed, "920412")
                    .map(|pair| pair.0);
            if let Some(plain_race) = plain_race {
                match first_change(&race, &plain_race) {
                    Some((tick, 0, HP_COL)) => {
                        println!("{tag}: first hp change tick {tick} on the owner");
                        owner_only += 1;
                    }
                    Some((tick, runner, column)) => problems.push(format!(
                        "{tag}: first change is runner {runner} column {column} at tick {tick}, \
                         expected the owner hp column"
                    )),
                    None => problems.push(format!("{tag}: the debuff fan delivered nothing")),
                }
            }
        }

        let mut v77_field = base_value("examples/v77-race.jsonl");
        {
            let Value::Object(map) = &mut v77_field else { unreachable!() };
            map.insert("courseId".into(), Value::Number(10504.0));
        }
        inject(&mut v77_field, "920411", &def("550.0"), 0);
        let mut v77_plain = base_value("examples/v77-race.jsonl");
        {
            let Value::Object(map) = &mut v77_plain else { unreachable!() };
            map.insert("courseId".into(), Value::Number(10504.0));
        }
        let tag = format!("v77 teamless seed {seed}");
        if let Some((race, fan_events)) = parity(&tag, &mut problems, &request_text(&v77_field), seed, "920411") {
            if fan_events.is_empty() {
                problems.push(format!("{tag}: the teamless arm never fired, the scene is vacuous"));
            }
            if let Some(plain_race) =
                parity(&format!("{tag} plain"), &mut problems, &request_text(&v77_plain), seed, "920411")
                    .map(|pair| pair.0)
            {
                if first_change(&race, &plain_race).is_some() {
                    problems.push(format!("{tag}: the teamless fan healed someone"));
                } else {
                    teamless += 1;
                }
            }
        }
    }
    println!("team fan pins: team healed {healed}, owner-only debuff {owner_only}, teamless inert {teamless}");
    if healed < SEEDS.len() {
        problems.push("the team fan must heal the owner's team on every seed".into());
    }
    if owner_only < SEEDS.len() {
        problems.push("the debuff fan must keep the owner only on every seed".into());
    }
    if teamless < SEEDS.len() {
        problems.push("a zero-team owner must select nobody on every seed".into());
    }
    if !problems.is_empty() {
        panic!("{} problems:\n{}", problems.len(), problems.join("\n"));
    }
}
