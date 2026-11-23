//! Target code 7 on the retail lane: the engine's fallback fan arm hands it
//! an empty list (race_field.rs:455), so the fire is real, its event is
//! logged and the effect delivers to nobody.
//!
//! The crafted pin pairs the empty fan with a self recovery inside one
//! window that always opens, so the fire is observable while the fan half
//! must move nothing. The real member 100851 rides v77: its fan pairs a
//! self chara-token effect with the empty fan, and the kernel must agree
//! with the engine whether that window opens or not.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value(file: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(file)).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("base parses")
}

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

/// One scene through kernel and CPU oracle: frames, full reply and the id's
/// event stream. Returns the race and the engine's event stream.
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

#[test]
fn empty_fan_fire_delivers_nobody() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let def = json::parse(
        r#"{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{"condition":"distance_rate>=90","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{"type":21,"target":7,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":-500,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]},
{"type":9,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":4500,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}]}]}"#,
    )
    .expect("crafted skill json");
    for &seed in &[1i64, 2] {
        let mut problems: Vec<String> = Vec::new();
        let mut with = base_value("examples/v77-race.jsonl");
        inject(&mut with, "669701", &def, 0);
        let without = base_value("examples/v77-race.jsonl");
        let tag = format!("669701 empty fan seed {seed}");
        let (fired, fired_events) =
            match parity(&tag, &mut problems, &request_text(&with), seed, "669701") {
                Some(pair) => pair,
                None => {
                    problems.push(format!("{tag}: no parity run"));
                    continue;
                }
            };
        let plain = match parity(
            &format!("{tag} plain"),
            &mut problems,
            &request_text(&without),
            seed,
            "669701",
        ) {
            Some(pair) => pair.0,
            None => continue,
        };
        if fired_events.is_empty() {
            problems.push(format!("{tag}: the pin never fired, the scene is vacuous"));
        }
        if first_change(&fired, &plain).is_none() {
            problems.push(format!("{tag}: the fired pin moved nothing"));
        }
        println!("{tag}: fired {fired_events:?}");
        if !problems.is_empty() {
            panic!("{tag}: {} problems:\n{}", problems.len(), problems.join("\n"));
        }
    }
}

#[test]
fn real_empty_fan_member_holds() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let def = json::parse(
        r#"{"alternatives":[{"baseDuration":50000,"condition":"distance_rate>=66&distance_rate<=68&order_rate>=60&order_rate<=90&temptation_count==0","effects":[{"modifier":2500,"target":1,"type":22},{"modifier":-500,"target":7,"type":21}],"precondition":""}],"rarity":5,"wisdomCheck":0}"#,
    )
    .expect("real skill json");
    let mut problems: Vec<String> = Vec::new();
    let mut member_fires = 0usize;
    for &seed in &[1i64, 2, 3, 4] {
        let mut with = base_value("examples/v77-race.jsonl");
        inject(&mut with, "100851", &def, 0);
        let without = base_value("examples/v77-race.jsonl");
        let tag = format!("100851 seed {seed} runner 0");
        let (fired, fired_events) =
            match parity(&tag, &mut problems, &request_text(&with), seed, "100851") {
                Some(pair) => pair,
                None => {
                    problems.push(format!("{tag}: no parity run"));
                    continue;
                }
            };
        let plain = match parity(
            &format!("{tag} plain"),
            &mut problems,
            &request_text(&without),
            seed,
            "100851",
        ) {
            Some(pair) => pair.0,
            None => continue,
        };
        if !fired_events.is_empty() {
            member_fires += 1;
            if first_change(&fired, &plain).is_none() {
                problems.push(format!("{tag}: the fired member moved nothing"));
            }
        } else if first_change(&fired, &plain).is_some() {
            problems.push(format!("{tag}: the unfired member moved the field"));
        }
        println!("{tag}: fired {fired_events:?}");
    }
    println!("100851: fired on {member_fires} of 4 seeds");
    if !problems.is_empty() {
        panic!("100851: {} problems:\n{}", problems.len(), problems.join("\n"));
    }
}
