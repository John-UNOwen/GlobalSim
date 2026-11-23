//! The static phase-window words on the retail lane: phase_firsthalf,
//! phase_firstquarter and phase_laterhalf fold to plain region filters, the
//! same windows the engine intersects (activation_conditions.rs:936-958).
//!
//! The four crafted pins race on the v77 field: the first quarter of phase 1
//! and the first half of phase 1 fire a recovery inside their window, the
//! later half of phase 0 fires early, and the first quarter of phase 0
//! crossed with remain_distance<=10 has an empty intersection, so the engine
//! drops that alternative and the kernel must never fire it.
//!
//! The real members 110771 and 110801 ride cm-10808 seed 1: the folded
//! precondition window lets 110771 fire at 77.7 s, and both members race
//! frame for frame against the engine's own windows.

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

/// A recovery that fires inside one folded phase window.
fn window_def(condition: &str) -> Value {
    json::parse(&format!(
        r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{{"type":9,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":4500,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}]}}],
"isGeneralSkill":true}}"#
    ))
    .expect("crafted skill json")
}

#[test]
fn phase_window_pins_fire_frame_for_frame() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let shapes: [(&str, &str, bool); 4] = [
        ("669501", "phase_firstquarter==1", true),
        ("669502", "phase_firsthalf==1", true),
        ("669503", "phase_laterhalf==0", true),
        ("669504", "phase_firstquarter==0&remain_distance<=10", false),
    ];
    for (id, condition, fires) in shapes {
        let def = window_def(condition);
        let mut problems: Vec<String> = Vec::new();
        for &seed in &[1i64, 2] {
            let mut with = base_value("examples/v77-race.jsonl");
            inject(&mut with, id, &def, 0);
            let without = base_value("examples/v77-race.jsonl");
            let tag = format!("{id} {condition} seed {seed}");
            let (fired, fired_events) =
                match parity(&tag, &mut problems, &request_text(&with), seed, id) {
                    Some(pair) => pair,
                    None => {
                        problems.push(format!("{tag}: no parity run"));
                        continue;
                    }
                };
            let plain =
                match parity(&format!("{tag} plain"), &mut problems, &request_text(&without), seed, id)
                {
                    Some(pair) => pair.0,
                    None => continue,
                };
            if fires {
                if fired_events.is_empty() {
                    problems.push(format!("{tag}: the pin never fired, the scene is vacuous"));
                }
                if first_change(&fired, &plain).is_none() {
                    problems.push(format!("{tag}: the window fired and moved nothing"));
                }
            } else {
                if !fired_events.is_empty() {
                    problems.push(format!("{tag}: the empty intersection fired: {fired_events:?}"));
                }
                if first_change(&fired, &plain).is_some() {
                    problems.push(format!("{tag}: the dead record moved the field"));
                }
            }
            println!("{tag}: fired {fired_events:?}");
        }
        if !problems.is_empty() {
            panic!("{id}: {} problems:\n{}", problems.len(), problems.join("\n"));
        }
    }
}

/// The real members ride the CM fields. 110771 fires on cm-10808 once the
/// first-half phase window opens. 110801 needs ground_type==2: the engine
/// fires it on no CM field or seed, and the kernel must agree on every one,
/// firing nothing and moving nothing. Each member must match the engine.
#[test]
fn real_phase_window_members_hold() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10808.json")).expect("geometry");
    globalsim::register_course_geometry(10808, geometry);
    let dirt =
        std::fs::read_to_string(repo_file("data/course_geometry/11103.json")).expect("geometry");
    globalsim::register_course_geometry(11103, dirt);
    let defs: &[(String, String, usize, usize, bool)] = &[
        (
            "110771".into(),
            r#"{"alternatives":[{"baseDuration":50000,"condition":"distance_rate>=45&order_rate>=20&order_rate<=50&remain_distance>=1000","effects":[{"modifier":4500,"target":1,"type":27}],"precondition":"phase_firsthalf==1&is_activate_heal_skill==1"},{"baseDuration":50000,"condition":"distance_rate>=45&order_rate>=20&order_rate<=50","effects":[{"modifier":2500,"target":1,"type":27}],"precondition":"phase_firsthalf==1&is_activate_heal_skill==1"}],"rarity":5,"wisdomCheck":0}"#.into(),
            10808,
            0,
            true,
        ),
        (
            "110801".into(),
            r#"{"alternatives":[{"baseDuration":30000,"condition":"ground_type==2&phase_laterhalf==0&order_rate<=30","effects":[{"modifier":4500,"target":1,"type":27}],"precondition":""},{"baseDuration":50000,"condition":"is_activate_other_skill_detail==1&distance_rate>=40&distance_rate<=50&overtake_target_time>=2","effects":[{"modifier":2500,"target":1,"type":22}],"precondition":""}],"rarity":5,"wisdomCheck":0}"#.into(),
            11103,
            0,
            false,
        ),
    ];
    for (id, text, course, runner, expect_fire) in defs {
        let def = json::parse(text).expect("real skill json");
        let mut problems: Vec<String> = Vec::new();
        let mut member_fires = 0usize;
        for &seed in &[1i64, 2, 3, 4] {
            let file = format!("examples/cm-{course}-race.jsonl");
            let mut with = base_value(&file);
            inject(&mut with, id, &def, *runner);
            let without = base_value(&file);
            let tag = format!("{id} course {course} seed {seed} runner {runner}");
            let (fired, fired_events) =
                match parity(&tag, &mut problems, &request_text(&with), seed, id) {
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
                id,
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
        if *expect_fire && member_fires == 0 {
            problems.push(format!("{id}: the member fired on no seed, the scene is vacuous"));
        }
        if !*expect_fire && member_fires > 0 {
            problems.push(format!("{id}: the engine-dead member fired {member_fires} times"));
        }
        if !problems.is_empty() {
            panic!("{id}: {} problems:\n{}", problems.len(), problems.join("\n"));
        }
    }
}
