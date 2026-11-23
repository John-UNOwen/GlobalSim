//! The all-stat bump (AllStatusUp, kind 32) on the retail lane. The engine
//! chains five accumulators and fields, the raw pair beside stamina and
//! power, as one effect (race_solver.rs:2569-2577). Construction fires are
//! host-baked through the bump words or replayed after the wisdom lottery,
//! and any other fire runs the chains in the kernel dispatch.
//!
//! The real members ride the v77 field: 203691 fires its rush and five-stat
//! drop at construction, 102602211 rides the wisdom lottery with a delay
//! factor beside its bump, and the real 104301221 needs a dirt soft course
//! the sample fields never give it, so it must fire nothing and move nothing.
//! The crafted pins force the mixed-chain order and the mid-race dispatch.

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
    inject_level(value, id, def, runner, None);
}

/// The level variant files the entry as an object so the compiler and the
/// engine both fold the fallback stat multiplier at that level, which is
/// what the STAT table (race_solver_builder.rs:387-388) answers with.
fn inject_level(value: &mut Value, id: &str, def: &Value, runner: usize, level: Option<i64>) {
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
        match level {
            Some(l) => held.push(json::parse(&format!(
                "{{\"skillId\":\"{id}\",\"level\":{l}}}"
            ))
            .expect("entry json")),
            None => held.push(Value::String(id.into())),
        }
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

/// One alternative with the given condition and effect list.
fn one_alt(condition: &str, effects: &str, rarity: i64, wisdom_check: i64) -> Value {
    json::parse(&format!(
        r#"{{"rarity":{rarity},"wisdomCheck":{wisdom_check},
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":-1,
"effects":[{effects}]}}]}}"#
    ))
    .expect("crafted skill json")
}

#[test]
fn real_allstat_members_hold() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    // The lottery member shifts the shared stream by its draw alone, so its
    // unfired seeds hold parity only, while an unfired plain member must also
    // move nothing.
    let defs: &[(String, Value, usize, bool, bool)] = &[
        (
            "203691".into(),
            one_alt("always==1",
                r#"{"type":29,"target":1,"modifier":400000},{"type":32,"target":1,"modifier":-200000}"#,
                1, 0),
            0,
            true,
            false,
        ),
        (
            "102602211".into(),
            one_alt("always==1",
                r#"{"type":10,"target":1,"modifier":4000},{"type":32,"target":1,"modifier":100000}"#,
                6, 1),
            1,
            true,
            true,
        ),
        (
            "104301221".into(),
            json::parse(
                r#"{"alternatives":[{"baseDuration":-1,"condition":"ground_type==2&ground_condition>=3","effects":[{"modifier":400000,"target":1,"type":1},{"modifier":400000,"target":1,"type":3},{"modifier":200000,"target":1,"type":32}],"precondition":""},{"baseDuration":-1,"condition":"ground_type==2&ground_condition<=2","effects":[{"modifier":400000,"target":1,"type":1},{"modifier":400000,"target":1,"type":3}],"precondition":""}],"rarity":6,"wisdomCheck":0}"#,
            )
            .expect("real skill json"),
            2,
            false,
            false,
        ),
    ];
    for (id, def, runner, expect_fire, lottery) in defs {
        let mut problems: Vec<String> = Vec::new();
        let mut member_fires = 0usize;
        for &seed in &[1i64, 2, 3, 4] {
            let mut with = base_value("examples/v77-race.jsonl");
            inject(&mut with, id, def, *runner);
            let without = base_value("examples/v77-race.jsonl");
            let tag = format!("{id} seed {seed} runner {runner}");
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
            } else if !*lottery && first_change(&fired, &plain).is_some() {
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

/// The fixed 104301221 mixes SpeedUp, PowerUp and AllStatusUp in one
/// constructor fire. The real condition needs a dirt soft course no sample
/// field gives, so the pin lifts it to the v77 ground: the engine chains
/// each accumulator in effect order, and a partial host bake would chain
/// the shared stats from an already-clamped seed instead. The entry rides
/// level 5, where the stat and other fallback multipliers differ, so the
/// level-less defs also pin the stat table.
#[test]
fn static_mixed_chain_ignites_in_effect_order() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let def = one_alt("ground_condition<=2",
        r#"{"type":1,"target":1,"modifier":400000},{"type":3,"target":1,"modifier":400000},{"type":32,"target":1,"modifier":200000}"#,
        6, 0);
    for &seed in &[1i64, 2] {
        let mut problems: Vec<String> = Vec::new();
        let mut with = base_value("examples/v77-race.jsonl");
        inject_level(&mut with, "669601", &def, 0, Some(5));
        let without = base_value("examples/v77-race.jsonl");
        let tag = format!("669601 mixed chain seed {seed}");
        let (fired, fired_events) =
            match parity(&tag, &mut problems, &request_text(&with), seed, "669601") {
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
            "669601",
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

/// A live-gated AllStatusUp fires its five chains mid-race. The dispatch
/// arm moves every accumulator, field, raw entry and derived cache the
/// single-stat arms move, in the engine's one-effect order.
#[test]
fn live_allstat_fire_runs_the_dispatch_chain() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let def = json::parse(
        r#"{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{"condition":"remain_distance<=500","precondition":"","baseDuration":-1,
"cooldown":5000000,"effects":[{"type":32,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":100000,"levelModifiers":[1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08]}]}]}"#,
    )
    .expect("crafted skill json");
    for &seed in &[1i64, 2] {
        let mut problems: Vec<String> = Vec::new();
        let mut with = base_value("examples/v77-race.jsonl");
        inject(&mut with, "669602", &def, 0);
        let without = base_value("examples/v77-race.jsonl");
        let tag = format!("669602 live allstat seed {seed}");
        let (fired, fired_events) =
            match parity(&tag, &mut problems, &request_text(&with), seed, "669602") {
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
            "669602",
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
