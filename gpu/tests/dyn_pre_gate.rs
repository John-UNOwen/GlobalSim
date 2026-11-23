//! Multi-group dynamic preconditions, the Naginata pair (handoff X14).
//!
//! A '@' precondition is an Or over groups, and the engine pairs each
//! group's region chain with its program inside the combined closure
//! (activation_conditions.rs:1454-1465, the watch at race_solver.rs:2267).
//! The fold keeps every group's chain contiguous and emits one program:
//! OP_REG membership AND the group's dynamic leaves, groups folded with
//! OP_OR. 120111 and 920111 are the shipped pair, and the only shipped
//! skills whose groups carry dynamics.
//!
//! Each rides cm-10506 seed 382661117 runner 0, the sweep case where the
//! pair fired (at tick ~1780, from the random pool) while the refusal
//! forced a fallback. The fold must fire them identically: frames through
//! to_bits(), the full reply byte-equal, and the pair's detail events equal
//! and non-empty on both lanes.

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

fn table_skill(id: &str) -> Value {
    let text = std::fs::read_to_string(repo_file("data/skill_data.json")).expect("table");
    let table: Value = json::parse(&text).expect("table parses");
    table.get(id).cloned().unwrap_or_else(|| panic!("skill_data.json has no {id}"))
}

fn inject(value: &mut Value, ids: &[&str], runner: usize) {
    let Value::Object(map) = value else { unreachable!() };
    {
        let Value::Object(sd) = map
            .entry("skillData".to_string())
            .or_insert_with(|| Value::Object(Default::default()))
        else {
            unreachable!()
        };
        for id in ids {
            sd.entry((*id).to_string())
                .or_insert_with(|| table_skill(id));
        }
    }
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    let Value::Object(h) = &mut horses[runner] else { unreachable!() };
    let mut held: Vec<Value> = h
        .get("skills")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for id in ids {
        let present = held
            .iter()
            .any(|e| e.get("skillId").and_then(Value::as_str) == Some(id) || e.as_str() == Some(id));
        if !present {
            held.push(Value::String((*id).into()));
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

/// Kernel versus both CPU oracles for one scene: every frame through
/// to_bits(), the full reply byte-equal, and each id's detail-event stream
/// equal on both lanes. Returns the kernel's race for the caller's checks.
fn parity(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
    ids: &[&str],
) -> Option<race_exec::RaceGpu> {
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
    for id in ids {
        let engine_events = events(&engine, id);
        let gpu_events = events(&gpu_reply, id);
        if engine_events != gpu_events {
            problems.push(format!(
                "{tag} {id}: events differ, engine {engine_events:?}, kernel {gpu_events:?}"
            ));
        }
        if engine_events.is_empty() {
            problems.push(format!("{tag} {id}: the skill never fired, the scene is vacuous"));
        }
    }
    Some((*race).clone())
}

/// The pair, alone and together, on their sweep case.
#[test]
fn naginata_pair_fires_frame_for_frame() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10506.json")).expect("geometry");
    globalsim::register_course_geometry(10506, geometry);
    let seed = 382661117i64;
    for ids in [&["120111"][..], &["920111"][..], &["120111", "920111"][..]] {
        let mut with = base_value("examples/cm-10506-race.jsonl");
        inject(&mut with, ids, 0);
        let tag = format!("{} seed {seed} runner 0", ids.join("+"));
        let mut problems: Vec<String> = Vec::new();
        parity(&tag, &mut problems, &request_text(&with), seed, ids);
        println!("{tag}");
        if !problems.is_empty() {
            panic!("{tag}: {} problems:\n{}", problems.len(), problems.join("\n"));
        }
    }
}

/// The same pair on a second seed, to keep the random draw honest: the
/// fold must latch identically whichever way the pool picks.
#[test]
fn naginata_pair_holds_on_a_second_seed() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10506.json")).expect("geometry");
    globalsim::register_course_geometry(10506, geometry);
    let seed = 2i64;
    let ids = ["120111", "920111"];
    let mut with = base_value("examples/cm-10506-race.jsonl");
    inject(&mut with, &ids, 0);
    let mut problems: Vec<String> = Vec::new();
    let tag = format!("{} seed {seed} runner 0", ids.join("+"));
    parity(&tag, &mut problems, &request_text(&with), seed, &ids);
    println!("{tag}");
    if !problems.is_empty() {
        panic!("{tag}: {} problems:\n{}", problems.len(), problems.join("\n"));
    }
}
