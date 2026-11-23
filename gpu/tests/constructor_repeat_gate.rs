//! The engine's constructor sweeps (handoff X8a). A retail solver activates
//! its immediate records inside the solver constructor, in two mixed-priority
//! sweeps (race_solver.rs:1008-1022). The v24 repeat exists because the first
//! sweep can read a gate short: an `activate_count_start` condition sees the
//! count the first sweep's fires raised, so the gated record activates in the
//! repeat, still inside the constructor. cm-10903 runner 2 is the case: 201601
//! needs `activate_count_start >= 3` and activates only once its three
//! siblings have activated. A constructor fire also runs the engine's
//! non-retail duration chain, because the retail config is not attached yet
//! (race_solver.rs:2506-2508), so the reply duration is an f64 product, not
//! the f32-rounded retail one.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// cm-10903 at `seed` with every runner stripped, so runner 2 alone holds
/// the constructor quartet and nothing else can raise or gate a count.
fn stripped_field(keep: &[&str], seed: i64) -> Value {
    let text = std::fs::read_to_string(repo_file("examples/cm-10903-race.jsonl")).expect("field");
    let mut root: Value = json::parse(text.lines().next().expect("first line")).expect("field parses");
    let Value::Object(map) = &mut root else { unreachable!() };
    map.insert("seed".into(), Value::Number(seed as f64));
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for (i, h) in horses.iter_mut().enumerate() {
        let Value::Object(hm) = h else { unreachable!() };
        let Some(Value::Array(skills)) = hm.get_mut("skills") else { continue };
        let mine: Vec<Value> = if i == 2 {
            skills
                .iter()
                .filter(|s| keep.contains(&s.get("skillId").and_then(Value::as_str).unwrap_or("")))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        *skills = mine;
    }
    root
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn compare_frames(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64, race: &race_exec::RaceGpu) {
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
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, race_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(race_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        COLUMNS[column],
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// A skillDetailEvents entry for runner 2 and `sid` at t 0: the constructor
/// pass, not the first loop scan.
fn constructor_fire(reply: &str, sid: i64) -> bool {
    let value = json::parse(reply).expect("reply parses");
    value
        .get("skillDetailEvents")
        .and_then(Value::as_array)
        .map(|events| {
            events.iter().any(|e| {
                e.get("horseIndex").and_then(Value::as_f64) == Some(2.0)
                    && e.get("skillId").and_then(Value::as_f64) == Some(sid as f64)
                    && e.get("t").and_then(Value::as_f64) == Some(0.0)
            })
        })
        .unwrap_or(false)
}

/// The reply duration the engine recorded for the activation of `sid` on
/// runner 2, or None when the id never appears in skillEvents.
fn recorded_duration(reply: &str, sid: i64) -> Option<f64> {
    let value = json::parse(reply).expect("reply parses");
    value
        .get("skillEvents")
        .and_then(Value::as_array)
        .and_then(|events| {
            events.iter().find_map(|e| {
                if e.get("horseIndex").and_then(Value::as_f64) == Some(2.0)
                    && e.get("skillId").and_then(Value::as_f64) == Some(sid as f64)
                {
                    e.get("duration").and_then(Value::as_f64)
                } else {
                    None
                }
            })
        })
}

#[test]
fn the_constructor_repeat_fires_and_matches_the_engine() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10903.json")).expect("geometry");
    globalsim::register_course_geometry(10903, geometry);
    let keep = ["200132", "201532", "201601", "202161"];
    let seeds: Vec<i64> = vec![1, 2, 3];
    let requests: Vec<(String, i64)> =
        seeds.iter().map(|&seed| (request_text(&stripped_field(&keep, seed)), seed)).collect();

    let full = race_exec::run_full(&requests).expect("full kernel run");
    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    let mut f64_duration = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = format!("quartet seed {seed}");
        let race = &full.races[index];
        compare_frames(&tag, &mut problems, request, *seed, race);
        let gpu = match race.reply.as_ref().map(|_| full_reply::build(request, *seed, race)) {
            Some(Ok(gpu)) => gpu,
            Some(Err(why)) => {
                problems.push(format!("{tag}: gpu reply failed: {why}"));
                continue;
            }
            None => {
                problems.push(format!("{tag}: no reply channel"));
                continue;
            }
        };
        let engine = match dep_cpu::run_cpu_binary(request, *seed) {
            Ok(engine) => engine,
            Err(why) => {
                problems.push(format!("{tag}: engine reply failed: {why}"));
                continue;
            }
        };
        if gpu != engine.reply {
            let differ = gpu.bytes().zip(engine.reply.bytes()).position(|(a, b)| a != b).unwrap_or(gpu.len());
            problems.push(format!("{tag}: reply differs at byte {differ}"));
        }
        if constructor_fire(&engine.reply, 201601) {
            fired += 1;
        }
        // A duration that survives the f32 round trip would let an f32-only
        // kernel chain pass this gate with the reply agreeing by luck.
        if let Some(d) = recorded_duration(&engine.reply, 201601) {
            if d != (d as f32) as f64 {
                f64_duration += 1;
            }
        }
    }
    if fired == 0 {
        problems.push("no seed activated 201601 in the constructor, the gate proves nothing".into());
    }
    if f64_duration == 0 {
        problems.push("no seed held a non-f32 constructor duration, the f64 chain is unexercised".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
