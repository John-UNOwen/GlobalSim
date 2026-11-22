//! The straight_front_type region port (handoff E1a, family 2), on the retail
//! lane.
//!
//! The engine compiles the filter to the course's own straight spans: keep
//! every straight whose frontType equals the arg, as plain regions, and reject
//! any arg that truncates outside 1 or 2 (activation_conditions.rs:1100-1106,
//! frontType read at lib.rs:131). The kernel needs no new reader: the compiler
//! resolves the spans from the same course JSON the engine reads and installs
//! them on the region chain, so the record is window-gated in the same places
//! the engine windows it.
//!
//! Wire scenes A to D run the four CM ids on the 10701 course. All four
//! combine the new leaf with a mid-race term while 10701's only type-2
//! straight is the first 300 m, so the leaf narrows the region chain there but
//! no alternative ever reaches its activation, and the scenes prove compile
//! fidelity.
//!
//! The behavior pins run on 10504, whose type-2 straight is the mid-course
//! 875 to 1200 m and whose type-1 straights bracket the course. Both crafted
//! leaves gate on a remain_distance term that is false at the constructor
//! pass, so the activation lands at runtime. The type-2 scene fires in the
//! mid-course straight, the type-1 scene fires only in the closing straight,
//! and the two windows are three quarters of a lap apart, so the frames answer
//! differently from each other and from the plain race. The empty pin runs the
//! type-2 leaf on 10505, a course with no type-2 straight at all: the engine
//! resolves no window, the compiler marks the arm dead, and the race must
//! answer exactly the plain one. Every run answers bit-exact frames plus a
//! byte-equal full reply.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value(course: i32) -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    let mut value = json::parse(text.lines().next().expect("first line")).expect("v77 parses");
    if let Value::Object(map) = &mut value {
        map.insert("courseId".into(), Value::Number(course as f64));
    }
    value
}

/// The exact wire definition of one family id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// A crafted definition, the same shape the CM wire entries use.
fn crafted(condition: &str) -> Value {
    json::parse(&format!(
        r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{{"type":27,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":4500.0,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}]}}],
"isGeneralSkill":true}}"#
    ))
    .expect("crafted skill json")
}

/// Horse 0 carries the named skill; the rest of the field is the plain v77.
fn field_value(course: i32, seed: i64, id: &str, def: &Value) -> Value {
    let mut value = base_value(course);
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[0] else { unreachable!() };
        h.insert("skills".into(), json::arr(vec![Value::String(id.into())]));
    }
    value
}

fn plain_value(course: i32, seed: i64) -> Value {
    let mut value = base_value(course);
    if let Value::Object(map) = &mut value {
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn compare_frames(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
) {
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
    'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} column {column}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// Runs the kernel, checks it against the CPU oracle and the full reply, and
/// returns the frames so the caller can compare scenes against each other.
fn run_kernel(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64) -> Option<race_exec::RaceGpu> {
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
    compare_frames(tag, problems, request, seed, race);
    let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
    let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
    let engine = dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply;
    if gpu_reply != engine {
        problems.push(format!("{tag}: full reply differs from the engine"));
    }
    Some((*race).clone())
}

fn frames_equal(a: &race_exec::RaceGpu, b: &race_exec::RaceGpu) -> bool {
    a.frames.len() == b.frames.len()
        && a.frames.iter().zip(b.frames.iter()).all(|(fa, fb)| {
            fa.len() == fb.len()
                && fa.iter().zip(fb).all(|(ra, rb)| {
                    ra.len() == rb.len()
                        && ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits())
                })
        })
}

#[test]
fn retail_straight_front_type_windows_bit_exact() {
    for course in [10701, 10504, 10505] {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry");
        globalsim::register_course_geometry(course, geometry);
    }
    let mut problems: Vec<String> = Vec::new();
    let two = crafted("straight_front_type==2&remain_distance<=1100");
    let one = crafted("straight_front_type==1&remain_distance<=300");
    let dead = crafted("straight_front_type==2&remain_distance<=200");
    for &seed in &SEEDS {
        // Definition fidelity: the four CM ids on their own course.
        for id in ["100681", "900681", "110201", "910201"] {
            let req = request_text(&field_value(10701, seed, id, &cm_skill(id)));
            let _ = run_kernel(&format!("{id} seed {seed}"), &mut problems, &req, seed);
        }
        let plain = match race_exec::run(&[(request_text(&plain_value(10504, seed)), seed)]) {
            Ok(run) => run.races[0].clone(),
            Err(why) => {
                problems.push(format!("plain seed {seed}: kernel run failed: {why}"));
                continue;
            }
        };
        if plain.fallback.is_some() {
            problems.push(format!("plain seed {seed}: control refused"));
        }
        let t2 = request_text(&field_value(10504, seed, "920272", &two));
        let Some(f2) = run_kernel(&format!("front2 seed {seed}"), &mut problems, &t2, seed) else {
            continue;
        };
        let t1 = request_text(&field_value(10504, seed, "920271", &one));
        let Some(f1) = run_kernel(&format!("front1 seed {seed}"), &mut problems, &t1, seed) else {
            continue;
        };
        let td = request_text(&field_value(10505, seed, "920270", &dead));
        let Some(fd) = run_kernel(&format!("front2 empty seed {seed}"), &mut problems, &td, seed) else {
            continue;
        };
        let plain5 = match race_exec::run(&[(request_text(&plain_value(10505, seed)), seed)]) {
            Ok(run) => run.races[0].clone(),
            Err(why) => {
                problems.push(format!("plain 10505 seed {seed}: kernel run failed: {why}"));
                continue;
            }
        };
        if plain5.fallback.is_some() {
            problems.push(format!("plain 10505 seed {seed}: control refused"));
        }
        if frames_equal(&f2, &plain) {
            problems.push(format!("seed {seed}: front2 stayed shut, the window is vacuous"));
        }
        if frames_equal(&f1, &plain) {
            problems.push(format!("seed {seed}: front1 stayed shut, the window is vacuous"));
        }
        if frames_equal(&f1, &f2) {
            problems.push(format!("seed {seed}: front1 and front2 answer alike, the frontType filter is ignored"));
        }
        if !frames_equal(&fd, &plain5) {
            problems.push(format!("seed {seed}: front2 fired on a course without any type-2 straight"));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
