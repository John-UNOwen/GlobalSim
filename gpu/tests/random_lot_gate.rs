//! The random_lot lottery gate (handoff E1a, family 4), on the retail lane.
//!
//! random_lot is sampled like the other retail random terms: at init the
//! shared stream rolls integer(0,100) once per arm and the arm opens for the
//! whole race when the roll lands strictly under the argument
//! (retail_initialization.rs:192). The retail parser replaces every random
//! term with its trigger, so the live solver-value filter never runs
//! (condition_parser.rs:199-205). A miss spends the draw and leaves the
//! record inert; a hit gives the window (0, distance), which intersects the
//! arm's other terms exactly like the engine's region identity.
//!
//! Both polarities are pinned without relying on luck: random_lot==100 can
//! only ever hit (the roll tops out at 99) and random_lot==0 can never hit,
//! and the two scenes draw the same amount, so a frame difference between
//! them is a fire and nothing else. A miss still spends the draw, so a
//! lottery scene legitimately differs from the plain field; the draw
//! position itself is pinned by bit-exactness against the CPU oracle. The
//! four wire ids (201561/201562 random_lot==50&post_number==7, 202441 with
//! its ==30/==60 arm pair, 202442 with ==15/==30) replay bit-exact with
//! byte-equal full replies, so the strict bound rides on the real rolls.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];
const WIRES: [&str; 4] = ["201561", "201562", "202441", "202442"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// A crafted definition with one lottery condition, the CM def shape.
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

/// Horse 0 carries the named skill; an empty id leaves the field plain. The
/// four wire definitions already ship in the v77 skillData.
fn field_value(seed: i64, id: &str) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        if !id.is_empty() {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[0] else { unreachable!() };
            let mut ids: Vec<Value> = h
                .get("skills")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            ids.push(Value::String(id.into()));
            h.insert("skills".into(), Value::Array(ids));
        }
    }
    value
}

/// Horse 0 carries the crafted lottery skill under id 920291.
fn crafted_field(seed: i64, condition: &str) -> String {
    let mut value = field_value(seed, "");
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert("920291".into(), crafted(condition));
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[0] else { unreachable!() };
        let mut ids: Vec<Value> = h
            .get("skills")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        ids.push(Value::String("920291".into()));
        h.insert("skills".into(), Value::Array(ids));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// Runs the kernel, checks it against the CPU oracle and the byte-equal full
/// reply, and returns the frames so the caller can compare scenes.
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
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return None;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        return Some((*race).clone());
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
fn retail_random_lottery_answer_bit_exact() {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut polarity_seen = 0usize;

    for &seed in &SEEDS {
        let _ = run_kernel(
            &format!("plain seed {seed}"),
            &mut problems,
            &request_text(&field_value(seed, "")),
            seed,
        );
        for id in WIRES {
            let req = request_text(&field_value(seed, id));
            let _ = run_kernel(&format!("{id} seed {seed}"), &mut problems, &req, seed);
        }

        // The roll lands in 0..99, so ==100 always opens the arm and ==0
        // never does. Both scenes draw the same amount from the shared
        // stream, so any frame difference between them is a fire.
        let always = match run_kernel(
            &format!("lottery open seed {seed}"),
            &mut problems,
            &crafted_field(seed, "random_lot==100"),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let dead = match run_kernel(
            &format!("lottery shut seed {seed}"),
            &mut problems,
            &crafted_field(seed, "random_lot==0"),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        if frames_equal(&always, &dead) {
            problems.push(format!("seed {seed}: the open lottery moved no frame beyond the shut one"));
        } else {
            polarity_seen += 1;
        }

        // The wire threshold scene, for bit-exactness on a real roll.
        let _ = run_kernel(
            &format!("lottery fifty seed {seed}"),
            &mut problems,
            &crafted_field(seed, "random_lot==50"),
            seed,
        );
    }
    println!("random_lot pins: open beat shut on {polarity_seen}/{} seeds", SEEDS.len());
    if polarity_seen != SEEDS.len() {
        problems.push("the guaranteed-open lottery must fire on every seed".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
