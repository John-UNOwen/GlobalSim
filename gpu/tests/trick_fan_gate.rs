//! Targets 19 and 20, the kakari-pool trick fan (handoff X5 family 1), on the
//! retail lane.
//!
//! The three wire ids are the CM field's Trick variants: a phase-1 Recovery
//! debuff that walks the rank order to the kakari runners behind (20) or
//! ahead (19) of the holder and draws up to targetValue 10 of them from the
//! holder's r128 stream (race_field.rs:429-438). Each scene puts one id on a
//! runner of the v77 field, every frame compares through to_bits() against
//! the CPU oracle, and the full reply must equal the engine's, so the drawn
//! receiver set has to match runner for runner, not just in count.
//!
//! The non-vacuity control is the same field without the skill. A scene's
//! receivers are the runners whose hp first drops below the plain field's
//! inside the fire window, and a scene with two of them proves the repeated
//! draw, not just a lucky first pick.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 6] = [1, 2, 3, 4, 5, 6];
const HP_COL: usize = 2;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// The exact wire definition of one Trick id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// The v77 field with one Trick id pushed onto the holder's skill list.
fn field_value(seed: i64, id: &str, def: &Value, holder: usize) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[holder] else { unreachable!() };
        let Value::Array(skills) = h.get_mut("skills").expect("skills") else { unreachable!() };
        skills.push(json::obj(vec![
            ("skillId", Value::String(id.into())),
            ("level", Value::Number(1.0)),
        ]));
    }
    value
}

/// The same field with the group present but the skill on nobody.
fn plain_value(seed: i64) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
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
            for (_column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// The fan's receiver set: the runners other than the holder whose hp first
/// falls below the plain field's within two ticks of that first drop. Late
/// drain drift starts far after the fire window and does not count.
fn receiver_set(plain: &[Vec<[f64; 6]>], race: &[Vec<[f64; 6]>], holder: usize) -> Vec<usize> {
    let mut first: Option<usize> = None;
    let mut dropped: Vec<(usize, usize)> = Vec::new();
    for (tick, (plain_frame, gpu_frame)) in plain.iter().zip(race.iter()).enumerate() {
        for (runner, (plain_row, gpu_row)) in plain_frame.iter().zip(gpu_frame.iter()).enumerate() {
            if runner == holder || dropped.iter().any(|(r, _)| *r == runner) {
                continue;
            }
            if plain_row[HP_COL] - gpu_row[HP_COL] > 0.5 {
                dropped.push((runner, tick));
                if first.is_none() {
                    first = Some(tick);
                }
            }
        }
    }
    match first {
        Some(t0) => dropped
            .into_iter()
            .filter(|&(_, t)| t <= t0 + 2)
            .map(|(r, _)| r)
            .collect(),
        None => Vec::new(),
    }
}

/// One Trick scene over the seeds: bit-exactness, reply equality, and the
/// receiver sets against the field without the skill.
fn run_scene(
    name: &str,
    id: &str,
    holder: usize,
    problems: &mut Vec<String>,
    max_receivers: &mut usize,
) {
    let def = cm_skill(id);
    let mut fired = 0usize;
    for &seed in &SEEDS {
        let tag = format!("{name} seed {seed}");
        let request = request_text(&field_value(seed, id, &def, holder));
        let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        compare_frames(&tag, problems, &request, seed, race);
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("{tag}: full reply differs from the engine"));
        }
        let plain = race_exec::run(&[(request_text(&plain_value(seed)), seed)])
            .expect("plain kernel run");
        let plain_race = &plain.races[0];
        if plain_race.fallback.is_some() {
            problems.push(format!("{tag}: plain control refused"));
            continue;
        }
        let moved = plain_race.frames.len() != race.frames.len()
            || plain_race.frames.iter().zip(race.frames.iter()).any(|(a, b)| a != b);
        if !moved {
            continue;
        }
        let receivers = receiver_set(&plain_race.frames, &race.frames, holder);
        if !receivers.is_empty() {
            fired += 1;
            *max_receivers = (*max_receivers).max(receivers.len());
            println!("scene {name} seed {seed}: debuff landed on runners {receivers:?}");
        }
    }
    if fired == 0 {
        problems.push(format!("scene {name} ({id}): the fan never delivered, the scene is vacuous"));
    }
}

#[test]
fn retail_trick_fan_scenes_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut max_receivers = 0usize;
    let scenes: [(&str, &str, usize); 3] = [
        ("trick behind 0", "200771", 0),
        ("trick behind 1", "200772", 1),
        ("trick ahead", "200781", 4),
    ];
    for (name, id, holder) in scenes {
        run_scene(name, id, holder, &mut problems, &mut max_receivers);
    }
    if max_receivers < 2 {
        problems.push(format!(
            "no scene delivered to two runners (max {max_receivers}), the drawn fan is not exercised"
        ));
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
