//! The raceInactive carnival entries (handoff X5 family 3), on the retail lane.
//!
//! The app's skill table carries 1000011 and 1000012 with an empty
//! alternatives list, and the retail builder installs nothing for them while
//! raceInactive is set (race_solver_builder.rs:497-500). The compiler now
//! takes the same branch instead of refusing, so a field carrying either id
//! must compile, run on the GPU, and answer the engine frame for frame with
//! the identical reply.
//!
//! The non-vacuity control strips raceInactive from a copy of the entry: with
//! no installed alternatives and no inactive flag the engine errors at build,
//! and the GPU keeps refusing that shape, so the acceptance is keyed to the
//! flag rather than to empty lists in general.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// The exact wire entry of one carnival id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// The v77 field with the entry installed and pushed onto runner 0.
fn field_value(seed: i64, id: &str, def: &Value) -> Value {
    let mut value = {
        let text =
            std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
        json::parse(text.lines().next().expect("first line")).expect("v77 parses")
    };
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[0] else { unreachable!() };
        let Value::Array(skills) = h.get_mut("skills").expect("skills") else { unreachable!() };
        skills.push(json::obj(vec![
            ("skillId", Value::String(id.into())),
            ("level", Value::Number(1.0)),
        ]));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

#[test]
fn race_inactive_entries_run_bit_exact_and_the_flag_gates_them() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    for id in ["1000011", "1000012"] {
        let def = cm_skill(id);
        for &seed in &SEEDS {
            let tag = format!("{id} seed {seed}");
            let request = request_text(&field_value(seed, id, &def));
            let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
            let race = &run.races[0];
            if let Some(reason) = &race.fallback {
                problems.push(format!("{tag}: fallback {reason}"));
                continue;
            }
            let cpu = match dep_cpu::run_cpu_race(&request, seed) {
                Ok(cpu) => cpu,
                Err(reason) => {
                    problems.push(format!("{tag}: cpu race failed: {reason}"));
                    continue;
                }
            };
            if cpu.frames.len() != race.frames.len() {
                problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
                continue;
            }
            'frames: for (tick, (cpu_frame, gpu_frame)) in
                cpu.frames.iter().zip(race.frames.iter()).enumerate()
            {
                for (runner, (cpu_row, gpu_row)) in
                    cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
                {
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
            let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
            let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
            let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
            if gpu_reply != engine {
                problems.push(format!("{tag}: full reply differs from the engine"));
            }
        }
        // The control: the same entry without the inactive flag, which the
        // engine refuses to build and the compiler still refuses to carry.
        let mut bare = def.clone();
        if let Value::Object(map) = &mut bare {
            map.remove("raceInactive");
        }
        let request = request_text(&field_value(SEEDS[0], id, &bare));
        let run = race_exec::run_full(&[(request, SEEDS[0])]).expect("control kernel run");
        match &run.races[0].fallback {
            Some(reason) if reason.contains("no installed activation alternatives") => {}
            Some(other) => {
                problems.push(format!("{id} control: unexpected refusal {other}"));
            }
            None => {
                problems.push(format!(
                    "{id} control: the entry without raceInactive compiled, the flag does not gate"
                ));
            }
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
