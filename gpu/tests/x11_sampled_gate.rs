//! A retail-sampled skill pinned on the dirty cm-11103 field (handoff X11).
//!
//! 210051 (phase_random==0, a retail sampled trigger, type 28 and type 8 with
//! valueUsage 7 tier factors) is defined by the field but carried by nobody.
//! Injected on runner 0 at seed 1614418797 it fires once its sampled window
//! opens and its tiered self buff lands, a single-tick divergence the kernel
//! once got wrong. The scene is the busy field with 210051 added to runner 0,
//! every frame compared through to_bits() and the full reply compared byte for
//! byte to the engine's.
//!
//! Non-vacuity has two sides. The engine fires 210051 on runner 0 in this
//! scene, so parity is not over a dead skill. And the same field with 210051
//! absent runs a different race, so the tiered buff is what moves it.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const DIRT: i32 = 11103;
const SEED: i64 = 1614418797;
const SKILL: &str = "210051";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// The first cm-11103 request, seeded, with 210051 put on (or left off) runner
/// 0 while the rest of the field stays as dumped.
fn field_value(carry: bool) -> String {
    let text = std::fs::read_to_string(repo_file("examples/cm-11103-race.jsonl")).expect("field");
    let mut value = json::parse(text.lines().next().expect("first line")).expect("field parses");
    let Value::Object(map) = &mut value else { unreachable!() };
    map.insert("seed".into(), Value::Number(SEED as f64));
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    let Value::Object(r0) = &mut horses[0] else { unreachable!() };
    let mut skills = match r0.get("skills").cloned().unwrap_or_else(|| json::arr(Vec::new())) {
        Value::Array(s) => s,
        _ => unreachable!(),
    };
    skills.retain(|row| row.get("skillId").and_then(Value::as_str) != Some(SKILL));
    if carry {
        skills.push(json::obj(vec![
            ("skillId", Value::String(SKILL.into())),
            ("level", Value::Number(3.0)),
        ]));
    }
    r0.insert("skills".into(), Value::Array(skills));
    request_text(&value)
}

/// Whether the engine reply lists runner 0 activating the pinned id.
fn skill_fired(reply: &str, id: i64) -> bool {
    let root = json::parse(reply).expect("reply parses");
    root.get("skillEvents")
        .and_then(Value::as_array)
        .map(|events| {
            events.iter().any(|row| {
                row.get("horseIndex").and_then(Value::as_f64) == Some(0.0)
                    && row.get("skillId").and_then(Value::as_f64) == Some(id as f64)
            })
        })
        .unwrap_or(false)
}

fn frames(a: &race_exec::RaceGpu) -> Vec<Vec<[f64; 6]>> {
    a.frames.clone()
}

#[test]
fn retail_sampled_skill_on_dirty_field_answers_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{DIRT}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(DIRT, geometry);
    let mut problems: Vec<String> = Vec::new();

    let carry = field_value(true);
    let run = race_exec::run_full(&[(carry.clone(), SEED)]).expect("kernel run");
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("carry seed {SEED}: fallback {reason}"));
    } else {
        if let Ok(cpu) = dep_cpu::run_cpu_race(&carry, SEED) {
            if cpu.frames.len() == race.frames.len() {
                'frames: for (tick, (cpu_frame, gpu_frame)) in
                    cpu.frames.iter().zip(race.frames.iter()).enumerate()
                {
                    for (runner, (cpu_row, gpu_row)) in
                        cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
                    {
                        for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                            if cpu.to_bits() != gpu.to_bits() {
                                problems.push(format!(
                                    "carry seed {SEED} tick {tick} runner {runner} column {column}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                                    cpu.to_bits(),
                                    gpu.to_bits()
                                ));
                                break 'frames;
                            }
                        }
                    }
                }
            } else {
                problems.push(format!("carry seed {SEED}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            }
        } else {
            problems.push(format!("carry seed {SEED}: cpu race failed"));
        }
        let full = dep_cpu::with_retain_all_frames(&carry).expect("retain request");
        let gpu_reply = full_reply::build(&full, SEED, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(&carry, SEED).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("carry seed {SEED}: full reply differs from the engine"));
        }
        if !skill_fired(&engine, SKILL.parse().unwrap()) {
            problems.push(format!("carry seed {SEED}: {SKILL} never fired, the pin is vacuous"));
        }
    }

    // The control: the same field without 210051 runs a different race, so the
    // pinned skill's tiered buff is what the scene turns on.
    let plain = field_value(false);
    let plain_run = race_exec::run_full(&[(plain.clone(), SEED)]).expect("kernel control run");
    if let Some(reason) = &plain_run.races[0].fallback {
        problems.push(format!("control seed {SEED}: fallback {reason}"));
    } else if race.fallback.is_none() && frames(&plain_run.races[0]) == frames(race) {
        problems.push(format!("control seed {SEED}: carrying {SKILL} did not change the race"));
    }

    assert!(problems.is_empty(), "{problems:#?}");
}
