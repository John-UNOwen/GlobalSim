//! The Sharp Turns trio pinned on the real cm-10903 mile (handoff X5).
//!
//! 202241, 202242 and 202243 are the only app-table skills whose condition
//! outran the old four-arm cap: each carries one alternative whose condition
//! is a six-arm top-level '@' chain of track_id folds, with a whole-race
//! self effect (+60, +40 and -40 wisdom). The three ids ship in the field's
//! own skillData but nobody on the mile carries them, and every dumped field
//! sits on a track none of the arms names, so the skill stays dead there.
//! The scene re-points the dumped race to course 11302, a 1600 m mile on
//! track 10103, names one id to each of the first three runners, and compares
//! every frame through to_bits() plus the full reply byte for byte to the
//! engine's.
//!
//! Non-vacuity has two sides. The engine fires all three ids in this scene,
//! so parity is not over dead records. And the same re-pointed field without
//! them runs a different race, so the pinned trio is what moves it.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const COURSE: i32 = 11302;
const SEEDS: [i64; 3] = [1, 2, 3];
const PINS: [(usize, &str); 3] = [(0, "202241"), (1, "202242"), (2, "202243")];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// The first cm-10903 request, re-pointed to the Sharp Turns mile 11302 and
/// seeded, with the trio put on (or left off) the first three runners while
/// the rest of the field stays as dumped. The definitions are the field's own
/// skillData entries.
fn field_value(seed: i64, carry: bool) -> String {
    let text = std::fs::read_to_string(repo_file("examples/cm-10903-race.jsonl")).expect("field");
    let mut value = json::parse(text.lines().next().expect("first line")).expect("field parses");
    let Value::Object(map) = &mut value else { unreachable!() };
    map.insert("seed".into(), Value::Number(seed as f64));
    map.insert("courseId".into(), Value::Number(COURSE as f64));
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for (runner, id) in PINS {
        let Value::Object(row) = &mut horses[runner] else { unreachable!() };
        let mut skills = match row.get("skills").cloned().unwrap_or_else(|| json::arr(Vec::new())) {
            Value::Array(s) => s,
            _ => unreachable!(),
        };
        skills.retain(|s| s.get("skillId").and_then(Value::as_str) != Some(id));
        if carry {
            skills.push(json::obj(vec![
                ("skillId", Value::String(id.into())),
                ("level", Value::Number(3.0)),
            ]));
        }
        row.insert("skills".into(), Value::Array(skills));
    }
    request_text(&value)
}

/// Whether the engine reply lists the runner activating the pinned id.
fn skill_fired(reply: &str, runner: usize, id: i64) -> bool {
    let root = json::parse(reply).expect("reply parses");
    root.get("skillEvents")
        .and_then(Value::as_array)
        .map(|events| {
            events.iter().any(|row| {
                row.get("horseIndex").and_then(Value::as_f64) == Some(runner as f64)
                    && row.get("skillId").and_then(Value::as_f64) == Some(id as f64)
            })
        })
        .unwrap_or(false)
}

fn frames(a: &race_exec::RaceGpu) -> Vec<Vec<[f64; 6]>> {
    a.frames.clone()
}

#[test]
fn sharp_turns_on_a_repointed_mile_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();

    for seed in SEEDS {
        let carry = field_value(seed, true);
        let run = race_exec::run_full(&[(carry.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("carry seed {seed}: fallback {reason}"));
            continue;
        }
        if let Ok(cpu) = dep_cpu::run_cpu_race(&carry, seed) {
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
                                    "carry seed {seed} tick {tick} runner {runner} column {column}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                                    cpu.to_bits(),
                                    gpu.to_bits()
                                ));
                                break 'frames;
                            }
                        }
                    }
                }
            } else {
                problems.push(format!("carry seed {seed}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            }
        } else {
            problems.push(format!("carry seed {seed}: cpu race failed"));
        }
        let full = dep_cpu::with_retain_all_frames(&carry).expect("retain request");
        let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(&carry, seed).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("carry seed {seed}: full reply differs from the engine"));
        }
        for (runner, id) in PINS {
            if !skill_fired(&engine, runner, id.parse().unwrap()) {
                problems.push(format!(
                    "carry seed {seed}: {id} on runner {runner} never fired, the pin is vacuous"
                ));
            }
        }

        // The control: the same re-pointed field without the trio runs a
        // different race, so the pinned wisdom swings are what move it.
        let plain = field_value(seed, false);
        let plain_run = race_exec::run_full(&[(plain.clone(), seed)]).expect("kernel control run");
        if let Some(reason) = &plain_run.races[0].fallback {
            problems.push(format!("control seed {seed}: fallback {reason}"));
        } else if frames(&plain_run.races[0]) == frames(race) {
            problems.push(format!("control seed {seed}: carrying the trio did not change the race"));
        }
    }

    assert!(problems.is_empty(), "{problems:#?}");
    println!("sharp turns gate: {} seed(s) bit exact, trio fired, present moved every race", SEEDS.len());
}
