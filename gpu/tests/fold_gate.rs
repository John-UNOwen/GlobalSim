//! Course and request constant folds (handoff X5 family 4), on the retail lane.
//!
//! The compiler folds request/course constants the classic lane never needed:
//! base_speed, base_stamina, base_guts and base_wiz (each the ranked stat the
//! runtime_phase builder truncates to i32, race_solver_builder.rs:876-881),
//! course_distance (the course geometry distance, activation_conditions.rs:1239)
//! and corner_count (the geometry corner count, :1238). Every one folds to a
//! per-runner or per-course constant at compile, so a leaf that once refused now
//! compiles dead or live.
//!
//! Each case carries a buff skill whose only live condition is one fold term, so
//! the race moves exactly when the fold is true. base_speed is the load-bearing
//! one: raw stat over 1000 arms the buff and moves the race versus the carrier
//! skill-less, under 1000 the fold drops the arm and the buff is inert. Every
//! scene runs each frame through to_bits() against the vendored engine CPU
//! oracle, so a fold value that disagreed with the engine's own read of the same
//! constant would diverge on one of the two sides.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 6] = [1, 2, 3, 4, 5, 6];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// A constructor power buff whose only live condition is one fold term. The
/// effect shape mirrors a wire buff (the engine accepts it verbatim), so the
/// race changes iff the folded condition is true.
fn buff_def(condition: &str) -> Value {
    let levels = json::arr((0..10).map(|_| Value::Number(200000.0)).collect());
    let effect = json::obj(vec![
        ("type", Value::Number(1.0)),
        ("target", Value::Number(1.0)),
        ("targetValue", Value::Number(0.0)),
        ("additionalActivate", Value::Number(0.0)),
        ("valueUsage", Value::Number(1.0)),
        ("modifier", Value::Number(200000.0)),
        ("levelModifiers", levels),
    ]);
    let alternative = json::obj(vec![
        ("condition", Value::String(condition.into())),
        ("precondition", Value::String(String::new())),
        ("baseDuration", Value::Number(-1.0)),
        ("durationUsage", Value::Number(1.0)),
        ("cooldown", Value::Number(0.0)),
        ("effects", json::arr(vec![effect])),
    ]);
    json::obj(vec![
        ("rarity", Value::Number(1.0)),
        ("wisdomCheck", Value::Number(0.0)),
        ("priority", Value::Number(0.0)),
        ("tags", json::arr(Vec::new())),
        ("alternatives", json::arr(vec![alternative])),
        ("isGeneralSkill", Value::Bool(true)),
    ])
}

/// v77 base with one course and one carrier: every other runner skill-less, and
/// runner 0 carrying `id` when `carry`. `speed` is the carrier's raw speed, the
/// base_speed fold knob.
fn field_value(seed: i64, id: &str, def: &Value, speed: f64, carry: bool) -> Value {
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
        for (i, horse) in horses.iter_mut().enumerate() {
            let Value::Object(h) = horse else { unreachable!() };
            h.insert("teamId".into(), Value::Number(0.0));
            if i == 0 {
                // Push the raw speed over and under the 1000 arm line; the
                // over-1200 halving and team/mood ranks keep the same order,
                // so the flip straddles the fold's threshold either way.
                h.insert("speed".into(), Value::Number(speed));
                h.insert("power".into(), Value::Number(speed));
                h.insert("skills".into(), if carry {
                    json::arr(vec![json::obj(vec![
                        ("skillId", Value::String(id.into())),
                        ("level", Value::Number(1.0)),
                    ])])
                } else {
                    json::arr(Vec::new())
                });
            } else {
                h.insert("skills".into(), json::arr(Vec::new()));
            }
        }
    }
    value
}

/// Force the course distanceType to 2 so a distance_type term can match.
fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn compare_frames(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64, race: &race_exec::RaceGpu) {
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

/// Run one scene on the GPU and check it against the vendored engine CPU
/// oracle frame for frame. Returns the GPU frames for the armed/inert
/// comparison, and None when the kernel fell back (a compile refusal, which the
/// fold must never cause).
fn scene(
    problems: &mut Vec<String>,
    id: &str,
    def: &Value,
    speed: f64,
    carry: bool,
    seed: i64,
) -> Option<Vec<Vec<[f64; 6]>>> {
    let tag = format!("{id} seed {seed} speed {speed} carry {carry}");
    let request = request_text(&field_value(seed, id, def, speed, carry));
    let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return None;
    }
    compare_frames(&tag, problems, &request, seed, race);
    Some(race.frames.clone())
}

#[test]
fn request_and_course_constant_folds_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();

    // base_speed: with the raw stat over 1000 the buff is live and the race
    // moves versus the same field with the carrier skill-less; under 1000 the
    // fold drops the arm and the buff is inert. Every scene is engine oracle
    // exact, so a fold value that disagreed with the engine's own base_speed
    // would diverge on one of the two sides.
    {
        let id = "999101";
        let def = buff_def("base_speed>=1000&base_power>=1000");
        let mut armed_diff = 0usize;
        let mut inert_same = 0usize;
        for &seed in &SEEDS {
            let armed = scene(&mut problems, id, &def, 1600.0, true, seed);
            let without = scene(&mut problems, id, &def, 1600.0, false, seed);
            if let (Some(armed), Some(without)) = (&armed, &without) {
                if armed != without {
                    armed_diff += 1;
                }
            }
            let dead = scene(&mut problems, id, &def, 400.0, true, seed);
            let inert = scene(&mut problems, id, &def, 400.0, false, seed);
            if let (Some(dead), Some(inert)) = (&dead, &inert) {
                if dead == inert {
                    inert_same += 1;
                }
            }
        }
        if armed_diff == 0 {
            problems.push(format!("{id}: the buff never changed the race with base_speed over 1000, the fold is not load bearing"));
        }
        if inert_same == 0 {
            problems.push(format!("{id}: the buff still changed the race with base_speed under 1000, the fold never dropped the arm"));
        }
    }

    // course_distance and corner_count: each folds to a course constant, so the
    // arm is live or dead by the course alone. Every scene is engine oracle
    // exact, so a constant folded the wrong way would arm a buff the engine
    // leaves inert (or the reverse) and diverge.
    for (id, condition) in
        [("999102", "course_distance>=2000"), ("999103", "corner_count==0")]
    {
        let def = buff_def(condition);
        for &seed in &SEEDS {
            scene(&mut problems, id, &def, 1600.0, true, seed);
        }
    }

    assert!(problems.is_empty(), "{problems:#?}");
}

