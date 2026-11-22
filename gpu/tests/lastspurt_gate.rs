//! The lastspurt calc-result leaf (handoff E1a, family 5), on the retail lane.
//!
//! The engine compares the last spurt plan result, remapped 1 to 2, 2 to 1
//! and anything else to 0, and fails while no plan calc has run yet
//! (activation_conditions.rs:876-887). The calcs start at phase 2
//! (race_solver_retail.rs:937-940) and the kernel mirrors the plan state in
//! the spurt cells its physics already maintains (retail_core.cl
//! spurt_state_update). The new reader returns NaN before the first calc, so
//! every comparator fails exactly where the engine's count>0 gate fails.
//!
//! Result 1 means the runner's HP could cover the whole remainder at max
//! spurt speed (retail_spurt.rs:69-70); every v77 runner reaches it and the
//! ==2 side is live on the plain field. Result 2, the planned mid-course
//! spurt, needs a stamina-starved runner: the ==1 side rides horse 1 with its
//! stamina at 300, measured stable across the seeds. Each side answers
//! against a control with the same field and no skill, exactly one of the
//! ==1 and ==2 scenes on the shared horse 0 answers differently from the
//! plain race, and a leaf reading any constant would collapse those pins.
//!
//! Wire scenes run 100131, 900131 and 100991 on 10504. Their ==2 alternative
//! also gates on distance_type==4, false on every course the lane simulates,
//! so the leaf narrows a dead arm there and the scenes prove compile
//! fidelity. Every run answers bit-exact frames plus a byte-equal full reply.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
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

/// Horse carrier carries the named skill; the rest of the field is the plain
/// v77. A stamina pair replaces that runner's own stat for the remap-side pin.
fn field_value(seed: i64, carrier: usize, id: &str, def: &Value, stamina: Option<(usize, i64)>) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        if let Some((runner, stat)) = stamina {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[runner] else { unreachable!() };
            h.insert("stamina".into(), Value::Number(stat as f64));
        }
        if !id.is_empty() {
            let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
            sd.insert(id.into(), def.clone());
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[carrier] else { unreachable!() };
            h.insert("skills".into(), json::arr(vec![Value::String(id.into())]));
        }
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
fn retail_lastspurt_leaf_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let two = crafted("lastspurt==2");
    let one = crafted("lastspurt==1");
    let mut two_fired = 0usize;
    let mut one_side_fired = 0usize;
    for &seed in &SEEDS {
        // Definition fidelity: the three CM ids on this course.
        for id in ["100131", "900131", "100991"] {
            let req = request_text(&field_value(seed, 0, id, &cm_skill(id), None));
            let _ = run_kernel(&format!("{id} seed {seed}"), &mut problems, &req, seed);
        }
        let plain = match run_kernel(
            &format!("plain seed {seed}"),
            &mut problems,
            &request_text(&field_value(seed, 0, "", &Value::Null, None)),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let low_ctrl = match run_kernel(
            &format!("low-stamina control seed {seed}"),
            &mut problems,
            &request_text(&field_value(seed, 0, "", &Value::Null, Some((1, 300)))),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let t2 = request_text(&field_value(seed, 0, "920282", &two, None));
        let Some(f2) = run_kernel(&format!("spurt2 seed {seed}"), &mut problems, &t2, seed) else {
            continue;
        };
        let t1 = request_text(&field_value(seed, 0, "920281", &one, None));
        let Some(f1) = run_kernel(&format!("spurt1 seed {seed}"), &mut problems, &t1, seed) else {
            continue;
        };
        let t11 = request_text(&field_value(seed, 1, "920281", &one, Some((1, 300))));
        let Some(f11) = run_kernel(&format!("spurt1 low-stamina seed {seed}"), &mut problems, &t11, seed) else {
            continue;
        };
        let fired2 = !frames_equal(&f2, &plain);
        let fired1 = !frames_equal(&f1, &plain);
        if fired2 == fired1 {
            problems.push(format!(
                "seed {seed}: lastspurt==1 and lastspurt==2 on one runner both {}",
                if fired2 { "fired" } else { "stayed shut" }
            ));
        }
        if fired2 {
            two_fired += 1;
        }
        if !frames_equal(&f11, &low_ctrl) {
            one_side_fired += 1;
        } else {
            problems.push(format!("seed {seed}: lastspurt==1 on the low-stamina runner stayed shut"));
        }
    }
    if two_fired == 0 {
        problems.push("lastspurt==2 never fired, the remap side is unproven".into());
    }
    if one_side_fired == 0 {
        problems.push("lastspurt==1 never fired, the remap side is unproven".into());
    }
    println!("lastspurt pins: ==2 fired {two_fired}/{}, ==1 side fired {one_side_fired}/{}", SEEDS.len(), SEEDS.len());
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
