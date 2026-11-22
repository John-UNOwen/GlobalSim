//! The is_basis_distance fold (handoff E1, port 3). Skill 200132, read from
//! the CM field (condition is_basis_distance==1, a self stamina up), rides
//! horse 0 of the v77 roster. The engine folds the condition on the course
//! distance and it is true exactly for a multiple of 400 m
//! (activation_conditions.rs:714-720). Course 10504 (2,000 m) answers the
//! live record and course 10102 (1,500 m) the dead one. Frames compare
//! through to_bits() against the CPU engine; the live course must move the
//! replay against the plain roster, and the dead course must not.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec, skills_compile};
use json::Value;

const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// The v77 request with the CM skill 200132 added to horse 0.
fn basis_field_value(course: i32, seed: i64) -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    let mut value = json::parse(text.lines().next().expect("first line")).expect("v77 parses");
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let cm_text =
            std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
        let cm: Value = json::parse(cm_text.lines().next().expect("first line")).expect("cm parses");
        let skill = cm
            .get("skillData")
            .and_then(|sd| sd.get("200132"))
            .expect("cm skill 200132")
            .clone();
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert("200132".into(), skill);
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h0) = &mut horses[0] else { unreachable!() };
        h0.insert("skills".into(), json::arr(vec![Value::String("200132".into())]));
    }
    value
}

fn plain_request(course: i32, seed: i64) -> String {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    let mut value = json::parse(text.lines().next().expect("first line")).expect("v77 parses");
    if let Value::Object(map) = &mut value {
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn frames_equal(a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(fa, fb)| {
            fa.len() == fb.len()
                && fa.iter().zip(fb).all(|(ra, rb)| {
                    ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits())
                })
        })
}

/// One course over the seeds: no fallback, frames bit-exact against the
/// engine, and equal or moved against the plain roster as asked.
fn run_course(course: i32, moves: bool, problems: &mut Vec<String>) {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
    for &seed in &SEEDS {
        let tag = format!("course {course} seed {seed}");
        let request = request_text(&basis_field_value(course, seed));
        let out = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
        let race = &out.races[0];
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
            problems.push(format!(
                "{tag}: tick count cpu {} gpu {}",
                cpu.frames.len(),
                race.frames.len()
            ));
            continue;
        }
        let first = cpu.frames.iter().zip(&race.frames).enumerate().find_map(|(tick, (cf, gf))| {
            cf.iter().zip(gf).enumerate().find_map(|(runner, (cr, gr))| {
                cr.iter().zip(gr).enumerate().find_map(|(column, (x, y))| {
                    if x.to_bits() == y.to_bits() {
                        None
                    } else {
                        Some(format!("{tag} tick {tick} runner {runner} column {column}: cpu {x:?} ({}), gpu {y:?} ({})", x.to_bits(), y.to_bits()))
                    }
                })
            })
        });
        if let Some(problem) = first {
            problems.push(problem);
        }
        let plain = race_exec::run(&[(plain_request(course, seed), seed)]).expect("plain kernel run");
        let plain_race = &plain.races[0];
        if plain_race.fallback.is_some() {
            problems.push(format!("{tag}: plain control refused"));
            continue;
        }
        let moved = !frames_equal(&plain_race.frames, &race.frames);
        if moved != moves {
            problems.push(format!(
                "{tag}: the record is {} against the plain roster, expected {}",
                if moved { "live" } else { "dead" },
                if moves { "live" } else { "dead" }
            ));
        }
    }
}

#[test]
fn is_basis_distance_folds_on_the_course_distance_bit_exact() {
    let mut problems: Vec<String> = Vec::new();
    run_course(10504, true, &mut problems);
    run_course(10102, false, &mut problems);

    // Compile only: both courses answer without a refusal (the dead course
    // still packs the record, its always-false leaf is skipped at runtime,
    // which the plain-roster frame checks above settle).
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let extras =
        skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    for course in ["10504", "10102"] {
        let root = basis_field_value(course.parse().expect("course id"), 1);
        let plan = skills_compile::plan_retail_with(&root, &courses.get(course).expect("course"), &extras);
        if let Some(reason) = &plan.reason {
            problems.push(format!("course {course}: compile refused {reason}"));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// With both ports in, the unstripped CM request compiles and its seed 1
/// replay is frame-exact against the CPU engine.
#[test]
fn cm_field_answers_with_zero_refusals() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, geometry);
    let text = std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let request = text.lines().next().expect("first line").to_string();
    let out = race_exec::run(&[(request.clone(), 1i64)]).expect("kernel run");
    let race = &out.races[0];
    if let Some(reason) = &race.fallback {
        panic!("the full CM field refused: {reason}");
    }
    let cpu = dep_cpu::run_cpu_race(&request, 1).expect("cpu race");
    let mut problems: Vec<String> = Vec::new();
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
    } else if let Some(first) = cpu.frames.iter().zip(&race.frames).enumerate().find_map(|(tick, (cf, gf))| {
        cf.iter().zip(gf).enumerate().find_map(|(runner, (cr, gr))| {
            cr.iter().zip(gr).enumerate().find_map(|(column, (x, y))| {
                if x.to_bits() == y.to_bits() {
                    None
                } else {
                    Some(format!("tick {tick} runner {runner} column {column}: cpu {x:?} ({}), gpu {y:?} ({})", x.to_bits(), y.to_bits()))
                }
            })
        })
    }) {
        problems.push(first);
    }
    for (runner, cpu_time) in cpu.finish_times.iter().enumerate() {
        let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
        if cpu_time.to_bits() != gpu.to_bits() {
            problems.push(format!("runner {runner} finishTimeRaw: cpu {cpu_time:?}, gpu {gpu:?}"));
        }
    }
    if !problems.is_empty() {
        panic!("the full CM race differs:\n{}", problems.join("\n"));
    }
    println!("cm field: {} ticks replayed frame exact", race.frames.len());
}
