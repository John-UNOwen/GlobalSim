//! Wave 17: the classic-lane base physics drift, closed. The section modifier
//! bake in classic_exec.rs divided by 100 after multiplying by base, while
//! race_solver.rs:1057-1058 divides the factor by 100 first. The associativity
//! moved one section term by one ulp, and on some stat bands the target speed
//! (BTS + section) then rounded one ulp high, so a runner pinned to its target
//! ran 1 to 2 ulp fast forever. These scenes are exactly the ones that diverged
//! on the old bake: they are bit-exact now and stay in the suite.
//!
//! The engine (`dep_cpu::run_cpu_race` frames plus `run_cpu_binary` reply) is
//! the oracle. Frames are compared column for column by `to_bits`, and the full
//! reply byte for byte.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, json};
use json::Value;

// High-stamina stats that arm the drift: skills cleared, stamina 30000, so the
// runner runs to its target instead of burning down.
const B_STAMINA: f64 = 30000.0;
const B_WISDOM: f64 = 1000.0;
const B_SEEDS: [i64; 3] = [1, 2, 3];

// Scenes that drifted on the old bake: plain v77 field at the seed and tick the
// wave 16 report pinned, then all four 2500 m courses on the high-stamina field
// across seeds 1 to 3 (the shared bit pair 19.39033049220090 lived there).
const PLAIN: [(i32, i64); 5] = [(10105, 2), (10404, 1), (10810, 1), (10810, 2), (10908, 1)];
const HIGH_STAMINA_COURSES: [i32; 4] = [10312, 10506, 10511, 10607];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The v77 field on the classic lane with the retail flags off, the course and
/// seed overlaid, and optionally every horse rewritten to fixed wisdom and
/// stamina with skills cleared.
fn classic_request(course: i32, seed: i64, rewrite: Option<(f64, f64)>) -> String {
    let mut value = {
        let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
        json::parse(text.lines().next().expect("first line")).expect("v77 parses")
    };
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), Value::Bool(false));
        }
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        if let Some((wisdom, stamina)) = rewrite {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            for h in horses.iter_mut() {
                let Value::Object(hm) = h else { unreachable!() };
                hm.insert("skills".into(), json::arr(vec![]));
                if let Some(Value::Array(sd)) = hm.get_mut("skillData") {
                    sd.clear();
                }
                for key in ["wisdom", "stamina", "power", "guts", "speed"] {
                    let want = if key == "wisdom" {
                        wisdom
                    } else if key == "stamina" {
                        stamina
                    } else {
                        600.0
                    };
                    if hm.contains_key(key) {
                        hm.insert(key.into(), json::num(want));
                    }
                }
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// One scene: no fallback, every replay column of every frame equal by bits
/// against the CPU engine, and the full reply byte-equal. Returns the first
/// diverging tick, column, and runner, if any.
fn assert_scene_bit_exact(
    course: i32,
    seed: i64,
    rewrite: Option<(f64, f64)>,
    problems: &mut Vec<String>,
) {
    let tag = format!("course {course} seed {seed}{}", if rewrite.is_some() { " high-stamina" } else { "" });
    let request = classic_request(course, seed, rewrite);
    let out = classic_exec::run(&[(request.clone(), seed)]).expect("classic kernel run");
    let race = out.races.into_iter().next().expect("one race");
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: kernel fallback {reason}"));
        return;
    }
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{tag}: frame count {} vs {}",
            cpu.frames.len(),
            race.frames.len()
        ));
    } else {
        'frame: for (tick, (fa, fb)) in cpu.frames.iter().zip(&race.frames).enumerate() {
            if fa.len() != fb.len() {
                problems.push(format!("{tag}: tick {tick} runner count {} vs {}", fa.len(), fb.len()));
                break 'frame;
            }
            for (r, (ra, rb)) in fa.iter().zip(fb).enumerate() {
                for (col, (x, y)) in ra.iter().zip(rb).enumerate() {
                    if x.to_bits() != y.to_bits() {
                        problems.push(format!(
                            "{tag}: first frame divergence tick {tick} runner {r} column {col}: \
                             engine {x} kernel {y}"
                        ));
                        break 'frame;
                    }
                }
            }
        }
    }
    let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
    let gpu_reply = classic_reply::build(&full, seed, &race).expect("full reply build");
    let engine_reply = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
    if gpu_reply != engine_reply {
        problems.push(format!("{tag}: full reply differs from the engine"));
    }
    println!("{tag}: frames and reply bit-exact, {} ticks", race.ticks);
}

#[test]
fn classic_drift_scenes_answer_bit_exact() {
    let mut courses: Vec<i32> = PLAIN.iter().map(|&(c, _)| c).collect();
    courses.extend(HIGH_STAMINA_COURSES);
    courses.sort_unstable();
    courses.dedup();
    for course in courses {
        register_geometry(course);
    }
    let mut problems: Vec<String> = Vec::new();
    for (course, seed) in PLAIN {
        assert_scene_bit_exact(course, seed, None, &mut problems);
    }
    for course in HIGH_STAMINA_COURSES {
        for seed in B_SEEDS {
            assert_scene_bit_exact(course, seed, Some((B_WISDOM, B_STAMINA)), &mut problems);
        }
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
