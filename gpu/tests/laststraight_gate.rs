//! The last_straight_random sampler (handoff E1a, family 7), on the retail lane.
//!
//! The engine draws one distance_window over a straight at init: the last
//! corner must sit within the final 1000 m, and the row is the first straight,
//! scanned in reverse, whose start reaches that corner's start; no corner, a
//! far corner or no qualifying row takes no window and spends no draw
//! (retail_initialization.rs:167-182). The compiler resolves the row from the
//! course and kills the arm when the sampler would take nothing; the kernel
//! then draws the window on the runner's shared init bank exactly like the
//! sibling samplers.
//!
//! Course 10504 (last corner 1450, resolved row (1690, 2000)) gives the live
//! side: the crafted condition fires on every seed as horse 0 crosses the
//! drawn 10 m window. Course 10613 ends in a turn, no straight starts at or
//! after its last corner, so the same skill is a provable dead arm and no
//! seed changes a frame. A leaf that ignored the draw would fire in both, and
//! one that ignored the row resolution would fire (or fall back) on 10613.
//!
//! Wire scene: 200411 (last_straight_random==1&distance_diff_rate>=75) on
//! horse 0 of 10504, bit-exact frames plus a byte-equal full reply.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec, skills_compile};
use json::Value;

const LIVE_COURSE: i32 = 10504;
const DEAD_COURSE: i32 = 10613;
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

/// A crafted definition with one sampled condition, the CM def shape.
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
/// 200411 definition already ships in the v77 skillData.
fn field_value(course: i32, seed: i64, id: &str) -> Value {
    let mut value = base_value(course);
    {
        let Value::Object(map) = &mut value else { unreachable!() };
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

/// A plan request for the compile-side probes: one crafted skill on runner 0
/// over the real wire course.
fn plan_for(course_id: &str, condition: &str) -> skills_compile::SkillsPlan {
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get(course_id).expect("wire course").clone();
    let root_text = format!(
        r#"{{"skillData":{{"920281":{}}},"horses":[{{"skills":["920281"]}}]}}"#,
        json::to_string(&crafted(condition))
    );
    let root = json::parse(&root_text).expect("root parses");
    skills_compile::plan(&root, &course)
}

#[test]
fn retail_last_straight_random_answer_bit_exact() {
    for course in [LIVE_COURSE, DEAD_COURSE] {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry");
        globalsim::register_course_geometry(course, geometry);
    }
    let mut problems: Vec<String> = Vec::new();

    // Compile-side: the live course lowers a sampled term, an argument the
    // sampler rejects stays a refusal, and the turn-ended course collapses
    // the arm without a draw instead of falling back.
    if plan_for("10504", "last_straight_random==1").reason.is_some() {
        problems.push("last_straight_random==1 refused on a qualifying course".into());
    }
    if plan_for("10613", "last_straight_random==1").reason.is_some() {
        problems.push("the turn-ended course must collapse the arm, not refuse".into());
    }
    match plan_for("10504", "last_straight_random==2").reason {
        Some(reason) if reason.contains("last_straight_random") => {}
        other => problems.push(format!("last_straight_random==2 must refuse by name, got {other:?}")),
    }

    let mut live_fired = 0usize;
    for &seed in &SEEDS {
        let req = request_text(&field_value(LIVE_COURSE, seed, "200411"));
        let _ = run_kernel(&format!("200411 seed {seed}"), &mut problems, &req, seed);
        let plain = match run_kernel(
            &format!("plain seed {seed}"),
            &mut problems,
            &request_text(&field_value(LIVE_COURSE, seed, "")),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let mut live = field_value(LIVE_COURSE, seed, "");
        {
            let Value::Object(map) = &mut live else { unreachable!() };
            let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
            sd.insert("920281".into(), crafted("last_straight_random==1"));
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[0] else { unreachable!() };
            h.insert("skills".into(), json::arr(vec![Value::String("920281".into())]));
        }
        let scene = match run_kernel(
            &format!("drawn window seed {seed}"),
            &mut problems,
            &request_text(&live),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        if frames_equal(&scene, &plain) {
            problems.push(format!("seed {seed}: the sampled window never fired on {LIVE_COURSE}"));
        } else {
            live_fired += 1;
        }
        let dead_plain = match run_kernel(
            &format!("10613 plain seed {seed}"),
            &mut problems,
            &request_text(&field_value(DEAD_COURSE, seed, "")),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let mut dead = field_value(DEAD_COURSE, seed, "");
        {
            let Value::Object(map) = &mut dead else { unreachable!() };
            let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
            sd.insert("920281".into(), crafted("last_straight_random==1"));
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[0] else { unreachable!() };
            h.insert("skills".into(), json::arr(vec![Value::String("920281".into())]));
        }
        let dead_scene = match run_kernel(
            &format!("10613 dead arm seed {seed}"),
            &mut problems,
            &request_text(&dead),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        if !frames_equal(&dead_scene, &dead_plain) {
            problems.push(format!(
                "seed {seed}: the dead arm fired on the cornerless course {DEAD_COURSE}"
            ));
        }
    }
    println!("last_straight_random pins: fired {live_fired}/{}, dead arm shut all seeds", SEEDS.len());
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
