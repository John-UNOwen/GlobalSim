//! X6: scenes the five CM fields cannot reach, on both lanes.
//!
//! The five-field skill sweep leaves skills whose condition no CM field meets.
//! This file pins crafted scenes that satisfy them: a cornerless course
//! (course 10301, `data/course_data.json` has `"corners":[]`, the engine's
//! `corner_count` is `c.corners.len()` at activation_conditions.rs:1238), a
//! night field (`RaceTime::Night = 4`, race_parameters.rs:29-35, parsed from
//! `"time":"Night"` at :82-91, read by `v_time` at :1247), and a roster with
//! five carriers of one skill (`same_skill_horse_count` counts the roster's
//! params at :1673).
//!
//! The engine is the oracle: frames are compared column by column through
//! `to_bits()`, and the reply byte for byte, on the retail lane and, where it
//! takes the request, the classic lane.
//!
//! Course 10301 is also the regression scene for the cornerless retail launch:
//! the engine's final-corner window is an `Option` (src/retail_geometry.rs:176),
//! so a course with no corners has none, and the retail launch must flag the
//! window absent instead of panicking. With no empty-in move the engine's lane
//! target stays the gate lane all race (src/race_solver_retail.rs:1095), so the
//! frames also pin that the kernel starts its lane target from the gate.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, full_reply, race_exec};

const CM_10701: &str = "examples/cm-10701-race.jsonl";
const V77: &str = "examples/v77-race.jsonl";
const STRAIGHT_COURSE: i32 = 10301;
const CORNERED_COURSE: i32 = 10201;
const HOME_COURSE: i32 = 10701;
const STRAIGHT_SEED: i64 = 317734447;
const CLASSIC_SEED: i64 = 12345;
const NIGHT_SKILLS: [&str; 3] = ["202231", "202232", "202233"];
const STRAIGHT_SKILLS: [&str; 3] = ["202991", "202992", "202993"];
const SHARE_SKILL: &str = "201631";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn load_field(path: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(path)).expect("request file");
    json::parse(text.lines().next().expect("first line")).expect("field parses")
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn register_geometry(course: i32) {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The CM field with the course and seed overlaid, the retail scene base.
fn retail_field(course: i32, seed: i64) -> Value {
    let mut value = load_field(CM_10701);
    let Value::Object(map) = &mut value else { unreachable!() };
    map.insert("courseId".into(), Value::Number(course as f64));
    map.insert("seed".into(), Value::Number(seed as f64));
    value
}

fn set_time(value: &mut Value, time: &str) {
    let Value::Object(map) = value else { unreachable!() };
    map.insert("time".into(), Value::String(time.into()));
}

fn skill_def(id: &str) -> Value {
    let text = std::fs::read_to_string(repo_file("data/skill_data.json")).expect("skill table");
    let table: Value = json::parse(&text).expect("skill table parses");
    table.get(id).unwrap_or_else(|| panic!("{id} in the engine skill table")).clone()
}

/// Copy the engine table's defs into the field's skillData, the sweep's
/// injection shape. The app table in the CM field has no 202991..202993 def.
fn inject_defs(value: &mut Value, ids: &[&str]) {
    let Value::Object(map) = value else { unreachable!() };
    let Value::Object(table) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
    for id in ids {
        table.insert((*id).into(), skill_def(id));
    }
}

fn add_skill(value: &mut Value, runner: usize, id: &str) {
    let Value::Object(map) = value else { unreachable!() };
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    let Value::Object(horse) = &mut horses[runner] else { unreachable!() };
    let Value::Array(skills) = horse.get_mut("skills").expect("skills") else { unreachable!() };
    let mut entry = std::collections::BTreeMap::new();
    entry.insert("skillId".to_string(), Value::String(id.into()));
    entry.insert("level".to_string(), Value::Number(1.0));
    skills.push(Value::Object(entry));
}

/// The classic field: the v77 request with the retail flags off and the course
/// and seed overlaid, the shape parity_audit's classic lane uses.
fn classic_request(course: i32, seed: i64) -> String {
    let mut value = load_field(V77);
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
    request_text(&value)
}

/// The first frame cell whose bits differ, if any. The CPU rows are the six
/// replay columns, the kernel rows may carry extra channels, so columns zip.
fn first_divergence<const N: usize, const M: usize>(
    want: &[Vec<[f64; N]>],
    got: &[Vec<[f64; M]>],
) -> Option<(usize, usize, usize, f64, f64)> {
    for (tick, (a, b)) in want.iter().zip(got).enumerate() {
        for (runner, (ra, rb)) in a.iter().zip(b).enumerate() {
            for (column, (x, y)) in ra.iter().zip(rb).enumerate() {
                if x.to_bits() != y.to_bits() {
                    return Some((tick, runner, column, *x, *y));
                }
            }
        }
    }
    None
}

/// Whether two races moved any frame bit, the control's evidence that a fired
/// skill reached the replay.
fn frames_differ(a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>]) -> bool {
    a.len() != b.len()
        || a.iter().zip(b).any(|(x, y)| {
            x.iter().zip(y).any(|(p, q)| {
                p.iter().zip(q).any(|(u, v)| u.to_bits() != v.to_bits())
            })
        })
}

/// Whether every runner's lane column holds one value across the frames.
fn lane_held(frames: &[Vec<[f64; 6]>]) -> bool {
    let Some(first) = frames.first() else { return false };
    first.iter().enumerate().all(|(runner, row)| {
        let lane = row[3].to_bits();
        frames.iter().all(|frame| frame[runner][3].to_bits() == lane)
    })
}

/// Compare one retail scene against the engine: frames bit-exact and the full
/// reply byte-equal, reported as problems.
fn check_retail(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
) -> Option<race_exec::RaceGpu> {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(reason) => {
            problems.push(format!("{tag}: retail launch failed: {reason}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: retail fallback {reason}"));
        return None;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: engine race failed: {reason}"));
            return None;
        }
    };
    let (want, got) = (&cpu.frames, &race.frames);
    if want.len() != got.len() {
        problems.push(format!(
            "{tag}: tick count engine {} retail {} (engine finishers {})",
            want.len(),
            got.len(),
            cpu.finishers
        ));
    }
    if let Some((tick, runner, column, cpu_value, gpu_value)) = first_divergence(want, got) {
        problems.push(format!(
            "{tag}: first frame divergence tick {tick} runner {runner} column {column}: engine {cpu_value} ({}) retail {gpu_value} ({})",
            cpu_value.to_bits(),
            gpu_value.to_bits()
        ));
    }
    let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
    let gpu_reply = match full_reply::build(&full, seed, race) {
        Ok(reply) => reply,
        Err(reason) => {
            problems.push(format!("{tag}: full reply build failed: {reason}"));
            return None;
        }
    };
    let engine = dep_cpu::run_cpu_binary(request, seed).expect("engine binary").reply;
    if gpu_reply != engine {
        let at = gpu_reply.bytes().zip(engine.bytes()).take_while(|(a, b)| a == b).count();
        problems.push(format!("{tag}: full reply differs at byte {at}"));
    }
    Some(race.clone())
}

/// Compare one classic scene against the engine the same way.
fn check_classic(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
) -> Option<classic_exec::ClassicGpu> {
    let run = match classic_exec::run(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(reason) => {
            problems.push(format!("{tag}: classic launch failed: {reason}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: classic fallback {reason}"));
        return None;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: engine race failed: {reason}"));
            return None;
        }
    };
    let (want, got) = (&cpu.frames, &race.frames);
    if want.len() != got.len() {
        problems.push(format!("{tag}: tick count engine {} classic {}", want.len(), got.len()));
    } else if let Some((tick, runner, column, cpu_value, gpu_value)) = first_divergence(want, got) {
        problems.push(format!(
            "{tag}: first frame divergence tick {tick} runner {runner} column {column}: engine {cpu_value} ({}) classic {gpu_value} ({})",
            cpu_value.to_bits(),
            gpu_value.to_bits()
        ));
    }
    let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
    let gpu_reply = match classic_reply::build(&full, seed, race) {
        Ok(reply) => reply,
        Err(reason) => {
            problems.push(format!("{tag}: classic reply build failed: {reason}"));
            return None;
        }
    };
    let engine = dep_cpu::run_cpu_binary(request, seed).expect("engine binary").reply;
    if gpu_reply != engine {
        let at = gpu_reply.bytes().zip(engine.bytes()).take_while(|(a, b)| a == b).count();
        problems.push(format!("{tag}: full reply differs at byte {at}"));
    }
    Some(race.clone())
}

/// The engine's detail events for one skill id, each as (horse, detail, t).
fn fired(cpu: &dep_cpu::CpuRace, id: i64) -> Vec<(usize, i64, f64)> {
    cpu.detail_events
        .iter()
        .filter(|(_, skill, _, _)| *skill == id)
        .map(|(horse, _, detail, t)| (*horse, *detail, *t))
        .collect()
}

fn skill_id(id: &str) -> i64 {
    id.parse().expect("numeric skill id")
}

fn engine_race(request: &str, seed: i64) -> dep_cpu::CpuRace {
    dep_cpu::run_cpu_race(request, seed).expect("engine race")
}

#[test]
fn cornerless_course_answers_bit_exact_on_both_lanes() {
    let courses: Value = json::parse(
        &std::fs::read_to_string(repo_file("data/course_data.json")).expect("course table"),
    )
    .expect("course table parses");
    let corners = courses
        .get("10301")
        .and_then(|course| course.get("corners"))
        .and_then(Value::as_array)
        .expect("course 10301 corners");
    assert!(corners.is_empty(), "course 10301 is the cornerless scene");
    register_geometry(STRAIGHT_COURSE);

    let mut problems: Vec<String> = Vec::new();
    let field = retail_field(STRAIGHT_COURSE, STRAIGHT_SEED);
    assert_eq!(
        field.get("experimentalRetailGeometry").and_then(Value::as_bool),
        Some(true),
        "the geometry table build is what panicked on the cornerless course"
    );
    let retail = check_retail("cornerless 10301 retail", &mut problems, &request_text(&field), STRAIGHT_SEED);
    let classic = check_classic(
        "cornerless 10301 classic",
        &mut problems,
        &classic_request(STRAIGHT_COURSE, CLASSIC_SEED),
        CLASSIC_SEED,
    );
    if let Some(race) = &retail {
        if race.frames.is_empty() {
            problems.push("cornerless 10301 retail: empty frame list".into());
        }
        if race.finish_step_all.iter().any(Option::is_none) {
            problems.push("cornerless 10301 retail: a runner never finished".into());
        }
        if !lane_held(&race.frames) {
            problems.push("cornerless 10301 retail: a lane moved, the null lane controller scene is gone".into());
        }
    }
    if let Some(race) = &classic {
        if race.frames.is_empty() {
            problems.push("cornerless 10301 classic: empty frame list".into());
        }
        if race.finish_step_all.iter().any(Option::is_none) {
            problems.push("cornerless 10301 classic: a runner never finished".into());
        }
    }
    assert!(problems.is_empty(), "{} cornerless problem(s):\n{}", problems.len(), problems.join("\n"));
}

#[test]
fn night_skills_fire_and_answer_bit_exact() {
    register_geometry(HOME_COURSE);
    let build = |time: &str| {
        let mut value = retail_field(HOME_COURSE, STRAIGHT_SEED);
        inject_defs(&mut value, &NIGHT_SKILLS);
        for id in NIGHT_SKILLS {
            add_skill(&mut value, 0, id);
        }
        set_time(&mut value, time);
        request_text(&value)
    };
    let night = build("Night");
    let midday = build("Midday");

    let mut problems: Vec<String> = Vec::new();
    let night_cpu = engine_race(&night, STRAIGHT_SEED);
    for id in NIGHT_SKILLS {
        let hits = fired(&night_cpu, skill_id(id));
        if !hits.iter().any(|&(horse, detail, _)| horse == 0 && detail == 0) {
            problems.push(format!("night scene: {id} did not fire on runner 0 at Night ({hits:?})"));
        }
    }
    let midday_cpu = engine_race(&midday, STRAIGHT_SEED);
    for id in NIGHT_SKILLS {
        if !fired(&midday_cpu, skill_id(id)).is_empty() {
            problems.push(format!("night control: {id} fired at Midday, the scene is not gated by time"));
        }
    }
    let full = check_retail("10701 night", &mut problems, &night, STRAIGHT_SEED);
    let control = check_retail("10701 midday", &mut problems, &midday, STRAIGHT_SEED);
    if let (Some(full), Some(control)) = (&full, &control) {
        if !frames_differ(&full.frames, &control.frames) {
            problems.push("night control: the fires moved no frame against Midday, the scene is vacuous".into());
        }
    }
    assert!(problems.is_empty(), "{} night problem(s):\n{}", problems.len(), problems.join("\n"));
}

#[test]
fn cornerless_course_skills_fire_and_answer_bit_exact() {
    register_geometry(STRAIGHT_COURSE);
    register_geometry(CORNERED_COURSE);
    let build = |course: i32, add: bool| {
        let mut value = retail_field(course, STRAIGHT_SEED);
        inject_defs(&mut value, &STRAIGHT_SKILLS);
        if add {
            for id in STRAIGHT_SKILLS {
                add_skill(&mut value, 0, id);
            }
        }
        request_text(&value)
    };
    let straight = build(STRAIGHT_COURSE, true);
    let cornered = build(CORNERED_COURSE, true);
    let plain = build(STRAIGHT_COURSE, false);

    let mut problems: Vec<String> = Vec::new();
    let straight_cpu = engine_race(&straight, STRAIGHT_SEED);
    for id in STRAIGHT_SKILLS {
        let hits = fired(&straight_cpu, skill_id(id));
        if !hits.iter().any(|&(horse, detail, _)| horse == 0 && detail == 0) {
            problems.push(format!("straight scene: {id} did not fire on runner 0 at 10301 ({hits:?})"));
        }
    }
    let cornered_cpu = engine_race(&cornered, STRAIGHT_SEED);
    for id in STRAIGHT_SKILLS {
        if !fired(&cornered_cpu, skill_id(id)).is_empty() {
            problems.push(format!("straight control: {id} fired on 10201, the scene is not gated by corner count"));
        }
    }
    let full = check_retail("10301 straight skills", &mut problems, &straight, STRAIGHT_SEED);
    let plain_race = check_retail("10301 plain", &mut problems, &plain, STRAIGHT_SEED);
    let control = check_retail("10201 cornered skills", &mut problems, &cornered, STRAIGHT_SEED);
    if let (Some(full), Some(plain)) = (&full, &plain_race) {
        if !frames_differ(&full.frames, &plain.frames) {
            problems.push("straight control: the fires moved no frame against the same course without them".into());
        }
    }
    if control.is_none() {
        problems.push("straight control: the cornered scene did not answer".into());
    }
    assert!(problems.is_empty(), "{} straight problem(s):\n{}", problems.len(), problems.join("\n"));
}

#[test]
fn five_shared_carriers_fire_and_answer_bit_exact() {
    register_geometry(HOME_COURSE);
    let build = |carriers: usize| {
        let mut value = retail_field(HOME_COURSE, STRAIGHT_SEED);
        for runner in 0..carriers {
            add_skill(&mut value, runner, SHARE_SKILL);
        }
        request_text(&value)
    };
    let five = build(5);
    let four = build(4);

    let mut problems: Vec<String> = Vec::new();
    let five_cpu = engine_race(&five, STRAIGHT_SEED);
    let hits = fired(&five_cpu, skill_id(SHARE_SKILL));
    for runner in 0..5 {
        if !hits.iter().any(|&(horse, detail, _)| horse == runner && detail == 0) {
            problems.push(format!("five-carrier scene: {SHARE_SKILL} did not fire on runner {runner} ({hits:?})"));
        }
    }
    let four_cpu = engine_race(&four, STRAIGHT_SEED);
    if !fired(&four_cpu, skill_id(SHARE_SKILL)).is_empty() {
        problems.push("five-carrier control: the skill fired with four carriers, the scene is not gated by the count".into());
    }
    let full = check_retail("10701 five carriers", &mut problems, &five, STRAIGHT_SEED);
    let control = check_retail("10701 four carriers", &mut problems, &four, STRAIGHT_SEED);
    if let (Some(full), Some(control)) = (&full, &control) {
        if !frames_differ(&full.frames, &control.frames) {
            problems.push("five-carrier control: the fifth fire moved no frame, the scene is vacuous".into());
        }
    }
    assert!(problems.is_empty(), "{} carrier problem(s):\n{}", problems.len(), problems.join("\n"));
}
