//! The is_exist_chara_id fold gate (handoff E1a, family 3), on the retail lane.
//!
//! is_exist_chara_id is a request-constant fold: the engine keeps the region
//! set when the argument lands in the roster's charaId set and drops it
//! otherwise, and missing charaId defaults to 0 (activation_conditions.rs:
//! 1584-1588, lib.rs:336). A false fold empties the alternative, and the
//! builder skips it before touching its effects
//! (race_solver_builder.rs:556-571), which is why the 3000xx wire skills'
//! target-22 fan installs nothing even though the kernel has no reader.
//!
//! The gate folds both ways on both wire fields. The v77 roster carries no
//! charaId at all, so the set is {0}: ==0 fires and 9999 stays inert there.
//! The CM roster carries real ids, so ==1010 fires and 1002, the wire
//! argument, stays inert. A live target-22 arm must still refuse by name.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const V77_COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];
const WIRES: [&str; 4] = ["300071", "300081", "300091", "300101"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value(file: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(file)).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("base parses")
}

/// A crafted definition with one is_exist_chara_id condition, the CM def
/// shape, and a self effect unless a custom tail json is given.
fn crafted(condition: &str, effects: &str) -> Value {
    json::parse(&format!(
        r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{effects}]}}],
"isGeneralSkill":true}}"#
    ))
    .expect("crafted skill json")
}

const SELF_EFFECT: &str =
    r#"{"type":27,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":4500.0,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}"#;

/// Horse 0 carries the named skill; an empty id leaves the field plain. The
/// four wire definitions already ship in both skillData tables.
fn field_value(file: &str, course: Option<i32>, seed: i64, id: &str) -> Value {
    let mut value = base_value(file);
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        if let Some(course) = course {
            map.insert("courseId".into(), Value::Number(course as f64));
        }
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

/// Horse 0 carries the crafted skill under id 920391, beside its own.
fn crafted_field(file: &str, course: Option<i32>, seed: i64, condition: &str, effects: &str) -> String {
    let mut value = field_value(file, course, seed, "");
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert("920391".into(), crafted(condition, effects));
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[0] else { unreachable!() };
        let mut ids: Vec<Value> = h
            .get("skills")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        ids.push(Value::String("920391".into()));
        h.insert("skills".into(), Value::Array(ids));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// Runs the kernel and checks it against the CPU oracle. With full_reply on,
/// the byte-equal full reply is checked too. Returns the frames, or None
/// when the race fell back or a check failed.
fn run_kernel(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
    full_reply: bool,
    expect_fallback: Option<&str>,
) -> Option<race_exec::RaceGpu> {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(why) => {
            problems.push(format!("{tag}: kernel run failed: {why}"));
            return None;
        }
    };
    let race = &run.races[0];
    match (&race.fallback, expect_fallback) {
        (Some(reason), Some(want)) => {
            if !reason.contains(want) {
                problems.push(format!("{tag}: fallback {reason} does not name {want}"));
            }
            return None;
        }
        (Some(reason), None) => {
            problems.push(format!("{tag}: fallback {reason}"));
            return None;
        }
        (None, Some(want)) => {
            problems.push(format!("{tag}: raced on the kernel, expected a refusal naming {want}"));
            return None;
        }
        (None, None) => {}
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return None;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        return Some((*race).clone());
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
    if full_reply {
        let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
        let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("{tag}: full reply differs from the engine"));
        }
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
fn retail_chara_existence_folds_both_ways() {
    let v77_geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{V77_COURSE}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(V77_COURSE, v77_geometry);
    let cm_geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, cm_geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    let mut inert = 0usize;

    for &seed in &SEEDS {
        // The v77 field: no charaId on any horse, so the set is {0}.
        for id in WIRES {
            let req = request_text(&field_value("examples/v77-race.jsonl", Some(V77_COURSE), seed, id));
            let _ = run_kernel(&format!("v77 {id} seed {seed}"), &mut problems, &req, seed, true, None);
        }
        let plain = match run_kernel(
            &format!("v77 plain seed {seed}"),
            &mut problems,
            &request_text(&field_value("examples/v77-race.jsonl", Some(V77_COURSE), seed, "")),
            seed,
            false,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        let hit = match run_kernel(
            &format!("v77 zero seed {seed}"),
            &mut problems,
            &crafted_field("examples/v77-race.jsonl", Some(V77_COURSE), seed, "is_exist_chara_id==0", SELF_EFFECT),
            seed,
            true,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        if frames_equal(&hit, &plain) {
            problems.push(format!("v77 seed {seed}: the missing-charaId default 0 did not fold true"));
        } else {
            fired += 1;
        }
        let miss = match run_kernel(
            &format!("v77 miss seed {seed}"),
            &mut problems,
            &crafted_field("examples/v77-race.jsonl", Some(V77_COURSE), seed, "is_exist_chara_id==9999", SELF_EFFECT),
            seed,
            true,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        if !frames_equal(&miss, &plain) {
            problems.push(format!("v77 seed {seed}: an absent charaId fired anyway"));
        } else {
            inert += 1;
        }
        let negated = match run_kernel(
            &format!("v77 negated seed {seed}"),
            &mut problems,
            &crafted_field("examples/v77-race.jsonl", Some(V77_COURSE), seed, "is_exist_chara_id!=9999", SELF_EFFECT),
            seed,
            true,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        if frames_equal(&negated, &plain) {
            problems.push(format!("v77 seed {seed}: the != fold of an absent id did not fire"));
        }

        // The CM field: real charaIds, 1010 on a horse and 1002 on none.
        for id in WIRES {
            let req = request_text(&field_value("examples/cm-10701-race.jsonl", None, seed, id));
            let _ = run_kernel(&format!("cm {id} seed {seed}"), &mut problems, &req, seed, false, None);
        }
        let cm_plain = match run_kernel(
            &format!("cm plain seed {seed}"),
            &mut problems,
            &request_text(&field_value("examples/cm-10701-race.jsonl", None, seed, "")),
            seed,
            false,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        let cm_hit = match run_kernel(
            &format!("cm hit seed {seed}"),
            &mut problems,
            &crafted_field("examples/cm-10701-race.jsonl", None, seed, "is_exist_chara_id==1010", SELF_EFFECT),
            seed,
            false,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        if frames_equal(&cm_hit, &cm_plain) {
            problems.push(format!("cm seed {seed}: a carried charaId did not fold true"));
        } else {
            fired += 1;
        }
        let cm_miss = match run_kernel(
            &format!("cm miss seed {seed}"),
            &mut problems,
            &crafted_field("examples/cm-10701-race.jsonl", None, seed, "is_exist_chara_id==1002", SELF_EFFECT),
            seed,
            false,
            None,
        ) {
            Some(race) => race,
            None => continue,
        };
        if !frames_equal(&cm_miss, &cm_plain) {
            problems.push(format!("cm seed {seed}: the wire argument 1002 fired anyway"));
        } else {
            inert += 1;
        }

        // A live arm with the chara fan must refuse, the kernel has no
        // reader for target 22.
        let fan = crafted_field(
            "examples/cm-10701-race.jsonl",
            None,
            seed,
            "is_exist_chara_id==1010",
            r#"{"type":1,"target":22,"targetValue":1010,"additionalActivate":0,
"valueUsage":1,"modifier":400000.0,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}"#,
        );
        let _ = run_kernel(
            &format!("cm fan seed {seed}"),
            &mut problems,
            &fan,
            seed,
            false,
            Some("target 22"),
        );
    }
    println!(
        "chara_id pins: fired {fired}, held inert {inert}, absent-id refusal pinned on every seed"
    );
    if fired < 2 * SEEDS.len() as usize {
        problems.push("each field's carried id must fold true on every seed".into());
    }
    if inert < 2 * SEEDS.len() as usize {
        problems.push("each field's absent id must fold false on every seed".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
