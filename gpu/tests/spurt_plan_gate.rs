//! Constructor stat bumps feed the frozen plan words (handoff X2, X9 and X10).
//! The engine chains a construction-pass SpeedUp, GutsUp or WisdomUp bump
//! into the f32 accumulator (race_solver.rs:703) before enable_retail_runtime
//! freezes base_target_speed and last_spurt_speed from the bumped field, so
//! the kernel must apply the same bump before those words are computed. Two
//! regressions here went unnoticed for waves: the injected sprint skill 202441
//! lost its GutsUp bump to the accumulator (X9, first diff tick 10 runner 7
//! column 0), and the bare 10808 medium lost the 201641 SpeedUp bump, moving
//! base_target_speed[2] by 0.022337 and clamping the spurt-window target at
//! tick 1176 runner 0 column 0 (X2).
//!
//! Scenes: the cm-10701 sprint with 202441 injected on runner 7 and its bare
//! control, the cm-10808 medium bare (the 200781 and 202362 gated skills
//! stripped, reproducing X2), and the same medium with 202441 or 202442
//! injected on runner 0 (reproducing X10). Every frame compares through
//! to_bits(), the full reply must be byte-equal to the engine's, and each
//! skill must move the replay against its control on at least one seed,
//! keeping the scenes non-vacuous.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const SPRINT: i32 = 10701;
const MEDIUM: i32 = 10808;
const SPRINT_SEEDS: [i64; 3] = [317734447, 317734448, 317734446];
const MEDIUM_SEEDS: [i64; 3] = [1817412797, 1817412798, 1817412796];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn field_root(file: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(file)).expect("field");
    json::parse(text.lines().next().expect("first line")).expect("field parses")
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// The field's first request with the wire seed replaced.
fn seeded(root: &Value, seed: i64) -> Value {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    m.insert("seed".into(), Value::Number(seed as f64));
    v
}

/// The request with skill `id` added to runner `runner`'s list.
fn inject(root: &Value, runner: usize, id: &str) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    let Value::Object(h) = &mut hs[runner] else { unreachable!() };
    if !matches!(h.get("skills"), Some(Value::Array(_))) {
        h.insert("skills".into(), Value::Array(Vec::new()));
    }
    let Some(Value::Array(s)) = h.get_mut("skills") else { unreachable!() };
    let mut one = std::collections::BTreeMap::new();
    one.insert("skillId".to_string(), Value::String(id.to_string()));
    one.insert("level".to_string(), Value::Number(1.0));
    s.push(Value::Object(one));
    request_text(&v)
}

/// The request with these skill ids removed from every runner.
fn strip(root: &Value, ids: &[&str]) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    for h in hs.iter_mut() {
        let Value::Object(hm) = h else { unreachable!() };
        if let Some(Value::Array(s)) = hm.get_mut("skills") {
            s.retain(|x| x.get("skillId").and_then(Value::as_str).is_none_or(|id| !ids.contains(&id)));
        }
    }
    request_text(&v)
}

/// One kernel race: frames bit-exact against the engine and the full reply
/// byte-equal. Returns the race for the control comparison.
fn check(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64) -> Option<race_exec::RaceGpu> {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(reason) => {
            problems.push(format!("{tag}: kernel run failed: {reason}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return Some(race.clone());
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return Some(race.clone());
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
    } else {
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
    let full = dep_cpu::with_retain_all_frames(request).expect("retain request");
    let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
    let engine = dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply;
    if gpu_reply != engine {
        problems.push(format!("{tag}: full reply differs from the engine"));
    }
    Some(race.clone())
}

fn frames_differ(a: &race_exec::RaceGpu, b: &race_exec::RaceGpu) -> bool {
    a.frames.len() != b.frames.len()
        || a.frames.iter().zip(b.frames.iter()).any(|(x, y)| x != y)
}

#[test]
fn sprint_injected_gutsup_plan_words_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{SPRINT}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(SPRINT, geometry);
    let root = field_root("examples/cm-10701-race.jsonl");
    let mut problems: Vec<String> = Vec::new();
    let mut moved = 0usize;
    for &seed in &SPRINT_SEEDS {
        let tag = format!("10701 injected 202441 seed {seed}");
        let injected = {
            let v = seeded(&root, seed);
            inject(&v, 7, "202441")
        };
        let bare = request_text(&seeded(&root, seed));
        let raced = check(&tag, &mut problems, &injected, seed);
        let control = check(&format!("10701 bare seed {seed}"), &mut problems, &bare, seed);
        if let (Some(a), Some(b)) = (&raced, &control) {
            if frames_differ(a, b) {
                moved += 1;
            }
        }
    }
    if moved == 0 {
        problems.push("202441 moved no seed against the bare control, the scene is vacuous".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn medium_bare_speedup_plan_words_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{MEDIUM}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(MEDIUM, geometry);
    let root = field_root("examples/cm-10808-race.jsonl");
    let mut problems: Vec<String> = Vec::new();
    let mut moved = 0usize;
    for &seed in &MEDIUM_SEEDS {
        let tag = format!("10808 bare seed {seed}");
        let bare = {
            let v = seeded(&root, seed);
            strip(&v, &["200781", "202362"])
        };
        let raced = check(&tag, &mut problems, &bare, seed);
        let control = {
            let v = seeded(&root, seed);
            strip(&v, &["200781", "202362", "201641"])
        };
        let plain = check(&format!("10808 bare without 201641 seed {seed}"), &mut problems, &control, seed);
        if let (Some(a), Some(b)) = (&raced, &plain) {
            if frames_differ(a, b) {
                moved += 1;
            }
        }
    }
    if moved == 0 {
        problems.push("the 201641 SpeedUp moved no seed against the control, the scene is vacuous".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// X10: the lottery-gated Risk-Maker and Risk-Taker fire in the engine's
/// constructor pass, before enable_retail_runtime freezes base_target_speed
/// and last_spurt_speed, so their +80 speed bump is already in the frozen
/// plan words. The kernel constructor scans run after the freeze, and with
/// the stale base_target_speed[2] the phase-2 target clamped at tick 1168
/// (202441) and tick 1177 (202442), runner 0, on the first seed.
#[test]
fn medium_injected_risk_skills_plan_words_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{MEDIUM}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(MEDIUM, geometry);
    let root = field_root("examples/cm-10808-race.jsonl");
    for id in ["202441", "202442"] {
        let mut problems: Vec<String> = Vec::new();
        let mut head_moved = false;
        for &seed in &MEDIUM_SEEDS {
            let tag = format!("10808 injected {id} seed {seed}");
            let bare = {
                let v = seeded(&root, seed);
                strip(&v, &["200781", "202362"])
            };
            let injected = {
                let v = seeded(&root, seed);
                let v = inject(&v, 0, id);
                strip(&json::parse(&v).expect("injected"), &["200781", "202362"])
            };
            let raced = check(&tag, &mut problems, &injected, seed);
            let control = check(&format!("10808 bare seed {seed}"), &mut problems, &bare, seed);
            if let (Some(a), Some(b)) = (&raced, &control) {
                if seed == MEDIUM_SEEDS[0] && frames_differ(a, b) {
                    head_moved = true;
                }
            }
        }
        if !head_moved {
            problems.push(format!(
                "{id} moved no frames on seed {}, the scene is vacuous",
                MEDIUM_SEEDS[0]
            ));
        }
        if !problems.is_empty() {
            panic!("{} mismatching races for {id}:\n{}", problems.len(), problems.join("\n"));
        }
    }
}
