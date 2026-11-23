//! Wave 20: the follow-up arm. Skill 10661 Engines LIT! carries a second
//! alternative gated by is_activate_other_skill_detail==1, which the engine
//! reads as the runner's used_skills set for the record's own id
//! (activation_conditions.rs:705-711, race_solver.rs:2669). The compiler used
//! to fold that arm false, and even with the leaf ported the kernel parked
//! the follow-up record on the plain group's cooldown, so the follow-up
//! never fired. The sweep confirmed it on cm-10808 runner 0 seed 1817412797:
//! the engine fires detail 0 at t 38.83 and detail 1 at t 91.17, the kernel
//! diverged at tick 1369 runner 0 column 0. The scene pins both details,
//! every frame through to_bits(), the full reply byte-equal, and a control
//! whose alternatives hold detail 0 only: its frames must differ, so the
//! follow-up fire itself moves the replay.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const COURSE: i32 = 10808;
const SEED: i64 = 1817412797;
const ID: &str = "10661";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn field_root() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/cm-10808-race.jsonl")).expect("field");
    json::parse(text.lines().next().expect("first line")).expect("field parses")
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// The engine table's def for the id, optionally keeping only the first
/// alternative.
fn table_def(first_alt_only: bool) -> Value {
    let text = std::fs::read_to_string(repo_file("data/skill_data.json")).expect("table");
    let table: Value = json::parse(&text).expect("table parses");
    let mut def = table.get(ID).expect("id in the table").clone();
    if first_alt_only {
        let Value::Object(m) = &mut def else { unreachable!() };
        if let Some(Value::Array(alts)) = m.get_mut("alternatives") {
            alts.truncate(1);
        }
    }
    def
}

/// The field request with the def added to skillData and the skill on runner
/// 0, the sweep's injection shape.
fn injected(root: &Value, first_alt_only: bool) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    {
        let Value::Object(sd) = m.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(ID.into(), table_def(first_alt_only));
    }
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    let Value::Object(h) = &mut hs[0] else { unreachable!() };
    let mut one = std::collections::BTreeMap::new();
    one.insert("skillId".to_string(), Value::String(ID.into()));
    one.insert("level".to_string(), Value::Number(1.0));
    let Some(Value::Array(s)) = h.get_mut("skills") else { unreachable!() };
    s.push(Value::Object(one));
    request_text(&v)
}

/// One race: frames bit-exact against the engine, full reply byte-equal.
fn check(tag: &str, problems: &mut Vec<String>, request: &str) -> Option<race_exec::RaceGpu> {
    let run = match race_exec::run_full(&[(request.to_string(), SEED)]) {
        Ok(run) => run,
        Err(reason) => {
            problems.push(format!("{tag}: kernel run failed: {reason}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return None;
    }
    let cpu = match dep_cpu::run_cpu_race(request, SEED) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return None;
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
    let gpu_reply = full_reply::build(&full, SEED, race).expect("full reply build");
    let engine = dep_cpu::run_cpu_binary(request, SEED).expect("cpu binary").reply;
    if gpu_reply != engine {
        problems.push(format!("{tag}: full reply differs from the engine"));
    }
    Some(race.clone())
}

#[test]
fn medium_followup_alternative_fires_and_answers_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let root = field_root();
    assert_eq!(
        root.get("seed").and_then(Value::as_i64),
        Some(SEED),
        "the field carries the scene seed"
    );
    let mut problems: Vec<String> = Vec::new();
    let request = injected(&root, false);
    let fired = {
        let cpu = dep_cpu::run_cpu_race(&request, SEED).expect("cpu race");
        let d0 = cpu.detail_events.iter().any(|&(h, s, d, _)| h == 0 && s.to_string() == ID && d == 0);
        let d1 = cpu.detail_events.iter().any(|&(h, s, d, _)| h == 0 && s.to_string() == ID && d == 1);
        (d0, d1)
    };
    if !fired.0 || !fired.1 {
        problems.push(format!(
            "the engine did not fire both {ID} details on runner 0 (detail 0 {0}, detail 1 {1}), the scene is vacuous",
            fired.0, fired.1
        ));
    }
    let full = check(&format!("10808 injected {ID} seed {SEED}"), &mut problems, &request);
    let control_request = injected(&root, true);
    let control = check(
        &format!("10808 injected {ID} detail-0-only seed {SEED}"),
        &mut problems,
        &control_request,
    );
    if let (Some(a), Some(b)) = (&full, &control) {
        let moved = a.frames.len() != b.frames.len()
            || a.frames.iter().zip(b.frames.iter()).any(|(x, y)| x != y);
        if !moved {
            problems.push(format!(
                "the {ID} follow-up moved no frames against the detail-0-only control, the scene is vacuous"
            ));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
