//! The section pacemaker reads the order published before the rank pass
//! (handoff X3). The engine's update computes it from the order the previous
//! rank pass published, while the kernel computed it from the order the same
//! field_update had just republished. On the skill-less cm-11103 dirt field
//! the section pacemaker picked runner 1 where the engine kept runner 7,
//! runner 1's PaceDown -0.5 acceleration flipped at tick 95, and the kernel
//! finished 5 ticks short of the engine (1632 ticks). The scene is the field
//! with every skill removed on three seeds, frames compared through to_bits()
//! and the full reply compared byte for byte to the engine's.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const DIRT: i32 = 11103;
const SEEDS: [i64; 3] = [1614418797, 1614418798, 1614418796];

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

/// The request with every runner's skill list emptied.
fn strip_all_skills(root: &Value) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    for h in hs.iter_mut() {
        let Value::Object(hm) = h else { unreachable!() };
        hm.insert("skills".into(), Value::Array(Vec::new()));
    }
    request_text(&v)
}

/// One kernel race: frames bit-exact against the engine and the full reply
/// byte-equal.
fn check(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64) {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(reason) => {
            problems.push(format!("{tag}: kernel run failed: {reason}"));
            return;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return;
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
}

#[test]
fn dirt_bare_pacemaker_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{DIRT}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(DIRT, geometry);
    let root = field_root("examples/cm-11103-race.jsonl");
    let mut problems: Vec<String> = Vec::new();
    for &seed in &SEEDS {
        let request = strip_all_skills(&seeded(&root, seed));
        check(&format!("11103 bare seed {seed}"), &mut problems, &request, seed);
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
