//! The sampled-window skill set from the long-field reduce (handoff X8b).
//! Five of runner 4's seven skills are gated by corner_random, phase_random,
//! is_final_corner_random or straight_random, windows the engine hands out
//! from the shared int32 stream. Reduced to nine skills across runners 3 and
//! 4, cm-10506 diverged at tick 1360, runner 3, column 0. The bump-chain fix
//! that closed X2 and X9 also closed this set, and this gate keeps it that
//! way. The scene runs the field's seed and its neighbours, frames compared
//! through to_bits(), the full reply compared byte for byte, and the set must
//! move the replay against the fully stripped field so the scene stays
//! non-vacuous.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};

const LONG: i32 = 10506;
const SEEDS: [i64; 3] = [382661117, 382661118, 382661116];
const KEEP: [(usize, &[&str]); 2] = [
    (3, &["200012", "200142"]),
    (4, &["200352", "200382", "200532", "200541", "200552", "201192", "201242"]),
];

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

/// The field at `seed`, `None` strips every skill, `Some` keeps only the
/// listed ids on their runners.
fn scene(root: &Value, seed: i64, keep: Option<&[(usize, &[&str])]>) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    m.insert("seed".into(), Value::Number(seed as f64));
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    for (i, h) in hs.iter_mut().enumerate() {
        let Value::Object(hm) = h else { unreachable!() };
        let Some(Value::Array(s)) = hm.get_mut("skills") else { continue };
        let mine: Vec<Value> = match keep.and_then(|k| k.iter().find(|(r, _)| *r == i)) {
            Some((_, ids)) => s
                .iter()
                .filter(|x| x.get("skillId").and_then(Value::as_str).is_some_and(|id| ids.contains(&id)))
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        *s = mine;
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
fn long_sampled_skill_set_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{LONG}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(LONG, geometry);
    let root = field_root("examples/cm-10506-race.jsonl");
    let mut problems: Vec<String> = Vec::new();
    let mut head_moved = false;
    for &seed in &SEEDS {
        let set = scene(&root, seed, Some(&KEEP));
        let bare = scene(&root, seed, None);
        let raced = check(&format!("10506 set seed {seed}"), &mut problems, &set, seed);
        let control = check(&format!("10506 stripped seed {seed}"), &mut problems, &bare, seed);
        if let (Some(a), Some(b)) = (&raced, &control) {
            if seed == SEEDS[0] && frames_differ(a, b) {
                head_moved = true;
            }
        }
    }
    if !head_moved {
        problems.push(format!(
            "the nine-skill set moved no frames on seed {}, the scene is vacuous",
            SEEDS[0]
        ));
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
