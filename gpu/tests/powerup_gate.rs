//! PowerUp (effect type 3) on the retail lane (handoff E1, port 2). Horse 0
//! of the v77 roster takes one synthetic kind 3 skill per scene: a mid-race
//! fire, a constructor fire (the "always>=0" fold), a wisdom-gated
//! constructor fire (the lottery decides per seed, the kernel bakes the bump
//! when the roll enables), and a delivered fire (target 4, which the engine
//! rewrites to the self target at race_field.rs:465-477). Every frame
//! compares through to_bits(), the full reply must be byte-equal to the
//! engine's, and frames must move against the plain roster, at least over
//! the seeds for the lottery scene.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// The v77 request with a kind 3 skill on horse 0, its condition, wisdom
/// check and effect target taken from the scene.
fn powerup_value(seed: i64, condition: &str, wisdom: i64, target: i64, target_value: i64) -> Value {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let skill_text = format!(
            r#"{{"rarity":1,"wisdomCheck":{wisdom},"priority":0,"tags":[403],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":-1,
"durationUsage":1,"cooldown":0,
"effects":[{{"type":3,"target":{target},"targetValue":{target_value},"additionalActivate":0,
"valueUsage":1,"modifier":600000}}]}}],"isGeneralSkill":true}}"#
        );
        let skill = json::parse(&skill_text).expect("skill json");
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert("901301".into(), skill);
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h0) = &mut horses[0] else { unreachable!() };
        h0.insert("skills".into(), json::arr(vec![Value::String("901301".into())]));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn plain_request(seed: i64) -> String {
    request_text(&powerup_value(seed, "hp_per<=-1", 0, 1, 0))
}

fn compare_frames(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64, race: &race_exec::RaceGpu) {
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
                        "{tag} tick {tick} runner {runner}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// One scene over the seeds: frames bit-exact, full reply byte-equal, and
/// how many seeds the skill moved the replay. A cross-target effect sends
/// to at most `target_value` receivers.
fn run_scene(
    name: &str,
    condition: &str,
    wisdom: i64,
    target: i64,
    target_value: i64,
    problems: &mut Vec<String>,
) -> usize {
    let mut changed = 0usize;
    for &seed in &SEEDS {
        let tag = format!("{name} seed {seed}");
        let request = request_text(&powerup_value(seed, condition, wisdom, target, target_value));
        let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        compare_frames(&tag, problems, &request, seed, race);
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("{tag}: full reply differs from the engine"));
        }
        let plain = race_exec::run(&[(plain_request(seed), seed)]).expect("plain kernel run");
        let plain_race = &plain.races[0];
        if plain_race.fallback.is_some() {
            problems.push(format!("{tag}: plain control refused"));
            continue;
        }
        let differs = plain_race.frames.len() != race.frames.len()
            || plain_race.frames.iter().zip(race.frames.iter()).any(|(a, b)| a != b);
        if differs {
            changed += 1;
        }
    }
    println!("scene {name}: the PowerUp moved the replay in {changed} of {} seeds", SEEDS.len());
    changed
}

#[test]
fn retail_powerup_scenes_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut totals = 0usize;
    totals += run_scene("mid-race", "hp_per<=99", 0, 1, 0, &mut problems);
    totals += run_scene("constructor", "always>=0", 0, 1, 0, &mut problems);
    let gated = run_scene("gated constructor", "always>=0", 1, 1, 0, &mut problems);
    if gated == 0 {
        problems.push("the wisdom lottery enabled the gated scene on no seed, the scene is vacuous".into());
    }
    totals += gated;
    totals += run_scene("delivered", "hp_per<=99", 0, 4, 4, &mut problems);
    if totals == 0 {
        problems.push("no scene changed any replay, the PowerUp never fired".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
