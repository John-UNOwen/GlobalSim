//! The previous-tick advantage marks and the heal-history fan (handoff X5
//! family 2), on the retail lane, carried by the exact wire skills.
//!
//! 202361 and 202362 arm on ground_type==2, distance_rate>=50 and
//! is_other_character_activate_advantage_skill==9, then fan a CurrentSpeed
//! debuff to the targetValue 1 most recent self heals (race_field.rs:387-392,
//! 442-453). The v77 field runs on dirt 10108 with every other runner's skill
//! list emptied, so the only Recovery marks and the only heal history come
//! from one planted self-heal skill on runner 1.
//!
//! Each carrying scene compares every frame through to_bits() against the CPU
//! oracle and the full reply with the engine, so the mark timing and the fan
//! receiver both have to match. The non-vacuity control is the same field
//! without the planted heal: with no kind-9 mark anywhere the engine must
//! never fire the carried skill, and it does fire it with the heal planted.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10108;
const SEEDS: [i64; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// The exact wire definition of one id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// Real self-heal (Recovery) ids spread across runners 1..8 so a mid-race
/// heal lands a kind-9 mark within reach of a runner past half distance. Each
/// is wisdomCheck 0, so it is always enabled and fires once live when its
/// condition opens.
const HEALS: [&str; 4] = ["10321", "10621", "110301", "110011"];

/// Runner 0 carries one id, runners 1..8 each carry one `heals` entry
/// round-robin (empty for the no-mark control), every other runner runs
/// skill-less, and no runner has a team.
fn field_value(seed: i64, id: &str, def: &Value, heals: &[(&str, Value)]) -> Value {
    let mut value = {
        let text =
            std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
        json::parse(text.lines().next().expect("first line")).expect("v77 parses")
    };
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        for (hid, hdef) in heals {
            sd.insert((*hid).into(), (*hdef).clone());
        }
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (i, horse) in horses.iter_mut().enumerate() {
            let Value::Object(h) = horse else { unreachable!() };
            h.insert("teamId".into(), Value::Number(0.0));
            if i == 0 {
                // Push the wisdom lottery over its enable line (chance is
                // 100 - 9000 / base_wisdom, retail_initialization.rs:247) so a
                // carried wisdomCheck skill is enabled and can fire.
                h.insert("wisdom".into(), Value::Number(9000.0));
            }
            let skills = if i == 0 {
                json::arr(vec![json::obj(vec![
                    ("skillId", Value::String(id.into())),
                    ("level", Value::Number(1.0)),
                ])])
            } else if !heals.is_empty() && (1..=8).contains(&i) {
                let (hid, _) = heals[(i - 1) % heals.len()];
                json::arr(vec![json::obj(vec![
                    ("skillId", Value::String(hid.into())),
                    ("level", Value::Number(1.0)),
                ])])
            } else {
                json::arr(Vec::new())
            };
            h.insert("skills".into(), skills);
        }
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
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
            for (_column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
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

/// The engine reply lists an activation in skillEvents; runner 0 firing the
/// carried id is the firing signal, identical on the GPU by reply parity.
fn skill_fired(reply: &str, id: i64) -> bool {
    let root = json::parse(reply).expect("reply parses");
    let Some(events) = root.get("skillEvents").and_then(Value::as_array) else {
        return false;
    };
    events.iter().any(|row| {
        row.get("horseIndex").and_then(Value::as_f64) == Some(0.0)
            && row.get("skillId").and_then(Value::as_f64) == Some(id as f64)
    })
}

#[test]
fn advantage_marks_and_heal_fan_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    for (name, id) in [("202361", "202361"), ("202362", "202362")] {
        let def = cm_skill(id);
        let heal_defs: Vec<(&str, Value)> =
            HEALS.iter().map(|h| (*h, cm_skill(h))).collect();
        let mut fired = 0usize;
        let mut control_fired = 0usize;
        for &seed in &SEEDS {
            for with_heal in [true, false] {
                let heals: &[(&str, Value)] = if with_heal { &heal_defs } else { &[] };
                let tag = format!("{name} seed {seed} heal {with_heal}");
                let request = request_text(&field_value(seed, id, &def, heals));
                let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
                let race = &run.races[0];
                if let Some(reason) = &race.fallback {
                    problems.push(format!("{tag}: fallback {reason}"));
                    continue;
                }
                compare_frames(&tag, &mut problems, &request, seed, race);
                let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
                let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
                let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
                if gpu_reply != engine {
                    problems.push(format!("{tag}: full reply differs from the engine"));
                }
                if skill_fired(&engine, id.parse::<i64>().unwrap()) {
                    if with_heal {
                        fired += 1;
                    } else {
                        control_fired += 1;
                    }
                }
            }
        }
        println!("scene {name}: fired with the heal on {fired} seed(s), on {control_fired} without");
        if fired == 0 {
            problems.push(format!("scene {name}: the carried skill never fired with a mark present, the scene is vacuous"));
        }
        if control_fired != 0 {
            problems.push(format!(
                "scene {name}: the skill fired on {control_fired} seed(s) with no kind-9 mark, the condition term does no work"
            ));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
