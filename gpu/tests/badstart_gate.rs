//! The is_badstart live leaf (handoff E1a, family 1), on the retail lane.
//!
//! The engine keeps is_badstart as a live check of the runner's own start
//! delay against 0.08 s (activation_conditions.rs:722-732). The kernel mirrors
//! that value in the sd start-delay word, seeded from the initialization word
//! (race_solver_retail.rs:623, lib.rs:743), and the leaf reads it every scan.
//!
//! The leaf is read through a remain_distance<=1500 term, so the record stays
//! shut at the constructor pass (the runner has the whole course ahead) and
//! fires once, partway in, where the kernel and the engine read the same retail
//! start delay. The engine evaluates a record that is already open at
//! construction against the classic race_solver.rs:995 draw of start_delay, a
//! value the retail lane overwrites at lib.rs:743 before the first tick and the
//! kernel never reproduces. Every real wire id keeps a traffic gate or a false
//! static term in front of is_badstart, so none of them is open at construction
//! (100261 gates on order and the final corner, and 202191 and 202192 gate on
//! distance_type==4, which folds dead on every course the lane simulates), and
//! the gate follows that same discipline by reading the leaf at runtime.
//!
//! Both comparison directions and both delay sides are pinned from the field's
//! own init delays. Each seed runs four scenes that carry the ==0 or the ==1
//! leaf on the lowest-delay runner (lo) or the highest-delay runner (hi):
//!   op0@lo, op0@hi, op1@lo, op1@hi
//! A scene fires exactly when the delay read by that runner on that seed
//! satisfies the operator, the delay being read from the engine reply, so the
//! expectation is derived, not guessed. Across the seeds the four cells of the
//! truth table all appear. The wire 100261 and its crafted ==1 mirror run as
//! scenes A and B for definition fidelity. Every run answers bit-exact frames
//! plus a byte-equal full reply.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// The exact wire definition of one family id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// A crafted definition, the same shape the CM wire entries use.
fn crafted(condition: &str, kind: i64, modifier: f64) -> Value {
    json::parse(&format!(
        r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{{"type":{kind},"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":{modifier},"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}]}}],
"isGeneralSkill":true}}"#
    ))
    .expect("crafted skill json")
}

/// One runner carries the named skills; the rest of the field is the plain v77.
fn field_value(seed: i64, carrier: usize, id: &str, def: &Value) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[carrier] else { unreachable!() };
        h.insert("skills".into(), json::arr(vec![Value::String(id.into())]));
    }
    value
}

/// The wire 100261, or its ==1 mirror, on horse 0.
fn wire_value(seed: i64, id: &str, def: &Value) -> Value {
    field_value(seed, 0, id, def)
}

fn plain_value(seed: i64) -> Value {
    let mut value = base_value();
    if let Value::Object(map) = &mut value {
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
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

/// The engine reply's per-runner startDelayTime, the init delay each leaf reads.
fn start_delays(request: &str, seed: i64) -> Vec<f64> {
    let reply = dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply;
    let root = json::parse(&reply).expect("engine reply parses");
    let Value::Array(results) = root.get("results").expect("results") else {
        unreachable!("the engine reply carries the results array")
    };
    results
        .iter()
        .map(|r| r.get("startDelayTime").and_then(Value::as_f64).expect("startDelayTime"))
        .collect()
}

#[test]
fn retail_badstart_leaf_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();

    // The wire 100261 and its ==1 mirror, carried for definition fidelity.
    let def_100261 = cm_skill("100261");
    let cm_text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let cm_root: Value = json::parse(cm_text.lines().next().expect("first line")).expect("cm parses");
    let mut def_mirror = cm_root
        .get("skillData")
        .and_then(|s| s.get("100261"))
        .cloned()
        .expect("cm skill 100261");
    if let Value::Object(map) = &mut def_mirror {
        let Value::Array(alts) = map.get_mut("alternatives").expect("alternatives") else {
            unreachable!()
        };
        for alt in alts {
            let Value::Object(a) = alt else { unreachable!() };
            let cond = a.get("condition").and_then(Value::as_str).expect("condition");
            a.insert("condition".into(), Value::String(cond.replace("is_badstart==0", "is_badstart==1")));
        }
    }

    let op0 = crafted("remain_distance<=1500&is_badstart==0", 27, 4500.0);
    let op1 = crafted("remain_distance<=1500&is_badstart==1", 27, 4500.0);

    // Truth-table tallies: [operator][true|false], summed over the seeds.
    let mut tally = [[0usize; 2]; 2];
    for &seed in &SEEDS {
        let plain_tag = format!("plain seed {seed}");
        let plain = match race_exec::run(&[(request_text(&plain_value(seed)), seed)]) {
            Ok(run) => run.races[0].clone(),
            Err(why) => {
                problems.push(format!("{plain_tag}: kernel run failed: {why}"));
                continue;
            }
        };
        if plain.fallback.is_some() {
            problems.push(format!("{plain_tag}: control refused"));
        }
        let delays = start_delays(&request_text(&plain_value(seed)), seed);
        let mut lo = 0usize;
        let mut hi = 0usize;
        for (i, &d) in delays.iter().enumerate() {
            if d < delays[lo] {
                lo = i;
            }
            if d > delays[hi] {
                hi = i;
            }
        }

        // Definition fidelity: the wire 100261 and its mirror, bit-exact only.
        let a_req = request_text(&wire_value(seed, "100261", &def_100261));
        let _ = run_kernel(&format!("100261 seed {seed}"), &mut problems, &a_req, seed);
        let b_req = request_text(&wire_value(seed, "920261", &def_mirror));
        let _ = run_kernel(&format!("100261 mirror seed {seed}"), &mut problems, &b_req, seed);

        // The four truth-table cells, on the field's own two delay extremes.
        let cells = [("0@lo", 0usize, lo), ("0@hi", 0, hi), ("1@lo", 1, lo), ("1@hi", 1, hi)];
        for (name, op, runner) in cells {
            let (id, def) = if op == 0 { ("920262", &op0) } else { ("920263", &op1) };
            let tag = format!("op{name} seed {seed} runner {runner}");
            let req = request_text(&field_value(seed, runner, id, def));
            let Some(race) = run_kernel(&tag, &mut problems, &req, seed) else { continue };
            let delay = delays[runner];
            let expect_fire = if op == 0 { delay <= 0.08 } else { delay > 0.08 };
            let fired = !frames_equal(&race, &plain);
            if fired != expect_fire {
                problems.push(format!(
                    "{tag}: is_badstart=={op} {} but delay {delay} says {}",
                    if fired { "fired" } else { "stayed shut" },
                    if expect_fire { "it should fire" } else { "it should stay shut" }
                ));
            }
            tally[op][usize::from(fired)] += 1;
        }
    }

    println!(
        "badstart truth table over {} seeds: ==0 fired {} shut {}, ==1 fired {} shut {}",
        SEEDS.len(),
        tally[0][1],
        tally[0][0],
        tally[1][1],
        tally[1][0]
    );
    for op in 0..2 {
        if tally[op][1] == 0 || tally[op][0] == 0 {
            problems.push(format!(
                "is_badstart=={op} never reached both outcomes (fired {}, shut {})",
                tally[op][1],
                tally[op][0]
            ));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
