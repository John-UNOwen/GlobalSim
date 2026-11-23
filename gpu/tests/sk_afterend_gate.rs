//! The engine's detail-zero placeholder (handoff X1). When a retail skill's
//! first alternative collapses, the builder still installs a detail-zero record
//! with that alternative's effects, one window past the course end, cooldown 0
//! (race_solver_builder.rs:594-625). The scan can never reach it, but a
//! kind-37 random-gold request reserves it and the reservation fire bypasses
//! the window, so a collapsed skill can still activate. cm-10903 runner 7 is
//! the pair: 564 Escapades requests every eligible gold, Dominator's own
//! trigger never opens on this course.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec, skills_compile};
use json::Value;

const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn has(flags: u32, bit: u32) -> bool {
    flags & bit != 0
}

/// cm-10903 at `seed` with every runner stripped, so runner 7 alone holds
/// `keep` and nothing else can reserve or fire.
fn stripped_field(keep: &[&str], seed: i64) -> Value {
    let text = std::fs::read_to_string(repo_file("examples/cm-10903-race.jsonl")).expect("field");
    let mut root: Value = json::parse(text.lines().next().expect("first line")).expect("field parses");
    let Value::Object(map) = &mut root else { unreachable!() };
    map.insert("seed".into(), Value::Number(seed as f64));
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for (i, h) in horses.iter_mut().enumerate() {
        let Value::Object(hm) = h else { unreachable!() };
        let Some(Value::Array(skills)) = hm.get_mut("skills") else { continue };
        let mine: Vec<Value> = if i == 7 {
            skills
                .iter()
                .filter(|s| keep.contains(&s.get("skillId").and_then(Value::as_str).unwrap_or("")))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        *skills = mine;
    }
    root
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

fn course_entry(root: &Value) -> Value {
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course_id = root.get("courseId").and_then(Value::as_i64).expect("course id");
    courses.get(&course_id.to_string()).cloned().expect("course entry")
}

/// The one record of `sid`: its flags, its baked cooldown double and its
/// effect count.
fn record_of(plan: &skills_compile::SkillsPlan, sid: u32) -> (u32, f64, usize) {
    let found: Vec<usize> = (0..plan.words[skills_compile::SK_NREC] as usize)
        .filter(|rec| {
            plan.words[skills_compile::SK_RECORDS + skills_compile::SK_RSTRIDE * rec] == sid
        })
        .collect();
    assert_eq!(found.len(), 1, "the stripped field carries one record of {sid}");
    let at = skills_compile::SK_RECORDS + skills_compile::SK_RSTRIDE * found[0];
    let hi = plan.words[at + 14] as u64;
    let cooldown = f64::from_bits((hi << 32) | plan.words[at + 13] as u64);
    (plan.words[at + 2], cooldown, plan.words[at + 5] as usize)
}

#[test]
fn the_collapsed_gold_keeps_its_effects_and_zero_cooldown() {
    let root = stripped_field(&["201151", "910071"], 1);
    let extras =
        skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let plan = skills_compile::plan_retail_with(&root, &course_entry(&root), &extras);
    assert_eq!(plan.reason, None, "the pair field compiles");

    let (flags, cooldown, effects) = record_of(&plan, 201151);
    assert!(has(flags, skills_compile::SK_FLAG_DEAD), "Dominator's alternative still collapses");
    assert!(
        has(flags, skills_compile::SK_FLAG_AFTEREND),
        "the placeholder is marked, so the kernel gives it the after-end window"
    );
    let pool = skills_compile::SK_FLAG_PRIMARY
        | skills_compile::SK_FLAG_GENERAL
        | skills_compile::SK_FLAG_RSELIG
        | skills_compile::SK_FLAG_RANGOLD;
    assert_eq!(flags & pool, pool, "the placeholder stays a random-gold candidate");
    assert!(effects > 0, "the placeholder keeps the collapsed alternative's effects");
    assert_eq!(cooldown, 0.0, "the builder's placeholder carries cooldown 0, not the wire's 500");

    let (flags, cooldown, _) = record_of(&plan, 910071);
    assert_eq!(flags & skills_compile::SK_FLAG_AFTEREND, 0, "Escapades fires on its own trigger");
    assert_eq!(cooldown, 500.0, "a live record keeps the wire cooldown of 5000000 ms");
}

/// The pool the reservation pass keeps per runner is SK_PICKMAX wide, so a
/// field that offers more candidates is refused instead of silently truncated.
#[test]
fn a_pool_past_the_pick_cap_is_refused_by_name() {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("v77");
    let mut root: Value = json::parse(text.lines().next().expect("first line")).expect("v77 parses");
    let Value::Object(map) = &mut root else { unreachable!() };
    let Value::Object(skill_data) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
    let template = skill_data.get("201151").expect("v77 carries 201151").clone();
    let mut skills: Vec<Value> = Vec::new();
    for n in 0..skills_compile::SK_PICKMAX + 1 {
        let mut def = template.clone();
        if let Value::Object(entry) = &mut def {
            if let Some(Value::Array(alts)) = entry.get_mut("alternatives") {
                let alt = alts.first_mut().expect("201151 carries one alternative");
                if let Value::Object(a) = alt {
                    a.insert("condition".into(), Value::String(format!("phase_random=={}", n % 4)));
                }
            }
        }
        let id = format!("96{n:03}");
        skill_data.insert(id.clone(), def);
        skills.push(Value::String(id));
    }
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for (i, h) in horses.iter_mut().enumerate() {
        let Value::Object(hm) = h else { unreachable!() };
        let owned: Vec<Value> = if i == 0 { skills.clone() } else { Vec::new() };
        hm.insert("skills".into(), json::arr(owned));
    }
    let extras =
        skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let plan = skills_compile::plan_retail_with(&root, &course_entry(&root), &extras);
    let reason = plan.reason.expect("one runner over the pool cap is refused");
    assert!(reason.contains("pool slots"), "the refusal names the pool cap: {reason}");
}

/// The kernel's pool arrays and the host's refusal share one number.
#[test]
fn the_kernel_pool_array_is_the_host_cap() {
    let text = include_str!("../src/kernels/skills.cl");
    let line = text
        .lines()
        .find(|l| l.starts_with("#define SK_PICKMAX"))
        .expect("skills.cl defines SK_PICKMAX");
    let cap: u32 = line
        .split_whitespace()
        .last()
        .expect("the define carries a value")
        .parse()
        .expect("the value is a number");
    assert_eq!(cap as usize, skills_compile::SK_PICKMAX, "host and kernel agree on the pool width");
}

fn compare_frames(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64, race: &race_exec::RaceGpu) {
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
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, race_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(race_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        COLUMNS[column],
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

fn detail_count(reply: &str, sid: i64) -> usize {
    let value = json::parse(reply).expect("reply parses");
    value
        .get("skillDetailEvents")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter(|e| e.get("skillId").and_then(Value::as_i64) == Some(sid)).count())
        .unwrap_or(0)
}

#[test]
fn the_reserved_placeholder_fires_and_matches_the_engine() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10903.json")).expect("geometry");
    globalsim::register_course_geometry(10903, geometry);
    let mut tags: Vec<String> = Vec::new();
    let mut requests: Vec<(String, i64)> = Vec::new();
    let pair_seeds: Vec<i64> = vec![1, 2, 3];
    for &seed in &pair_seeds {
        let root = stripped_field(&["201151", "910071"], seed);
        tags.push(format!("pair seed {seed}"));
        requests.push((request_text(&root), seed));
    }
    // Alone, the collapsed gold must never fire: its own window never opens.
    let alone = stripped_field(&["201151"], 1);
    tags.push("dominator alone seed 1".into());
    requests.push((request_text(&alone), 1));
    // The same field without skills answers as before.
    let plain = stripped_field(&[], 1);
    tags.push("no skills seed 1".into());
    requests.push((request_text(&plain), 1));

    let out = race_exec::run(&requests).expect("kernel run");
    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        compare_frames(&tags[index], &mut problems, request, *seed, &out.races[index]);
    }

    // The pair replays must reach the full reply channel, and both sides must
    // report the same activations of the collapsed gold.
    let full = race_exec::run_full(&requests[..pair_seeds.len()]).expect("full kernel run");
    for index in 0..pair_seeds.len() {
        let (request, seed) = &requests[index];
        let tag = &tags[index];
        let race = &full.races[index];
        let gpu = match race.reply.as_ref().map(|_| full_reply::build(request, *seed, race)) {
            Some(Ok(gpu)) => gpu,
            Some(Err(why)) => {
                problems.push(format!("{tag}: gpu reply failed: {why}"));
                continue;
            }
            None => {
                problems.push(format!("{tag}: no reply channel"));
                continue;
            }
        };
        let engine = match dep_cpu::run_cpu_binary(request, *seed) {
            Ok(engine) => engine,
            Err(why) => {
                problems.push(format!("{tag}: engine reply failed: {why}"));
                continue;
            }
        };
        let (gpu_n, engine_n) = (detail_count(&gpu, 201151), detail_count(&engine.reply, 201151));
        println!("{tag}: Dominator detail events gpu {gpu_n} engine {engine_n}");
        if gpu_n != engine_n {
            problems.push(format!("{tag}: Dominator detail events gpu {gpu_n} engine {engine_n}"));
        }
        if engine_n > 0 {
            fired += 1;
        }
    }
    if fired == 0 {
        problems.push("no seed activated the placeholder in the engine, the gate proves nothing".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
