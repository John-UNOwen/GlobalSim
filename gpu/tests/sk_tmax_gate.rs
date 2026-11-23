//! The tag-cap buckets (handoff X4). Horse 0 of the v77 roster takes 140
//! synthetic skills whose random trigger terms each add a tag row, a tag pool
//! past the fixed 4096 bytes the lane grew with. A compile-only check proves
//! the field crossed the old cap and landed in the second bucket. The replay
//! compares every frame with the CPU engine through to_bits() and against the
//! plain roster, so the wider pool cannot pass by changing nothing. The
//! second case puts both tag buckets into one batch: the launch must split
//! them and answer both bit exactly. The third case compiles the five real CM
//! fields whole, with no skill stripped for capacity.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec, skills_compile};
use json::Value;

const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// The v77 request with horse 0 replaced by `count` clones of the 200341
/// definition gated on `phase_random`, so every record spends a tag row
/// (`T<id>:<alt>:condition:phase_random`) and a wisdom row. One effect each
/// keeps the effect pool inside its first bucket, so only the tag pool can
/// overflow.
fn wide_tag_value(course: i32, seed: i64, count: usize) -> Value {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(skill_data) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        let template = skill_data.get("200341").expect("v77 skillData has 200341").clone();
        let mut skills: Vec<Value> = Vec::new();
        for n in 0..count {
            let mut def = template.clone();
            if let Value::Object(entry) = &mut def {
                // Not a general skill, so the pool the reservation pass keeps
                // stays out of the way and only the tag pool can overflow.
                entry.insert("isGeneralSkill".into(), Value::Bool(false));
                if let Some(Value::Array(alts)) = entry.get_mut("alternatives") {
                    let alt = alts.first_mut().expect("200341 carries one alternative");
                    if let Value::Object(a) = alt {
                        let first = a
                            .get("effects")
                            .and_then(Value::as_array)
                            .and_then(|e| e.first().cloned())
                            .expect("200341 carries one effect");
                        a.insert("effects".into(), json::arr(vec![first]));
                        a.insert("condition".into(), Value::String(format!("phase_random=={}", n % 4)));
                    }
                }
            }
            let id = format!("95{n:04}");
            skill_data.insert(id.clone(), def);
            skills.push(Value::String(id));
        }
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h0) = &mut horses[0] else { unreachable!() };
        h0.insert("skills".into(), json::arr(skills));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
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
    for (runner, cpu) in cpu.finish_times.iter().enumerate() {
        let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
        if cpu.to_bits() != gpu.to_bits() {
            problems.push(format!(
                "{tag} runner {runner} finishTimeRaw: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                cpu.to_bits(),
                gpu.to_bits()
            ));
        }
    }
}

#[test]
fn wide_tag_field_rides_the_second_bucket_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);

    // Compile only: the fixed pool still refuses, the ladder answers at the
    // second tag bucket with the same tag count.
    let root = wide_tag_value(10504, 1, 140);
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    let fixed = skills_compile::plan(&root, &course);
    assert_eq!(
        fixed.reason.as_deref(),
        Some("skill table overflow"),
        "the fixed 4096-byte tag pool must still refuse this field"
    );
    assert!(fixed.tags_used > 4096, "the field really passes the old cap: {} bytes", fixed.tags_used);
    let extras = skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let laddered = skills_compile::plan_retail_with(&root, &course, &extras);
    assert_eq!(laddered.reason, None, "the ladder compiles the field");
    assert_eq!(laddered.tmax, 8192, "one doubling covers the field");
    assert_eq!(laddered.emax, skills_compile::SK_EMAX_BUCKETS[0], "the effect pool stayed inside its first bucket");
    assert!(laddered.words[skills_compile::SK_NTAG] > 4096, "every tag row is in the table");

    // Replay: every frame bit exact, and different from the plain roster.
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut plain: Vec<(String, i64)> = Vec::new();
    for &seed in &[1i64, 2, 3, 4] {
        let request = request_text(&wide_tag_value(10504, seed, 140));
        requests.push((request, seed));
        plain.push((
            race_exec::multi_request(&request_text(&wide_tag_value(10504, seed, 0)), 10504, seed).expect("plain request"),
            seed,
        ));
    }
    let out = race_exec::run(&requests).expect("kernel run");
    let plain_out = race_exec::run(&plain).expect("plain kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut changed = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = format!("seed {seed}");
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason} kernel_flag {} arg {}", race.kernel_flag, race.kernel_arg));
            continue;
        }
        compare_frames(&tag, &mut problems, request, *seed, race);
        let plain_race = &plain_out.races[index];
        let differs = plain_race.frames.len() != race.frames.len()
            || plain_race.frames.iter().zip(race.frames.iter()).any(|(a, b)| a != b);
        if differs {
            changed += 1;
        }
    }
    println!("the 140-tag field changed the replay in {changed} of {} races", requests.len());
    if changed == 0 {
        problems.push("no race changed against the plain roster, the skills never fired".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn one_batch_with_two_tag_buckets_answers_both_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    // 100 random terms stay inside 4096 bytes, 140 pass it. One batch holds
    // both, so the launch splits the groups by (effect, tag) bucket pair.
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for (count, want_tmax) in [(100usize, 4096usize), (140, 8192)] {
        let root = wide_tag_value(10504, 1, count);
        let extras =
            skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
        let plan = skills_compile::plan_retail_with(&root, &course, &extras);
        assert_eq!(plan.reason, None, "{count} tags land in a tag bucket");
        assert_eq!(plan.tmax, want_tmax, "{count} tags land in tag bucket {want_tmax}");
        for seed in [1i64, 2] {
            let request = request_text(&wide_tag_value(10504, seed, count));
            requests.push((request, seed));
            tags.push(format!("{count} tags bucket {want_tmax} seed {seed}"));
        }
    }
    let out = race_exec::run(&requests).expect("kernel run");
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = &tags[index];
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        compare_frames(tag, &mut problems, request, *seed, race);
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Drop every copy of the named skill from every runner, string or entry.
fn strip_skill(root: &mut Value, id: &str) -> usize {
    let held = |x: &Value| x.as_str() == Some(id) || x.get("skillId").and_then(Value::as_str) == Some(id);
    let mut removed = 0usize;
    let Value::Object(map) = root else { unreachable!() };
    let Some(Value::Array(horses)) = map.get_mut("horses") else { unreachable!() };
    for h in horses.iter_mut() {
        if let Value::Object(hm) = h {
            if let Some(Value::Array(skills)) = hm.get_mut("skills") {
                let before = skills.len();
                skills.retain(|x| !held(x));
                removed += before - skills.len();
            }
        }
    }
    removed
}

/// Strip the skills the compiler refuses by name (the X5 families), so only a
/// capacity overflow can keep a CM field from compiling whole.
fn strip_refused(root: &mut Value) -> usize {
    let extras = skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course_id = root.get("courseId").and_then(Value::as_i64).expect("course id");
    let course = courses.get(&course_id.to_string()).cloned().expect("course entry");
    let mut removed = 0usize;
    loop {
        let plan = skills_compile::plan_retail_with(root, &course, &extras);
        let Some(reason) = &plan.reason else { return removed };
        assert!(!plan.effects_overflow && !plan.tags_overflow, "the CM field still overflows a cap: {reason}");
        let Some(skill) = reason.split_whitespace().skip_while(|w| *w != "skill").nth(1) else {
            panic!("the CM field refused without a skill id: {reason}");
        };
        let hit = strip_skill(root, skill);
        assert!(hit > 0, "strip stuck on {skill}: {reason}");
        removed += hit;
    }
}

#[test]
fn the_five_cm_fields_compile_whole_on_the_tag_ladder() {
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let extras = skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    // course -> the tag bucket the field is expected to need.
    let fields = [
        ("cm-10701-race.jsonl", 10701i32, 4096usize),
        ("cm-10903-race.jsonl", 10903, 8192),
        ("cm-10808-race.jsonl", 10808, 8192),
        ("cm-10506-race.jsonl", 10506, 4096),
        ("cm-11103-race.jsonl", 11103, 8192),
    ];
    for (file, course_id, want_tmax) in fields {
        let text = std::fs::read_to_string(repo_file(&format!("examples/{file}"))).expect("field");
        let mut root: Value = json::parse(text.lines().next().expect("first line")).expect("field parses");
        let stripped = strip_refused(&mut root);
        let course = courses.get(&course_id.to_string()).cloned().expect("course entry");
        let plan = skills_compile::plan_retail_with(&root, &course, &extras);
        assert_eq!(plan.reason, None, "{file} compiles whole after {stripped} named strips");
        assert_eq!(plan.tmax, want_tmax, "{file} rides tag bucket {want_tmax}");
        println!("{file}: tags {} bytes, tag bucket {}", plan.tags_used, plan.tmax);
    }
}
