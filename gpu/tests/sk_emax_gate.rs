//! The effect-cap buckets (handoff E1, port 1). Horse 0 of the v77 roster
//! takes 25 synthetic skills of eight effects each, a 200-entry effect pool
//! past the fixed 192-slot pool the classic lane keeps, sized like the real
//! CM field (199). A compile-only check proves the field crossed the old cap
//! and landed in the first retail bucket. The replay compares every frame
//! with the CPU engine through to_bits() and against the plain roster, so the
//! wider table cannot pass by changing nothing. The second case puts two
//! different buckets into one batch: the launch must split them and answer
//! both bit exactly.

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
/// definition, each alternative rewritten to eight copies of its first
/// effect and gated by its own `distance_rate` band so the skills activate
/// spread over the race instead of one tick (the per-kind timer list caps at
/// SK_SEQN entries). 200341 is one alternative with one effect, so the field
/// compiles `count * 8` effects.
fn wide_field_value(course: i32, seed: i64, count: usize) -> Value {
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
                if let Some(Value::Array(alts)) = entry.get_mut("alternatives") {
                    let alt = alts.first_mut().expect("200341 carries one alternative");
                    if let Value::Object(a) = alt {
                        let first = a
                            .get("effects")
                            .and_then(Value::as_array)
                            .and_then(|e| e.first().cloned())
                            .expect("200341 carries one effect");
                        a.insert("effects".into(), json::arr(vec![first; 8]));
                        a.insert(
                            "condition".into(),
                            json::Value::String(format!("distance_rate>={}&distance_rate<={}", 4 * n, 4 * n + 3)),
                        );
                    }
                }
            }
            let id = format!("90{n:04}");
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
    'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
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
fn two_hundred_effects_ride_the_first_bucket_bit_exact() {
    for &course in &[10504i32, 10510] {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json"))).expect("geometry");
        globalsim::register_course_geometry(course, geometry);
    }

    // Compile only: the field crosses the fixed pool and the retail ladder
    // answers it at the first bucket with the same effect count.
    let root = wide_field_value(10504, 1, 25);
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    let fixed = skills_compile::plan(&root, &course);
    assert_eq!(
        fixed.reason.as_deref(),
        Some("skill table overflow: too many effects"),
        "the fixed 192-slot pool must still refuse this field"
    );
    let extras = skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let laddered = skills_compile::plan_retail_with(&root, &course, &extras);
    assert_eq!(laddered.reason, None, "the ladder compiles the field");
    assert_eq!(laddered.emax, 256, "one bucket covers the CM-sized field");
    assert_eq!(laddered.words[skills_compile::SK_NEFF], 200, "25 skills of eight effects");

    // Replay: every frame bit exact, and different from the plain roster.
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut plain: Vec<(String, i64)> = Vec::new();
    for &course in &[10504i32, 10510] {
        for &seed in &[1i64, 2, 3, 4] {
            let request = request_text(&wide_field_value(course, seed, 25));
            requests.push((request, seed));
            plain.push((race_exec::multi_request(
                &request_text(&wide_field_value(course, seed, 0)),
                course,
                seed,
            ).expect("plain request"), seed));
        }
    }
    let out = race_exec::run(&requests).expect("kernel run");
    let plain_out = race_exec::run(&plain).expect("plain kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut changed = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        let course = dep_cpu::course_id(request).expect("course id");
        let tag = format!("course {course} seed {seed}");
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
    println!("the 200-effect field changed the replay in {changed} of {} races", requests.len());
    if changed == 0 {
        problems.push("no race changed against the plain roster, the skills never fired".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn one_batch_with_two_buckets_answers_both_bit_exact() {
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    // 25 skills fill the 200-entry pool into bucket 256, 33 skills overflow
    // it into bucket 512. One batch holds both, so the launch splits groups.
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for (count, want_bucket) in [(25usize, 256usize), (33, 512)] {
        let root = wide_field_value(10504, 1, count);
        let extras =
            skills_compile::ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
        let plan = skills_compile::plan_retail_with(&root, &course, &extras);
        assert_eq!(plan.reason, None, "{count} skills compile at some bucket");
        assert_eq!(plan.emax, want_bucket, "{count} skills land in bucket {want_bucket}");
        for seed in [1i64, 2] {
            let request = request_text(&wide_field_value(10504, seed, count));
            requests.push((request, seed));
            tags.push(format!("{count} skills bucket {want_bucket} seed {seed}"));
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
