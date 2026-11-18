//! The skill gate (handoff step 7): the race kernel runs the numeric skill
//! table end to end (init triggers and wisdom, timers, cooldowns, the scan
//! gate, the fire bookkeeping and the receiver folds) and replays the CPU
//! engine bit for bit. Six real skills ride on horse 0 of the v77 roster,
//! sampled conditions included, on the 2000 m course 10504 and the 2400 m
//! course 10510, seeds 1..4. Every frame value compares through to_bits()
//! and no race may fall back. One negative case checks that an unsupported
//! construct flags the whole race and names the construct. A third case
//! runs one skill per race over the full course and seed matrix in a single
//! batch. Batching is a tested dimension: the init-pass stride bug this port
//! found stayed invisible in solo batches and only crossed race boundaries
//! inside one launch.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec};

const SKILLS: [i64; 6] = [200361, 200341, 200333, 10071, 100401, 110591];
const COURSES: [i32; 2] = [10504, 10510];
const SEEDS: [i64; 4] = [1, 2, 3, 4];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_line() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

#[test]
fn six_skills_on_horse_zero_replay_bit_exact() {
    let base = v77_line();
    let mut requests: Vec<(String, i64)> = Vec::new();
    for &course in &COURSES {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
        for &seed in &SEEDS {
            let request =
                race_exec::skills_request(&base, course, seed, 0, &SKILLS, None).expect("skills request");
            requests.push((request, seed));
        }
    }

    // The same roster without the skills, for the fired-check: a skill set
    // that never changes any frame would pass the comparison vacuously.
    let mut plain: Vec<(String, i64)> = Vec::new();
    for (request, seed) in &requests {
        let course = dep_cpu::course_id(request).expect("course id");
        plain.push((race_exec::multi_request(&base, course, *seed).expect("plain request"), *seed));
    }

    let out = race_exec::run(&requests).expect("kernel run");
    let plain_out = race_exec::run(&plain).expect("plain kernel run");
    println!("device {} build {:.1} ms, {} races", out.device.name, out.build_ms, out.races.len());

    let mut problems: Vec<String> = Vec::new();
    let mut changed = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        let course = dep_cpu::course_id(request).expect("course id");
        let tag = format!("course {course} seed {seed}");
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        assert_eq!(cpu.runners, 9, "{tag}: v77 roster");
        if race.ticks == 0 || race.ticks > race_exec::RACE_MAX_TICKS - 2 {
            problems.push(format!("{tag}: implausible tick count {}", race.ticks));
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
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
        let plain_race = &plain_out.races[index];
        let differs = plain_race.frames.len() != race.frames.len()
            || plain_race
                .frames
                .iter()
                .zip(race.frames.iter())
                .any(|(a, b)| a != b);
        if differs {
            changed += 1;
        }
    }
    println!("skills changed the replay in {changed} of {} races", requests.len());
    if changed == 0 {
        problems.push("no race changed against the plain roster, the skills never fired".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn one_skill_per_race_over_the_matrix_bit_exact() {
    let base = v77_line();
    for &course in &COURSES {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
    }
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for &skill in &SKILLS {
        for &course in &COURSES {
            for &seed in &[1i64, 3] {
                let request =
                    race_exec::skills_request(&base, course, seed, 0, &[skill], None).expect("request");
                requests.push((request, seed));
                tags.push(format!("skill {skill} course {course} seed {seed}"));
            }
        }
    }
    let out = race_exec::run(&requests).expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = &tags[index];
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        // The engine reply's skillDetailEvents is the observation channel:
        // a horse-0 skill that the engine activated at least once.
        if cpu.detail_events.iter().any(|(runner, _, _, _)| *runner == 0) {
            fired += 1;
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
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
    }
    println!("engine activated a horse-0 skill in {fired} of {} races", requests.len());
    if fired == 0 {
        problems.push("the engine activated no horse-0 skill anywhere, the matrix compares vacuously".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn unsupported_construct_flags_the_whole_race() {
    let base = v77_line();
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);
    // duration_usage 4 keeps the delayed running-instance path the kernel has
    // no model for, so the whole race flags and the reason names the use.
    let request = race_exec::skills_request(&base, 10504, 1, 0, &SKILLS, Some((200333, 4)))
        .expect("patched request");
    let out = race_exec::run(&[(request.clone(), 1i64)]).expect("kernel run");
    let race = &out.races[0];
    match &race.fallback {
        Some(reason) => {
            assert!(
                reason.contains("duration_usage 4"),
                "fallback reason should name the construct, got {reason}"
            );
        }
        None => panic!("the patched race ran on the kernel without flagging"),
    }
}

/// A condition naming a random trigger the kernel has no sampler for must
/// refuse the whole race by that name, not run on a wrong or empty window. The
/// merged skill set the app ships carries several such terms (random_lot,
/// distance_rate_after_random, corner_random), so a refusal names the term.
#[test]
fn unsupported_random_term_refuses_by_name() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);
    // Rewrite one otherwise-accepted skill's condition to an unported random
    // term so the only reason to flag is the term name itself.
    let mut value = json::parse(&base).expect("base parses");
    if let json::Value::Object(map) = &mut value {
        if let Some(json::Value::Object(sd)) = map.get_mut("skillData") {
            if let Some(json::Value::Object(entry)) = sd.get_mut("200333") {
                if let Some(json::Value::Array(alts)) = entry.get_mut("alternatives") {
                    if let Some(json::Value::Object(alt)) = alts.get_mut(0) {
                        alt.insert(
                            "condition".into(),
                            json::Value::String("random_lot==1".into()),
                        );
                    }
                }
            }
        }
    }
    let mut patched = String::new();
    json::to_string_into(&value, &mut patched);
    let request = race_exec::skills_request(&patched, 10504, 1, 0, &SKILLS, None).expect("patched request");
    let out = race_exec::run(&[(request.clone(), 1i64)]).expect("kernel run");
    match &out.races[0].fallback {
        Some(reason) => assert!(
            reason.contains("random_lot"),
            "fallback reason should name the unsupported term, got {reason}"
        ),
        None => panic!("a race with an unported random term ran on the kernel"),
    }
}
