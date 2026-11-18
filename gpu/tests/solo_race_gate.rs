//! The solo race gate (handoff step 6 items 3-5 minus skills): the race-loop
//! kernel replays the CPU engine bit for bit over 24 no-skill single-runner
//! races: three courses, seeds 1..8, every frame value by `to_bits()`, the
//! tick count and finishTimeRaw.

#![cfg(feature = "race")]

use globalsim_gpu::{comparator, dep_cpu, race_exec};
use globalsim_gpu::race::RaceRunner;

const COURSES: [i32; 3] = [10504, 10604, 11004];
const SEEDS: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
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
fn solo_races_are_bit_exact_over_24_races() {
    let base = v77_line();
    let mut requests: Vec<(String, i64)> = Vec::new();
    for &course in &COURSES {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
        for &seed in &SEEDS {
            requests.push((race_exec::solo_request(&base, course, seed).expect("solo request"), seed));
        }
    }

    let out = race_exec::run(&requests).expect("kernel run");
    println!("device {} build {:.1} ms, {} races", out.device.name, out.build_ms, out.races.len());

    let mut problems: Vec<String> = Vec::new();
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
        assert_eq!(cpu.runners, 1, "{tag}: solo field");
        if race.ticks == 0 || race.ticks > race_exec::RACE_MAX_TICKS - 2 {
            problems.push(format!("{tag}: implausible tick count {}", race.ticks));
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (cpu_row, gpu_row) in cpu_frame.iter().zip(gpu_frame.iter()) {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                            COLUMNS[column],
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                        break 'frames;
                    }
                }
            }
        }
        match (cpu.finish_times.first(), race.finish_raw) {
            (Some(cpu), Some(gpu)) if cpu.to_bits() == gpu.to_bits() => {}
            (cpu, gpu) => problems.push(format!(
                "{tag}: finishTimeRaw cpu {:?} gpu {:?}",
                cpu.map(|v| v.to_bits()),
                gpu.map(f64::to_bits)
            )),
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
#[ignore = "multi-runner duel and spot are not ported; documents the first divergence"]
fn v77_multi_runner_race_stays_flagged() {
    let request = v77_line();
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);

    let cpu = dep_cpu::run_cpu_race(&request, 7).expect("cpu race");
    assert_eq!(cpu.runners, 9);
    let mut runner = race_exec::GpuRace;
    let outcome = runner.run_race(&request, 7);
    assert!(outcome.fallback.is_some(), "the multi-runner kernel must stay flagged");

    let mismatch = comparator::compare(&cpu.frames, &outcome)
        .expect_err("a flagged outcome never matches the whole race");
    println!("first divergence: {mismatch}");
    assert!(matches!(mismatch, comparator::Mismatch::RunnerCount { .. } | comparator::Mismatch::TickCount { .. }));
}
