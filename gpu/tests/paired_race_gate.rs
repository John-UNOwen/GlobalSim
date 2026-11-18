//! The paired-mode race gate: the same 9-horse v77 roster as the multi
//! suite, requested with `pairedRandomStreams: true`, where the engine gives
//! every runner its own RUN substream. Every frame value, the tick count and
//! finishTimeRaw compare through `to_bits()` against the CPU engine with zero
//! fallback. A separate check proves the paired frames differ from the
//! unpaired ones, so the gate cannot pass on a missed mode switch.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, race_exec};

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

/// Every tick, every runner, every column compared through to_bits(), plus
/// equal frame counts and equal finishTimeRaw bits. Returns the compared
/// value count, or the problem list when the race mismatched.
fn compare_race(
    tag: &str,
    cpu: &dep_cpu::CpuRace,
    race: &race_exec::RaceGpu,
) -> (usize, Vec<String>) {
    let mut problems: Vec<String> = Vec::new();
    let mut values = 0usize;
    if race.ticks == 0 || race.ticks > race_exec::RACE_MAX_TICKS - 2 {
        problems.push(format!("{tag}: implausible tick count {}", race.ticks));
    }
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        return (values, problems);
    }
    if cpu.frames.len() - 1 != race.ticks {
        problems.push(format!("{tag}: loop ticks gpu {} cpu {}", race.ticks, cpu.frames.len() - 1));
    }
    'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                values += 1;
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
    (values, problems)
}

#[test]
fn v77_paired_races_are_bit_exact() {
    let base = v77_line();
    let plan: [(i32, &[i64]); 3] = [
        (10504, &[1, 2, 3, 4, 5, 6, 7, 8]),
        (10604, &[1, 2, 3, 4]),
        (11004, &[1, 2, 3, 4]),
    ];
    let mut requests: Vec<(String, i64)> = Vec::new();
    for &(course, seeds) in &plan {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
        for &seed in seeds {
            requests.push((race_exec::paired_request(&base, course, seed).expect("paired request"), seed));
        }
    }

    let out = race_exec::run(&requests).expect("kernel run");
    println!("device {} build {:.1} ms, {} races", out.device.name, out.build_ms, out.races.len());

    let mut problems: Vec<String> = Vec::new();
    let mut values = 0usize;
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
        assert!(cpu.runners <= race_exec::RACE_MAXR, "{tag}: roster inside the kernel");
        let (count, race_problems) = compare_race(&tag, &cpu, race);
        println!("{tag}: {} ticks, {} compared values", cpu.frames.len() - 1, count);
        values += count;
        problems.extend(race_problems);
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
    println!("{} compared values over {} paired races", values, requests.len());
}

/// The mode switch is real: for the first planned race the CPU engine's
/// unpaired frames differ from its paired ones, at least in one reported
/// field value. The paired gate above already pins the GPU to the paired
/// CPU frames, so a differing field value pins the GPU to the other stream
/// set as well.
#[test]
fn paired_requests_change_the_race() {
    let base = v77_line();
    let (course, seed) = (10504, 1);
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .expect("geometry file");
    globalsim::register_course_geometry(course, geometry);
    let paired = race_exec::paired_request(&base, course, seed).expect("paired request");
    let unpaired = race_exec::multi_request(&base, course, seed).expect("unpaired request");

    let paired = dep_cpu::run_cpu_race(&paired, seed).expect("paired cpu race");
    let unpaired = dep_cpu::run_cpu_race(&unpaired, seed).expect("unpaired cpu race");

    let mut difference: Option<String> = None;
    'frames: for (tick, (paired_frame, unpaired_frame)) in
        paired.frames.iter().zip(unpaired.frames.iter()).enumerate()
    {
        for (runner, (paired_row, unpaired_row)) in
            paired_frame.iter().zip(unpaired_frame.iter()).enumerate()
        {
            for (column, (paired, unpaired)) in
                paired_row.iter().zip(unpaired_row.iter()).enumerate()
            {
                if paired.to_bits() != unpaired.to_bits() {
                    difference = Some(format!(
                        "course {course} seed {seed} tick {tick} runner {runner} {}: unpaired {unpaired:?} (0x{:016x}), paired {paired:?} (0x{:016x})",
                        COLUMNS[column],
                        unpaired.to_bits(),
                        paired.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
    let difference = match difference {
        Some(found) => found,
        None => panic!("paired and unpaired frames are identical, the paired keying did not reach the race loop"),
    };
    println!("{difference}");
}
