//! C2 local work size gate: the race launch forced to several work-group sizes
//! must answer bit for bit like the driver's choice and like the CPU engine.
//!
//! One work item is one whole race and lanes share no `__local` memory or
//! barriers, so the work-group size only schedules. Forcing a size also pads
//! the 1-D launch to a work-group multiple, so the batch of three races runs
//! under group sizes that leave idle tail lanes (5 pads to 5, 8 to 8, 32 to 32).
//! Those lanes must return without touching any buffer, which is what this gate
//! checks: every forced size equals the unforced one and the oracle.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, race_exec};

const CM: &str = "examples/cm-10701-race.jsonl";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn bits(frames: &[Vec<[f64; 6]>]) -> Vec<u64> {
    frames
        .iter()
        .flat_map(|frame| frame.iter())
        .flat_map(|row| row.iter())
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn forced_local_work_sizes_are_bit_exact() {
    let base = std::fs::read_to_string(repo_file(CM))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry file");
    globalsim::register_course_geometry(10701, geometry.clone());
    let mut ctx = race_exec::RaceContext::open().expect("context");
    ctx.register_geometry(10701, &geometry).expect("geometry");
    let requests: Vec<(String, i64)> = (1..=3i64)
        .map(|s| (race_exec::multi_request(&base, 10701, s).expect("request"), s))
        .collect();

    // Sizes below and above the three-race count: 5 and 32 pad the launch and
    // leave idle lanes, 1 divides it exactly. None keeps the pinned default.
    let sizes: [Option<usize>; 5] = [None, Some(1), Some(5), Some(8), Some(32)];
    let mut problems: Vec<String> = Vec::new();
    let mut reference: Option<Vec<Vec<u64>>> = None;
    for size in sizes {
        ctx.set_local_work_size(size);
        let out = ctx.run_requests(&requests, None).expect("launch");
        let tag = match size {
            Some(n) => format!("local {n}"),
            None => format!("pinned default {}", race_exec::RACE_LOCAL_WORK_SIZE),
        };
        let per_race: Vec<Vec<u64>> = out.races.iter().map(|r| bits(&r.frames)).collect();
        for (index, race) in out.races.iter().enumerate() {
            if let Some(reason) = &race.fallback {
                problems.push(format!("{tag}: seed {} fell back: {reason}", requests[index].1));
            }
        }
        match &reference {
            Some(want) => {
                for index in 0..per_race.len() {
                    if want[index] != per_race[index] {
                        problems.push(format!("{tag}: seed {} frames differ from the pinned default", requests[index].1));
                    }
                }
            }
            None => reference = Some(per_race),
        }
    }

    // The pinned default column is already engine bit-exact (the equivalence and
    // slice gates), so pin it here against the CPU oracle to anchor the chain.
    let cpu = dep_cpu::run_cpu_race(&requests[0].0, requests[0].1).expect("cpu race");
    ctx.set_local_work_size(Some(race_exec::RACE_LOCAL_WORK_SIZE));
    let pinned = ctx.run_requests(&requests, None).expect("pinned launch");
    if bits(&cpu.frames) != bits(&pinned.races[0].frames) {
        problems.push("pinned default seed 1 frames differ from the CPU engine".to_string());
    }
    if !problems.is_empty() {
        panic!("{} local-size mismatches:\n{}", problems.len(), problems.join("\n"));
    }
    println!("local work sizes {:?}: all bit-exact, {} ticks", sizes.map(|s| s.unwrap_or(0)), cpu.frames.len());
}
