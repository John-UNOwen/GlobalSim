//! F7 slice gate: a launch split into tick ranges must answer bit for bit
//! like one launch and like the CPU engine. The CM field (real builds, 159
//! skills) and the v77 field run under fixed and adaptive slice lengths,
//! plus the case where the finish tick lands exactly on a slice boundary.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, dep_cpu, json, race_exec};
use race_exec::SlicePolicy;

const CM: &str = "examples/cm-10701-race.jsonl";
const V77: &str = "examples/v77-race.jsonl";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn request_line(rel: &str) -> String {
    std::fs::read_to_string(repo_file(rel))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

/// Every replay column of every frame, as raw bits.
fn bits(frames: &[Vec<[f64; 6]>]) -> Vec<u64> {
    frames
        .iter()
        .flat_map(|frame| frame.iter())
        .flat_map(|row| row.iter())
        .map(|v| v.to_bits())
        .collect()
}

/// The same batch under one policy, every race unflagged.
fn run(
    ctx: &mut race_exec::RaceContext,
    requests: &[(String, i64)],
    policy: SlicePolicy,
) -> race_exec::RaceRun {
    let out = ctx
        .run_requests(requests, Some(policy))
        .expect("sliced launch run");
    for (index, race) in out.races.iter().enumerate() {
        assert!(race.fallback.is_none(), "seed {} fell back: {:?}", requests[index].1, race.fallback);
    }
    out
}

/// Bit-identical `tag` frames against the baseline's, column by column.
fn compare(problems: &mut Vec<String>, tag: &str, want: &[u64], got: &[u64]) {
    if want != got {
        let diff = want.iter().zip(got).position(|(a, b)| a != b);
        problems.push(match diff {
            Some(i) => format!("{tag}: frame bit {i} of {} differs (want {:#x}, got {:#x})", want.len(), want[i], got[i]),
            None => format!("{tag}: {} values vs {}", want.len(), got.len()),
        });
    }
}

fn fail(problems: &[String]) {
    panic!("{} mismatching runs:\n{}", problems.len(), problems.join("\n"));
}

#[test]
fn cm_field_slicing_is_bit_exact() {
    let base = request_line(CM);
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry file");
    globalsim::register_course_geometry(10701, geometry.clone());
    let mut ctx = race_exec::RaceContext::open().expect("context");
    ctx.register_geometry(10701, &geometry).expect("geometry");
    let seeds: Vec<i64> = (1..=4).collect();
    let requests: Vec<(String, i64)> = seeds
        .iter()
        .map(|&s| (race_exec::multi_request(&base, 10701, s).expect("request"), s))
        .collect();

    let one = run(&mut ctx, &requests, SlicePolicy::Off);
    let fixed = run(&mut ctx, &requests, SlicePolicy::Fixed(100));
    let adaptive = run(&mut ctx, &requests, SlicePolicy::Adaptive { target_ms: 500.0 });
    assert!(fixed.stage_ms.slices >= 9, "100-tick slices over an 837-tick race need at least 9 launches");
    assert!(adaptive.stage_ms.slices >= 2, "adaptive slicing must split the launch");

    let mut problems: Vec<String> = Vec::new();
    for index in 0..requests.len() {
        let tag = format!("seed {}", requests[index].1);
        compare(&mut problems, &format!("{tag} fixed"), &bits(&one.races[index].frames), &bits(&fixed.races[index].frames));
        compare(&mut problems, &format!("{tag} adaptive"), &bits(&one.races[index].frames), &bits(&adaptive.races[index].frames));
    }
    if !problems.is_empty() {
        fail(&problems);
    }

    // The one-shot column is already engine bit-exact (equivalence gates),
    // so pin the sliced runs against the CPU oracle directly as well.
    let cpu = dep_cpu::run_cpu_race(&requests[0].0, requests[0].1).expect("cpu race");
    let race = &fixed.races[0];
    assert_eq!(cpu.frames.len(), race.frames.len(), "tick count sliced vs cpu");
    compare(&mut problems, "seed 1 cpu", &bits(&cpu.frames), &bits(&race.frames));
    for (runner, cpu_raw) in cpu.finish_times.iter().enumerate() {
        let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
        if cpu_raw.to_bits() != gpu.to_bits() {
            problems.push(format!("seed 1 runner {runner} finishTimeRaw: cpu {cpu_raw:?}, gpu {gpu:?}"));
        }
    }
    if !problems.is_empty() {
        fail(&problems);
    }
    println!(
        "cm slicing: {} fixed launches max {:.1} ms, {} adaptive launches max {:.1} ms, {} ticks",
        fixed.stage_ms.slices, fixed.stage_ms.slice_max_ms,
        adaptive.stage_ms.slices, adaptive.stage_ms.slice_max_ms,
        one.races[0].ticks,
    );
}

/// The finish tick lands exactly on a slice boundary: a slice length that
/// divides the race's tick count (and, once, equals it), so the last tick
/// of the race is the last tick of the last slice.
#[test]
fn slice_boundary_at_the_finish_tick() {
    let mut problems: Vec<String> = Vec::new();
    for (file, course) in [(V77, 10504), (CM, 10701)] {
        let base = request_line(file);
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json"))).expect("geometry file");
        globalsim::register_course_geometry(course, geometry.clone());
        let mut ctx = race_exec::RaceContext::open().expect("context");
        ctx.register_geometry(course, &geometry).expect("geometry");
        let requests = vec![(race_exec::multi_request(&base, course, 1).expect("request"), 1i64)];

        let one = run(&mut ctx, &requests, SlicePolicy::Off);
        let ticks = one.races[0].ticks;
        let parts = (2..=ticks).find(|d| ticks % d == 0).expect("a divisor");
        let exact = run(&mut ctx, &requests, SlicePolicy::Fixed((ticks / parts) as u32));
        assert_eq!(exact.stage_ms.slices, parts, "the dividing slice length must need exactly {parts} launches");
        compare(&mut problems, &format!("{file} boundary slices"), &bits(&one.races[0].frames), &bits(&exact.races[0].frames));

        let whole = run(&mut ctx, &requests, SlicePolicy::Fixed(ticks as u32));
        assert_eq!(whole.stage_ms.slices, 1, "one slice ending at the finish tick");
        compare(&mut problems, &format!("{file} single exact slice"), &bits(&one.races[0].frames), &bits(&whole.races[0].frames));
        println!("{file}: {ticks} ticks, boundary slice length {}", ticks / parts);
    }
    if !problems.is_empty() {
        fail(&problems);
    }
}

/// Wide classic-lane launches measured for the F7 timeout decision: the
/// classic chunk budget caps one launch (512 MiB of frames), so these are
/// the widest classic launches the lane produces. Prints kernel time and
/// tick counts for a normal and the longest course.
#[test]
#[ignore = "diagnostic: times wide classic launches, needs the B580"]
fn classic_wide_launch_time() {
    // The v77 field on classic physics: the retail flags cleared.
    let mut value = json::parse(&request_line(V77)).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    let per_race = globalsim_gpu::classic_exec::CLS_MAX_TICKS * 9 * globalsim_gpu::classic_exec::CLS_FRAME_COLS * 8;
    let races = (512usize << 20) / per_race;
    for course in [10504, 10507] {
        let requests: Vec<(String, i64)> = (1..=races as i64)
            .map(|s| (race_exec::multi_request(&text, course, s).expect("request"), s))
            .collect();
        let out = classic_exec::run(&requests).expect("wide classic launch");
        let flagged = out.races.iter().filter(|r| r.fallback.is_some()).count();
        let max_ticks = out.races.iter().map(|r| r.ticks).max().unwrap_or(0);
        println!(
            "classic wide course {course}: {} races, max {max_ticks} ticks, flagged {flagged}, launch {:.0} ms over {} slices, slowest slice {:.0} ms",
            out.races.len(),
            out.launch_ms,
            out.slices,
            out.slice_max_ms,
        );
    }
}
