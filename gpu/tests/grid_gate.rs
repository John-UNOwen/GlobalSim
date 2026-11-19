//! Grid mode gates: the kernel's dense-rank grid starts must replay
//! frame-for-frame bit-exact against the CPU engine at 21, 24, 40 and 120
//! horses, invalid grid shapes must fall back quoting the engine's own
//! message, and the grid and retail starts at 21 horses must disagree at
//! frame 0 while each stays bit-exact against the CPU.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 2] = [1, 2];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];
const GRID_REASON: &str =
    "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0.";

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_request() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The v77 base horse copied to `n` runners with gateIndex 0..n-1, the only
/// field the grid start depends on beyond the gate sort.
fn roster(n: usize) -> Vec<Value> {
    let base = json::parse(&v77_request()).expect("v77 parses");
    let json::Value::Object(map) = &base else { unreachable!() };
    let json::Value::Array(horses) = map.get("horses").expect("horses") else {
        unreachable!()
    };
    let template = horses[0].clone();
    (0..n)
        .map(|i| {
            let mut horse = template.clone();
            let json::Value::Object(hm) = &mut horse else { unreachable!() };
            hm.insert("gateIndex".into(), Value::Number(i as f64));
            horse
        })
        .collect()
}

/// One request: the v77 shared fields (runtime flag, skillData, ground,
/// distance) with the roster rebuilt to `n` horses and the grid object
/// written verbatim when given.
fn grid_request(course: i32, seed: i64, n: usize, grid: Option<&str>) -> String {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        map.insert("horses".into(), json::arr(roster(n)));
        map.remove("pairedRandomStreams");
        if let Some(shape) = grid {
            map.insert("experimentalGrid".into(), json::parse(shape).expect("grid shape parses"));
            // The engine refuses geometry with grid starts (api.rs:68-71),
            // so a runnable grid request never carries the geometry flag.
            map.insert("experimentalRetailGeometry".into(), Value::Bool(false));
        } else {
            map.remove("experimentalGrid");
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn compare_race(
    tag: &str,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
    problems: &mut Vec<String>,
) {
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
        problems.push(format!(
            "{tag}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in
        cpu.frames.iter().zip(race.frames.iter()).enumerate()
    {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
            for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cv.to_bits() != gv.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                        COLUMNS[column],
                        cv.to_bits(),
                        gv.to_bits()
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

const GRID20: &str = r#"{"columns":20,"rowSpacing":3}"#;

/// Grid starts replay bit-exact at the mandatory rosters, plus 120 horses.
#[test]
fn grid_races_replay_bit_exact() {
    register_geometry(10504);
    register_geometry(10604);
    let mut problems = Vec::new();
    for &n in &[21usize, 24, 40, 120] {
        // One launch needs one uniform runner count, so batch per roster size.
        let mut batch: Vec<(String, i64)> = Vec::new();
        let mut tags: Vec<String> = Vec::new();
        for course in [10504, 10604] {
            for &seed in &SEEDS {
                batch.push((grid_request(course, seed, n, Some(GRID20)), seed));
                tags.push(format!("grid course {course} n {n} seed {seed}"));
            }
        }
        let out = race_exec::run(&batch).expect("kernel run");
        for (index, (request, seed)) in batch.iter().enumerate() {
            let race = &out.races[index];
            let tag = &tags[index];
            if race.fallback.is_none() && !race.grid {
                problems.push(format!("{tag}: the kernel ran a grid race without the grid flag"));
            }
            compare_race(tag, request, *seed, race, &mut problems);
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} grid gate problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// At 21 horses the grid start and the retail gate start must disagree at
/// frame 0 while both replay bit-exact against the CPU engine.
#[test]
fn grid_and_retail_starts_differ_at_21_horses() {
    register_geometry(10504);
    let seed = 1i64;
    let grid = grid_request(10504, seed, 21, Some(GRID20));
    let retail = grid_request(10504, seed, 21, None);
    let out = race_exec::run(&[(grid.clone(), seed), (retail.clone(), seed)]).expect("kernel run");
    let mut problems = Vec::new();
    compare_race("grid21", &grid, seed, &out.races[0], &mut problems);
    compare_race("retail21", &retail, seed, &out.races[1], &mut problems);
    if let (Some(g), Some(r)) = (out.races[0].frames.first(), out.races[1].frames.first()) {
        let same = g
            .iter()
            .zip(r.iter())
            .all(|(grow, rrow)| grow.iter().zip(rrow).all(|(a, b)| a.to_bits() == b.to_bits()));
        if same {
            problems.push("grid and retail frame 0 are identical at 21 horses".into());
        }
    } else {
        problems.push("a 21-horse race produced no frames".into());
    }
    if !out.races[0].grid || out.races[1].grid {
        problems.push("the grid flag is wrong on the 21-horse control pair".into());
    }
    if !problems.is_empty() {
        panic!(
            "{} grid control pair problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// Grid with the world-geometry flag must fall back with the engine's own
/// refusal (api.rs:68-71), and the engine must refuse the same request.
#[test]
fn grid_with_geometry_quotes_the_engine_reason() {
    register_geometry(10504);
    let mut request = grid_request(10504, 1, 40, Some(GRID20));
    {
        let mut value = json::parse(&request).expect("request parses");
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert(
            "experimentalRetailGeometry".into(),
            Value::Bool(true),
        );
        request.clear();
        json::to_string_into(&value, &mut request);
    }
    let mut problems = Vec::new();
    let out = race_exec::run(&[(request.clone(), 1)]).expect("kernel run");
    const CONFLICT: &str = "World geometry requires the experimental runtime and standard starting grid";
    match &out.races[0].fallback {
        None => problems.push("grid plus geometry ran on the kernel".into()),
        Some(reason) if reason != CONFLICT => {
            problems.push(format!("fallback names {reason}, not the engine's message"));
        }
        Some(_) => {}
    }
    match dep_cpu::run_cpu_race(&request, 1) {
        Ok(_) => problems.push("the engine accepted grid plus geometry".into()),
        Err(reason) => {
            if !reason.contains(CONFLICT) {
                problems.push(format!("engine reason {reason} lacks the expected message"));
            }
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} grid geometry conflict problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// Invalid grid shapes fall back naming the engine's message, and the engine
/// itself rejects the same request with that message.
#[test]
fn invalid_grid_shapes_quote_the_engine_reason() {
    register_geometry(10504);
    let shapes: [(&str, usize); 3] = [
        (r#"{"columns":20}"#, 40),
        (r#"{"columns":19,"rowSpacing":3}"#, 40),
        (GRID20, 121),
    ];
    let mut problems = Vec::new();
    for (index, (shape, n)) in shapes.iter().enumerate() {
        let seed = (index + 1) as i64;
        let request = grid_request(10504, seed, *n, Some(shape));
        let out = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
        let race = &out.races[0];
        match &race.fallback {
            None => problems.push(format!("shape {shape} n {n}: the kernel ran an invalid grid")),
            Some(reason) if reason != GRID_REASON => {
                problems.push(format!(
                    "shape {shape} n {n}: fallback names {reason}, not the engine's message"
                ));
            }
            Some(_) => {}
        }
        match dep_cpu::run_cpu_race(&request, seed) {
            Ok(_) => problems.push(format!("shape {shape} n {n}: the engine accepted an invalid grid")),
            Err(reason) => {
                let quoted = GRID_REASON.trim_end_matches('.');
                if !reason.contains(quoted) {
                    problems.push(format!(
                        "shape {shape} n {n}: engine reason {reason} lacks the expected message"
                    ));
                }
            }
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} invalid grid problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}
