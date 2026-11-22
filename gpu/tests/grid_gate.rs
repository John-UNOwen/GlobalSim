//! Grid mode gates: an invalid grid shape must fall back quoting the engine's
//! own message. The engine's grid band is 21 to 120 horses, past the lane's
//! 18-horse cap, so a valid grid roster refuses by size and the CPU engine
//! answers it (pinned in `equivalence_gate` and `facade_gate`).

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

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

const GRID20: &str = r#"{"columns":20,"rowSpacing":3}"#;

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
