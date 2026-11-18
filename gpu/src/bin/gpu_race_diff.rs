//! Whole-race comparison CLI (handoff steps 4 and 5): runs the CPU engine
//! and the GPU race runner on one request and seed, prints the race summary,
//! the first divergent tick with both bit patterns, and the fallback flag.

use globalsim_gpu::{comparator, dep_cpu, race};
use race::RaceRunner;

struct Config {
    request: String,
    geometry: String,
    seed: Option<i64>,
    cpu_only: bool,
}

fn parse_args() -> Result<Config, String> {
    let usage = "usage: gpu-race-diff --request <file> --geometry <file> [--seed N] [--engine cpu-only]";
    let mut request = None;
    let mut geometry = None;
    let mut seed = None;
    let mut cpu_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| -> Result<String, String> {
            args.next().ok_or_else(|| format!("{flag} needs a value\n{usage}"))
        };
        match arg.as_str() {
            "--request" => request = Some(value("--request")?),
            "--geometry" => geometry = Some(value("--geometry")?),
            "--seed" => {
                let text = value("--seed")?;
                seed = Some(text.parse().map_err(|_| format!("--seed needs an integer: {text}"))?);
            }
            "--engine" => match value("--engine")?.as_str() {
                "cpu-only" => cpu_only = true,
                other => return Err(format!("unknown --engine value {other:?}\n{usage}")),
            },
            "--help" | "-h" => return Err(usage.into()),
            other => return Err(format!("unknown argument {other:?}\n{usage}")),
        }
    }
    Ok(Config {
        request: request.ok_or_else(|| format!("--request is required\n{usage}"))?,
        geometry: geometry.ok_or_else(|| format!("--geometry is required\n{usage}"))?,
        seed,
        cpu_only,
    })
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("reading {path} failed: {e}"))
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    // Race request files are JSONL: one request per line, this tool takes the first.
    let request = read(&config.request)?
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or("request file has no JSON line")?
        .to_string();
    let course = dep_cpu::course_id(&request)?;
    // External-geometry builds resolve the course at preparation, so register before.
    globalsim::register_course_geometry(course, read(&config.geometry)?);
    let seed = match config.seed {
        Some(seed) => seed,
        None => dep_cpu::request_seed(&request)?,
    };

    let cpu = dep_cpu::run_cpu_race(&request, seed)?;
    println!(
        "cpu race: {} ticks, {} runners, {} finishers (course {course}, seed {seed})",
        cpu.frames.len(),
        cpu.runners,
        cpu.finishers
    );
    if config.cpu_only {
        return Ok(());
    }

    let mut runner = race::RestrictedStub;
    let outcome = runner.run_race(&request, seed);
    match comparator::compare(&cpu.frames, &outcome) {
        Ok(()) => println!("gpu race: every tick and every field equal, no fallback flag"),
        Err(mismatch) => println!("gpu race: {mismatch}"),
    }
    match &outcome.fallback {
        Some(reason) => println!("fallback: {reason:?}"),
        None => println!("fallback: none"),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gpu-race-diff: {error}");
        std::process::exit(1);
    }
}
