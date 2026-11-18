//! Skill cost probe: kernel-only launch time of race batches with no skills,
//! six skills on one runner and six skills on two runners, at a fixed runner
//! budget (default 4096). Reports ms per launch and ms per race tick. The
//! 512-byte tag block caps the table at about 30 sampled skills per race,
//! so "six skills on every runner" (9 runners) is beyond the table and the
//! stress case uses two runners instead.

use globalsim_gpu::race_exec;

const SKILLS: [i64; 6] = [200361, 200341, 200333, 10071, 100401, 110591];
const COURSES: [i32; 2] = [10504, 10510];
const SEEDS: [i64; 4] = [1, 2, 3, 4];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut runners_target: usize = 4096;
    let mut warm = 3usize;
    let mut iters = 10usize;
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| -> Result<usize, String> {
            let text = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
            text.parse().map_err(|_| format!("{flag} needs an integer: {text}"))
        };
        match arg.as_str() {
            "--runners" => runners_target = value("--runners")?,
            "--warm" => warm = value("--warm")?,
            "--iters" => iters = value("--iters")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let base = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .map_err(|e| format!("request file: {e}"))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    for &course in &COURSES {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .map_err(|e| format!("geometry: {e}"))?;
        globalsim::register_course_geometry(course, geometry);
    }
    // v77 carries nine runners, so the batch size rounds the runner budget up.
    let batch = runners_target.div_ceil(9);
    println!("device probe batch {batch} races x 9 runners, warm {warm}, timed {iters}");

    let mut cases: Vec<(&str, Vec<(String, i64)>)> = Vec::new();
    for &skills_on in &[0usize, 1, 2] {
        let mut requests: Vec<(String, i64)> = Vec::with_capacity(batch);
        for i in 0..batch {
            let course = COURSES[i % COURSES.len()];
            let seed = SEEDS[(i / COURSES.len()) % SEEDS.len()];
            let mut request = race_exec::multi_request(&base, course, seed)?;
            for horse in 0..skills_on {
                request = race_exec::skills_request(&request, course, seed, horse, &SKILLS, None)?;
            }
            requests.push((request, seed));
        }
        cases.push((match skills_on {
            0 => "no skills",
            1 => "6 skills on 1 runner",
            _ => "6 skills on 2 runners",
        }, requests));
    }

    for (name, requests) in &cases {
        let probe = race_exec::run(&requests[0..1]).map_err(|e| format!("{name} probe: {e}"))?;
        if let Some(reason) = &probe.races[0].fallback {
            return Err(format!("{name}: the probe race flagged: {reason}"));
        }
        let bench = race_exec::run_kernel_only(requests, warm, iters)?;
        let ms_per_tick = bench.launch_ms / bench.total_ticks as f64;
        println!(
            "{name}: device {} launch {:.2} ms, ticks {}, {:.4} us per race tick, {:.1} ns per runner tick",
            bench.device,
            bench.launch_ms,
            bench.total_ticks,
            ms_per_tick * 1e3,
            ms_per_tick * 1e6 / 9.0,
        );
    }
    Ok(())
}
