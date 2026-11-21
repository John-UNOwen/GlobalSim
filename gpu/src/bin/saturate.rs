//! Full-device throughput. The other benches launch at most a few hundred
//! races (the 512 MiB frame budget caps a 9-runner launch at 193 races, one
//! work item each), which leaves most of a GPU idle. This one widens the
//! launch step by step up to what device memory holds, and prints races per
//! second for each width beside the CPU engine on every core, same request,
//! same seeds.
//!
//! Run (needs the Intel oneAPI LIB on the environment, and no other GPU or
//! CPU-heavy jobs while it runs):
//!   cargo run --release --features race --bin saturate -- [--course 10504] [--cpu-races 400] [--launches 3] [--max-races N]
//!
//! Columns:
//! - kernel-only: back-to-back launches of the same buffers with one sync
//!   (`run_prepared_burst`), so the device never idles between launches.
//!   This is the device's ceiling for the workload.
//! - end to end: one `run_prepared` call (upload, launch, readback, decode),
//!   what the app's batch lane pays.
//! - items/CU: work items per compute unit. One race is one work item, so
//!   this is how much parallel work each compute unit was given.

use globalsim_gpu::{dep_cpu, race_exec};
use std::time::Instant;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

struct Args {
    course: i32,
    cpu_races: usize,
    launches: usize,
    max_races: Option<usize>,
}

fn args() -> Result<Args, String> {
    let mut a = Args { course: 10504, cpu_races: 400, launches: 3, max_races: None };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--course" => a.course = value()?.parse().map_err(|e| format!("--course: {e}"))?,
            "--cpu-races" => a.cpu_races = value()?.parse().map_err(|e| format!("--cpu-races: {e}"))?,
            "--launches" => a.launches = value()?.parse().map_err(|e| format!("--launches: {e}"))?,
            "--max-races" => a.max_races = Some(value()?.parse().map_err(|e| format!("--max-races: {e}"))?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

/// The CPU engine over `races` seeds on `threads` threads, each thread with
/// its own prepared race (prepared once, as the app's workers do). Returns
/// wall milliseconds.
fn cpu_run(request: &str, races: usize, threads: usize) -> Result<f64, String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || -> Result<(), String> {
                    let mut prepared = globalsim::PreparedRace::new(request)?;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= races {
                            return Ok(());
                        }
                        prepared.run(i as i64 + 1)?;
                    }
                })
            })
            .collect();
        handles.into_iter().try_for_each(|h| h.join().map_err(|_| "cpu thread panicked".to_string())?)
    })?;
    Ok(start.elapsed().as_secs_f64() * 1e3)
}

fn main() -> Result<(), String> {
    let a = args()?;
    let base = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .map_err(|e| format!("request file: {e}"))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    let request = race_exec::multi_request(&base, a.course, 1)?;
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{}.json", a.course)))
        .map_err(|e| format!("geometry {}: {e}", a.course))?;
    globalsim::register_course_geometry(a.course, geometry.clone());

    // CPU first, so its timings never share the machine with the device run.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("workload: v77 line 1, course {}, seeds 1..N", a.course);
    println!("\nCPU engine (PreparedRace::run, no retained frames)");
    cpu_run(&request, cores.min(a.cpu_races), cores)?; // warm
    let one_n = (a.cpu_races / cores).max(8);
    let one_ms = cpu_run(&request, one_n, 1)?;
    let all_ms = cpu_run(&request, a.cpu_races, cores)?;
    let cpu_one = one_n as f64 / one_ms * 1e3;
    let cpu_all = a.cpu_races as f64 / all_ms * 1e3;
    println!("  1 thread : {one_n} races in {one_ms:.0} ms = {cpu_one:.1} races/s ({:.2} ms/race)", one_ms / one_n as f64);
    println!("  {cores} threads: {} races in {all_ms:.0} ms = {cpu_all:.1} races/s", a.cpu_races);

    let mut ctx = race_exec::RaceContext::open()?;
    ctx.register_geometry(a.course, &geometry)?;
    let device = ctx.device();
    let compute_units = device.compute_units.max(1) as usize;
    let (global_mem, max_alloc) = ctx.device_memory()?;
    println!(
        "\ndevice {} ({:?}), {} compute units, {:.2} GiB global memory, {:.2} GiB max single allocation",
        device.name, device.platform, compute_units,
        global_mem as f64 / (1u64 << 30) as f64, max_alloc as f64 / (1u64 << 30) as f64,
    );

    let template = ctx.prepare(&request, 1)?;
    if let Some(reason) = template.host_fallback_reason() {
        return Err(format!("the workload falls back on the GPU: {reason}"));
    }
    let n = template.runners();

    // Device bytes one race slot takes in a launch, and which buffer is the
    // largest (it must fit one allocation).
    let frames = race_exec::RACE_MAX_TICKS * n * 6 * 8;
    let scr_u = race_exec::SCR_U_WORDS * 4;
    let scr_d = race_exec::SRL_WORDS * 8;
    let per_race = frames + scr_u + scr_d
        + race_exec::RI_STRIDE * 4 + race_exec::RIL_STRIDE * 8 + race_exec::SK_STRIDE * 4
        + race_exec::ST_STRIDE * 4 + race_exec::EV_STRIDE * 4;
    let largest = frames.max(scr_u).max(scr_d);
    let by_memory = (global_mem as f64 * 0.85 / per_race as f64) as usize;
    let by_alloc = (max_alloc as usize) / largest;
    let cap = a.max_races.unwrap_or(by_memory.min(by_alloc)).max(1);
    println!(
        "one race slot: {:.2} MiB (frames {:.2}, scratch {:.2} + {:.2}); memory holds about {by_memory}, one allocation {by_alloc}; sweeping up to {cap}",
        per_race as f64 / 1048576.0, frames as f64 / 1048576.0, scr_u as f64 / 1048576.0, scr_d as f64 / 1048576.0,
    );

    // Correctness guard: the seed-1 race of a batched launch equals the CPU
    // engine frame for frame, so the timed launches run a known-good chain.
    {
        let mut prepared: Vec<race_exec::Prepared> = (1..=8).map(|s| template.with_seed(s)).collect();
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            return Err(format!("guard: seed 1 fell back: {reason}"));
        }
        let cpu = dep_cpu::run_cpu_race(&request, 1)?;
        let bits = |f: &Vec<Vec<[f64; 6]>>| f.iter().flatten().flatten().map(|v| v.to_bits()).collect::<Vec<u64>>();
        if bits(&race.frames) != bits(&cpu.frames) {
            return Err("guard: GPU seed 1 frames differ from the CPU engine".into());
        }
        println!("guard: GPU seed 1 matches the CPU engine frame for frame ({} ticks)", cpu.frames.len());
    }

    let mut sizes: Vec<usize> = vec![32, 64, 128, 193, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192];
    sizes.retain(|&m| m < cap);
    sizes.push(cap);

    println!("\nkernel-only, {} back-to-back launches per width", a.launches);
    println!(
        "  {:>6} {:>8} {:>11} {:>10} {:>9} {:>8} {:>8}",
        "races", "items/CU", "launch ms", "races/s", "vs 1 core", "vs CPU", "clock",
    );
    let mut best = (0usize, 0f64);
    for &m in &sizes {
        let budget_mb = (m * frames).div_ceil(1 << 20) + 1;
        std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", budget_mb.to_string());
        let mut prepared: Vec<race_exec::Prepared> = (1..=m as i32).map(|s| template.with_seed(s)).collect();
        let started = chrono_now();
        let (_, bench) = ctx.run_prepared_burst(&mut prepared, 1, a.launches)?;
        let bench = bench.ok_or("burst returned no timing")?;
        let rate = m as f64 / bench.launch_ms * 1e3;
        if rate > best.1 {
            best = (m, rate);
        }
        println!(
            "  {:>6} {:>8.1} {:>11.1} {:>10.1} {:>8.2}x {:>7.2}x {:>8}",
            m, m as f64 / compute_units as f64, bench.launch_ms, rate, rate / cpu_one, rate / cpu_all, started,
        );
    }

    // End to end at the best width: what one app batch pays per race.
    let (m, _) = best;
    let budget_mb = (m * frames).div_ceil(1 << 20) + 1;
    std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", budget_mb.to_string());
    let mut prepared: Vec<race_exec::Prepared> = (1..=m as i32).map(|s| template.with_seed(s)).collect();
    let t = Instant::now();
    let (run, _) = ctx.run_prepared(&mut prepared, None)?;
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let flagged = run.races.iter().filter(|r| r.fallback.is_some()).count();
    let s = &run.stage_ms;
    let rate = m as f64 / wall * 1e3;
    println!(
        "\nend to end at {m} races: {wall:.0} ms = {rate:.1} races/s ({:.2}x the {cores}-thread CPU); upload {:.0}, launch {:.0}, readback {:.0}, decode {:.0} ms; {flagged} flagged",
        rate / cpu_all, s.upload_ms, s.launch_ms, s.readback_ms, s.decode_ms,
    );
    Ok(())
}

/// Wall-clock HH:MM:SS (local time is not in std; UTC is enough to line the
/// rows up with an external utilization log).
fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{:02}:{:02}:{:02}", secs / 3600 % 24, secs / 60 % 60, secs % 60)
}
