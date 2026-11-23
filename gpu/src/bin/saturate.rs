//! Full-device throughput. The other benches launch at most a few hundred
//! races (the 512 MiB frame budget caps a 9-runner launch at 193 races, one
//! work item each), which leaves most of a GPU idle. This one widens the
//! launch step by step up to what device memory holds, and prints races per
//! second for each width beside the CPU engine on every core, same request,
//! same seeds.
//!
//! Run (needs the Intel oneAPI LIB on the environment, and no other GPU or
//! CPU-heavy jobs while it runs):
//!   cargo run --release --features race --bin saturate -- [--request examples/cm-10701-race.jsonl] [--course ID] [--cpu-races 400] [--launches 3] [--max-races N] [--compact]
//!
//! Workload: the first line of `--request` (a path from the repo root), run
//! at seeds 1..N. The default is a real Champions Meeting race (course 10701,
//! Chukyo 1200 m turf, 9 runners, 159 skills), made from a game race dump by
//! `tools/race_dump_to_request.py`. `--request examples/v77-race.jsonl` is
//! the earlier workload: 9 identical skill-less runners on course 10504.
//! `--course` overrides the request's course id.
//!
//! A request the GPU lane refuses stops with the lane's reason. With
//! `--strip-refused` the bench removes what the lane refuses, one step at a
//! time, and prints each removal: a skill the refusal names, or, for a skill
//! table overflow, the last skill of the runner holding the most. Both the
//! CPU and the GPU then run the stripped request.
//!
//! With `--compact` every launched race keeps one frame row (the app's
//! compact lane), and the slot model, guard comparison and sweep follow it.
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
    request: String,
    course: Option<i32>,
    cpu_races: usize,
    launches: usize,
    max_races: Option<usize>,
    strip_refused: bool,
    compact: bool,
}

fn args() -> Result<Args, String> {
    let mut a = Args { request: "examples/cm-10701-race.jsonl".into(), course: None, cpu_races: 400, launches: 3, max_races: None, strip_refused: false, compact: false };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--request" => a.request = value()?,
            "--course" => a.course = Some(value()?.parse().map_err(|e| format!("--course: {e}"))?),
            "--cpu-races" => a.cpu_races = value()?.parse().map_err(|e| format!("--cpu-races: {e}"))?,
            "--launches" => a.launches = value()?.parse().map_err(|e| format!("--launches: {e}"))?,
            "--strip-refused" => a.strip_refused = true,
            "--compact" => a.compact = true,
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
    let base = std::fs::read_to_string(repo_file(&a.request))
        .map_err(|e| format!("request file {}: {e}", a.request))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    let course = match a.course {
        Some(id) => id,
        None => globalsim_gpu::json::parse(&base)?
            .get("courseId")
            .and_then(|v| v.as_f64())
            .ok_or("request has no courseId; pass --course")? as i32,
    };
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .map_err(|e| format!("geometry {course}: {e}"))?;
    globalsim::register_course_geometry(course, geometry.clone());
    let mut ctx = race_exec::RaceContext::open()?;
    ctx.register_geometry(course, &geometry)?;
    let (base, removed) = gpu_runnable(&mut ctx, &base, course, a.strip_refused)?;
    let request = race_exec::multi_request(&base, course, 1)?;
    let (runners, skills) = roster(&base)?;

    // CPU first, so its timings never share the machine with the device run.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("workload: {} line 1, course {course}, {runners} runners, {skills} skills, seeds 1..N", a.request);
    if !removed.is_empty() {
        println!("stripped for the GPU lane ({} skills removed):", removed.len());
        for line in &removed {
            println!("  {line}");
        }
    }
    println!("\nCPU engine (PreparedRace::run, no retained frames)");
    cpu_run(&request, cores.min(a.cpu_races), cores)?; // warm
    let one_n = (a.cpu_races / cores).max(8);
    let one_ms = cpu_run(&request, one_n, 1)?;
    let all_ms = cpu_run(&request, a.cpu_races, cores)?;
    let cpu_one = one_n as f64 / one_ms * 1e3;
    let cpu_all = a.cpu_races as f64 / all_ms * 1e3;
    println!("  1 thread : {one_n} races in {one_ms:.0} ms = {cpu_one:.1} races/s ({:.2} ms/race)", one_ms / one_n as f64);
    println!("  {cores} threads: {} races in {all_ms:.0} ms = {cpu_all:.1} races/s", a.cpu_races);

    let device = ctx.device();
    let compute_units = device.compute_units.max(1) as usize;
    let (global_mem, max_alloc) = ctx.device_memory()?;
    println!(
        "\ndevice {} ({:?}), {} compute units, {:.2} GiB global memory, {:.2} GiB max single allocation",
        device.name, device.platform, compute_units,
        global_mem as f64 / (1u64 << 30) as f64, max_alloc as f64 / (1u64 << 30) as f64,
    );

    let mut template = ctx.prepare(&request, 1)?;
    if let Some(reason) = template.host_fallback_reason() {
        return Err(format!("the workload falls back on the GPU: {reason}"));
    }
    let n = template.runners();

    // One race slot in bytes. Scratch follows the record bucket, frames the
    // course tick bound and the kept row count. The request table holds the
    // pd row and skill table once per distinct request (F1), not per race.
    let rcap = race_exec::sk_rcap(template.records());
    let tick_bound = race_exec::course_tick_bound(template.course_distance());
    let scr_u = race_exec::scr_u_words(rcap, false) * 4;
    let scr_d = race_exec::srl_words(rcap) * 8;
    let tail = race_exec::RI_STRIDE * 4 + race_exec::ST_STRIDE * 4 + race_exec::EV_STRIDE * 4;
    let slot = |rows: usize| tick_bound * rows * 6 * 8 + scr_u + scr_d + tail;
    // With --compact the launched races keep one frame row, so the sweep
    // sizes against that slot.
    let rows = if a.compact { 1 } else { n };
    let frames = tick_bound * rows * 6 * 8;
    let skw = globalsim_gpu::skills_compile::sk_stride_at(template.sk_emax(), template.sk_tmax());
    let req_table = race_exec::RIL_STRIDE * 8 + skw * 4;
    let per_race = slot(rows);
    let largest = frames.max(scr_u).max(scr_d);
    let by_memory = ((global_mem as f64 * 0.85 - req_table as f64) / per_race as f64) as usize;
    let by_alloc = (max_alloc as usize) / largest;
    let cap = a.max_races.unwrap_or(by_memory.min(by_alloc)).max(1);
    println!(
        "one race slot: {:.2} MiB (frames {:.2}, scratch {:.2} + {:.2}), request table {:.2} MiB per distinct request (skill table bucket {}); memory holds about {by_memory}, one allocation {by_alloc}; sweeping up to {cap}",
        per_race as f64 / 1048576.0, frames as f64 / 1048576.0, scr_u as f64 / 1048576.0, scr_d as f64 / 1048576.0,
        req_table as f64 / 1048576.0, template.sk_emax(),
    );
    // The frame block is the only region a launch shape changes, so the
    // shapes differ in slot size alone.
    for (label, r) in [("full frames", n), ("compact 1 row", 1), ("compact 2 rows", 2.min(n))] {
        let s = slot(r);
        let holds = ((global_mem as f64 * 0.85 - req_table as f64) / s as f64) as usize;
        println!(
            "shape {label} ({r} frame rows): slot {:.2} MiB (frames {:.2}), memory holds about {holds}",
            s as f64 / 1048576.0,
            (tick_bound * r * 6 * 8) as f64 / 1048576.0,
        );
    }
    if a.compact {
        if !template.set_frame_rows(&[0]) {
            return Err("--compact: the template refused the one-row mask".into());
        }
    }

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
        // A compact launch returns only the kept rows, so the comparison
        // reads those rows out of the CPU's full frames.
        let kept = prepared[0].kept_frame_rows().to_vec();
        let bits = |f: &Vec<Vec<[f64; 6]>>| -> Vec<u64> {
            f.iter()
                .flat_map(|tick| {
                    if kept.is_empty() {
                        tick.iter().flat_map(|v| v.map(|x| x.to_bits())).collect::<Vec<u64>>()
                    } else {
                        kept.iter().flat_map(|&r| tick[r].map(|x| x.to_bits())).collect::<Vec<u64>>()
                    }
                })
                .collect()
        };
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

    // End to end at the best width: what one app batch pays per race, in
    // both slice regimes (F7), so the launch-boundary cost is on the table.
    let (m, _) = best;
    let budget_mb = (m * frames).div_ceil(1 << 20) + 1;
    std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", budget_mb.to_string());
    ctx.set_slice_policy(Some(race_exec::SlicePolicy::Off));
    let mut prepared: Vec<race_exec::Prepared> = (1..=m as i32).map(|s| template.with_seed(s)).collect();
    let t = Instant::now();
    let (run, _) = ctx.run_prepared(&mut prepared, None)?;
    drop(prepared);
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let flagged = run.races.iter().filter(|r| r.fallback.is_some()).count();
    let s = &run.stage_ms;
    let rate = m as f64 / wall * 1e3;
    println!(
        "\nend to end at {m} races, one launch: {wall:.0} ms = {rate:.1} races/s ({:.2}x the {cores}-thread CPU); upload {:.0}, launch {:.0}, readback {:.0}, decode {:.0} ms; {flagged} flagged",
        rate / cpu_all, s.upload_ms, s.launch_ms, s.readback_ms, s.decode_ms,
    );

    // Second pass on slices: each race carries its geometry as a shared Arc,
    // so a launched Prepared can go through the launch again (F3).
    ctx.set_slice_policy(None);
    let mut prepared: Vec<race_exec::Prepared> = (1..=m as i32).map(|s| template.with_seed(s)).collect();
    let t = Instant::now();
    let (run, _) = ctx.run_prepared(&mut prepared, None)?;
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let flagged = run.races.iter().filter(|r| r.fallback.is_some()).count();
    let s = &run.stage_ms;
    let rate = m as f64 / wall * 1e3;
    println!(
        "end to end at {m} races, sliced: {wall:.0} ms = {rate:.1} races/s ({:.2}x the {cores}-thread CPU); {flagged} flagged",
        rate / cpu_all,
    );
    println!(
        "  slice launches {}, slowest {:.0} ms; upload {:.0}, launch {:.0}, readback {:.0}, decode {:.0} ms",
        s.slices, s.slice_max_ms, s.upload_ms, s.launch_ms, s.readback_ms, s.decode_ms,
    );
    if flagged > 0 {
        return Err(format!("{flagged} races fell back in the sliced end-to-end run"));
    }
    Ok(())
}

/// The request as the GPU lane will run it. A refusal is an error unless
/// `strip` is set, in which case the refused part is removed and the bench
/// goes on; each removal is returned as one line naming the runner, the skill
/// and the lane's reason.
fn gpu_runnable(ctx: &mut race_exec::RaceContext, base: &str, course: i32, strip: bool) -> Result<(String, Vec<String>), String> {
    use globalsim_gpu::json::{self, Value};
    let mut value = json::parse(base)?;
    let mut removed = Vec::new();
    loop {
        let mut text = String::new();
        json::to_string_into(&value, &mut text);
        let prepared = ctx.prepare(&race_exec::multi_request(&text, course, 1)?, 1)?;
        let Some(reason) = prepared.host_fallback_reason().map(str::to_string) else {
            return Ok((text, removed));
        };
        if !strip {
            return Err(format!("the workload falls back on the GPU: {reason} (--strip-refused benches it without the refused skills)"));
        }
        if removed.len() >= 64 {
            return Err(format!("still refused after 64 removals: {reason}"));
        }
        let named = reason
            .split_whitespace()
            .skip_while(|w| *w != "skill")
            .nth(1)
            .filter(|w| w.chars().all(|c| c.is_ascii_digit()))
            .map(str::to_string);
        let Value::Object(map) = &mut value else { return Err("request is not an object".into()) };
        let Some(Value::Array(horses)) = map.get_mut("horses") else { return Err("request has no horses".into()) };
        let id_of = |s: &Value| s.get("skillId").and_then(|v| v.as_str()).map(str::to_string);
        let before = removed.len();
        match named {
            Some(sid) => {
                for (i, h) in horses.iter_mut().enumerate() {
                    if let Some(skills) = skills_of(h) {
                        let had = skills.len();
                        skills.retain(|s| id_of(s).as_deref() != Some(sid.as_str()));
                        if skills.len() != had {
                            removed.push(format!("runner {i}: {sid} ({reason})"));
                        }
                    }
                }
            }
            None if reason.contains("overflow") => {
                let most = horses
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, h)| skills_of(h).map(|s| (i, s.len())))
                    .max_by_key(|&(i, n)| (n, std::cmp::Reverse(i)))
                    .map(|(i, _)| i);
                if let Some(i) = most {
                    if let Some(skill) = skills_of(&mut horses[i]).and_then(|s| s.pop()) {
                        removed.push(format!("runner {i}: {} ({reason})", id_of(&skill).unwrap_or_default()));
                    }
                }
            }
            None => {}
        }
        if removed.len() == before {
            return Err(format!("cannot strip this refusal: {reason}"));
        }
    }
}

/// A runner's skills array, when it has one.
fn skills_of(h: &mut globalsim_gpu::json::Value) -> Option<&mut Vec<globalsim_gpu::json::Value>> {
    use globalsim_gpu::json::Value;
    match h {
        Value::Object(hm) => match hm.get_mut("skills") {
            Some(Value::Array(s)) => Some(s),
            _ => None,
        },
        _ => None,
    }
}

/// Runner and skill counts of a request, for the workload line.
fn roster(request: &str) -> Result<(usize, usize), String> {
    let value = globalsim_gpu::json::parse(request)?;
    let horses = value.get_array("horses")?;
    let skills = horses
        .iter()
        .map(|h| h.get("skills").and_then(|s| s.as_array()).map_or(0, |s| s.len()))
        .sum();
    Ok((horses.len(), skills))
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
