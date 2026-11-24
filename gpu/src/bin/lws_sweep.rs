//! Local work size sweep (C2 handoff route). One warm process, one batch, one
//! lane: prepare a fixed request once, then for each candidate work-group size
//! relaunch it and print the kernel-only cost. Two numbers drive the choice:
//! kernel-only microseconds per race-tick at a narrow fixed batch, and kernel-only
//! races per second at the widest launch the device memory holds.
//!
//! A local work size only groups work items into work groups. The race kernel
//! gives each work item one whole race and lanes share no `__local` memory or
//! barriers, so the value changes scheduling and no arithmetic: every candidate
//! is bit-exact. The guard at the top checks the seed-1 race against the CPU
//! engine before any timed launch.
//!
//! Run (from `gpu/`, needs the Intel oneAPI LIB on the environment and the GPU
//! to itself):
//!   cargo run --release --features race --bin lws_sweep -- [--request FILE]
//!     [--course ID] [--locals 0,8,16,32,64] [--widths 32,cap] [--launches 3]
//!     [--compact] [--skew] [--max-races N]
//!
//! `--locals` are work-group sizes; `0`, `a` or `auto` leaves the group size to
//! the driver (the pre-C2 launch). `--widths` are race counts; `cap` or `full`
//! is the widest launch device memory holds, the number saturate sweeps to.

use globalsim_gpu::{dep_cpu, race_exec};

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

struct Args {
    request: String,
    course: Option<i32>,
    locals: Vec<usize>,
    widths: Vec<String>,
    launches: usize,
    compact: bool,
    max_races: Option<usize>,
    skew: bool,
}

fn parse_locals(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|w| {
            let w = w.trim();
            match w {
                "" => Err("empty local size".to_string()),
                "a" | "auto" | "none" => Ok(0),
                _ => w.parse::<usize>().map_err(|e| format!("--locals {w}: {e}")),
            }
        })
        .collect()
}

fn args() -> Result<Args, String> {
    let mut a = Args {
        request: "examples/cm-10701-race.jsonl".into(),
        course: None,
        locals: vec![0, 4, 8, 16, 24, 32, 48, 64, 96, 128],
        widths: vec!["32".into(), "cap".into()],
        launches: 3,
        compact: false,
        max_races: None,
        skew: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--request" => a.request = value()?,
            "--course" => a.course = Some(value()?.parse().map_err(|e| format!("--course: {e}"))?),
            "--locals" => a.locals = parse_locals(&value()?)?,
            "--widths" => a.widths = value()?.split(',').map(|w| w.trim().to_string()).collect(),
            "--launches" => a.launches = value()?.parse().map_err(|e| format!("--launches: {e}"))?,
            "--compact" => a.compact = true,
            "--skew" => a.skew = true,
            "--max-races" => a.max_races = Some(value()?.parse().map_err(|e| format!("--max-races: {e}"))?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
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
    let request = race_exec::multi_request(&base, course, 1)?;

    let device = ctx.device();
    let compute_units = device.compute_units.max(1) as usize;
    let (global_mem, max_alloc) = ctx.device_memory()?;
    let max_group = device.max_work_group_size;
    println!(
        "device {} ({}), {} compute units, {:.2} GiB memory, max work group {max_group}",
        device.name, device.driver_version, compute_units,
        global_mem as f64 / (1u64 << 30) as f64,
    );

    let mut template = ctx.prepare(&request, 1)?;
    if let Some(reason) = template.host_fallback_reason() {
        return Err(format!("the workload falls back on the GPU: {reason}"));
    }
    let n = template.runners();

    // One race slot, the same model saturate uses, so "cap" is the width that
    // tool sweeps to. Frames follow the tick bound and kept rows; scratch
    // follows the record bucket.
    let rcap = race_exec::sk_rcap(template.records());
    let tick_bound = race_exec::course_tick_bound(template.course_distance());
    let scr_u = race_exec::scr_u_words(rcap, false) * 4;
    let scr_d = race_exec::srl_words(rcap) * 8;
    let tail = race_exec::RI_STRIDE * 4 + race_exec::ST_STRIDE * 4 + race_exec::EV_STRIDE * 4;
    let rows = if a.compact { 1 } else { n };
    let frames = tick_bound * rows * 6 * 8;
    let per_race = frames + scr_u + scr_d + tail;
    let largest = frames.max(scr_u).max(scr_d);
    let skw = globalsim_gpu::skills_compile::sk_stride_at(template.sk_emax(), template.sk_tmax());
    let req_table = race_exec::RIL_STRIDE * 8 + skw * 4;
    let by_memory = ((global_mem as f64 * 0.85 - req_table as f64) / per_race as f64) as usize;
    let by_alloc = (max_alloc as usize) / largest;
    let cap = a.max_races.unwrap_or(by_memory.min(by_alloc)).max(1);
    println!(
        "slot {:.2} MiB (frames {:.2}, scratch {:.2} + {:.2}), {rows} frame rows, records {}; memory cap {by_memory}, one alloc {by_alloc}, cap {cap}",
        per_race as f64 / 1048576.0, frames as f64 / 1048576.0,
        scr_u as f64 / 1048576.0, scr_d as f64 / 1048576.0, template.records(),
    );
    if a.compact {
        if !template.set_frame_rows(&[0]) {
            return Err("--compact: the template refused the one-row mask".into());
        }
    }

    // Bit-exact guard at the narrowest width: the seed-1 race of a launched
    // batch equals the CPU engine frame for frame on the kept rows (compact
    // keeps row 0, the full path every runner).
    let cpu = dep_cpu::run_cpu_race(&request, 1)?;
    let kept_rows: Vec<usize> = if a.compact { vec![0] } else { (0..cpu.frames[0].len()).collect() };
    let cpu_bits: Vec<u64> = cpu
        .frames
        .iter()
        .flat_map(|tick| kept_rows.iter().flat_map(|&row| tick[row].map(|x| x.to_bits())))
        .collect();
    {
        let mut prepared: Vec<race_exec::Prepared> = (1..=8i32).map(|s| template.with_seed(s)).collect();
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            return Err(format!("guard: seed 1 fell back: {reason}"));
        }
        let gpu_bits: Vec<u64> = race
            .frames
            .iter()
            .flat_map(|tick| tick.iter().flat_map(|v| v.map(|x| x.to_bits())))
            .collect();
        if gpu_bits != cpu_bits {
            return Err("guard: GPU seed 1 frames differ from the CPU engine".into());
        }
        println!("guard: GPU seed 1 matches the CPU engine frame for frame ({} ticks)", cpu.frames.len());
    }

    let widths: Vec<usize> = a
        .widths
        .iter()
        .map(|w| match w.as_str() {
            "cap" | "full" | "max" => Ok(cap),
            _ => w.parse::<usize>().map_err(|e| format!("--widths {w}: {e}")),
        })
        .collect::<Result<_, _>>()?;

    // One chunk per width: the frame budget is set to the width's own frame
    // bytes so the whole batch is one launch and the number is that launch.
    println!(
        "\nkernel-only, {} back-to-back launches per point (us/race-tick and races/s)",
        a.launches
    );
    println!(
        "  {:>6} {:>6} {:>11} {:>9} {:>11} {:>9}",
        "local", "races", "launch ms", "races/s", "us/tick", "items/CU",
    );
    for &lws in &a.locals {
        ctx.set_local_work_size(Some(lws));
        let label = if lws == 0 { "auto".to_string() } else { lws.to_string() };
        for &w in &widths {
            let budget_mb = (w * frames).div_ceil(1 << 20) + 1;
            std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", budget_mb.to_string());
            let mut prepared: Vec<race_exec::Prepared> = (1..=w as i32).map(|s| template.with_seed(s)).collect();
            let (_, bench) = ctx.run_prepared_burst(&mut prepared, 1, a.launches)?;
            let bench = bench.ok_or("burst returned no timing")?;
            let rate = w as f64 / bench.launch_ms * 1e3;
            let us_tick = bench.launch_ms / bench.total_ticks as f64 * 1e3;
            println!(
                "  {:>6} {:>6} {:>11.1} {:>9.1} {:>11.3} {:>9.1}",
                label, w, bench.launch_ms, rate, us_tick, w as f64 / compute_units as f64,
            );
        }
    }

    // Race finish lengths in one decoded launch, the skew F8 would refill.
    // The device ticks a launch wall until its longest race ends, so the gap
    // between p50 and max is the straggler tail a refill could reuse.
    if let Some(&capw) = widths.last().filter(|_| a.skew) {
        ctx.set_local_work_size(None);
        let budget_mb = (capw * frames).div_ceil(1 << 20) + 1;
        std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", budget_mb.to_string());
        let mut prepared: Vec<race_exec::Prepared> = (1..=capw as i32).map(|s| template.with_seed(s)).collect();
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        let mut ticks: Vec<usize> = run.races.iter().map(|r| r.ticks).collect();
        ticks.sort_unstable();
        let pick = |q: f64| ticks[((ticks.len() as f64 - 1.0) * q) as usize];
        let mean = ticks.iter().sum::<usize>() as f64 / ticks.len() as f64;
        let last = *ticks.last().unwrap();
        let used: u64 = ticks.iter().map(|&t| t as u64).sum();
        let busy = used as f64 / (ticks.len() as f64 * last.max(1) as f64);
        println!(
            "\nskew of {} races: ticks p50 {} p95 {} max {} mean {:.0}; a single launch holds the device {:.1}% of its {:.0}-tick wall",
            ticks.len(), pick(0.5), pick(0.95), last, mean, busy * 100.0, last,
        );
    }
    Ok(())
}
