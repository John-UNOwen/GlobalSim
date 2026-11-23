//! Skill path profile: host phase table for a fixed race batch, kernel-only
//! per-tick cost at several widths, occupancy facts from the OpenCL API, and
//! the CM full-reply byte check. The saturate bench sweeps widths. This one
//! fixes the batch the app-style lane actually sends and splits one
//! `run_prepared` into its stages.
//!
//! Run (from `gpu/`):
//!   cargo run --release --features race --bin sk_profile -- [--request FILE]
//!     [--course ID] [--widths 1,8,32,64,96] [--iters 3]
//!     [--journal] [--full] [--device-info]
//!     [--repeat N] (run the facade's `run_repeated` on one request with N
//!     seeds, `--iters` batches, and stop: the app's batch-lane cost per
//!     message, with a digest pinning every outcome's bytes)
//! `--journal` launches one race with the diagnostics journal on and prints
//! the event histogram. `--full` runs the CM request through `run_full` and
//! byte-compares `full_reply::build` against the engine's retained reply.

use globalsim_gpu::{dep_cpu, full_reply, race_exec};
use std::time::Instant;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

struct Args {
    request: String,
    course: Option<i32>,
    widths: Vec<usize>,
    iters: usize,
    journal: bool,
    full: bool,
    device_info: bool,
    no_plan: bool,
    repeat: usize,
}

fn args() -> Result<Args, String> {
    let mut a = Args {
        request: "examples/cm-10701-race.jsonl".into(),
        course: None,
        widths: vec![1, 8, 32, 64, 96],
        iters: 3,
        journal: false,
        full: false,
        device_info: false,
        no_plan: false,
        repeat: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--request" => a.request = value()?,
            "--course" => a.course = Some(value()?.parse().map_err(|e| format!("--course: {e}"))?),
            "--widths" => {
                a.widths = value()?
                    .split(',')
                    .map(|w| w.trim().parse::<usize>().map_err(|e| format!("--widths: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--iters" => a.iters = value()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--journal" => a.journal = true,
            "--full" => a.full = true,
            "--device-info" => a.device_info = true,
            "--no-plan" => a.no_plan = true,
            "--repeat" => a.repeat = value()?.parse().map_err(|e| format!("--repeat: {e}"))?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

/// FNV-1a over bytes, so one digest line can pin every outcome of a batch.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The app's batch lane: one request through the facade's `run_repeated`
/// (one request, N seeds per batch), batch after batch. The wall per batch
/// is what the addon pays per message. The digest folds every outcome's
/// meta and packet bytes, so it catches a byte change between batches,
/// builds, and cache hits.
fn repeat_bench(text: &str, course: i32, geometry: &str, seeds: usize, batches: usize) -> Result<(), String> {
    let mut gpu = globalsim_gpu::facade::GpuRaces::new()?;
    gpu.register_geometry(course, geometry)?;
    let all: Vec<i32> = (1..=seeds as i32).collect();
    println!("repeat lane: {seeds} seeds per batch, {batches} batches, one request");
    let mut first = String::new();
    for k in 1..=batches {
        let t = Instant::now();
        let outcomes = gpu.run_repeated(text, &all)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for o in &outcomes {
            if let Some(m) = &o.meta {
                fnv(&mut h, m.as_bytes());
            }
            if let Some(p) = &o.packet {
                fnv(&mut h, p);
            }
            if let Some(f) = &o.fallback {
                fnv(&mut h, f.as_bytes());
            }
        }
        let flagged = outcomes.iter().filter(|o| o.fallback.is_some()).count();
        let d = format!("{h:016x}");
        println!("repeat {seeds} seeds batch {k}: {ms:.1} ms, {flagged} flagged, digest {d}");
        if k == 1 {
            first = d.clone();
        } else if d != first {
            return Err(format!("repeat batch {k} digest differs from batch 1"));
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let a = args()?;
    let text = std::fs::read_to_string(repo_file(&a.request))
        .map_err(|e| format!("request file: {e}"))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    let course = match a.course {
        Some(id) => id,
        None => globalsim_gpu::json::parse(&text)?
            .get("courseId")
            .and_then(|v| v.as_f64())
            .ok_or("request has no courseId, pass --course")? as i32,
    };
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .map_err(|e| format!("geometry {course}: {e}"))?;
    globalsim::register_course_geometry(course, geometry.clone());

    if a.repeat > 0 {
        return repeat_bench(&text, course, &geometry, a.repeat, a.iters.max(2));
    }

    let mut ctx = race_exec::RaceContext::open()?;
    ctx.set_plan_enabled(!a.no_plan);
    ctx.register_geometry(course, &geometry)?;
    let device_name = ctx.device().name.clone();
    let compute_units = ctx.device().compute_units;
    let max_group = ctx.device().max_work_group_size;
    println!("device {device_name}, {compute_units} compute units, max work group {max_group}");

    let t = Instant::now();
    let template = ctx.prepare(&race_exec::multi_request(&text, course, 1)?, 1)?;
    println!("prepare first: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    if let Some(reason) = template.host_fallback_reason() {
        return Err(format!("the workload falls back on the GPU: {reason}"));
    }
    let records = template.records();
    println!(
        "runners {}, records {}, rcap {}, emax {}, tmax {}",
        template.runners(),
        records,
        race_exec::sk_rcap(records),
        template.sk_emax(),
        template.sk_tmax(),
    );

    if a.device_info {
        device_info(template.sk_emax(), template.sk_tmax(), race_exec::sk_rcap(records));
    }

    // Prepare cost per race: the with_seed copy the repeated-seed batches use
    // against the full prepare that re-flattens and re-compiles the request.
    let batch = *a.widths.iter().max().unwrap_or(&64).max(&64);
    let t = Instant::now();
    let copies: Vec<race_exec::Prepared> = (1..=batch as i32).map(|s| template.with_seed(s)).collect();
    println!("clone prepare: {:.1} ms for {batch} races", t.elapsed().as_secs_f64() * 1e3);
    drop(copies);
    let t = Instant::now();
    let fresh: Vec<race_exec::Prepared> = (1..=64)
        .map(|s| ctx.prepare(&race_exec::multi_request(&text, course, s as i64)?, s as i64))
        .collect::<Result<_, _>>()?;
    println!("full prepare: {:.1} ms for {} races", t.elapsed().as_secs_f64() * 1e3, fresh.len());
    drop(fresh);

    // Host phase table: one timed run_prepared per width after a warm run,
    // with the default adaptive launch policy. A launch takes the geometry
    // out of the prepared races, so every timed call gets fresh copies.
    for &width in &a.widths {
        if width == 0 {
            continue;
        }
        let fresh = || -> Vec<race_exec::Prepared> {
            (1..=width as i32).map(|s| template.with_seed(s)).collect()
        };
        let _ = ctx.run_prepared(&mut fresh(), None)?;
        let mut best: Option<(race_exec::StageTimings, usize)> = None;
        for _ in 0..a.iters {
            let mut prepared = fresh();
            let (run, _) = ctx.run_prepared(&mut prepared, None)?;
            let flagged: Vec<usize> = run
                .races
                .iter()
                .enumerate()
                .filter(|(_, r)| r.fallback.is_some())
                .map(|(slot, _)| slot)
                .collect();
            if !flagged.is_empty() {
                let first = &run.races[flagged[0]];
                println!(
                    "batch {width}: {} of {} flagged, slot {} ticks {} flag {} arg {} {:?}",
                    flagged.len(),
                    run.races.len(),
                    flagged[0],
                    first.ticks,
                    first.kernel_flag,
                    first.kernel_arg,
                    first.fallback,
                );
                continue;
            }
            let ticks = run.races.iter().map(|r| r.ticks).sum();
            let s = run.stage_ms;
            best = Some(match best {
                Some((cur, _)) if cur.total_ms < s.total_ms => (cur, ticks),
                _ => (s, ticks),
            });
        }
        let Some((s, ticks)) = best else { continue };
        println!(
            "batch {width}: prepass {:.1}, tables {:.1}, upload {:.1}, launch {:.1}, readback {:.1}, decode {:.1}, total {:.1} ms, slices {}, slice max {:.1} ms, {:.1} races/s",
            s.prepass_ms,
            s.tables_ms,
            s.upload_ms,
            s.launch_ms,
            s.readback_ms,
            s.decode_ms,
            s.total_ms,
            s.slices,
            s.slice_max_ms,
            width as f64 / s.total_ms * 1e3,
        );
        println!("  ticks {ticks}, {:.1} us per race tick over the launch phase", s.launch_ms / ticks as f64 * 1e3);
    }

    // Kernel-only per-tick cost: warm then timed launches over reused
    // buffers, so the number is the device cost per race tick alone.
    println!("\nkernel-only (warm 1, timed {}), per-tick device cost", a.iters);
    for &width in &a.widths {
        if width == 0 {
            continue;
        }
        let mut prepared: Vec<race_exec::Prepared> =
            (1..=width as i32).map(|s| template.with_seed(s)).collect();
        let (_, bench) = ctx.run_prepared(&mut prepared, Some((1, a.iters)))?;
        let bench = bench.ok_or("no launchable race")?;
        println!(
            "  {width:>5} races: launch {:.1} ms, ticks {}, {:.3} us per race tick, {:.1} items/CU",
            bench.launch_ms,
            bench.total_ticks,
            bench.launch_ms / bench.total_ticks as f64 * 1e3,
            width as f64 / compute_units.max(1) as f64,
        );
    }

    if a.journal {
        journal_histogram(&text, course)?;
    }
    if a.full {
        cm_full_check(&text, course)?;
    }
    Ok(())
}

/// The diagnostics journal for one race: op histogram and fire density, read
/// back through the debug-scratch launch path.
fn journal_histogram(request: &str, course: i32) -> Result<(), String> {
    let req = race_exec::multi_request(request, course, 1)?;
    let run = race_exec::run_with_skill_scratch(&[(req.clone(), 1)])?;
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        return Err(format!("journal race flagged: {reason}"));
    }
    let sk = race.debug_sk.as_ref().ok_or("no skill scratch")?;
    let mut hist = std::collections::BTreeMap::<u32, usize>::new();
    for rec in sk.journal.chunks_exact(10) {
        *hist.entry(rec[0]).or_default() += 1;
    }
    let name = |op: u32| match op {
        1 => "FIRE",
        2 => "FAN ",
        3 => "DIE ",
        4 => "RET ",
        5 => "FLIP",
        6 => "EXT ",
        _ => "?   ",
    };
    println!(
        "journal one race: ticks {}, journal {} records (window cap {})",
        race.ticks,
        sk.journal.len() / 10,
        race_exec::SK_JRN_N,
    );
    for (op, count) in &hist {
        println!("  {} {op}: {count}", name(*op));
    }
    let per = hist.get(&1).copied().unwrap_or(0) as f64 / race.ticks.max(1) as f64;
    println!("  fires per tick: {per:.3}");
    let engine = dep_cpu::run_cpu_binary(&req, 1).map_err(|e| format!("cpu binary: {e}"))?.reply;
    let value = globalsim_gpu::json::parse(&engine).map_err(|e| e.to_string())?;
    let engine_fires = value
        .get("skillEvents")
        .and_then(|v| v.as_array())
        .map_or(0, Vec::len);
    println!("  engine skillEvents: {engine_fires}");
    Ok(())
}

/// The CM full-reply check: the engine's retained reply and the lane's
/// assembled reply must be the same bytes.
fn cm_full_check(request: &str, course: i32) -> Result<(), String> {
    let req = race_exec::multi_request(request, course, 1)?;
    let run = race_exec::run_full(&[(req.clone(), 1)])?;
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        return Err(format!("full-reply race flagged: {reason}"));
    }
    let full = dep_cpu::with_retain_all_frames(&req).map_err(|e| format!("retain request: {e}"))?;
    let gpu = full_reply::build(&full, 1, race).map_err(|e| format!("full reply build: {e}"))?;
    let engine = dep_cpu::run_cpu_binary(&req, 1).map_err(|e| format!("cpu binary: {e}"))?.reply;
    let (g, e) = (gpu.as_bytes(), engine.as_bytes());
    if g == e {
        println!("full reply: bit-exact ({} bytes)", g.len());
        Ok(())
    } else {
        let at = g
            .iter()
            .zip(e)
            .position(|(x, y)| x != y)
            .unwrap_or_else(|| g.len().min(e.len()));
        Err(format!(
            "full reply diverges at byte {at} (gpu {} bytes, engine {} bytes)",
            g.len(),
            e.len()
        ))
    }
}

/// Occupancy evidence the OpenCL API exposes for the race kernel: binary
/// size, max work group, local and private bytes per work item. Raw
/// clGetKernelInfo values sit above the CL 1.0 enum ocl-core wraps.
fn device_info(emax: usize, tmax: usize, rcap: usize) {
    use globalsim_gpu::gpu::RaceShape;
    use ocl::core::{ProgramInfo, ProgramInfoResult};
    let spike = match globalsim_gpu::gpu::Spike::open() {
        Ok(s) => s,
        Err(e) => {
            println!("device info: spike failed: {e}");
            return;
        }
    };
    for shape in [RaceShape::Sampling, RaceShape::Full] {
        let program = match spike.build_race_program(emax, tmax, rcap, shape) {
            Ok(p) => p,
            Err(e) => {
                println!("device info: program build failed: {e}");
                continue;
            }
        };
        match program.info(ProgramInfo::BinarySizes) {
            Ok(ProgramInfoResult::BinarySizes(sizes)) => {
                println!("{shape:?} program binary sizes: {sizes:?}")
            }
            Ok(_) => println!("{shape:?} program binary sizes: other"),
            Err(e) => println!("{shape:?} program binary sizes: {e}"),
        }
    }
    let program = match spike.build_race_program(emax, tmax, rcap, RaceShape::Sampling) {
        Ok(p) => p,
        Err(e) => {
            println!("device info: program build failed: {e}");
            return;
        }
    };
    let queue = spike.queue();
    let u32s = |name: &str| -> Result<ocl::Buffer<u32>, String> {
        ocl::Buffer::<u32>::builder().queue(queue.clone()).len(4096).build().map_err(|e| format!("{name}: {e}"))
    };
    let f64s = |name: &str| -> Result<ocl::Buffer<f64>, String> {
        ocl::Buffer::<f64>::builder().queue(queue.clone()).len(64).build().map_err(|e| format!("{name}: {e}"))
    };
    let f32s = |name: &str| -> Result<ocl::Buffer<f32>, String> {
        ocl::Buffer::<f32>::builder().queue(queue.clone()).len(64).build().map_err(|e| format!("{name}: {e}"))
    };
    let kernel = (|| -> Result<ocl::Kernel, String> {
        let a1 = u32s("ri")?;
        let a2 = f64s("pd")?;
        let a3 = u32s("scr_u")?;
        let a4 = u32s("rc")?;
        let a5 = f64s("scr_d")?;
        let a6 = f64s("frames")?;
        let a7 = u32s("status")?;
        let a8 = u32s("cs")?;
        let a9 = f64s("sg_x")?;
        let a10 = f64s("sg_xd")?;
        let a11 = f64s("sg_z")?;
        let a12 = f64s("sg_zd")?;
        let a13 = f32s("sg_q")?;
        let a14 = f32s("sg_qd")?;
        let a15 = f32s("lm_b")?;
        let a16 = f32s("lm_l")?;
        let a17 = f32s("ce_b")?;
        let a18 = f32s("ce_e")?;
        let a19 = u32s("ce_i")?;
        let a20 = u32s("ce_f")?;
        let a21 = u32s("sk")?;
        let a22 = u32s("events")?;
        ocl::Kernel::builder()
            .program(&program)
            .name("race")
            .queue(queue.clone())
            .global_work_size(1)
            .arg(&a1)
            .arg(&a2)
            .arg(&a3)
            .arg(&a4)
            .arg(&a5)
            .arg(&a6)
            .arg(&a7)
            .arg(&a8)
            .arg(&a9)
            .arg(&a10)
            .arg(&a11)
            .arg(&a12)
            .arg(&a13)
            .arg(&a14)
            .arg(&a15)
            .arg(&a16)
            .arg(&a17)
            .arg(&a18)
            .arg(&a19)
            .arg(&a20)
            .arg(&a21)
            .arg(&a22)
            .arg(1u32)
            .arg(u32::MAX)
            .arg(0u32)
            .build()
            .map_err(|e| e.to_string())
    })();
    let kernel = match kernel {
        Ok(k) => k,
        Err(e) => {
            println!("kernel info: builder refused the kernel: {e}");
            return;
        }
    };
    let raw = |param: u32, label: &str, width: usize| unsafe {
        let mut value = vec![0u8; width];
        let mut got = 0usize;
        let err = ocl::core::ffi::clGetKernelInfo(
            kernel.as_ptr().cast(),
            param,
            width,
            value.as_mut_ptr().cast(),
            &mut got,
        );
        if err == 0 && got > 0 {
            let words = match width {
                4 => value[..got]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64)
                    .collect::<Vec<_>>(),
                8 => value[..got]
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                24 => value[..got]
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            println!("kernel {label}: {words:?}");
        } else {
            println!("kernel {label}: unavailable (err {err}, {got} bytes)");
        }
    };
    raw(0x1181, "CL_KERNEL_WORK_GROUP_SIZE", 8);
    raw(0x1182, "CL_KERNEL_COMPILE_WORK_GROUP_SIZE", 24);
    raw(0x1183, "CL_KERNEL_LOCAL_MEM_SIZE", 8);
    raw(0x1184, "CL_KERNEL_PRIVATE_MEM_SIZE (spills)", 8);
}
