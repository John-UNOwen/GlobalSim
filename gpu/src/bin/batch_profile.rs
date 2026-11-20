//! Batch lane cost profile: where facade `run_batch` and `run_repeated` spend
//! their milliseconds on the v77 line-1 request (9 runners, geometry on), at
//! batch sizes 1, 8, 32, 64 and 96. Stage times come from `run_prepared`
//! (`RaceRun::stage_ms`) over the same call sequence the facade drives, with
//! the prepare loop and the meta plus packet build timed around it.

use globalsim_gpu::facade::GpuRaces;
use globalsim_gpu::initialization::{self, Case, StatsCase};
use globalsim_gpu::race_exec::{self, Prepared, RaceContext};
use std::time::Instant;

const SIZES: [usize; 5] = [1, 8, 32, 64, 96];
const COURSE: i32 = 10504;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn main() -> Result<(), String> {
    let base = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .map_err(|e| format!("request file: {e}"))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .map_err(|e| format!("geometry: {e}"))?;
    globalsim::register_course_geometry(COURSE, geometry);
    let request = race_exec::multi_request(&base, COURSE, 1)?;

    // Program binary sizes: the fixed per-launch device cost tracks the
    // binary the driver has to bind into the GPU on each dispatch.
    {
        use ocl::core::{ProgramInfo, ProgramInfoResult};
        let spike = globalsim_gpu::gpu::Spike::open()?;
        for (name, source) in [
            ("restricted_tick", globalsim_gpu::gpu::KERNEL_SOURCE),
            ("initialization", globalsim_gpu::gpu::INITIALIZATION_KERNEL_SOURCE),
            ("race", globalsim_gpu::gpu::RACE_KERNEL_SOURCE),
        ] {
            let program = spike.build_program(source)?;
            let sizes = match program.info(ProgramInfo::BinarySizes).map_err(|e| e.to_string())? {
                ProgramInfoResult::BinarySizes(sizes) => sizes,
                _ => Vec::new(),
            };
            println!("program {name}: source {} KB, binary sizes {sizes:?}", source.len() / 1024);
        }
        // Program-load-at-dispatch isolation: launch a trivial `geometry`
        // query through the race program and through its own small program.
        let race_program = spike.build_program(globalsim_gpu::gpu::RACE_KERNEL_SOURCE)?;
        let geo_program = spike.build_program(globalsim_gpu::gpu::GEOMETRY_KERNEL_SOURCE)?;
        let queue = spike.queue();
        let u = ocl::Buffer::<u32>::builder().queue(queue.clone()).len(4096).build().map_err(|e| e.to_string())?;
        let d = ocl::Buffer::<f64>::builder().queue(queue.clone()).len(2 * 2002).build().map_err(|e| e.to_string())?;
        let f = ocl::Buffer::<f32>::builder().queue(queue.clone()).len(8 * 2002).build().map_err(|e| e.to_string())?;
        for (label, program) in [("race program", &race_program), ("geometry program", &geo_program)] {
            let kernel = ocl::Kernel::builder()
                .program(program)
                .name("geometry")
                .queue(queue.clone())
                .global_work_size(1)
                .arg(&u)
                .arg(&d)
                .arg(&d)
                .arg(&d)
                .arg(&d)
                .arg(&f)
                .arg(&f)
                .arg(&f)
                .arg(&f)
                .arg(&f)
                .arg(&f)
                .arg(&u)
                .arg(&u)
                .arg(&u)
                .arg(&u)
                .build()
                .map_err(|e| format!("{label} geometry kernel: {e}"))?;
            let mut first = 0f64;
            let mut steady = 0f64;
            for i in 0..4 {
                let t = std::time::Instant::now();
                unsafe { kernel.enq().map_err(|e| e.to_string())? };
                queue.finish().map_err(|e| e.to_string())?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if i == 0 {
                    first = ms;
                } else {
                    steady += ms / 3.;
                }
            }
            println!("one-item geometry launch via {label}: first {first:.1} ms, steady {steady:.1} ms");
        }
    }

    // Context open split: device gate, race program, init program, data.
    let mut cold_ctx = RaceContext::open()?;
    println!(
        "device {} single_fp_config {:#x} double_fp_config {:#x}",
        cold_ctx.device().name, cold_ctx.device().single_fp_config, cold_ctx.device().double_fp_config,
    );
    println!(
        "context open: spike (gate + restricted_tick) {:.1} ms, race program {:.1} ms, init program {:.1} ms, engine data {:.1} ms, total {:.1} ms",
        cold_ctx.open_ms.spike_ms,
        cold_ctx.open_ms.race_program_ms,
        cold_ctx.open_ms.init_program_ms,
        cold_ctx.open_ms.data_ms,
        cold_ctx.open_ms.total_ms,
    );

    // The first prepare on a fresh context loads and flattens the course.
    let t = Instant::now();
    cold_ctx.prepare(&request, 1)?;
    println!("cold prepare (course load + flatten, one course): {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    drop(cold_ctx);

    // What the per-batch stats pre-pass used to cost every call: a fresh
    // device open plus the init program compile, on top of the launch.
    let stats = StatsCase {
        raw: [1100., 1100., 1100., 1100., 1100.],
        mood: 0,
        rank: 0,
        surface: 1,
        strategy_apt: 1,
        ground: 1,
        css: vec![1, 2, 3],
        single_mode: false,
    };
    let cases: Vec<Case> = (0..9).map(|_| Case::Stats(stats.clone())).collect();
    let t = Instant::now();
    initialization::run(&cases)?;
    let legacy_prepass_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    initialization::run(&cases)?;
    println!(
        "initialization::run 9 cases (per-call Spike::open + init program build + launch), old pre-pass path: first {:.1} ms, second {:.1} ms",
        legacy_prepass_ms,
        t.elapsed().as_secs_f64() * 1e3,
    );

    let mut ctx = RaceContext::open()?;
    let mut facade = GpuRaces::new()?;

    // Kernel-only launch cost with reused buffers (the timing path inside
    // `run_prepared`): separates the per-launch device fixed cost from the
    // per-race marginal cost. Warm 3, timed 5 per size.
    println!("\nkernel-only (same buffers, warm 3 timed 5), ms per launch");
    println!("  {:>4}  {:>10}  {:>12}", "n", "launch_ms", "ms per race");
    for &n in &SIZES {
        let seeds: Vec<i32> = (1..=n as i32).collect();
        let requests: Vec<String> = (0..n).map(|_| request.clone()).collect();
        let mut prepared: Vec<Prepared> = Vec::with_capacity(n);
        for (request, &seed) in requests.iter().zip(&seeds) {
            prepared.push(ctx.prepare(request, seed as i64)?);
        }
        let (_, bench) = ctx.run_prepared(&mut prepared, Some((3, 5)))?;
        let bench = bench.ok_or("no launchable race")?;
        println!(
            "  {:>4}  {:>10.1}  {:>12.2}",
            n,
            bench.launch_ms,
            bench.launch_ms / n as f64,
        );
    }

    // Burst mode: ten launches enqueued back to back with a single sync at
    // the end, against the sync-per-launch timing path. If only the first
    // launch carries the ~430 ms, the cost is device wake-up between launches,
    // not work inside them.
    println!("\nkernel-only burst, ten launches enqueued then one sync, m=1, same buffers");
    for pass in 0..2 {
        let seeds: Vec<i32> = vec![1];
        let requests: Vec<String> = (0..1).map(|_| request.clone()).collect();
        let mut prepared: Vec<Prepared> = Vec::with_capacity(1);
        for (request, &seed) in requests.iter().zip(&seeds) {
            prepared.push(ctx.prepare(request, seed as i64)?);
        }
        let (run, bench) = ctx.run_prepared_burst(&mut prepared, 3, 10)?;
        no_fallback(&run)?;
        let bench = bench.ok_or("no launchable race")?;
        println!("  pass {pass}: {:.1} ms per launch, burst of 10", bench.launch_ms);
    }

    // Work proportionality: the same single-race launch on a 1000 m course,
    // about half the ticks of the 2000 m v77 line. If the ~429 ms floor
    // stands, it is per-dispatch, not work inside the kernel.
    {
        const SHORT: i32 = 11401;
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{SHORT}.json"))).map_err(|e| format!("geometry: {e}"))?;
        globalsim::register_course_geometry(SHORT, geometry);
        let short_request = race_exec::multi_request(&base, SHORT, 1)?;
        let mut prepared = vec![ctx.prepare(&short_request, 1)?];
        match ctx.run_prepared(&mut prepared, Some((2, 5)))? {
            (_, Some(bench)) => println!(
                "single race on course {SHORT} (1000 m): launch {:.1} ms, {} ticks",
                bench.launch_ms, bench.total_ticks,
            ),
            (run, None) => {
                let reason = run.races[0].fallback.clone().unwrap_or_default();
                println!("single race on course {SHORT}: skipped ({reason})");
            }
        }
    }

    let mut batch_rows: Vec<Row> = Vec::new();
    let mut repeat_rows: Vec<Row> = Vec::new();
    for &n in &SIZES {
        let seeds: Vec<i32> = (1..=n as i32).collect();
        let requests: Vec<String> = (0..n).map(|_| request.clone()).collect();

        // run_batch: one prepare per request, one launch. Warm once, timed once.
        drive_batch(&mut ctx, &requests, &seeds)?;
        let t = Instant::now();
        let mut prepared: Vec<Prepared> = Vec::with_capacity(n);
        for (request, &seed) in requests.iter().zip(&seeds) {
            prepared.push(ctx.prepare(request, seed as i64)?);
        }
        let prepare_ms = t.elapsed().as_secs_f64() * 1e3;
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        no_fallback(&run)?;
        let t = Instant::now();
        let batch_wall = facade.run_batch(&requests, &seeds)?.len();
        let batch_wall_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(batch_wall, n);
        batch_rows.push(Row { n, prepare_ms, run: run.stage_ms, wall_ms: batch_wall_ms });

        // run_repeated: one prepare, a clone per seed, one launch.
        drive_repeated(&mut ctx, &request, &seeds)?;
        let t = Instant::now();
        let template = ctx.prepare(&request, seeds[0] as i64)?;
        let mut prepared: Vec<Prepared> = seeds.iter().map(|&s| template.with_seed(s)).collect();
        let prepare_ms = t.elapsed().as_secs_f64() * 1e3;
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        no_fallback(&run)?;
        let t = Instant::now();
        let repeat_wall = facade.run_repeated(&request, &seeds)?.len();
        let repeat_wall_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(repeat_wall, n);
        repeat_rows.push(Row { n, prepare_ms, run: run.stage_ms, wall_ms: repeat_wall_ms });
    }

    println!("\nrun_batch, v77 line 1, course {COURSE}, 9 runners, one call per n (ms)");
    table(&batch_rows);
    println!("\nrun_repeated, same request, prepare once, a with_seed clone per seed (ms)");
    table(&repeat_rows);
    Ok(())
}

struct Row {
    n: usize,
    prepare_ms: f64,
    run: race_exec::StageTimings,
    wall_ms: f64,
}

fn table(rows: &[Row]) {
    println!(
        "  {:>4}  {:>9}  {:>7}  {:>6}  {:>7}  {:>11}  {:>8}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}",
        "n", "prepare", "prepass", "tables", "upload", "launch+sync", "readback", "decode", "meta+pkt", "inner", "wall", "ms per seed",
    );
    for row in rows {
        let s = &row.run;
        // The wall row is the facade call: its own prepare loop, the inner
        // path, and the compact meta plus packet build.
        let meta_pkt_ms = row.wall_ms - s.total_ms - row.prepare_ms;
        println!(
            "  {:>4}  {:>9.1}  {:>7.1}  {:>6.1}  {:>7.1}  {:>11.1}  {:>8.1}  {:>7.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.2}",
            row.n,
            row.prepare_ms,
            s.prepass_ms,
            s.tables_ms,
            s.upload_ms,
            s.launch_ms,
            s.readback_ms,
            s.decode_ms,
            meta_pkt_ms,
            s.total_ms,
            row.wall_ms,
            row.wall_ms / row.n as f64,
        );
    }
}

fn drive_batch(ctx: &mut RaceContext, requests: &[String], seeds: &[i32]) -> Result<(), String> {
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, &seed) in requests.iter().zip(seeds) {
        prepared.push(ctx.prepare(request, seed as i64)?);
    }
    let (run, _) = ctx.run_prepared(&mut prepared, None)?;
    no_fallback(&run)
}

fn drive_repeated(ctx: &mut RaceContext, request: &str, seeds: &[i32]) -> Result<(), String> {
    let template = ctx.prepare(request, seeds[0] as i64)?;
    let mut prepared: Vec<Prepared> = seeds.iter().map(|&s| template.with_seed(s)).collect();
    let (run, _) = ctx.run_prepared(&mut prepared, None)?;
    no_fallback(&run)
}

fn no_fallback(run: &race_exec::RaceRun) -> Result<(), String> {
    match run.races.iter().find_map(|race| race.fallback.as_ref()) {
        Some(reason) => Err(format!("profile race flagged: {reason}")),
        None => Ok(()),
    }
}
