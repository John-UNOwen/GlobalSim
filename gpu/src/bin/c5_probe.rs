//! C5 probe. Settles the one-work-group-per-race route with measurements on
//! the device: the current mapping (one work item per race, runners serial)
//! against one work group per race (one lane per runner), on the same
//! synthetic per-runner timer walk.
//!
//! Also prints the app field's per-runner record counts, times the production
//! kernel with identical seeds against distinct seeds, and calibrates how many
//! SIMD lanes the device actually issues for each work-group size.
//!
//! Run (from `gpu/`, needs the Intel oneAPI LIB and the GPU to itself):
//!   cargo run --release --features race --bin c5_probe -- records \
//!       [--request FILE] [--course ID]
//!   cargo run --release --features race --bin c5_probe -- seeds \
//!       [--request FILE] [--width 8] [--launches 10]
//!   cargo run --release --features race --bin c5_probe -- shape
//!   cargo run --release --features race --bin c5_probe -- mapping \
//!       [--pattern 16,16,15,...]

use std::time::Instant;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

const DEFAULT_APP: &str = "examples/app-compare-race.jsonl";
const DEFAULT_V77: &str = "examples/v77-race.jsonl";
// The app field's compiled record owners, runner 0 to 8 (printed by `records`).
const DEFAULT_PATTERN: &str = "16,16,15,15,15,15,15,0,15";

fn request_text(rel: &str) -> Result<String, String> {
    std::fs::read_to_string(repo_file(rel))
        .map_err(|e| format!("request file {rel}: {e}"))?
        .lines()
        .next()
        .ok_or_else(|| format!("request file {rel} is empty"))
        .map(|s| s.to_string())
}

fn request_course(text: &str) -> Result<i32, String> {
    globalsim_gpu::json::parse(text)?
        .get("courseId")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "request has no courseId".to_string())
        .map(|v| v as i32)
}

fn geometry(course: i32) -> Result<String, String> {
    std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
        .map_err(|e| format!("geometry {course}: {e}"))
}

/// Per-runner compiled record counts. The kernel's record table carries the
/// owner in word 1 (`retail_core.cl:4687-4689`), so the counts come from the
/// same bytes the kernel reads.
fn mode_records(request: &str, course_override: Option<i32>) -> Result<(), String> {
    let text = request_text(request)?;
    let course = match course_override {
        Some(c) => c,
        None => request_course(&text)?,
    };
    let geo = geometry(course)?;
    globalsim::register_course_geometry(course, geo.clone());
    let mut ctx = globalsim_gpu::race_exec::RaceContext::open()?;
    ctx.register_geometry(course, &geo)?;
    let template = ctx.prepare(&globalsim_gpu::race_exec::multi_request(&text, course, 1)?, 1)?;
    if let Some(reason) = template.host_fallback_reason() {
        return Err(format!("the workload falls back on the GPU: {reason}"));
    }
    let (_, _, sk) = template.host_input_bits();
    let nrec = sk[globalsim_gpu::skills_compile::SK_NREC] as usize;
    let mut counts = vec![0usize; template.runners()];
    for i in 0..nrec {
        let at = globalsim_gpu::skills_compile::SK_RECORDS
            + globalsim_gpu::skills_compile::SK_RSTRIDE * i;
        let owner = sk[at + 1] as usize;
        if owner < counts.len() {
            counts[owner] += 1;
        }
    }
    println!(
        "device {} ({}), runners {}, records {}, rcap {}",
        ctx.device().name,
        ctx.device().driver_version,
        template.runners(),
        nrec,
        globalsim_gpu::race_exec::sk_rcap(nrec),
    );
    println!("runner  records");
    for (i, &c) in counts.iter().enumerate() {
        println!("{i:>6}  {c:>7}");
    }
    let total: usize = counts.iter().sum();
    let max = counts.iter().copied().max().unwrap_or(0);
    let min = counts.iter().copied().min().unwrap_or(0);
    println!(
        "records {total}, per runner min {min} max {max} mean {:.2}",
        total as f64 / counts.len() as f64,
    );
    Ok(())
}

/// Same seed repeated against distinct seeds on the production kernel. Both
/// batches have the same width, so the difference is only whether the races in
/// a work group follow the same branch path.
fn mode_seeds(app_request: &str, width: usize, launches: usize) -> Result<(), String> {
    for request in [app_request, DEFAULT_V77] {
        let text = request_text(request)?;
        let course = request_course(&text)?;
        let geo = geometry(course)?;
        globalsim::register_course_geometry(course, geo.clone());
        let mut ctx = globalsim_gpu::race_exec::RaceContext::open()?;
        ctx.register_geometry(course, &geo)?;
        let template = ctx.prepare(&globalsim_gpu::race_exec::multi_request(&text, course, 1)?, 1)?;
        if let Some(reason) = template.host_fallback_reason() {
            return Err(format!("{request} falls back on the GPU: {reason}"));
        }
        std::env::set_var("GSIM_GPU_FRAME_BUDGET_MB", (width * 3 + 1).to_string());
        let run = |ctx: &mut globalsim_gpu::race_exec::RaceContext,
                   same: bool|
         -> Result<(f64, usize), String> {
            let mut prepared: Vec<globalsim_gpu::race_exec::Prepared> = (0..width)
                .map(|i| {
                    let seed: i32 = if same { 1 } else { 1 + i as i32 };
                    template.with_seed(seed)
                })
                .collect();
            let (_, bench) = ctx.run_prepared_burst(&mut prepared, 2, launches)?;
            let bench = bench.ok_or("burst returned no timing")?;
            Ok((bench.launch_ms, bench.total_ticks))
        };
        let (same_ms, same_ticks) = run(&mut ctx, true)?;
        let (diff_ms, diff_ticks) = run(&mut ctx, false)?;
        println!(
            "{request}: width {width}, {launches} launches\n  same seed: {same_ms:.2} ms, {:.3} us/race-tick\n  distinct:  {diff_ms:.2} ms, {:.3} us/race-tick\n  distinct/same {:.4}",
            same_ms / same_ticks as f64 * 1e3,
            diff_ms / diff_ticks as f64 * 1e3,
            diff_ms / same_ms,
        );
    }
    Ok(())
}

const KERNEL_SOURCE: &str = r#"
#define C5_NREC_MAX 16
#define C5_EFF_MAX 8
// Doubles per runner: 16 records x 8 timers, then 16 refire windows.
#define C5_DSTRIDE 144
// Words per runner: nrec, 16 slot masks, then spare.
#define C5_USTRIDE 20

// One tick for one runner: the hot record timer walk (retail_core.cl:4682-4736),
// with every receiver's order searched on expiry. Same body for both mappings.
static void c5_runner_step(__global double* d, __global uint* u,
    __global const uint* tok, const uint race, const uint runner,
    const uint lanes) {
    const long dbase = ((long)race * lanes + runner) * C5_DSTRIDE;
    const long ubase = ((long)race * lanes + runner) * C5_USTRIDE;
    const long tbase = (long)race * lanes * C5_EFF_MAX;
    const uint nrec = min(u[ubase], (uint)C5_NREC_MAX);
    for (uint c = 0; c < nrec; c++) {
        uint mask = u[ubase + 1u + c];
        if (mask == 0u) continue;
        const long doff = dbase + (long)c * C5_EFF_MAX;
        for (uint w = 0; w < C5_EFF_MAX; w++) {
            if ((mask & (1u << w)) == 0u) continue;
            double t = d[doff + w];
            t = (double)(float)(t + 0.0666);
            d[doff + w] = t;
            if (t >= 0.0) {
                // The search token is owner independent, so every lane that
                // expires on the same slot searches the same token.
                const uint token = c * C5_EFF_MAX + w;
                uint found = 0u;
                for (uint j = 0; j < lanes * C5_EFF_MAX; j++) {
                    if (tok[tbase + j] == token) {
                        found = 1u;
                        break;
                    }
                }
                const double window = d[doff + C5_EFF_MAX + c];
                d[doff + w] = -window;
                mask &= ~(1u << w);
                u[ubase + 1u + c] = mask;
                u[ubase + 17] += found;
            }
        }
    }
}

// One work item per race: the runner loop is serial inside the item, the
// cross-race lanes are what the SIMD unit runs side by side (today's mapping).
__kernel void c5_item(__global double* d, __global uint* u,
    __global const uint* tok, const uint races, const uint lanes,
    const uint steps) {
    const uint race = get_global_id(0);
    if (race >= races) return;
    for (uint s = 0; s < steps; s++) {
        for (uint r = 0; r < lanes; r++) {
            c5_runner_step(d, u, tok, race, r, lanes);
        }
    }
}

// One work group per race: one lane per runner (the C5 mapping).
__kernel void c5_lanes(__global double* d, __global uint* u,
    __global const uint* tok, const uint races, const uint lanes,
    const uint steps) {
    const uint race = get_group_id(0);
    const uint runner = get_local_id(0);
    if (race >= races || runner >= lanes) return;
    for (uint s = 0; s < steps; s++) {
        c5_runner_step(d, u, tok, race, runner, lanes);
    }
}

// A dependent f64 FMA chain, one per active lane, to measure how many lanes a
// work-group size actually issues. `active` masks lanes past the useful count.
__kernel void c5_flops(__global double* out, const uint iters,
    const uint active) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    if (lid >= active) return;
    double x = (double)(gid & 63) * 0.125 + 1.0;
    for (uint i = 0; i < iters; i++) {
        x = fma(x, 1.0000000001, 0.0000005);
    }
    out[gid] = x;
}

// Launch-shape facts the device actually uses, one row per work item.
__kernel void c5_info(__global uint* out) {
    const uint i = get_global_id(0);
    out[i * 4 + 0] = get_sub_group_size();
    out[i * 4 + 1] = get_num_sub_groups();
    out[i * 4 + 2] = get_local_size(0);
    out[i * 4 + 3] = get_global_size(0);
}
"#;

const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const CL_DEVICE_SUB_GROUP_SIZES_INTEL: u32 = 0x4108;

fn info_u32(device: &ocl::Device, param: u32) -> Vec<u32> {
    device
        .info_raw(param)
        .map(|raw| {
            raw.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .unwrap_or_default()
}

fn open_b580() -> Result<(ocl::Queue, ocl::Device, ocl::Context), String> {
    let mut seen = Vec::new();
    for platform in ocl::Platform::list() {
        let devices = ocl::Device::list(&platform, Some(ocl::DeviceType::GPU))
            .map_err(|e| e.to_string())?;
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "unknown".into());
            seen.push(name.clone());
            if !name.contains("B580") {
                continue;
            }
            let context = ocl::Context::builder()
                .platform(platform.clone())
                .devices(device.clone())
                .build()
                .map_err(|e| e.to_string())?;
            let queue = ocl::Queue::new(&context, device.clone(), None)
                .map_err(|e| e.to_string())?;
            return Ok((queue, device, context));
        }
    }
    Err(format!("no B580 device among {seen:?}"))
}

fn device_info(device: &ocl::Device) {
    let name = device.name().unwrap_or_else(|_| "unknown".into());
    let cu = info_u32(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    let wg = info_u32(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    let raw = device.info_raw(CL_DEVICE_SUB_GROUP_SIZES_INTEL).unwrap_or_default();
    // The extension declares a size_t array, so parse both widths and keep the
    // reading whose entries are all powers of two.
    let u64s: Vec<u64> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let u32s: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let printable = |v: &[u64]| v.iter().all(|&x| x > 0 && x.is_power_of_two());
    let subgroups = if printable(&u64s) { u64s } else { u32s.iter().map(|&v| v as u64).collect() };
    println!(
        "device {name}, compute units {:?}, max work group {:?}, subgroup sizes {subgroups:?}",
        cu.first(),
        wg.first(),
    );
}

fn report_launch_shapes(queue: &ocl::Queue, program: &ocl::Program) -> Result<(), String> {
    let info_buf = ocl::Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(32 * 4)
        .build()
        .map_err(|e| e.to_string())?;
    println!("work-group shape from the compiled kernel:");
    for &lws in &[8usize, 16, 18, 24, 32] {
        let mut kb = ocl::Kernel::builder();
        kb.program(program)
            .name("c5_info")
            .queue(queue.clone())
            .global_work_size(lws);
        kb.local_work_size(lws);
        kb.arg(&info_buf);
        let kernel = match kb.build() {
            Ok(k) => k,
            Err(e) => {
                println!("  local {lws}: build refused: {e}");
                continue;
            }
        };
        info_buf.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut host = vec![0u32; lws * 4];
        info_buf.cmd().read(&mut host).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        println!(
            "  local {lws}: subgroup {}, subgroups {}, local {}, global {}",
            host[0], host[1], host[2], host[3],
        );
    }
    Ok(())
}

/// Small xorshift so the synthetic state is reproducible across runs.
fn next_rand(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[derive(Clone, Copy, PartialEq)]
enum Vary {
    /// Every race and runner carries the same data.
    None,
    /// Whole races differ (the current mapping's divergence axis).
    Race,
    /// Runners inside a race differ (the C5 mapping's divergence axis).
    Runner,
}

struct MicroState {
    doubles: Vec<f64>,
    words: Vec<u32>,
    tokens: Vec<u32>,
}

fn base_mask(c: usize) -> u32 {
    // A jagged but deterministic slot mask. Same for every lane, so lockstep
    // and app-like states do the same number of timer operations.
    let a = ((0x9e37u32 >> ((c * 5) % 13)) & 0xff) | 1;
    let b = 1u32 << ((c * 3) % 8);
    (a | b) & 0xff | 1
}

/// Fill the state buffers. Per-runner record counts always follow `pattern`,
/// the app field's measured distribution.
fn synth_state(
    races: usize,
    lanes: usize,
    pattern: &[u32],
    vary: Vary,
    seed: u64,
) -> MicroState {
    let mut doubles = vec![0f64; races * lanes * 144];
    let mut words = vec![0u32; races * lanes * 20];
    let mut tokens = vec![0u32; races * lanes * 8];
    let mut rand = seed | 1;
    for race in 0..races {
        let mut race_rand = seed ^ (race as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        // The token list is one race's receiver orders: a random subset of the
        // record tokens, so an expiry's early-exit search depends on data.
        for j in 0..lanes * 8 {
            tokens[race * lanes * 8 + j] = (next_rand(&mut race_rand) % 128) as u32;
        }
        for runner in 0..lanes {
            let mut runner_rand = seed ^ (runner as u64).wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
            let nrec = pattern[runner % pattern.len()].min(16) as usize;
            let ub = (race * lanes + runner) * 20;
            let db = (race * lanes + runner) * 144;
            words[ub] = nrec as u32;
            for c in 0..nrec {
                words[ub + 1 + c] = base_mask(c);
                for w in 0..8 {
                    // The phase is when this timer reaches zero. Lockstep data
                    // is identical everywhere, Race varies it per race, Runner
                    // varies it per runner.
                    let base = (c * 8 + w) as f64 * 0.75;
                    let offset = match vary {
                        Vary::None => 0.0,
                        Vary::Race => (next_rand(&mut race_rand) % 1000) as f64 / 100.0,
                        Vary::Runner => (next_rand(&mut runner_rand) % 1000) as f64 / 100.0,
                    };
                    doubles[db + c * 8 + w] = -(base + offset);
                }
                doubles[db + 128 + c] = 20.0;
            }
        }
    }
    MicroState { doubles, words, tokens }
}

struct Timed {
    name: String,
    ms: f64,
    runner_steps: f64,
}

fn time_kernel(
    queue: &ocl::Queue,
    kernel: &ocl::Kernel,
    launches: usize,
    rounds: usize,
    name: &str,
    runner_steps: f64,
) -> Result<Timed, String> {
    for _ in 0..3 {
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
    }
    queue.finish().map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..launches {
            unsafe { kernel.enq().map_err(|e| e.to_string())? };
        }
        queue.finish().map_err(|e| e.to_string())?;
        samples.push(start.elapsed().as_secs_f64() * 1e3 / launches as f64);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    Ok(Timed {
        name: name.to_string(),
        ms: samples[samples.len() / 2],
        runner_steps,
    })
}

fn fnv_bits(h: &mut u64, bits: &[u64]) {
    for &b in bits {
        for byte in b.to_le_bytes() {
            *h ^= byte as u64;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
}

struct Buffers {
    d: ocl::Buffer<f64>,
    u: ocl::Buffer<u32>,
    tok: ocl::Buffer<u32>,
}

#[allow(clippy::too_many_arguments)]
fn build_kernel(
    queue: &ocl::Queue,
    program: &ocl::Program,
    name: &str,
    gws: usize,
    lws: usize,
    buffers: &Buffers,
    races: u32,
    lanes: u32,
    steps: u32,
) -> Result<ocl::Kernel, String> {
    let mut kb = ocl::Kernel::builder();
    kb.program(program)
        .name(name)
        .queue(queue.clone())
        .global_work_size(gws);
    if lws > 0 {
        kb.local_work_size(lws);
    }
    kb.arg(&buffers.d)
        .arg(&buffers.u)
        .arg(&buffers.tok)
        .arg(races)
        .arg(lanes)
        .arg(steps);
    kb.build().map_err(|e| e.to_string())
}

/// Shape calibration: a dependent f64 FMA chain per active lane. The rate per
/// active lane shows how many SIMD lanes the work-group size really issues.
fn mode_shape(launches: usize, rounds: usize) -> Result<(), String> {
    let (queue, device, context) = open_b580()?;
    device_info(&device);
    let options = globalsim_gpu::gpu::build_options();
    let program = ocl::Program::builder()
        .cmplr_opt(options)
        .source(KERNEL_SOURCE)
        .build(&context)
        .map_err(|e| e.to_string())?;
    report_launch_shapes(&queue, &program)?;
    let iters = 8192u32;
    println!("\nf64 FMA chain, {iters} iterations per active lane, one launch per sample");
    println!("  {:>5} {:>6} {:>7} {:>9} {:>12} {:>10}", "local", "active", "items", "launch ms", "ns per FMA", "vs 32/32");
    let mut baseline = None;
    for (lws, active) in [(8usize, 8u32), (16, 16), (18, 18), (24, 24), (32, 32), (32, 18)] {
        let gws = lws * 1024;
        let out = ocl::Buffer::<f64>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(gws)
            .build()
            .map_err(|e| e.to_string())?;
        let mut kb = ocl::Kernel::builder();
        kb.program(&program)
            .name("c5_flops")
            .queue(queue.clone())
            .global_work_size(gws);
        kb.local_work_size(lws);
        kb.arg(&out).arg(iters).arg(active);
        let kernel = kb.build().map_err(|e| e.to_string())?;
        for _ in 0..2 {
            unsafe { kernel.enq().map_err(|e| e.to_string())? };
        }
        queue.finish().map_err(|e| e.to_string())?;
        let mut samples = Vec::new();
        for _ in 0..rounds {
            let start = Instant::now();
            for _ in 0..launches {
                unsafe { kernel.enq().map_err(|e| e.to_string())? };
            }
            queue.finish().map_err(|e| e.to_string())?;
            samples.push(start.elapsed().as_secs_f64() * 1e3 / launches as f64);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let ms = samples[samples.len() / 2];
        let active_lanes = gws / lws * active as usize;
        let ns = ms * 1e6 / (active_lanes as f64 * iters as f64);
        let base = *baseline.get_or_insert(ns);
        println!(
            "  {lws:>5} {active:>6} {gws:>7} {ms:>9.3} {ns:>12.4} {:>9.3}x",
            ns / base,
        );
    }
    Ok(())
}

/// Mapping comparison: the timer walk run as one work item per race (runners
/// serial) and as one work group per race (one lane per runner). Both mappings
/// compute the same bytes (checked first), and both are timed at the same
/// number of races so the batch shape is the only difference.
fn mode_mapping(pattern: &[u32], launches: usize, rounds: usize) -> Result<(), String> {
    let (queue, device, context) = open_b580()?;
    device_info(&device);
    let options = globalsim_gpu::gpu::build_options();
    let program = ocl::Program::builder()
        .cmplr_opt(options)
        .source(KERNEL_SOURCE)
        .build(&context)
        .map_err(|e| e.to_string())?;
    report_launch_shapes(&queue, &program)?;

    const MAX_RACES: usize = 4096;
    let lock = synth_state(MAX_RACES, 18, pattern, Vary::None, 1);
    let buffers = Buffers {
        d: ocl::Buffer::<f64>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(lock.doubles.len())
            .copy_host_slice(&lock.doubles)
            .build()
            .map_err(|e| e.to_string())?,
        u: ocl::Buffer::<u32>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(lock.words.len())
            .copy_host_slice(&lock.words)
            .build()
            .map_err(|e| e.to_string())?,
        tok: ocl::Buffer::<u32>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(lock.tokens.len())
            .copy_host_slice(&lock.tokens)
            .build()
            .map_err(|e| e.to_string())?,
    };

    // Both mappings must produce the same state before either is timed.
    let verify_steps = 8u32;
    let mut hashes = Vec::new();
    for (name, lws, gws) in [
        ("c5_item", 8usize, 8usize),
        ("c5_lanes", 32, 32),
    ] {
        buffers.d.cmd().write(&lock.doubles).enq().map_err(|e| e.to_string())?;
        buffers.u.cmd().write(&lock.words).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        let kernel = build_kernel(
            &queue, &program, name, gws, lws, &buffers, 1, 18, verify_steps,
        )?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut host_d = vec![0f64; lock.doubles.len()];
        let mut host_u = vec![0u32; lock.words.len()];
        buffers.d.cmd().read(&mut host_d).enq().map_err(|e| e.to_string())?;
        buffers.u.cmd().read(&mut host_u).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        fnv_bits(&mut h, &host_d.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        fnv_bits(&mut h, &host_u.iter().map(|&v| v as u64).collect::<Vec<_>>());
        hashes.push(h);
    }
    println!(
        "mapping check: item {:#018x}, lanes {:#018x}, equal {}",
        hashes[0],
        hashes[1],
        hashes[0] == hashes[1],
    );
    if hashes[0] != hashes[1] {
        return Err("the two mappings changed the runner state differently".into());
    }

    // One race as a work group: the C5 unit. The same batches as work items:
    // the current unit at the pinned local size 8, then at 16 and 32 so the
    // comparison separates the mapping from the work-group width.
    let configs: [(&str, &str, usize, usize, usize, u32, Vary); 12] = [
        ("item 8 races lws8", "c5_item", 8, 8, 18, 512, Vary::None),
        ("item 16 races lws16", "c5_item", 16, 16, 18, 256, Vary::None),
        ("item 32 races lws32", "c5_item", 32, 32, 18, 128, Vary::None),
        ("item 8 races x9", "c5_item", 8, 8, 9, 512, Vary::None),
        ("lanes 1 race x18", "c5_lanes", 1, 32, 18, 512, Vary::None),
        ("lanes 1 race x9", "c5_lanes", 1, 32, 9, 512, Vary::None),
        ("lanes 1 x9 app", "c5_lanes", 1, 32, 9, 512, Vary::Runner),
        ("lanes 1 x9 group9", "c5_lanes", 1, 9, 9, 512, Vary::None),
        ("item 1024 x9", "c5_item", 1024, 8, 9, 64, Vary::None),
        ("lanes 1024 x9", "c5_lanes", 1024, 32, 9, 64, Vary::Runner),
        ("item 2048 x18", "c5_item", 2048, 8, 18, 32, Vary::None),
        ("lanes 2048 x18", "c5_lanes", 2048, 32, 18, 32, Vary::Runner),
    ];
    let mut timed = Vec::new();
    for (label, name, races, lws, lanes, steps, vary) in configs {
        let st = synth_state(races, lanes, pattern, vary, 2);
        buffers.d.cmd().write(&st.doubles).enq().map_err(|e| e.to_string())?;
        buffers.u.cmd().write(&st.words).enq().map_err(|e| e.to_string())?;
        buffers.tok.cmd().write(&st.tokens).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        let gws = if name == "c5_item" {
            races.div_ceil(lws) * lws
        } else {
            races * lws
        };
        let kernel = build_kernel(
            &queue, &program, name, gws, lws, &buffers, races as u32,
            lanes as u32, steps,
        )?;
        let runner_steps = (races * lanes * steps as usize) as f64;
        timed.push(time_kernel(&queue, &kernel, launches, rounds, label, runner_steps)?);
    }

    println!(
        "\n{launches} launches per sample, {rounds} rounds, median; runners per tick 18",
    );
    println!(
        "{:<24} {:>10} {:>14} {:>10}",
        "mapping", "launch ms", "ns/runner-step", "vs item8",
    );
    let base = timed
        .iter()
        .find(|t| t.name == "item 8 races lws8")
        .ok_or("missing the item baseline")?
        .ms;
    for t in &timed {
        let ns = t.ms * 1e6 / t.runner_steps;
        println!(
            "{:<24} {:>10.3} {:>14.3} {:>9.3}x",
            t.name,
            t.ms,
            ns,
            t.ms / base,
        );
    }
    Ok(())
}

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "mapping".into());
    let request = parse_flag(&args, "--request").unwrap_or_else(|| DEFAULT_APP.to_string());
    let course = parse_flag(&args, "--course").and_then(|v| v.parse().ok());
    let launches = parse_flag(&args, "--launches").and_then(|v| v.parse().ok()).unwrap_or(10);
    let rounds = parse_flag(&args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(3);
    match mode.as_str() {
        "records" => mode_records(&request, course),
        "seeds" => {
            let width = parse_flag(&args, "--width").and_then(|v| v.parse().ok()).unwrap_or(8);
            mode_seeds(&request, width.max(1), launches.max(2))
        }
        "shape" => mode_shape(launches.max(2), rounds.max(1)),
        "mapping" => {
            let pattern = parse_flag(&args, "--pattern")
                .unwrap_or_else(|| DEFAULT_PATTERN.into())
                .split(',')
                .filter_map(|v| v.trim().parse::<u32>().ok())
                .collect::<Vec<_>>();
            if pattern.is_empty() {
                return Err("--pattern needs at least one count".into());
            }
            mode_mapping(&pattern, launches.max(2), rounds.max(1))
        }
        other => Err(format!("unknown mode {other} (records, seeds, shape, mapping)")),
    }
}
