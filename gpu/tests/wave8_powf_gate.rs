//! Wave-8 powf equivalence gate: the ucrtbase FMA-variant port in
//! gpu/src/kernels/powf_core.cl must return, on the B580, the exact to_bits
//! of Rust std f64::powf on every (base, exponent) pair the live-guts
//! consumers can form. One mismatch stops wave 8: the near-match is not
//! shipped, the failing pair classes are reported instead.
//!
//! Ground truth is Rust f64::powf itself: on this host the ucrtbase dispatch
//! global (image+0x1007D4, read at runtime = 1) routes every f64::powf
//! through the FMA-variant body at pow+0x33BA1, which the port transcribes
//! instruction for instruction with fused fma().
//!
//! The domains are the consumer chains read straight from the engine: the
//! duel entry (guts*200.0f as f64) powf 0.708f->f64 and (guts*160.0f as f64)
//! powf 0.59f->f64 in f64 (src/race_solver.rs:3197-3201), the spot entry
//! (guts*500.0f as f64) powf 0.6f->f64 (src/race_solver.rs:2933-2936) and
//! its duration (700.0*guts) powf 0.5 in f64 (src/race_solver.rs:2940). The
//! guts grid sweeps every reachable f32 in [1, 2000] at a stride, dense
//! bands at the clamp ends, plus random pairs and two exponent fuzz bands
//! around the real exponents.
//!
//! The plan-closure f32 site (retail_physics.rs:362) gates against Rust
//! f32::powf through the gs_powf port (the ucrtbase powf body at +0x34D66, a
//! lower-precision pipeline than pow, which is why powf is not the f32
//! rounding of pow). The OpenCL built-in pow and powr run beside the ports
//! as measurement, not as candidates; the via64 column measures the
//! f64-route double rounding for the same f32 pairs.

#![cfg(feature = "race")]

use globalsim_gpu::gpu;
use ocl::{Buffer, Device, DeviceType, Platform, Program, Queue, flags};

const CORE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF
"#;

const DRIVER: &str = r#"
kernel void powf64(global const double* xs, global const double* ys,
                   global ulong* o_port, global ulong* o_clpow, global ulong* o_clpowr)
{
    const size_t i = get_global_id(0);
    const double x = xs[i];
    const double y = ys[i];
    o_port[i] = as_ulong(gs_pow(x, y));
    o_clpow[i] = as_ulong(pow(x, y));
    o_clpowr[i] = as_ulong(powr(x, y));
}

kernel void powf32(global const float* xs, global const float* ys,
                   global uint* o_port, global uint* o_clpow, global uint* o_clpowr,
                   global uint* o_via64)
{
    const size_t i = get_global_id(0);
    const float x = xs[i];
    const float y = ys[i];
    o_port[i] = (uint)as_int(gs_powf(x, y));
    o_clpow[i] = (uint)as_int(pow(x, y));
    o_clpowr[i] = (uint)as_int(powr(x, y));
    o_via64[i] = (uint)as_int(convert_float_rte(gs_pow(convert_float_rte(x), convert_float_rte(y))));
}
"#;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The same exactness gate the spike uses: first GPU whose metadata passes.
fn open_gated_device() -> Result<(Queue, Program), String> {
    const CL_DEVICE_NAME: u32 = 0x102B;
    const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
    const CL_DEVICE_SINGLE_FP_CONFIG: u32 = 0x101B;
    const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
    for platform in Platform::list() {
        for device in Device::list(&platform, Some(DeviceType::GPU)).map_err(|e| e.to_string())? {
            let read = |param: u32| -> String {
                String::from_utf8_lossy(&device.info_raw(param).unwrap_or_default()).into_owned()
            };
            let fp_config = |param: u32| -> u64 {
                device
                    .info_raw(param)
                    .ok()
                    .filter(|raw| raw.len() == 8)
                    .map_or(0, |raw| u64::from_le_bytes(raw[..8].try_into().unwrap()))
            };
            let rejections = gpu::fp_gate_rejections(
                &read(CL_DEVICE_EXTENSIONS),
                fp_config(CL_DEVICE_SINGLE_FP_CONFIG),
                fp_config(CL_DEVICE_DOUBLE_FP_CONFIG),
            );
            if !rejections.is_empty() {
                continue;
            }
            let context = ocl::Context::builder()
                .platform(platform)
                .devices(device.clone())
                .build()
                .map_err(|e| e.to_string())?;
            let queue = Queue::new(&context, device, None).map_err(|e| e.to_string())?;
            let program = Program::builder()
                .cmplr_opt(gpu::BUILD_OPTIONS)
                .source(format!(
                    "{CORE}{}\n{DRIVER}",
                    include_str!("../src/kernels/powf_core.cl")
                ))
                .build(&context)
                .map_err(|e| e.to_string())?;
            let _ = read(CL_DEVICE_NAME);
            return Ok((queue, program));
        }
    }
    Err("no GPU passes the exactness gate".into())
}

struct Report {
    domain: &'static str,
    pairs: u64,
    port: u64,
    cl_pow: u64,
    cl_powr: u64,
    via64: u64,
    samples: Vec<String>,
}

impl Report {
    fn new(domain: &'static str) -> Report {
        Report {
            domain,
            pairs: 0,
            port: 0,
            cl_pow: 0,
            cl_powr: 0,
            via64: 0,
            samples: Vec::new(),
        }
    }
    fn sample(&mut self, line: String) {
        if self.samples.len() < 8 {
            self.samples.push(line);
        }
    }
}

/// Split a u32/f64 bit word into the nearest-value summary for samples.
fn f64_word(bits: u64) -> String {
    format!("{} (0x{bits:016X})", f64::from_bits(bits))
}
fn f32_word(bits: u32) -> String {
    format!("{} (0x{bits:08X})", f32::from_bits(bits))
}

/// The reachable guts f32 grid: every stride-th value in [1, 2000], dense
/// bands around the clamp ends, plus random pairs.
fn guts_grid(coef: f32, stride: u32, random: u64, seed: u64) -> Vec<f32> {
    let _ = coef;
    let mut out: Vec<f32> = Vec::new();
    let first = 1.0f32.to_bits();
    let last = 2000.0f32.to_bits();
    let mut bits = first;
    while bits <= last {
        out.push(f32::from_bits(bits));
        bits = bits.saturating_add(stride);
    }
    for edge in [1.0f32, 2000.0] {
        let e = edge.to_bits() as i64;
        for d in -512i64..=512 {
            let b = e + d;
            if (first as i64..=last as i64).contains(&b) {
                out.push(f32::from_bits(b as u32));
            }
        }
    }
    // Random log-uniform guts: uniform mantissa over the octaves 2^0..2^11.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    for _ in 0..random {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mant = (state >> 40) as u32 & 0x7fffff;
        let exp = 127 + (state >> 16) as u32 % 11;
        let g = f32::from_bits((exp << 23) | mant);
        if (1.0..=2000.0).contains(&g) {
            out.push(g);
        }
    }
    out
}

fn run_f64_domain(
    queue: &Queue,
    program: &Program,
    name: &'static str,
    coef: f32,
    scale_f64: bool,
    exponent: f64,
    stride: u32,
    random: u64,
) -> Result<Report, String> {
    let mut report = Report::new(name);
    let guts = guts_grid(coef, stride, random, 0x9E3779B97F4A7C15 ^ name.as_bytes()[0] as u64);
    let mut xs: Vec<f64> = Vec::with_capacity(guts.len());
    let mut ys: Vec<f64> = Vec::with_capacity(guts.len());
    let mut want: Vec<u64> = Vec::with_capacity(guts.len());
    for g in &guts {
        let x = if scale_f64 {
            700.0_f64 * (*g as f64)
        } else {
            f64::from(*g * coef)
        };
        xs.push(x);
        ys.push(exponent);
        want.push(x.powf(exponent).to_bits());
    }
    let chunk = 4_000_000usize;
    let mut start = 0usize;
    while start < xs.len() {
        let end = (start + chunk).min(xs.len());
        let len = end - start;
        let xs_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&xs[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let ys_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&ys[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let o_port = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_pow = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_powr = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(program)
            .name("powf64")
            .queue(queue.clone())
            .global_work_size(len)
            .arg(&xs_buf)
            .arg(&ys_buf)
            .arg(&o_port)
            .arg(&o_pow)
            .arg(&o_powr)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut got_port = vec![0u64; len];
        let mut got_pow = vec![0u64; len];
        let mut got_powr = vec![0u64; len];
        o_port.cmd().read(&mut got_port).enq().map_err(|e| e.to_string())?;
        o_pow.cmd().read(&mut got_pow).enq().map_err(|e| e.to_string())?;
        o_powr.cmd().read(&mut got_powr).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        for i in 0..len {
            report.pairs += 1;
            let w = want[start + i];
            if got_port[i] != w {
                report.port += 1;
                report.sample(format!(
                    "{name} x={} y={} want={} gs_pow={}",
                    f64_word(xs[start + i].to_bits()),
                    f64_word(exponent.to_bits()),
                    f64_word(w),
                    f64_word(got_port[i])
                ));
            }
            if got_pow[i] != w {
                report.cl_pow += 1;
            }
            if got_powr[i] != w {
                report.cl_powr += 1;
            }
        }
        start = end;
    }
    Ok(report)
}

fn run_f32_domain(
    queue: &Queue,
    program: &Program,
    name: &'static str,
    coef: f32,
    exponent: f32,
    stride: u32,
    random: u64,
) -> Result<Report, String> {
    let mut report = Report::new(name);
    let guts = guts_grid(coef, stride, random, 0x100000001B3 ^ name.as_bytes()[0] as u64);
    let mut xs: Vec<f32> = Vec::with_capacity(guts.len());
    let mut ys: Vec<f32> = Vec::with_capacity(guts.len());
    let mut want: Vec<u32> = Vec::with_capacity(guts.len());
    for g in &guts {
        let x = g.clamp(1.0, 2000.0) * coef;
        xs.push(x);
        ys.push(exponent);
        want.push(x.powf(exponent).to_bits());
    }
    let chunk = 4_000_000usize;
    let mut start = 0usize;
    while start < xs.len() {
        let end = (start + chunk).min(xs.len());
        let len = end - start;
        let xs_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&xs[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let ys_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&ys[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let o_port = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_pow = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_powr = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_via64 = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(program)
            .name("powf32")
            .queue(queue.clone())
            .global_work_size(len)
            .arg(&xs_buf)
            .arg(&ys_buf)
            .arg(&o_port)
            .arg(&o_pow)
            .arg(&o_powr)
            .arg(&o_via64)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut got_port = vec![0u32; len];
        let mut got_pow = vec![0u32; len];
        let mut got_powr = vec![0u32; len];
        let mut got_via64 = vec![0u32; len];
        o_port.cmd().read(&mut got_port).enq().map_err(|e| e.to_string())?;
        o_pow.cmd().read(&mut got_pow).enq().map_err(|e| e.to_string())?;
        o_powr.cmd().read(&mut got_powr).enq().map_err(|e| e.to_string())?;
        o_via64.cmd().read(&mut got_via64).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        for i in 0..len {
            report.pairs += 1;
            let w = want[start + i];
            if got_port[i] != w {
                report.port += 1;
                report.sample(format!(
                    "{name} x={} y={} want={} gs_powf={}",
                    f32_word(xs[start + i].to_bits()),
                    f32_word(exponent.to_bits()),
                    f32_word(w),
                    f32_word(got_port[i])
                ));
            }
            if got_pow[i] != w {
                report.cl_pow += 1;
            }
            if got_powr[i] != w {
                report.cl_powr += 1;
            }
            if got_via64[i] != w {
                report.via64 += 1;
            }
        }
        start = end;
    }
    Ok(report)
}

/// Exponent fuzz: random base/exponent pairs around the reachable domains,
/// characterized against the port (reported, not silently tolerated).
fn run_fuzz(
    queue: &Queue,
    program: &Program,
    name: &'static str,
    lo: f64,
    hi: f64,
    e_lo: f64,
    e_hi: f64,
    count: u64,
) -> Result<Report, String> {
    let mut report = Report::new(name);
    let mut xs: Vec<f64> = Vec::with_capacity(count as usize);
    let mut ys: Vec<f64> = Vec::with_capacity(count as usize);
    let mut want: Vec<u64> = Vec::with_capacity(count as usize);
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..count {
        let x = lo + (hi - lo) * next();
        let y = e_lo + (e_hi - e_lo) * next();
        xs.push(x);
        ys.push(y);
        want.push(x.powf(y).to_bits());
    }
    let chunk = 4_000_000usize;
    let mut start = 0usize;
    while start < xs.len() {
        let end = (start + chunk).min(xs.len());
        let len = end - start;
        let xs_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&xs[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let ys_buf = Buffer::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).copy_host_slice(&ys[start..end]).len(len).build().map_err(|e| e.to_string())?;
        let o_port = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_pow = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let o_powr = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(len).build().map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(program)
            .name("powf64")
            .queue(queue.clone())
            .global_work_size(len)
            .arg(&xs_buf)
            .arg(&ys_buf)
            .arg(&o_port)
            .arg(&o_pow)
            .arg(&o_powr)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut got_port = vec![0u64; len];
        let mut got_pow = vec![0u64; len];
        let mut got_powr = vec![0u64; len];
        o_port.cmd().read(&mut got_port).enq().map_err(|e| e.to_string())?;
        o_pow.cmd().read(&mut got_pow).enq().map_err(|e| e.to_string())?;
        o_powr.cmd().read(&mut got_powr).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        for i in 0..len {
            report.pairs += 1;
            let w = want[start + i];
            if got_port[i] != w {
                report.port += 1;
                report.sample(format!(
                    "{name} x={} y={} want={} gs_pow={}",
                    f64_word(xs[start + i].to_bits()),
                    f64_word(ys[start + i].to_bits()),
                    f64_word(w),
                    f64_word(got_port[i])
                ));
            }
            if got_pow[i] != w {
                report.cl_pow += 1;
            }
            if got_powr[i] != w {
                report.cl_powr += 1;
            }
        }
        start = end;
    }
    Ok(report)
}

#[test]
fn wave8_powf_gate_matches_rust_pow_bit_for_bit() {
    let stride = env_u64("W8_STRIDE", 16) as u32;
    let random = env_u64("W8_RANDOM", 2_000_000);
    let fuzz = env_u64("W8_FUZZ", 1_000_000);
    let (queue, program) = match open_gated_device() {
        Ok(pair) => pair,
        Err(reason) => panic!("powf gate: {reason}"),
    };
    let mut reports: Vec<Report> = Vec::new();
    let push = |reports: &mut Vec<Report>, r: Result<Report, String>| {
        reports.push(r.unwrap_or_else(|e| panic!("powf gate launch: {e}")))
    };

    // The duration domain first: (700.0 * guts) powf 0.5, f64.
    push(
        &mut reports,
        run_f64_domain(&queue, &program, "duration 700*g ^0.5", 1.0, true, 0.5, stride, random),
    );
    push(
        &mut reports,
        run_f64_domain(
            &queue,
            &program,
            "spot (g*500f)^(0.6f)",
            500.0,
            false,
            f64::from(0.600000023841858_f64 as f32),
            stride,
            random,
        ),
    );
    push(
        &mut reports,
        run_f64_domain(
            &queue,
            &program,
            "duelT (g*200f)^(0.708f)",
            200.0,
            false,
            f64::from(0.708000004291534_f64 as f32),
            stride,
            random,
        ),
    );
    push(
        &mut reports,
        run_f64_domain(
            &queue,
            &program,
            "duelA (g*160f)^(0.59f)",
            160.0,
            false,
            f64::from(0.589999973773956_f64 as f32),
            stride,
            random,
        ),
    );
    // The plan closure site, f32 all the way (retail_physics.rs:362).
    push(
        &mut reports,
        run_f32_domain(
            &queue,
            &program,
            "plan (g*450f)^(0.597f)",
            450.0,
            0.597000002861023_f64 as f32,
            stride,
            random,
        ),
    );
    // Fuzz bands around the real exponents, characterization.
    push(
        &mut reports,
        run_fuzz(&queue, &program, "fuzz plan band", 450.0, 900_000.0, 0.55, 0.65, fuzz),
    );
    push(
        &mut reports,
        run_fuzz(&queue, &program, "fuzz duel band", 160.0, 400_000.0, 0.55, 0.72, fuzz),
    );

    let mut failures: Vec<String> = Vec::new();
    println!("wave8 powf gate: domain pairs port cl_pow cl_powr via64");
    for r in &reports {
        println!(
            "  {:<24} {:>9} {:>6} {:>7} {:>7} {}",
            r.domain,
            r.pairs,
            r.port,
            r.cl_pow,
            r.cl_powr,
            if r.domain.starts_with("plan") {
                format!("{:>7}", r.via64)
            } else {
                "        -".to_string()
            }
        );
        if r.port != 0 {
            failures.push(format!("{}: {} port mismatches", r.domain, r.port));
        }
    }
    for r in &reports {
        for s in &r.samples {
            failures.push(format!("  sample {s}"));
        }
    }
    if !failures.is_empty() {
        panic!("powf gate NOT green:\n{}", failures.join("\n"));
    }
}
