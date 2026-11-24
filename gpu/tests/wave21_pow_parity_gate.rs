//! Wave-21 pow parity gate: every live gs_pow / gs_powf consumer domain in
//! both lanes, gated pair for pair against Rust std f64::powf and f32::powf
//! on the B580. Wave 8 gated the gutter consumers the port shipped with. This
//! gate adds the classic-lane f64 literal exponents, the compete and
//! secure-lead ratios, the wisdom and percent chains whose bases drop below
//! 1, and the f32 plan closure over its live-guts range.
//!
//! Ground truth is Rust std powf itself, the same authority wave 8 used: the
//! port transcribes the ucrtbase FMA-variant pow and powf bodies instruction
//! for instruction, so a bit difference here is a port bug, not a rounding
//! choice.
//!
//! Consumer domains, engine site, lane and how the result is consumed:
//!
//! | consumer            | engine site                             | lane    | result |
//! | last spurt speed    | race_solver.rs:65  (450*guts)^0.597     | classic | f64 |
//! | spot modifier       | race_solver.rs:2938 (500*guts)^0.6      | classic | f64 |
//! | spot duration       | race_solver.rs:2940 (700*guts)^0.5      | both    | f64 |
//! | duel target speed   | race_solver.rs:3205 (200*guts)^0.708    | classic | f64 |
//! | duel accel          | race_solver.rs:3206 (160*guts)^0.59     | classic | f64 |
//! | compete power       | race_solver.rs:2981 (power/1500)^0.5    | both    | f64 |
//! | compete guts        | race_solver.rs:2981 (guts/3000)^0.2     | both    | f64 |
//! | secure lead         | race_solver.rs:3033 (guts/2000)^0.5     | both    | f64 |
//! | stamina keep wisdom | race_solver.rs:2776 wisdom^0.03         | both    | f64 |
//! | stamina keep chance | race_solver.rs:2777 (1-pct/100)^(dt/2)  | both    | f64 |
//! | competition wisdom  | race_solver.rs:1536 (1-pct/100)^(dt/2)  | both    | f64 |
//! | retail spot         | race_solver.rs:2934 500f32*guts, 0.6f32 | retail  | f32 store |
//! | retail duel target  | race_solver.rs:3198 200f32*guts, 0.708f32 | retail | f32 store |
//! | retail duel accel   | race_solver.rs:3198 160f32*guts, 0.59f32 | retail | f32 store |
//! | retail plan closure | retail_physics.rs:362 (bounded*450f)^0.597f | retail | f32 |
//!
//! The pair tests take host-built (base, exponent) pairs. That isolates the
//! port from every other arithmetic op in the consumer chain. The two-stage
//! chains are covered in two steps: the inner power is its own domain, then
//! the outer power gets pairs whose base comes from the host chain. A green
//! inner domain makes the base bits the device forms equal to the host bits,
//! so the outer pair list is the device's own pair list.
//!
//! The real-input test repeats the comparison over the five CM fields' actual
//! runner stats plus every construction stat bump the prepared plan carries,
//! so those pairs are values the shipping lane sees rather than a sweep.

#![cfg(feature = "race")]

use globalsim_gpu::{gpu, initialization, race_exec};
use ocl::{Buffer, Device, DeviceType, Platform, Program, Queue, flags};

const CORE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF
"#;

const DRIVER: &str = r#"
kernel void pow64(global const double* xs, global const double* ys, global ulong* o)
{
    const size_t i = get_global_id(0);
    o[i] = as_ulong(gs_pow(xs[i], ys[i]));
}

kernel void pow32(global const float* xs, global const float* ys, global uint* o)
{
    const size_t i = get_global_id(0);
    o[i] = (uint)as_int(gs_powf(xs[i], ys[i]));
}
"#;

/// dt is the engine tick constant (src/lib.rs:66), the consumers take dt/2.
fn half_dt() -> f64 {
    (1.0f64 / 15.0) / 2.0
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The same exactness gate the spike uses: first GPU whose metadata passes.
fn open_gated_device() -> Result<(Queue, Program), String> {
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
                .source(format!("{CORE}{}\n{DRIVER}", include_str!("../src/kernels/powf_core.cl")))
                .build(&context)
                .map_err(|e| e.to_string())?;
            return Ok((queue, program));
        }
    }
    Err("no GPU passes the exactness gate".into())
}

struct Report {
    domain: &'static str,
    lane: &'static str,
    pairs: u64,
    port: u64,
    samples: Vec<String>,
}

impl Report {
    fn new(domain: &'static str, lane: &'static str) -> Report {
        Report { domain, lane, pairs: 0, port: 0, samples: Vec::new() }
    }

    fn sample(&mut self, line: String) {
        if self.samples.len() < 8 {
            self.samples.push(line);
        }
    }
}

fn f64_word(bits: u64) -> String {
    format!("{} (0x{bits:016X})", f64::from_bits(bits))
}

fn f32_word(bits: u32) -> String {
    format!("{} (0x{bits:08X})", f32::from_bits(bits))
}

/// Deterministic LCG, the pair grids must not change run to run.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Every stride-th f32 in [lo, hi], dense bands of 512 ulps at both clamp
/// ends, and random log-uniform values across the octaves the range spans.
fn f32_bases(lo: f32, hi: f32, stride: u32, random: u64, seed: u64) -> Vec<f32> {
    let first = lo.to_bits();
    let last = hi.to_bits();
    let mut out: Vec<f32> =
        Vec::with_capacity(((last - first) / stride) as usize + 2_048 + random as usize);
    let mut bits = first;
    while bits <= last {
        out.push(f32::from_bits(bits));
        bits = bits.saturating_add(stride.max(1));
    }
    for edge in [lo, hi] {
        let e = edge.to_bits() as i64;
        for d in -512i64..=512 {
            let b = e + d;
            if (first as i64..=last as i64).contains(&b) {
                out.push(f32::from_bits(b as u32));
            }
        }
    }
    let mut rng = Lcg::new(seed);
    let exp_lo = (lo as f64).log2().floor() as i32;
    let exp_hi = (hi as f64).log2().floor() as i32;
    let span = (exp_hi - exp_lo + 1) as u32;
    for _ in 0..random {
        let mant = (rng.next_u64() >> 40) as u32 & 0x7f_ffff;
        let exp = (exp_lo + (rng.next_u64() as u32 % span) as i32 + 127) as u32;
        let value = f32::from_bits((exp << 23) | mant);
        if (lo..=hi).contains(&value) {
            out.push(value);
        }
    }
    out
}

/// The classic lane carries f64 stats, so its base grid adds random f64 values
/// over [1, 2500] and f64 ulp neighbourhoods of the f32 grid. The non-retail
/// add has no upper clamp, so the overflow band is reachable there.
fn f64_bases(f32_grid: &[f32], random: u64, seed: u64) -> Vec<f64> {
    let mut out: Vec<f64> =
        Vec::with_capacity(f32_grid.len() + random as usize + f32_grid.len() / 8);
    out.extend(f32_grid.iter().map(|g| f64::from(*g)));
    let mut rng = Lcg::new(seed);
    for _ in 0..random {
        out.push(1.0 + (2500.0 - 1.0) * rng.next_f64());
    }
    for g in f32_grid.iter().step_by(8) {
        let base = f64::from(*g);
        let ulp = (base.abs() * f64::EPSILON).max(f64::MIN_POSITIVE);
        out.push(base + ulp * ((rng.next_f64() - 0.5) * 8.0));
    }
    out
}

fn scaled(bases: &[f64], coef: f64, exponent: f64) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = bases.iter().map(|base| coef * base).collect();
    (xs, vec![exponent; bases.len()])
}

fn divided(bases: &[f64], divisor: f64, exponent: f64) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = bases.iter().map(|base| base / divisor).collect();
    (xs, vec![exponent; bases.len()])
}

/// The retail lane builds its pow argument from an f32 stat: one f32 product,
/// widened to f64 for pow, then narrowed to f32 on store.
fn retail_product(bases32: &[f32], coef: f32, exponent32: f32) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = bases32.iter().map(|base| f64::from(*base * coef)).collect();
    (xs, vec![f64::from(exponent32); bases32.len()])
}

/// The inner stamina-keep power: wisdom^0.03, f64.
fn keep_wisdom_pairs(bases: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = bases.iter().map(|base| base.max(1.0)).collect();
    (xs, vec![0.03; bases.len()])
}

/// The outer stamina-keep chance over the reachable wisdom range. The x bit
/// pattern is the host's: the inner domain above makes the device's pct equal.
fn keep_chance_pairs(bases: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let half = half_dt();
    let mut xs = Vec::with_capacity(bases.len());
    for base in bases {
        let wisdom = base.max(1.0);
        let pct = 30.0 * (wisdom / 1000.0 + wisdom.powf(0.03));
        xs.push(1.0 - pct.min(100.0) / 100.0);
    }
    (xs, vec![half; bases.len()])
}

/// The outer competition chance over the reachable wisdom range, coef 15.0 at
/// race_solver.rs:2973.
fn competition_chance_pairs(bases: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let half = half_dt();
    let mut xs = Vec::with_capacity(bases.len());
    for base in bases {
        let wisdom = base.max(1.0);
        let pct = (15.0 * (wisdom * 0.1).log10()).max(0.0);
        xs.push(1.0 - pct.min(100.0) / 100.0);
    }
    (xs, vec![half; bases.len()])
}

/// The percent argument itself: pct.min(100.0) can land anywhere in [0, 100],
/// so the outer chance also gets a uniform pct band with both clamp ends.
fn pct_pairs(count: u64) -> (Vec<f64>, Vec<f64>) {
    let half = half_dt();
    let mut pcts: Vec<f64> = vec![0.0, 1.0, 50.0, 99.0, 99.9, 99.999, 100.0];
    let mut rng = Lcg::new(0x7063_7421);
    for _ in 0..count {
        pcts.push(100.0 * rng.next_f64());
    }
    for edge in [0.0, 100.0] {
        for k in -512i64..=512 {
            let pct = edge + k as f64 * f64::EPSILON * 64.0;
            if (0.0..=100.0).contains(&pct) {
                pcts.push(pct);
            }
        }
    }
    let xs: Vec<f64> = pcts.iter().map(|pct| 1.0 - pct.min(100.0) / 100.0).collect();
    (xs, vec![half; pcts.len()])
}

fn run64(
    queue: &Queue, program: &Program, name: &'static str, lane: &'static str,
    xs: Vec<f64>, ys: Vec<f64>,
) -> Result<Report, String> {
    let mut report = Report::new(name, lane);
    let want: Vec<u64> = xs.iter().zip(&ys).map(|(x, y)| x.powf(*y).to_bits()).collect();
    let chunk = 4_000_000usize;
    let mut start = 0usize;
    while start < xs.len() {
        let end = (start + chunk).min(xs.len());
        let len = end - start;
        let xs_buf = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(&xs[start..end])
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let ys_buf = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(&ys[start..end])
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let out = Buffer::<u64>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(program)
            .name("pow64")
            .queue(queue.clone())
            .global_work_size(len)
            .arg(&xs_buf)
            .arg(&ys_buf)
            .arg(&out)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut got = vec![0u64; len];
        out.cmd().read(&mut got).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        for i in 0..len {
            report.pairs += 1;
            let wanted = want[start + i];
            if got[i] != wanted {
                report.port += 1;
                report.sample(format!(
                    "{name} x={} y={} want={} gs_pow={}",
                    f64_word(xs[start + i].to_bits()),
                    f64_word(ys[start + i].to_bits()),
                    f64_word(wanted),
                    f64_word(got[i]),
                ));
            }
        }
        start = end;
    }
    Ok(report)
}

fn run32(
    queue: &Queue, program: &Program, name: &'static str, lane: &'static str,
    xs: Vec<f32>, ys: Vec<f32>,
) -> Result<Report, String> {
    let mut report = Report::new(name, lane);
    let want: Vec<u32> = xs.iter().zip(&ys).map(|(x, y)| x.powf(*y).to_bits()).collect();
    let chunk = 4_000_000usize;
    let mut start = 0usize;
    while start < xs.len() {
        let end = (start + chunk).min(xs.len());
        let len = end - start;
        let xs_buf = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(&xs[start..end])
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let ys_buf = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(&ys[start..end])
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let out = Buffer::<u32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(len)
            .build()
            .map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(program)
            .name("pow32")
            .queue(queue.clone())
            .global_work_size(len)
            .arg(&xs_buf)
            .arg(&ys_buf)
            .arg(&out)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        queue.finish().map_err(|e| e.to_string())?;
        let mut got = vec![0u32; len];
        out.cmd().read(&mut got).enq().map_err(|e| e.to_string())?;
        queue.finish().map_err(|e| e.to_string())?;
        for i in 0..len {
            report.pairs += 1;
            let wanted = want[start + i];
            if got[i] != wanted {
                report.port += 1;
                report.sample(format!(
                    "{name} x={} y={} want={} gs_powf={}",
                    f32_word(xs[start + i].to_bits()),
                    f32_word(ys[start + i].to_bits()),
                    f32_word(wanted),
                    f32_word(got[i]),
                ));
            }
        }
        start = end;
    }
    Ok(report)
}

#[test]
fn every_live_pow_consumer_domain_is_bit_exact() {
    let stride = env_u64("W21_STRIDE", 16) as u32;
    let random32 = env_u64("W21_F32_RANDOM", 150_000);
    let random64 = env_u64("W21_RANDOM", 250_000);
    let pct_count = env_u64("W21_PCT", 200_000);
    let (queue, program) = match open_gated_device() {
        Ok(pair) => pair,
        Err(reason) => panic!("wave21 pow gate: {reason}"),
    };

    let g32 = f32_bases(1.0, 2000.0, stride, random32, 0x6732_3121);
    let p32 = f32_bases(1.0, 2000.0, stride, random32, 0x6732_3132);
    let w32 = f32_bases(1.0, 2000.0, stride, random32, 0x6732_3133);
    let g64 = f64_bases(&g32, random64, 0x6732_3141);
    let p64 = f64_bases(&p32, random64, 0x6732_3142);
    let w64 = f64_bases(&w32, random64, 0x6732_3143);
    println!(
        "wave21 pow gate grids: f32 {} bases, f64 {} bases, pct band {} points",
        g32.len(),
        g64.len(),
        pct_count + 7 + 1_026,
    );

    let mut reports: Vec<Report> = Vec::new();
    let push64 = |reports: &mut Vec<Report>, name, lane, pairs: (Vec<f64>, Vec<f64>)| {
        let report = run64(&queue, &program, name, lane, pairs.0, pairs.1)
            .unwrap_or_else(|e| panic!("wave21 launch failed: {e}"));
        reports.push(report);
    };
    let push32 = |reports: &mut Vec<Report>, name, lane, pairs: (Vec<f32>, Vec<f32>)| {
        let report = run32(&queue, &program, name, lane, pairs.0, pairs.1)
            .unwrap_or_else(|e| panic!("wave21 launch failed: {e}"));
        reports.push(report);
    };

    // Classic lane: f64 stats and f64 literal exponents.
    push64(&mut reports, "last spurt 450*g^0.597", "classic f64", scaled(&g64, 450.0, 0.597));
    push64(&mut reports, "spot 500*g^0.6", "classic f64", scaled(&g64, 500.0, 0.6));
    push64(&mut reports, "spot duration 700*g^0.5", "both f64", scaled(&g64, 700.0, 0.5));
    push64(&mut reports, "duel target 200*g^0.708", "classic f64", scaled(&g64, 200.0, 0.708));
    push64(&mut reports, "duel accel 160*g^0.59", "classic f64", scaled(&g64, 160.0, 0.59));
    push64(&mut reports, "compete power/1500^0.5", "both f64", divided(&p64, 1500.0, 0.5));
    push64(&mut reports, "compete guts/3000^0.2", "both f64", divided(&g64, 3000.0, 0.2));
    push64(&mut reports, "secure lead guts/2000^0.5", "both f64", divided(&g64, 2000.0, 0.5));
    push64(&mut reports, "stamina keep wisdom^0.03", "both f64", keep_wisdom_pairs(&w64));
    push64(&mut reports, "stamina keep chance^(dt/2)", "both f64", keep_chance_pairs(&w64));
    push64(&mut reports, "competition chance^(dt/2)", "both f64", competition_chance_pairs(&w64));
    push64(&mut reports, "pct band (1-pct/100)^(dt/2)", "both f64", pct_pairs(pct_count));
    // Retail lane: f32 stat argument, f32 exponent widened to f64, f32 store.
    push64(&mut reports, "retail spot 500f*g, 0.6f", "retail f32", retail_product(&g32, 500.0, 0.6));
    push64(&mut reports, "retail duel target 200f*g, 0.708f", "retail f32", retail_product(&g32, 200.0, 0.708));
    push64(&mut reports, "retail duel accel 160f*g, 0.59f", "retail f32", retail_product(&g32, 160.0, 0.59));
    // The plan closure, f32 all the way (retail_physics.rs:362).
    let plan_xs: Vec<f32> = g32
        .iter()
        .map(|base| base.clamp(1.0, 2000.0) * 450.0f32)
        .collect();
    let plan_ys: Vec<f32> = vec![0.597f32; plan_xs.len()];
    push32(&mut reports, "retail plan (bounded*450f)^0.597f", "retail f32", (plan_xs, plan_ys));

    let mut failures: Vec<String> = Vec::new();
    println!("wave21 pow gate: domain lane pairs mismatches");
    for report in &reports {
        println!(
            "  {:<38} {:<11} {:>10} {:>10}",
            report.domain, report.lane, report.pairs, report.port
        );
        if report.port != 0 {
            failures.push(format!("{}: {} mismatches", report.domain, report.port));
        }
    }
    for report in &reports {
        for sample in &report.samples {
            failures.push(format!("  sample {sample}"));
        }
    }
    if !failures.is_empty() {
        panic!("wave21 pow gate NOT green:\n{}", failures.join("\n"));
    }
}

/// One consumer's real-input (base, exponent) pairs and the precision the
/// engine keeps after the power.
struct RealPairs {
    consumer: &'static str,
    kind: &'static str,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl RealPairs {
    fn new(consumer: &'static str, kind: &'static str) -> RealPairs {
        RealPairs { consumer, kind, xs: Vec::new(), ys: Vec::new() }
    }

    fn push(&mut self, x: f64, y: f64) {
        self.xs.push(x);
        self.ys.push(y);
    }
}

fn bump_kinds(stat: usize) -> &'static [i32] {
    match stat {
        2 => &[3, 32],
        3 => &[4, 32],
        4 => &[5, 32],
        _ => &[],
    }
}

/// Retail-lane live values for one stat: the f32 accumulation of the plan's
/// construction bumps, clamped on read. The adjusted f64 seed is the live
/// value when no bump touches the field.
fn retail_stat_bases(adjusted: f64, stat: usize, bumps: &[(usize, i32, f64)]) -> Vec<f64> {
    let kinds = bump_kinds(stat);
    let mut out: Vec<f64> = vec![adjusted];
    let mut acc = adjusted as f32;
    let mut touched = false;
    for (_, kind, modifier) in bumps {
        if kinds.contains(kind) {
            acc += *modifier as f32;
            out.push(f64::from(acc).clamp(1.0, 2000.0));
            touched = true;
        }
    }
    if !touched {
        out.truncate(1);
    }
    out
}

/// Classic-lane live values for one stat: the f64 add with the max(1.0) floor,
/// no upper clamp.
fn classic_stat_bases(adjusted: f64, stat: usize, bumps: &[(usize, i32, f64)]) -> Vec<f64> {
    let kinds = bump_kinds(stat);
    let mut out: Vec<f64> = vec![adjusted];
    let mut field = adjusted;
    for (_, kind, modifier) in bumps {
        if kinds.contains(kind) {
            field = (field + modifier).max(1.0);
            out.push(field);
        }
    }
    out
}

const CM: [(&str, i32); 5] = [
    ("examples/cm-10506-race.jsonl", 10506),
    ("examples/cm-10701-race.jsonl", 10701),
    ("examples/cm-10808-race.jsonl", 10808),
    ("examples/cm-10903-race.jsonl", 10903),
    ("examples/cm-11103-race.jsonl", 11103),
];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// Build the real-input pair lists for one prepared request.
fn real_input_pairs(
    stats: &[initialization::StatsCase],
    bumps: &[(usize, i32, f64)],
) -> Vec<RealPairs> {
    let mut last_spurt = RealPairs::new("last spurt 450*g^0.597", "f64");
    let mut spot = RealPairs::new("spot classic 500*g^0.6", "f64");
    let mut duration = RealPairs::new("spot duration 700*g^0.5", "f64");
    let mut duel_t = RealPairs::new("duel target 200*g^0.708", "f64");
    let mut duel_a = RealPairs::new("duel accel 160*g^0.59", "f64");
    let mut retail_spot = RealPairs::new("retail spot 500f*g, 0.6f", "f64 narrow f32");
    let mut retail_t = RealPairs::new("retail duel target 200f*g, 0.708f", "f64 narrow f32");
    let mut retail_a = RealPairs::new("retail duel accel 160f*g, 0.59f", "f64 narrow f32");
    let mut plan = RealPairs::new("retail plan (bounded*450f)^0.597f", "f32");
    let mut compete_power = RealPairs::new("compete power/1500^0.5", "f64");
    let mut compete_guts = RealPairs::new("compete guts/3000^0.2", "f64");
    let mut secure = RealPairs::new("secure lead guts/2000^0.5", "f64");
    let mut keep_wisdom = RealPairs::new("stamina keep wisdom^0.03", "f64");
    let mut keep_chance = RealPairs::new("stamina keep chance^(dt/2)", "f64");
    let mut competition = RealPairs::new("competition chance^(dt/2)", "f64");

    for (runner, case) in stats.iter().enumerate() {
        let adjusted = initialization::stats_reference(case).adjusted_f64();
        let runner_bumps: Vec<(usize, i32, f64)> =
            bumps.iter().filter(|(r, _, _)| *r == runner).copied().collect();

        for base in retail_stat_bases(adjusted[3], 3, &runner_bumps) {
            let field = base as f32;
            retail_spot.push(f64::from(field * 500.0f32), f64::from(0.6f32));
            retail_t.push(f64::from(field * 200.0f32), f64::from(0.708f32));
            retail_a.push(f64::from(field * 160.0f32), f64::from(0.59f32));
            duration.push(700.0 * base, 0.5);
            let bounded = field.clamp(1.0, 2000.0);
            plan.push(f64::from(bounded * 450.0f32), f64::from(0.597f32));
        }
        for base in classic_stat_bases(adjusted[3], 3, &runner_bumps) {
            last_spurt.push(450.0 * base, 0.597);
            spot.push(500.0 * base, 0.6);
            duration.push(700.0 * base, 0.5);
            duel_t.push(200.0 * base, 0.708);
            duel_a.push(160.0 * base, 0.59);
            compete_guts.push(base / 3000.0, 0.2);
            secure.push(base / 2000.0, 0.5);
        }
        for base in retail_stat_bases(adjusted[2], 2, &runner_bumps) {
            compete_power.push(base / 1500.0, 0.5);
        }
        for base in classic_stat_bases(adjusted[2], 2, &runner_bumps) {
            compete_power.push(base / 1500.0, 0.5);
        }
        let mut wisdom: Vec<f64> = retail_stat_bases(adjusted[4], 4, &runner_bumps);
        wisdom.extend(classic_stat_bases(adjusted[4], 4, &runner_bumps));
        for base in wisdom {
            let wisdom = base.max(1.0);
            keep_wisdom.push(wisdom, 0.03);
            let pct = 30.0 * (wisdom / 1000.0 + wisdom.powf(0.03));
            keep_chance.push(1.0 - pct.min(100.0) / 100.0, half_dt());
            let pct = (15.0 * (wisdom * 0.1).log10()).max(0.0);
            competition.push(1.0 - pct.min(100.0) / 100.0, half_dt());
        }
    }

    vec![
        last_spurt,
        spot,
        duration,
        duel_t,
        duel_a,
        compete_power,
        compete_guts,
        secure,
        keep_wisdom,
        keep_chance,
        competition,
        retail_spot,
        retail_t,
        retail_a,
        plan,
    ]
}

#[test]
fn real_cm_field_pow_inputs_are_bit_exact() {
    let (queue, program) = match open_gated_device() {
        Ok(pair) => pair,
        Err(reason) => panic!("wave21 pow gate: {reason}"),
    };
    let mut ctx = race_exec::RaceContext::open().expect("context");
    for (_, course) in CM {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry file");
        globalsim::register_course_geometry(course, geometry.clone());
        ctx.register_geometry(course, &geometry).expect("geometry");
    }

    let dump_path = std::env::var_os("GSIM_POW_PAIRS_OUT").map(std::path::PathBuf::from);
    let mut dump = String::new();
    let mut rows: Vec<(String, &'static str, u64, u64)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for (rel, course) in CM {
        let line = std::fs::read_to_string(repo_file(rel))
            .expect("request file")
            .lines()
            .next()
            .expect("first line")
            .to_string();
        let request = race_exec::multi_request(&line, course, 1).expect("request");
        let prepared = ctx.prepare(&request, 1).expect("prepare");
        assert!(
            prepared.host_fallback_reason().is_none(),
            "course {course}: the real field fell back, {}",
            prepared.host_fallback_reason().unwrap_or("")
        );
        let stat_bumps = prepared
            .stat_bumps()
            .iter()
            .filter(|(_, kind, _)| matches!(*kind, 3 | 4 | 5 | 32))
            .count();
        println!(
            "  field {course}: {} runners, {} guts/power/wisdom bumps, live guts {}",
            prepared.runners(),
            stat_bumps,
            prepared.live_guts(),
        );
        let sets = real_input_pairs(prepared.stats_cases(), prepared.stat_bumps());

        for set in sets {
            let label = format!("{course} {}", set.consumer);
            match set.kind {
                "f32" => {
                    let xs: Vec<f32> = set.xs.iter().map(|x| *x as f32).collect();
                    let ys: Vec<f32> = set.ys.iter().map(|y| *y as f32).collect();
                    if dump_path.is_some() {
                        for (x, y) in xs.iter().zip(&ys) {
                            dump.push_str(&format!(
                                "{{\"consumer\":\"{}\",\"kind\":\"f32\",\"x\":\"{:08X}\",\"y\":\"{:08X}\",\"native\":\"{:08X}\"}}\n",
                                set.consumer,
                                x.to_bits(),
                                y.to_bits(),
                                x.powf(*y).to_bits(),
                            ));
                        }
                    }
                    let report = run32(&queue, &program, "real f32", set.kind, xs, ys)
                        .unwrap_or_else(|e| panic!("wave21 launch failed: {e}"));
                    rows.push((label, set.kind, report.pairs, report.port));
                    if report.port != 0 {
                        failures.push(format!("real {course} {}: {} mismatches", set.consumer, report.port));
                        failures.extend(report.samples.iter().cloned());
                    }
                }
                _ => {
                    if dump_path.is_some() {
                        for (x, y) in set.xs.iter().zip(&set.ys) {
                            dump.push_str(&format!(
                                "{{\"consumer\":\"{}\",\"kind\":\"{}\",\"x\":\"{:016X}\",\"y\":\"{:016X}\",\"native\":\"{:016X}\"}}\n",
                                set.consumer,
                                set.kind,
                                x.to_bits(),
                                y.to_bits(),
                                x.powf(*y).to_bits(),
                            ));
                        }
                    }
                    let report = run64(&queue, &program, "real f64", set.kind, set.xs, set.ys)
                        .unwrap_or_else(|e| panic!("wave21 launch failed: {e}"));
                    rows.push((label, set.kind, report.pairs, report.port));
                    if report.port != 0 {
                        failures.push(format!("real {course} {}: {} mismatches", set.consumer, report.port));
                        failures.extend(report.samples.iter().cloned());
                    }
                }
            }
        }
    }

    if let Some(path) = dump_path {
        std::fs::write(&path, &dump).expect("write the pair dump");
        println!("  pair dump {} lines: {}", dump.lines().count(), path.display());
    }
    println!("wave21 real CM fields: consumer kind pairs mismatches");
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let mut total = 0u64;
    for (label, kind, pairs, mismatches) in &rows {
        total += pairs;
        println!("  {:<42} {:<14} {:>8} {:>10}", label, kind, pairs, mismatches);
    }
    println!("  total pairs {total}");
    if !failures.is_empty() {
        panic!("wave21 real CM fields NOT green:\n{}", failures.join("\n"));
    }
}
