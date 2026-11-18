//! Host side of the RNG port (handoff step 6.1): input/output buffer layout,
//! the run through the gated device, and pure-Rust reference wrappers over the
//! engine's Rule30CARng and RetailRandom.
//!
//! The reference wrappers call the engine RNGs (`crate::random_gen` and
//! `crate::retail_random`). They do not reimplement the streams. Only
//! `uniform_guarded` and `substream_key` mirror a loop or a hash, because the
//! engine gives no capped-uniform entry point and returns no intermediate hash.

use crate::random_gen::Rule30CARng;
use crate::retail_random::RetailRandom;

pub const IN_STRIDE: usize = 16;
pub const OUT_STRIDE: usize = 80;
pub const TAG_CAP: usize = 64;

pub const MODE_RULE30: u32 = 0;
pub const MODE_RETAIL: u32 = 1;
pub const MODE_SUBSTREAM: u32 = 2;

pub const SERIES_NEXTINT: u32 = 0;
pub const SERIES_NEXTDOUBLE: u32 = 1;
pub const SERIES_INTEGER: u32 = 2;
pub const SERIES_REAL: u32 = 3;
pub const SERIES_REALDIST: u32 = 4;

pub const R30_N: usize = 64;
pub const R30_K: usize = 8;
pub const R_ARR: usize = 12;
pub const SUB_CHAIN: usize = 4;
pub const SUB_DBL: usize = 2;
pub const UNIFORM_CAP: u32 = 4096;
pub const SENTINEL: i32 = 2147483647;

pub const IS_MODE: usize = 0;
pub const IS_SERIES: usize = 1;
pub const IS_SEED_LO: usize = 2;
pub const IS_SEED_HI: usize = 3;
pub const IS_SEED_I32: usize = 4;
pub const IS_RUNNER: usize = 5;
pub const IS_REAL_LO: usize = 6;
pub const IS_REAL_HI: usize = 7;
pub const IS_INT_LO: usize = 8;
pub const IS_INT_HI: usize = 9;
pub const IS_TAG_LEN: usize = 10;

pub const OS_R_STATE: usize = 0;
pub const OS_R_A: usize = 56;
pub const OS_R_B: usize = 57;
pub const OS_R_DC: usize = 58;
pub const OS_SUB_SEED: usize = 60;
pub const OS_FLAG: usize = 63;
pub const OS_RES: usize = 64;
pub const OS_LO: usize = 65;
pub const OS_UNI: usize = 66;
pub const OS_I32: usize = 67;

pub const OL_RES: usize = 0;
pub const OL_SUB_DBL: usize = 8;

// The kernel's next_double multiplier, as the engine's own expression. Rust has
// no hex float literal, so rng_gate.rs ties the kernel's hex literal to this by
// comparing raw bits (0x3e00000000200000).
pub const RD_SCALE: f64 = 1.0 / 2147483647.0;

#[derive(Clone)]
pub struct RngItem {
    mode: u32,
    series: u32,
    seed_lo: u32,
    seed_hi: u32,
    seed_i32: i32,
    runner: u32,
    real_lo: f32,
    real_hi: f32,
    int_lo: i32,
    int_hi: i32,
    upper: f64,
    tag: [u8; TAG_CAP],
    tag_len: usize,
}

fn blank(mode: u32) -> RngItem {
    RngItem {
        mode,
        series: 0,
        seed_lo: 0,
        seed_hi: 0,
        seed_i32: 0,
        runner: 0,
        real_lo: 0.,
        real_hi: 0.,
        int_lo: 0,
        int_hi: 0,
        upper: 0.,
        tag: [0; TAG_CAP],
        tag_len: 0,
    }
}

impl RngItem {
    pub fn rule30(seed_lo: u32, seed_hi: u32, upper: f64) -> RngItem {
        let mut it = blank(MODE_RULE30);
        it.seed_lo = seed_lo;
        it.seed_hi = seed_hi;
        it.upper = upper;
        it
    }

    pub fn retail(series: u32, seed: i32, int_lo: i32, int_hi: i32, real_lo: f32, real_hi: f32) -> RngItem {
        let mut it = blank(MODE_RETAIL);
        it.series = series;
        it.seed_i32 = seed;
        it.int_lo = int_lo;
        it.int_hi = int_hi;
        it.real_lo = real_lo;
        it.real_hi = real_hi;
        it
    }

    pub fn substream(seed: i32, runner: u32, tag: &str) -> RngItem {
        let mut it = blank(MODE_SUBSTREAM);
        it.seed_i32 = seed;
        it.runner = runner;
        assert!(tag.len() <= TAG_CAP, "substream tag longer than TAG_CAP");
        it.tag[..tag.len()].copy_from_slice(tag.as_bytes());
        it.tag_len = tag.len();
        it
    }

    fn words(&self) -> [u32; IN_STRIDE] {
        let mut w = [0u32; IN_STRIDE];
        w[IS_MODE] = self.mode;
        w[IS_SERIES] = self.series;
        w[IS_SEED_LO] = self.seed_lo;
        w[IS_SEED_HI] = self.seed_hi;
        w[IS_SEED_I32] = self.seed_i32 as u32;
        w[IS_RUNNER] = self.runner;
        w[IS_REAL_LO] = self.real_lo.to_bits();
        w[IS_REAL_HI] = self.real_hi.to_bits();
        w[IS_INT_LO] = self.int_lo as u32;
        w[IS_INT_HI] = self.int_hi as u32;
        w[IS_TAG_LEN] = self.tag_len as u32;
        w
    }
}

pub struct Rule30Ref {
    pub hi: u32,
    pub lo: u32,
    pub int32: [u32; R30_K],
    pub random_bits: [u64; R30_K],
    pub uniform: i32,
    pub uniform_engine: i32,
    pub flag: u32,
}

pub fn rule30_reference(seed_lo: u32, seed_hi: u32, upper: f64) -> Rule30Ref {
    let mut a = Rule30CARng::new(seed_lo, seed_hi);
    let mut b = Rule30CARng::new(seed_lo, seed_hi);
    for _ in 0..R30_N {
        a.step();
        b.step();
    }
    let hi = a.hi;
    let lo = a.lo;
    let mut int32 = [0u32; R30_K];
    for k in 0..R30_K {
        int32[k] = a.int32();
    }
    let mut random_bits = [0u64; R30_K];
    for k in 0..R30_K {
        random_bits[k] = a.random().to_bits();
    }
    let (uniform, flag) = uniform_guarded(&mut a, upper, UNIFORM_CAP);
    // The uncapped engine call on a parallel stream that ran the same prefix
    // (N steps, K int32, K random) up to the uniform, so the gate can check the
    // guard never changed a non-degenerate answer. Skip it when the guard hit the
    // cap: the engine loop runs forever there, which is the case the cap guards.
    let uniform_engine = if flag == 0 {
        for _ in 0..R30_K {
            b.int32();
        }
        for _ in 0..R30_K {
            b.random();
        }
        b.uniform(upper)
    } else {
        uniform
    };
    Rule30Ref {
        hi,
        lo,
        int32,
        random_bits,
        uniform,
        uniform_engine,
        flag,
    }
}

pub fn uniform_guarded(rng: &mut Rule30CARng, upper: f64, cap: u32) -> (i32, u32) {
    let v = (upper - 1.0) as i64;
    let ored = (v | 1) as u32;
    let clz = ored.leading_zeros();
    let mask = 0xFFFF_FFFFu32 >> clz;
    for _ in 0..cap {
        let n_u32 = rng.int32() & mask;
        let n: i64 = if n_u32 >= 0x8000_0000 {
            n_u32 as i64 - 0x1_0000_0000
        } else {
            n_u32 as i64
        };
        if (n as f64) < upper {
            return (n as i32, 0);
        }
    }
    (SENTINEL, 1)
}

#[derive(Default)]
pub struct RetailRef {
    pub draw_count: u32,
    pub ints: [i32; R_ARR],
    pub dbl_bits: [u64; R_ARR],
    pub f32_bits: [u32; R_ARR],
}

pub fn retail_reference(series: u32, seed: i32, int_lo: i32, int_hi: i32, real_lo: f32, real_hi: f32) -> RetailRef {
    let mut r = RetailRandom::new(seed);
    let mut res = RetailRef::default();
    match series {
        SERIES_NEXTINT => {
            for i in 0..R_ARR {
                res.ints[i] = r.next_int();
            }
        }
        SERIES_NEXTDOUBLE => {
            for i in 0..R_ARR {
                res.dbl_bits[i] = r.next_double().to_bits();
            }
        }
        SERIES_INTEGER => {
            for i in 0..R_ARR {
                res.ints[i] = r.integer(int_lo, int_hi);
            }
        }
        SERIES_REAL => {
            for i in 0..R_ARR {
                res.f32_bits[i] = r.real(real_lo, real_hi).to_bits();
            }
        }
        _ => {
            for i in 0..R_ARR {
                res.f32_bits[i] = r.real_distance(real_lo, real_hi).to_bits();
            }
        }
    }
    res.draw_count = r.draw_count as u32;
    res
}

pub struct SubRef {
    pub hash: u64,
    pub seed: i32,
    pub ints: [i32; SUB_CHAIN],
    pub dbl_bits: [u64; SUB_DBL],
}

pub fn substream_key(seed: i32, runner: u32, tag: &str) -> (u64, i32) {
    let mut hash = 0xcbf29ce484222325u64 ^ (seed as u32 as u64);
    hash = hash.wrapping_mul(0x100000001b3) ^ runner as u64;
    for byte in tag.bytes() {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    (hash, (hash >> 33) as i32)
}

pub fn substream_reference(seed: i32, runner: u32, tag: &str) -> SubRef {
    // The chain comes from the engine's own substream(), so keying is pinned end
    // to end. The hash and derived seed are checked against substream_key.
    let (hash, derived) = substream_key(seed, runner, tag);
    let mut r = crate::retail_random::substream(seed, runner as usize, tag);
    let mut res = SubRef {
        hash,
        seed: derived,
        ints: [0; SUB_CHAIN],
        dbl_bits: [0; SUB_DBL],
    };
    for i in 0..SUB_CHAIN {
        res.ints[i] = r.next_int();
    }
    for i in 0..SUB_DBL {
        res.dbl_bits[i] = r.next_double().to_bits();
    }
    res
}

pub struct FixtureRow {
    pub seed: i32,
    pub integers: Vec<i64>,
    pub doubles: Vec<f64>,
    pub wide: Vec<i64>,
}

pub fn load_retail_fixture() -> Vec<FixtureRow> {
    let text = include_str!("../../tests/fixtures/retail_rng_dotnet.json").trim_start_matches('\u{feff}');
    let data = crate::json::parse(text).unwrap();
    let mut rows = Vec::new();
    for row in data.as_array().unwrap() {
        let integers: Vec<i64> = row.get_array("integers").unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let doubles: Vec<f64> = row.get_array("doubles").unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let wide: Vec<i64> = row.get_array("wide").unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        rows.push(FixtureRow { seed: row.get_i64("seed").unwrap() as i32, integers, doubles, wide });
    }
    rows
}

#[cfg(feature = "gpu")]
pub struct RngOutput {
    pub out_u: Vec<u32>,
    pub out_l: Vec<u64>,
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
}

#[cfg(feature = "gpu")]
pub fn run(items: &[RngItem]) -> Result<RngOutput, String> {
    use ocl::{flags, Buffer};

    let spike = crate::gpu::Spike::open()?;
    let program = spike.build_program(crate::gpu::RNG_KERNEL_SOURCE)?;
    let queue = spike.queue();
    let n = items.len();

    let in_u: Vec<u32> = items.iter().flat_map(|it| it.words()).collect();
    let in_upper: Vec<f64> = items.iter().map(|it| it.upper).collect();
    let in_tag: Vec<u8> = items.iter().flat_map(|it| it.tag).collect();

    let n_el = |k: usize| k * n;
    let b_in_u = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).len(n_el(IN_STRIDE)).build().map_err(|e| e.to_string())?;
    let b_in_upper = Buffer::<f64>::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).len(n_el(1)).build().map_err(|e| e.to_string())?;
    let b_in_tag = Buffer::<u8>::builder().queue(queue.clone()).flags(flags::MEM_READ_ONLY).len(n_el(TAG_CAP)).build().map_err(|e| e.to_string())?;
    let b_out_u = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(n_el(OUT_STRIDE)).build().map_err(|e| e.to_string())?;
    let b_out_l = Buffer::<u64>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(n_el(OUT_STRIDE)).build().map_err(|e| e.to_string())?;

    b_in_u.cmd().write(&in_u).enq().map_err(|e| e.to_string())?;
    b_in_upper.cmd().write(&in_upper).enq().map_err(|e| e.to_string())?;
    b_in_tag.cmd().write(&in_tag).enq().map_err(|e| e.to_string())?;

    let kernel = ocl::Kernel::builder()
        .program(&program)
        .name("rng")
        .queue(queue.clone())
        .global_work_size(n)
        .arg(&b_in_u)
        .arg(&b_in_upper)
        .arg(&b_in_tag)
        .arg(&b_out_u)
        .arg(&b_out_l)
        .build()
        .map_err(|e| e.to_string())?;
    unsafe { kernel.enq().map_err(|e| e.to_string())? };
    queue.finish().map_err(|e| e.to_string())?;

    let mut out_u = vec![0u32; n_el(OUT_STRIDE)];
    let mut out_l = vec![0u64; n_el(OUT_STRIDE)];
    b_out_u.cmd().read(&mut out_u).enq().map_err(|e| e.to_string())?;
    b_out_l.cmd().read(&mut out_l).enq().map_err(|e| e.to_string())?;

    Ok(RngOutput {
        out_u,
        out_l,
        device: spike.device,
        build_ms: spike.build_ms,
    })
}
