//! Handoff step 6.1 gate: the OpenCL `rng` kernel reproduces the engine's
//! Rule30CARng, RetailRandom and substream keying bit for bit on the gated
//! device, and the retail chain reproduces the independent .NET fixture.
//! Every float is compared through `to_bits()`.
#![cfg(feature = "gpu")]

use globalsim_gpu::rng;

fn spread_u64(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn next_double_constant_ties_the_kernel_literal_to_rust() {
    // The kernel embeds the hex literal `0x1.0000000200000p-31` as RD_SCALE.
    // Rust has no hex float literal, so rng::RD_SCALE is the engine expression
    // `1.0 / 2147483647.0`, and these asserts tie that expression, the kernel's
    // literal, and the engine's `1.0/i32::MAX as f64` to one bit pattern.
    assert_eq!(rng::RD_SCALE.to_bits(), (1.0 / i32::MAX as f64).to_bits());
    assert_eq!(rng::RD_SCALE.to_bits(), 0x3e00000000200000);
}

#[test]
fn rule30_kernel_matches_the_pure_rust_chain() {
    const N: usize = 1024;
    let uppers = [5.5f64, 1.0, 0.0, -0.5, 2.0, 100.0, 3.25, 10.0];
    let upper_of = |i: usize| {
        if i == 0 {
            -3.0e9
        } else {
            uppers[i % uppers.len()]
        }
    };
    let seeds: Vec<(u32, u32)> = (0..N)
        .map(|i| {
            let s = spread_u64(i as u64);
            (s as u32, (s >> 32) as u32)
        })
        .collect();

    let items: Vec<rng::RngItem> = seeds
        .iter()
        .enumerate()
        .map(|(i, (lo, hi))| rng::RngItem::rule30(*lo, *hi, upper_of(i)))
        .collect();
    let out = rng::run(&items).expect("rule30 kernel run");
    report_device(&out);
    let ou = |id: usize, slot: usize| out.out_u[id * rng::OUT_STRIDE + slot];
    let ol = |id: usize, slot: usize| out.out_l[id * rng::OUT_STRIDE + slot];

    let mut cap_hits = 0;
    for (i, &(lo, hi)) in seeds.iter().enumerate() {
        let upper = upper_of(i);
        let r = rng::rule30_reference(lo, hi, upper);
        let tag = || format!("rule30 seed({lo:#08x},{hi:#08x}) upper {upper} item {i}");
        assert_eq!(ou(i, rng::OS_RES), r.hi, "{} final hi", tag());
        assert_eq!(ou(i, rng::OS_LO), r.lo, "{} final lo", tag());
        assert_eq!(ou(i, rng::OS_FLAG), r.flag, "{} uniform flag", tag());
        let uni = ou(i, rng::OS_UNI) as i32;
        assert_eq!(uni, r.uniform, "{} uniform got {uni} want {}", tag(), r.uniform);
        if r.flag == 1 {
            cap_hits += 1;
            assert_eq!(uni, rng::SENTINEL, "{} cap sentinel", tag());
        } else {
            assert_eq!(r.uniform, r.uniform_engine, "{} guard differs from engine uniform", tag());
            assert_eq!(uni, r.uniform_engine, "{} kernel uniform differs from engine uniform", tag());
        }
        for k in 0..rng::R30_K {
            assert_eq!(ou(i, rng::OS_I32 + k), r.int32[k], "{} int32[{k}]", tag());
            assert_eq!(
                ol(i, rng::OL_RES + k),
                r.random_bits[k],
                "{} random[{k}] gpu {:#018x} rust {:#018x}",
                tag(),
                ol(i, rng::OL_RES + k),
                r.random_bits[k]
            );
        }
    }
    println!("{N} rule30 work items bit-identical ({cap_hits} uniform-cap hits, item 0 forces one)");
}

#[test]
fn retail_kernel_matches_the_pure_rust_chain() {
    let mut seeds: Vec<i32> = vec![0, 1, -1, i32::MAX, i32::MIN];
    for i in 0..1000 {
        seeds.push(spread_u64(i as u64 + 0x1234_5678) as i32);
    }

    struct Spec {
        seed: i32,
        series: u32,
        ilo: i32,
        ihi: i32,
        rlo: f32,
        rhi: f32,
    }
    let mut specs: Vec<Spec> = Vec::new();
    for &seed in &seeds {
        let push = |specs: &mut Vec<Spec>, series, ilo, ihi, rlo, rhi| {
            specs.push(Spec { seed, series, ilo, ihi, rlo, rhi })
        };
        push(&mut specs, rng::SERIES_NEXTINT, 0, 0, 0., 0.);
        push(&mut specs, rng::SERIES_NEXTDOUBLE, 0, 0, 0., 0.);
        push(&mut specs, rng::SERIES_INTEGER, i32::MIN, i32::MAX, 0., 0.);
        push(&mut specs, rng::SERIES_INTEGER, 5, 20, 0., 0.);
        push(&mut specs, rng::SERIES_REAL, 0, 0, 1.0, 5.0);
        push(&mut specs, rng::SERIES_REAL, 0, 0, 10.0, 3.0);
        push(&mut specs, rng::SERIES_REALDIST, 0, 0, -4.0, 7.0);
        push(&mut specs, rng::SERIES_REALDIST, 0, 0, 7.0, -4.0);
    }

    let items: Vec<rng::RngItem> = specs
        .iter()
        .map(|s| rng::RngItem::retail(s.series, s.seed, s.ilo, s.ihi, s.rlo, s.rhi))
        .collect();
    let out = rng::run(&items).expect("retail kernel run");
    report_device(&out);
    let ou = |id: usize, slot: usize| out.out_u[id * rng::OUT_STRIDE + slot];
    let ol = |id: usize, slot: usize| out.out_l[id * rng::OUT_STRIDE + slot];

    for (idx, s) in specs.iter().enumerate() {
        let r = rng::retail_reference(s.series, s.seed, s.ilo, s.ihi, s.rlo, s.rhi);
        let tag = || format!("retail seed {} series {} bounds ({},{}) real ({},{}) item {idx}", s.seed, s.series, s.ilo, s.ihi, s.rlo, s.rhi);
        assert_eq!(ou(idx, rng::OS_R_DC), r.draw_count, "{} draw_count", tag());
        assert_eq!(ou(idx, rng::OS_FLAG), 0, "{} flag must be clear", tag());
        match s.series {
            rng::SERIES_NEXTINT | rng::SERIES_INTEGER => {
                for i in 0..rng::R_ARR {
                    assert_eq!(ou(idx, rng::OS_RES + i) as i32, r.ints[i], "{} int[{i}]", tag());
                }
            }
            rng::SERIES_NEXTDOUBLE => {
                for i in 0..rng::R_ARR {
                    assert_eq!(
                        ol(idx, rng::OL_RES + i),
                        r.dbl_bits[i],
                        "{} dbl[{i}] gpu {:#018x} rust {:#018x}",
                        tag(),
                        ol(idx, rng::OL_RES + i),
                        r.dbl_bits[i]
                    );
                }
            }
            _ => {
                for i in 0..rng::R_ARR {
                    assert_eq!(
                        ou(idx, rng::OS_RES + i),
                        r.f32_bits[i],
                        "{} real[{i}] gpu {:#010x} rust {:#010x}",
                        tag(),
                        ou(idx, rng::OS_RES + i),
                        r.f32_bits[i]
                    );
                }
            }
        }
    }
    println!("{} retail series over {} seeds bit-identical (next_int, next_double, integer narrow and wide, real and real_distance swapped)", specs.len(), seeds.len());
}

#[test]
fn substream_kernel_matches_the_pure_rust_chain() {
    let tags = ["RUN", "T200361:0:condition:straight_random", "rushing", "init"];
    let mut specs: Vec<(i32, u32, &str)> = Vec::new();
    for si in 0..8i32 {
        let seed = spread_u64((si as u64) + 77) as i32;
        for runner in 0..8u32 {
            for tag in tags {
                specs.push((seed, runner, tag));
            }
        }
    }
    assert_eq!(specs.len(), 256);

    let items: Vec<rng::RngItem> = specs
        .iter()
        .map(|(s, r, t)| rng::RngItem::substream(*s, *r, t))
        .collect();
    let out = rng::run(&items).expect("substream kernel run");
    report_device(&out);
    let ou = |id: usize, slot: usize| out.out_u[id * rng::OUT_STRIDE + slot];
    let ol = |id: usize, slot: usize| out.out_l[id * rng::OUT_STRIDE + slot];

    for (idx, (seed, runner, tag)) in specs.iter().enumerate() {
        let r = rng::substream_reference(*seed, *runner, tag);
        let msg = || format!("substream seed {seed} runner {runner} tag {tag:?} item {idx}");
        assert_eq!(
            ol(idx, rng::OL_RES),
            r.hash,
            "{} hash gpu {:#018x} rust {:#018x}",
            msg(),
            ol(idx, rng::OL_RES),
            r.hash
        );
        assert_eq!(ou(idx, rng::OS_SUB_SEED) as i32, r.seed, "{} derived seed", msg());
        for i in 0..rng::SUB_CHAIN {
            assert_eq!(ou(idx, rng::OS_RES + i) as i32, r.ints[i], "{} derived int[{i}]", msg());
        }
        for i in 0..rng::SUB_DBL {
            assert_eq!(
                ol(idx, rng::OL_SUB_DBL + i),
                r.dbl_bits[i],
                "{} derived dbl[{i}] gpu {:#018x} rust {:#018x}",
                msg(),
                ol(idx, rng::OL_SUB_DBL + i),
                r.dbl_bits[i]
            );
        }
    }
    println!("256 substream keys bit-identical (derived seed, hash and derived retail chain)");
}

#[test]
fn kernel_reproduces_the_dotnet_retail_fixture() {
    let rows = rng::load_retail_fixture();
    let mut specs: Vec<(usize, u32, i32, i32)> = Vec::new();
    for ri in 0..rows.len() {
        specs.push((ri, rng::SERIES_NEXTINT, 0, 0));
        specs.push((ri, rng::SERIES_NEXTDOUBLE, 0, 0));
        specs.push((ri, rng::SERIES_INTEGER, i32::MIN, i32::MAX));
    }
    let items: Vec<rng::RngItem> = specs
        .iter()
        .map(|&(ri, series, ilo, ihi)| rng::RngItem::retail(series, rows[ri].seed, ilo, ihi, 0., 0.))
        .collect();
    let out = rng::run(&items).expect("fixture kernel run");
    report_device(&out);
    let ou = |id: usize, slot: usize| out.out_u[id * rng::OUT_STRIDE + slot];
    let ol = |id: usize, slot: usize| out.out_l[id * rng::OUT_STRIDE + slot];

    let mut checks = 0;
    for (idx, &(ri, series, _, _)) in specs.iter().enumerate() {
        let row = &rows[ri];
        match series {
            rng::SERIES_NEXTINT => {
                for i in 0..row.integers.len() {
                    assert_eq!(ou(idx, rng::OS_RES + i) as i32 as i64, row.integers[i], "fixture seed {} kernel next_int[{i}] gpu {} dotnet {}", row.seed, ou(idx, rng::OS_RES + i) as i32, row.integers[i]);
                    checks += 1;
                }
            }
            rng::SERIES_NEXTDOUBLE => {
                for i in 0..row.doubles.len() {
                    assert_eq!(ol(idx, rng::OL_RES + i), row.doubles[i].to_bits(), "fixture seed {} kernel next_double[{i}] gpu {:#018x} dotnet {:#018x}", row.seed, ol(idx, rng::OL_RES + i), row.doubles[i].to_bits());
                    checks += 1;
                }
            }
            _ => {
                for i in 0..row.wide.len() {
                    assert_eq!(ou(idx, rng::OS_RES + i) as i32 as i64, row.wide[i], "fixture seed {} kernel integer(MIN,MAX)[{i}] gpu {} dotnet {}", row.seed, ou(idx, rng::OS_RES + i) as i32, row.wide[i]);
                    checks += 1;
                }
            }
        }
    }
    println!("kernel pinned to the .NET fixture: {checks} values across {} seeds", rows.len());
}

fn report_device(out: &rng::RngOutput) {
    println!(
        "device: {} ({} CUs, single_fp_config {:#x}, double_fp_config {:#x}, build {:.1} ms)",
        out.device.name, out.device.compute_units, out.device.single_fp_config, out.device.double_fp_config, out.build_ms
    );
}
