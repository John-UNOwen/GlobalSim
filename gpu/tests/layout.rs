//! Struct-layout checks for the GPU spike, plus a cross-check of the packed
//! kernel parameters against the vendored engine's `Parameters` public methods.

use globalsim_gpu::reference;
use globalsim_gpu::retail_physics::{Acceleration, TargetSpeed};

/// Every runner-indexed region sizes to the game's 18-horse field. Host and
/// kernel take the cap from one constant (`gpu::ROSTER_CAP`, passed to the
/// kernels as `-D RACE_MAXR` / `-D CLS_MAXR`), and the literal pins below
/// follow it.
#[cfg(feature = "race")]
#[test]
fn default_layout_sizes_to_the_eighteen_horse_field() {
    use globalsim_gpu::{classic_exec as ce, gpu, race_exec as re};
    assert_eq!(gpu::ROSTER_CAP, 18);
    assert_eq!(re::RACE_MAXR, gpu::ROSTER_CAP);
    assert_eq!(ce::CLS_MAXR, gpu::ROSTER_CAP);
    assert!(gpu::build_options().contains("-D RACE_MAXR=18 -D CLS_MAXR=18"));
    assert_eq!(re::SK_STRIDE, 55872, "the skill table does not depend on the roster cap");
    // The default race program compiles the journal out, the diag variant
    // carries its 16,384-word window. The reply channels live in their own
    // rc buffer, outside the scratch stride, sized only for full requests.
    assert_eq!(re::SCR_U_WORDS, 265601);
    assert_eq!(re::SCR_U_WORDS_DIAG, 281985);
    assert_eq!(re::RC_WORDS, 100892);
    // Scratch follows the record bucket, 244 u32 words and 276 doubles per
    // record slot. A launch takes the smallest bucket that holds the batch.
    assert_eq!(re::sk_rcap(0), 64);
    assert_eq!(re::sk_rcap(64), 64);
    assert_eq!(re::sk_rcap(65), 128);
    assert_eq!(re::sk_rcap(960), 960);
    assert_eq!(re::sk_rcap(961), 960);
    assert_eq!(re::scr_u_words(64, false), 46977);
    assert_eq!(re::scr_u_words(128, false), 62593);
    assert_eq!(re::scr_u_words(256, false), 93825);
    // The start-delay column the is_badstart port adds widens the sd stride,
    // so every record bucket's d64 scratch grows by RACE_MAXR words again.
    assert_eq!(re::srl_words(64), 28244);
    assert_eq!(re::srl_words(128), 45908);
    assert_eq!(re::srl_words(256), 81236);
    // The frame window follows the course, clamped to the engine ceiling.
    assert_eq!(re::course_tick_bound(0.0), 640);
    assert_eq!(re::course_tick_bound(500.0), 1475);
    assert_eq!(re::course_tick_bound(1000.0), 2309);
    assert_eq!(re::course_tick_bound(1600.0), 3310);
    assert_eq!(re::course_tick_bound(3200.0), 5979);
    assert_eq!(re::course_tick_bound(100000.0), 6144);
    assert!(gpu::diag_build_options().ends_with("-D SK_DIAG"));
    assert!(!gpu::build_options().contains("SK_DIAG"));
    assert_eq!(ce::CLS_ST_STRIDE, 16 + 20 * ce::CLS_MAXR);
    assert_eq!(ce::CLS_RI_STRIDE, ce::CLS_RI_RUN0 + 48 * ce::CLS_MAXR);
    assert!(re::RI_REQ < re::RI_STRIDE);
}

/// The per-race ST block stride the host relaunch loop walks must equal the
/// stride the kernel writes with, or multi-race launches read the next race's
/// status and the slice loop spins. Check the host constant against the stride
/// resolved from the kernel's own ST_* defines.
#[cfg(feature = "race")]
#[test]
fn host_st_stride_matches_the_opencl_kernel() {
    use globalsim_gpu::race_exec as re;
    let source = include_str!("../src/kernels/retail_core.cl");
    let maxr = globalsim_gpu::race_exec::RACE_MAXR;
    let mut defs: Vec<(String, String)> = Vec::new();
    for line in source.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define ST_") else { continue };
        let rest = rest.split("//").next().unwrap_or(rest).trim();
        let mut it = rest.splitn(2, |c: char| c.is_whitespace());
        let name = it.next().unwrap_or("");
        let expr = it.next().unwrap_or("").trim().to_string();
        if name.is_empty() || expr.is_empty() {
            continue;
        }
        defs.push((format!("ST_{name}"), expr));
    }
    let mut val: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for _ in 0..defs.len() + 2 {
        let mut progressed = false;
        for (name, expr) in &defs {
            if val.contains_key(name) {
                continue;
            }
            if let Some(n) = eval_st_define(expr, &val, maxr) {
                val.insert(name.clone(), n);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    let stride = *val.get("ST_STRIDE").expect("kernel ST_STRIDE resolves");
    let max_word = val
        .iter()
        .filter(|(k, _)| k.as_str() != "ST_STRIDE")
        .map(|(_, v)| *v)
        .max()
        .expect("ST words present");
    assert!(
        stride > max_word,
        "kernel ST_STRIDE {stride} does not cover the last ST word {max_word}"
    );
    assert_eq!(
        stride, re::ST_STRIDE,
        "host ST_STRIDE {} differs from the kernel stride {}",
        re::ST_STRIDE, stride
    );
}

/// Resolve one ST_* #define body of names, RACE_MAXR, + and * to a word index.
/// None when a referenced name is not resolved yet.
fn eval_st_define(expr: &str, val: &std::collections::HashMap<String, usize>, maxr: usize) -> Option<usize> {
    let inner = expr.trim();
    let inner = inner.strip_prefix('(').and_then(|s| s.strip_suffix(')')).unwrap_or(inner);
    let mut total = 0usize;
    for term in inner.split('+') {
        let mut prod = 1usize;
        for factor in term.split('*') {
            let f = factor.trim();
            if f == "RACE_MAXR" {
                prod = prod.checked_mul(maxr)?;
            } else if let Ok(n) = f.parse::<usize>() {
                prod = prod.checked_mul(n)?;
            } else if let Some(&n) = val.get(f) {
                prod = prod.checked_mul(n)?;
            } else {
                return None;
            }
        }
        total += prod;
    }
    Some(total)
}

#[test]
fn kernel_structs_have_the_c_layout() {
    // repr(C) of plain f32/i32 fields has no padding on x86, so each struct's
    // size is the field count times four.
    assert_eq!(std::mem::size_of::<reference::KernelCase>(), 26 * 4);
    assert_eq!(std::mem::align_of::<reference::KernelCase>(), 4);
    assert_eq!(std::mem::size_of::<reference::KernelRow>(), 9 * 4);
    assert_eq!(std::mem::align_of::<reference::KernelRow>(), 4);
    assert_eq!(std::mem::size_of::<reference::KernelParams>(), 61 * 4);
    assert_eq!(std::mem::align_of::<reference::KernelParams>(), 4);
    const _: () = {
        assert!(std::mem::offset_of!(reference::KernelCase, speed_in) == 36);
        assert!(std::mem::offset_of!(reference::KernelCase, bonus) == 100);
        assert!(std::mem::offset_of!(reference::KernelRow, phase) == 24);
        assert!(std::mem::offset_of!(reference::KernelParams, course) == 0);
        assert!(std::mem::offset_of!(reference::KernelParams, hp_drain) == 224);
    };
}

#[test]
fn packed_kernel_params_match_the_engine_parameters() {
    // `Parameters` fields are private to the vendored module, so the packer
    // re-reads global_parameters.json. Compare the two through the engine's
    // public methods, covering every scalar in the pack.
    let p = reference::load_parameters();
    let packed = reference::kernel_params();
    for distance in [1000_f32, 1200., 1600., 2000., 3200.] {
        let course = || packed.course[0] - (distance - packed.course[1]) / packed.course[2];
        assert_eq!(p.course_speed(distance).to_bits(), course().to_bits(), "course_speed {distance}");
        let start_dash = course() * packed.minimum[0];
        assert_eq!(p.start_dash_speed(distance).to_bits(), start_dash.to_bits(), "start_dash {distance}");
        for guts in [1_f32, 333., 1234., 2000.] {
            let root = (((guts.clamp(1., 2000.) * packed.minimum[1]) as f64).sqrt() as f32) as f64;
            let minimum = (start_dash as f64 + root * packed.minimum[2] as f64) as f32;
            assert_eq!(p.minimum_speed(guts, distance).to_bits(), minimum.to_bits(), "minimum_speed {guts} {distance}");
        }
    }
    for strategy in 1..=5 {
        for phase in 0..3 {
            for speed in [1_f32, 100., 850., 1400., 2000.] {
                let (aptitude, distance) = (0.94_f32, 1200_f32);
                let expected = p.base_target_speed(speed, strategy, phase, aptitude, distance);
                let course = packed.course[0] - (distance - packed.course[1]) / packed.course[2];
                let mut base = (packed.phase_speed[strategy - 1][phase] as f64 * 0.01f32 as f64) * course as f64;
                if phase >= 2 {
                    let root = ((speed.clamp(1., 2000.) * packed.end_speed_sqrt) as f64).sqrt() as f32;
                    base += (((root as f64 * aptitude as f64) * packed.add_speed as f64) as f32) as f64;
                }
                assert_eq!(expected.to_bits(), (base as f32).to_bits(), "base_target_speed {strategy}/{phase}/{speed}");
            }
            // Uphill true here so accel[2] is covered, skipped by the restricted loop.
            let expected = p.acceleration(
                Acceleration { power: 1234., strategy, phase, ground: 1.1, distance: 0.92, uphill: true, start_dash: true },
                &[],
            );
            let root = ((1234_f32.clamp(1., 2000.) * packed.accel[0]) as f64).sqrt() as f32;
            let mut value = (((root as f64 * packed.accel[2] as f64) * 1.1f64) * 0.92f64 * packed.phase_accel[strategy - 1][phase] as f64) as f32;
            value += packed.accel[3];
            value = value.max(0.);
            assert_eq!(expected.to_bits(), value.to_bits(), "acceleration {strategy}/{phase}");
        }
    }
    for phase in [0usize, 1, 2] {
        for speed in [18_f32, 24., 30.] {
            let expected = p.hp_per_second(speed, 22.5, 1.0, 999., phase, 0.95);
            let gap = (speed - 22.5) + packed.hp_drain[0];
            let mut value = (((((gap * gap) / packed.hp_drain[1]) as f64) * packed.hp_drain[2] as f64 * 0.95f64) * 1.0f64) as f32;
            if phase >= 2 {
                let root = ((999_f32.clamp(1., 2000.) * packed.hp_drain[3]) as f64).sqrt() as f32 as f64;
                value *= (1.0f64 + packed.hp_drain[4] as f64 / root) as f32;
            }
            assert_eq!(expected.to_bits(), value.to_bits(), "hp_per_second {speed}/{phase}");
        }
    }
    // target_speed across every branch of the modifier chain and both clamps.
    let inputs = [
        (15_f32, 0_f32, false, false, true),
        (23., 1.2, false, false, true),
        (-1.2, 0., false, true, true),
        (31., 0., false, false, true),
        (21., 0., true, false, false),
    ];
    for (base, slope, moving_lane_skill, downhill, hp_remaining) in inputs {
        let minimum = 17.25;
        let expected = p.target_speed(
            TargetSpeed {
                base,
                section: 0.,
                pace: 1.,
                force_in: 0.,
                power: 1234.,
                moving_lane_skill,
                slope,
                downhill,
                hp_remaining,
                minimum,
            },
            &[],
        );
        let mut speed = (base + 0.) * 1. + 0.;
        if moving_lane_skill {
            speed += ((1234_f32.clamp(1., 2000.) * 0.0002f32) as f64).sqrt() as f32;
        }
        let correction = if slope > 0. {
            -(slope as f64 * (packed.slope[0] as f64 / 1234_f32.clamp(1., 2000.) as f64)) as f32
        } else if slope < 0. && downhill {
            (packed.slope[1] as f64 + slope.abs() as f64 / packed.slope[2] as f64) as f32
        } else {
            0.
        };
        speed += correction;
        speed += 0.;
        speed += 0.;
        if !hp_remaining {
            speed = minimum;
        }
        let actual = speed.min(30.).max(packed.target_min);
        assert_eq!(expected.to_bits(), actual.to_bits(), "target_speed {base} {slope}");
    }
}

/// The retail effect-cap buckets derive every region offset after the effect
/// pool, mirroring the skills.cl define chain, so a wider table needs no new
/// literals anywhere.
#[cfg(feature = "race")]
#[test]
fn effect_cap_buckets_derive_the_table_layout() {
    use globalsim_gpu::skills_compile as sk;
    assert_eq!(sk::SK_EMAX_BUCKETS, [256, 512, 1024, 2048]);
    assert_eq!(sk::sk_programs_at(sk::SK_EMAX), sk::SK_PROGRAMS);
    assert_eq!(sk::sk_stride_at(sk::SK_EMAX, sk::SK_TMAX), sk::SK_STRIDE);
    for bucket in sk::SK_EMAX_BUCKETS.iter().copied().chain([sk::SK_EMAX]) {
        let l = sk::sk_layout(bucket, sk::SK_TMAX);
        assert_eq!(l.emax, bucket);
        assert_eq!(l.programs, sk::SK_EFFECTS + sk::SK_ESTRIDE * bucket);
        assert_eq!(l.programs, sk::sk_programs_at(bucket));
        assert_eq!(l.arms, l.programs + sk::SK_PSTRIDE * sk::SK_PMAX);
        assert_eq!(l.regions, l.arms + sk::SK_ASTRIDE * sk::SK_AMAX);
        assert_eq!(l.terms, l.regions + sk::SK_GSTRIDE * sk::SK_GMAX);
        assert_eq!(l.tags, l.terms + sk::SK_MSTRIDE * sk::SK_MMAX);
        assert_eq!(l.stride, l.tags + sk::SK_TMAX + 16);
        assert_eq!(l.stride, sk::sk_stride_at(bucket, sk::SK_TMAX));
        assert!(l.programs <= l.arms, "regions stack at bucket {bucket}");
    }
    assert_eq!(sk::sk_stride_at(256, 4096), sk::SK_STRIDE + 640);
    assert_eq!(sk::sk_stride_at(2048, 4096), sk::SK_STRIDE + 18560);
}

/// The tag-cap buckets move only the stride tail after the tag pool, so every
/// region offset before it stays put across the ladder.
#[cfg(feature = "race")]
#[test]
fn tag_cap_buckets_move_only_the_stride_tail() {
    use globalsim_gpu::skills_compile as sk;
    assert_eq!(sk::SK_TMAX_BUCKETS, [4096, 8192]);
    assert_eq!(sk::SK_TMAX_BUCKETS[0], sk::SK_TMAX, "the first bucket is the define default");
    for bucket in sk::SK_TMAX_BUCKETS {
        let l = sk::sk_layout(sk::SK_EMAX_BUCKETS[0], bucket);
        assert_eq!(l.tmax, bucket);
        assert_eq!(l.tags, sk::sk_tags_at(sk::SK_EMAX_BUCKETS[0]));
        assert_eq!(l.stride, l.tags + bucket + 16);
        assert_eq!(l.stride, sk::sk_stride_at(sk::SK_EMAX_BUCKETS[0], bucket));
    }
    assert_eq!(sk::sk_stride_at(256, 8192), sk::sk_stride_at(256, 4096) + 4096);
    // A tag bucket past the default is the only thing the define set adds.
    let opts = globalsim_gpu::gpu::race_bucket_build_options;
    assert!(!opts(256, 4096, 960, false).contains("SK_TMAX"));
    assert!(opts(256, 8192, 960, false).ends_with("-D SK_TMAX=8192"));
}
