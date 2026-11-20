//! Struct-layout checks for the GPU spike, plus a cross-check of the packed
//! kernel parameters against the vendored engine's `Parameters` public methods.

use globalsim_gpu::reference;
use globalsim_gpu::retail_physics::{Acceleration, TargetSpeed};

#[cfg(feature = "race")]
#[test]
fn skill_scratch_and_table_sizes_match_the_kernel_mirrors() {
    // The kernel's retail_core.cl scratch map grows by the skill regions, and
    // the host mirror must keep the same layout the work item reads.
    use globalsim_gpu::race_exec as re;
    assert_eq!(re::SK_STRIDE, 55872);
    // The full scratch mirrors: stat rows, race state, banks, then the skill
    // state block, five token orders, the substream bank, the staging, the
    // additional-activation token pool and the per-runner running lists.
    // Sized to the grid-mode roster cap RACE_MAXR 120 and SK_RMAX 960.
    // Wave 6b grew the record state 8 to 20 words per record (plus 11520),
    // the token pool 128 x 6 (plus 768) and the running lists (plus 1080),
    // the scalar block 83 to 89 slots per runner (plus 720 plus alignment),
    // one staged value per effect slot of every record (plus 15360), and the
    // double block gained 6 columns of RACE_MAXR (SRL plus 720).
    // Wave 8 grew the pd runner block 9 to 11 words (RIL plus 240) and the
    // double block 5 more guts columns of RACE_MAXR (SRL plus 600).
    // Wave 9c grew the record state 20 to 24 words per record (plus 3840),
    // the double block 2 more stat columns (plus 1920) and the record double
    // block 148 to 276 for four-arm window chains (plus 122880).
    // The fan hunt journal adds 16384 words plus the tick mirror (plus 16385).
    // Wave 10 adds the reply channel block plus the shifts from the record
    // state 24 to 26 words and the scalar block 89 to 90 slots (plus 102978).
    assert_eq!(re::SCR_U_WORDS, 602999);
    assert_eq!(re::SRL_WORDS, 356840);
    // Input and status strides track RACE_MAXR.
    assert_eq!(re::RI_STRIDE, 1592);
    // Wave 7 grew the pd runner block from 8 to 9 words for the unclamped
    // wisdom accumulator word (RIL_STRIDE 2072 to 2192). Wave 8 added the
    // guts accumulator and the spot proficiency word (2192 to 2432).
    assert_eq!(re::RIL_STRIDE, 2432);
    // Wave 10 adds the snapshot, spurt and first-spurt columns (1080 words).
    assert_eq!(re::ST_STRIDE, 2168);
    // The pd runner block, race scalars and per-runner precompute block stack
    // in that order, each behind the last per the kernel's formulas.
    assert_eq!(re::PL_FRONT_LANE, re::PI_RUNNER0 + re::PI_RUN_STRIDE * re::RACE_MAXR);
    assert_eq!(re::PD_CF_DIST, re::PL_FRONT_LANE + 16);
    assert_eq!(re::PD_RUN0, re::PL_FRONT_LANE + 32);
    assert_eq!(re::RIL_STRIDE, re::PD_RUN0 + re::PD_RUN_STRIDE * re::RACE_MAXR);
    // The init program's per-case strides fit the 120-runner IM_INIT block.
    assert!(globalsim_gpu::initialization::OS_RUN0
        + globalsim_gpu::initialization::IRS_STRIDE
            * globalsim_gpu::initialization::MAX_RUNNERS
        <= globalsim_gpu::initialization::OS_CONT_I);
    assert!(
        globalsim_gpu::initialization::OUT_U_STRIDE
            >= globalsim_gpu::initialization::OS_CONT_I + 16 + 60 + 32
    );
    assert_eq!(
        globalsim_gpu::initialization::OUT_L_STRIDE,
        globalsim_gpu::initialization::MAX_RUNNERS + 24
    );
    // Table areas stay inside the stride, header first, then course lists,
    // records, effects, programs, arms, regions, terms and tags.
    assert!(16 <= globalsim_gpu::skills_compile::SK_STRAIGHTS);
    assert!(
        globalsim_gpu::skills_compile::SK_TAGS
            + globalsim_gpu::skills_compile::SK_TMAX
            <= globalsim_gpu::skills_compile::SK_STRIDE
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_RECORDS + globalsim_gpu::skills_compile::SK_RSTRIDE
            * globalsim_gpu::skills_compile::SK_RMAX,
        globalsim_gpu::skills_compile::SK_EFFECTS
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_EFFECTS + globalsim_gpu::skills_compile::SK_ESTRIDE
            * globalsim_gpu::skills_compile::SK_EMAX,
        globalsim_gpu::skills_compile::SK_PROGRAMS
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_PROGRAMS + globalsim_gpu::skills_compile::SK_PSTRIDE
            * globalsim_gpu::skills_compile::SK_PMAX,
        globalsim_gpu::skills_compile::SK_ARMS
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_ARMS + globalsim_gpu::skills_compile::SK_ASTRIDE
            * globalsim_gpu::skills_compile::SK_AMAX,
        globalsim_gpu::skills_compile::SK_REGIONS
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_REGIONS + globalsim_gpu::skills_compile::SK_GSTRIDE
            * globalsim_gpu::skills_compile::SK_GMAX,
        globalsim_gpu::skills_compile::SK_TERMS
    );
    assert_eq!(
        globalsim_gpu::skills_compile::SK_TERMS + globalsim_gpu::skills_compile::SK_MSTRIDE
            * globalsim_gpu::skills_compile::SK_MMAX,
        globalsim_gpu::skills_compile::SK_TAGS
    );
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
