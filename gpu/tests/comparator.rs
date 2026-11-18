//! The comparator contract, no device involved: a fake GPU outcome with one
//! injected off-by-one must report exactly the injected tick, runner, field
//! and both raw bit patterns.

use globalsim_gpu::race::{FallbackReason, RaceOutcome};
use globalsim_gpu::{comparator, dep_cpu::TickFrame};

fn cpu_frames() -> Vec<TickFrame> {
    (0..5)
        .map(|tick| {
            (0..2)
                .map(|runner| [tick as f64, runner as f64, 100. + tick as f64, 2., 1., -1.])
                .collect()
        })
        .collect()
}

#[test]
fn reports_the_injected_field_with_both_bit_patterns() {
    let cpu = cpu_frames();
    let mut gpu = cpu_frames();
    // Runner 1 of tick 3 comes back with an hp one step above the CPU value.
    gpu[3][1][2] += 1.0;
    let outcome = RaceOutcome { frames: gpu, fallback: None };

    let mismatch = comparator::compare(&cpu, &outcome).expect_err("the injected step must surface");
    let divergence = match &mismatch {
        comparator::Mismatch::Field(d) => d,
        other => panic!("expected a field mismatch, got {other:?}"),
    };
    assert_eq!((divergence.tick, divergence.runner, divergence.field), (3, 1, "hp"));
    assert_eq!(divergence.cpu_bits, 103.0_f64.to_bits());
    assert_eq!(divergence.gpu_bits, 104.0_f64.to_bits());
}

#[test]
fn an_equal_race_without_a_flag_is_a_match() {
    let cpu = cpu_frames();
    let outcome = RaceOutcome { frames: cpu_frames(), fallback: None };
    comparator::compare(&cpu, &outcome).expect("identical frames must match");
}

#[test]
fn equal_frames_with_a_flag_are_not_parity() {
    let cpu = cpu_frames();
    let outcome = RaceOutcome {
        frames: cpu_frames(),
        fallback: Some(FallbackReason::UnsupportedDomain("test domain".into())),
    };
    match comparator::compare(&cpu, &outcome) {
        Err(comparator::Mismatch::Flagged(_)) => {}
        other => panic!("expected a flagged mismatch, got {other:?}"),
    }
}
