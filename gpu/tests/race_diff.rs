//! The comparison tool on the shipped test request: the full CPU engine
//! race, the restricted GPU stub, and the fallback flag (handoff steps 4
//! and 5). Needs an OpenCL device that passes the exactness gate for the
//! stub to emit its ticks.
#![cfg(feature = "gpu")]

use globalsim_gpu::{comparator, dep_cpu, race, reference};
use race::RaceRunner;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

#[test]
fn v77_request_diverges_early_and_carries_a_fallback_flag() {
    let request = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string();
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10504.json"))
        .expect("geometry file");
    globalsim::register_course_geometry(
        dep_cpu::course_id(&request).expect("course id"),
        geometry,
    );

    let cpu = dep_cpu::run_cpu_race(&request, 7).expect("cpu race");
    assert!(cpu.frames.len() > 1000, "expected a dense whole race, got {} ticks", cpu.frames.len());
    assert_eq!(cpu.runners, 9);
    assert_eq!(cpu.finishers, 9, "the v77 race finishes on the track");

    let mut runner = race::RestrictedStub;
    let outcome = runner.run_race(&request, 7);
    match &outcome.fallback {
        Some(race::FallbackReason::UnsupportedDomain(domain)) => {
            assert_eq!(domain, "field/rng/geometry not ported");
        }
        other => panic!("expected the unsupported-domain flag, got {other:?}"),
    }
    assert_eq!(
        outcome.frames.len(),
        reference::ROWS,
        "the stub kernel emits its eight restricted ticks"
    );
    assert!(outcome.frames.iter().all(|tick| tick.len() == 1), "player lane only");

    let mismatch = comparator::compare(&cpu.frames, &outcome)
        .expect_err("the restricted stub must not match the whole race");
    println!("first divergence: {mismatch}");
    assert!(matches!(
        mismatch,
        comparator::Mismatch::Field(_)
            | comparator::Mismatch::RunnerCount { .. }
            | comparator::Mismatch::TickCount { .. }
    ));
}
