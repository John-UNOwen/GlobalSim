//! Launch coalescing gate: races that share a program bucket launch together
//! wherever they sit in the batch, and a race's answer never depends on which
//! other races share its launch. The parity sweep interleaves one skill per
//! case, so grouping only consecutive equal buckets used to launch near one
//! race per chunk.

#![cfg(feature = "race")]

use globalsim_gpu::race_exec;
use race_exec::SlicePolicy;

const CM: [(&str, i32); 5] = [
    ("examples/cm-10701-race.jsonl", 10701),
    ("examples/cm-10903-race.jsonl", 10903),
    ("examples/cm-10808-race.jsonl", 10808),
    ("examples/cm-10506-race.jsonl", 10506),
    ("examples/cm-11103-race.jsonl", 11103),
];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn request_line(course: i32) -> String {
    let rel = CM.iter().find(|(_, c)| *c == course).expect("course").0;
    std::fs::read_to_string(repo_file(rel))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

fn bits(frames: &[Vec<[f64; 6]>]) -> Vec<u64> {
    frames
        .iter()
        .flat_map(|frame| frame.iter())
        .flat_map(|row| row.iter())
        .map(|v| v.to_bits())
        .collect()
}

/// Interleave the five courses three times, so same-course races are never
/// adjacent. Consecutive grouping opened one launch per race, while bucket
/// grouping folds them into at most the (effect, tag) bucket count.
#[test]
fn interleaved_courses_coalesce_into_bucket_launches() {
    let mut ctx = race_exec::RaceContext::open().expect("context");
    for (_, course) in CM {
        let geometry = std::fs::read_to_string(repo_file(&format!(
            "data/course_geometry/{course}.json"
        )))
        .expect("geometry file");
        globalsim::register_course_geometry(course, geometry.clone());
        ctx.register_geometry(course, &geometry).expect("geometry");
    }

    // Solo reference per course at seed 1.
    let mut solo: Vec<Vec<u64>> = Vec::new();
    for (_, course) in CM {
        let req = race_exec::multi_request(&request_line(course), course, 1).expect("request");
        let one = ctx
            .run_requests(&[(req, 1i64)], Some(SlicePolicy::Off))
            .expect("solo run");
        assert!(one.races[0].fallback.is_none(), "course {course} solo fell back");
        solo.push(bits(&one.races[0].frames));
    }

    // Interleaved batch: course 0, 1, 2, 3, 4, 0, 1, ... over three seeds.
    let mut batch: Vec<(String, i64)> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    for round in 0..3i64 {
        for c in 0..CM.len() {
            let course = CM[c].1;
            let req = race_exec::multi_request(&request_line(course), course, 1 + round).expect("request");
            batch.push((req, 1 + round));
            order.push(c);
        }
    }
    let out = ctx
        .run_requests(&batch, Some(SlicePolicy::Off))
        .expect("interleaved run");

    // The seed-1 races are the same input as the solo runs, so each must
    // answer bit for bit the same. A coalescing reorder bug scrambles slots.
    let mut problems: Vec<String> = Vec::new();
    for (i, race) in out.races.iter().take(CM.len()).enumerate() {
        if race.fallback.is_none() && bits(&race.frames) != solo[i] {
            problems.push(format!("course {} interleaved launch diverges from solo", CM[i].1));
        }
    }

    // Coalesced launches stay under the (effect, tag) bucket count, not the
    // request count. Eight buckets exist, so interleaving 15 requests must
    // land in at most 8 launches where consecutive grouping needed close to 15.
    assert!(
        out.stage_ms.slices <= 8,
        "15 interleaved requests must coalesce into bucket launches, got {} slices",
        out.stage_ms.slices
    );
    assert!(out.stage_ms.slices >= 1, "expected at least one launch");
    if !problems.is_empty() {
        panic!("coalescing changed an answer:\n{}", problems.join("\n"));
    }
    println!(
        "coalesce gate: {} requests coalesced into {} bucket launches",
        batch.len(),
        out.stage_ms.slices
    );
}
