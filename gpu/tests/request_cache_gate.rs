//! Request-cache gate (F2 and F3): the stats pre-pass derives once per
//! request text per context, and a prepared request is parsed once per
//! context. Every launch of the request, any seed and any batch shape,
//! answers bit for bit what the fresh derivation and parse answered.

#![cfg(feature = "race")]

use globalsim_gpu::race_exec;
use race_exec::SlicePolicy;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn request_line(rel: &str) -> String {
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

/// One CM field (159 skills, mixed bump kinds) and the skill-less v77 line
/// (no bumps, the NaN accumulator path), both on their file's own course.
fn cm_context() -> race_exec::RaceContext {
    let mut ctx = race_exec::RaceContext::open().expect("context");
    for course in [10701, 10504] {
        let geometry = std::fs::read_to_string(repo_file(&format!(
            "data/course_geometry/{course}.json"
        )))
        .expect("geometry file");
        globalsim::register_course_geometry(course, geometry.clone());
        ctx.register_geometry(course, &geometry).expect("geometry");
    }
    ctx
}

/// The second launch of one request (its row now in the context cache)
/// answers bit for bit what the first, deriving launch answered. Seeds and
/// batch shapes mix across the two.
#[test]
fn second_launch_answers_bit_exact() {
    let mut ctx = cm_context();
    let text = request_line("examples/cm-10701-race.jsonl");
    let req = race_exec::multi_request(&text, 10701, 1).expect("request");

    let first = ctx
        .run_requests(&[(req.clone(), 1i64), (req.clone(), 2i64)], Some(SlicePolicy::Off))
        .expect("cold launch");
    for race in &first.races {
        assert!(race.fallback.is_none(), "CM launch fell back");
    }

    // Same seeds again: prepares attach the cached row, the launch derives
    // nothing. Frames must be identical, per race, bit for bit.
    let second = ctx
        .run_requests(&[(req.clone(), 1i64), (req.clone(), 2i64)], Some(SlicePolicy::Off))
        .expect("cached launch");
    for (a, b) in first.races.iter().zip(&second.races) {
        assert_eq!(bits(&a.frames), bits(&b.frames), "cached launch diverged");
    }

    // Clones of one prepare (the run_repeated shape) share the row too, and
    // a fresh third seed in the same launch answers like its own solo run.
    let template = ctx.prepare(&req, 5).expect("prepare");
    let mut batch = vec![template.with_seed(5), template.with_seed(6)];
    let (run, _) = ctx.run_prepared(&mut batch, None).expect("clone launch");
    for (i, race) in run.races.iter().enumerate() {
        let solo = ctx
            .run_requests(&[(req.clone(), 5 + i as i64)], Some(SlicePolicy::Off))
            .expect("solo run");
        assert_eq!(bits(&race.frames), bits(&solo.races[0].frames), "seed {} diverged", 5 + i);
    }
}

/// A launch mixing a cached request with an uncached one answers like the
/// two run alone, so the per-group cache never bleeds one request's row
/// into another.
#[test]
fn mixed_batch_shares_nothing_between_requests() {
    let mut ctx = cm_context();
    let cm = race_exec::multi_request(&request_line("examples/cm-10701-race.jsonl"), 10701, 1).expect("cm");
    let v77 = race_exec::multi_request(&request_line("examples/v77-race.jsonl"), 10504, 1).expect("v77");

    // Cache the CM row first (9-runner launch). The v77 line has 9 runners
    // too, so the mixed batch below really does share one launch.
    let cm_solo = ctx
        .run_requests(&[(cm.clone(), 3i64)], Some(SlicePolicy::Off))
        .expect("cm warm");
    assert!(cm_solo.races[0].fallback.is_none(), "cm fell back");

    let v77_solo = ctx
        .run_requests(&[(v77.clone(), 3i64)], Some(SlicePolicy::Off))
        .expect("v77 solo");
    assert!(v77_solo.races[0].fallback.is_none(), "v77 fell back");

    let mut batch: Vec<race_exec::Prepared> = Vec::new();
    for req in [&cm, &v77, &cm] {
        batch.push(ctx.prepare(req, 3).expect("prepare"));
    }
    let (run, _) = ctx.run_prepared(&mut batch, None).expect("mixed run");
    assert_eq!(bits(&run.races[0].frames), bits(&cm_solo.races[0].frames), "first cm");
    assert_eq!(bits(&run.races[1].frames), bits(&v77_solo.races[0].frames), "v77 between two cm");
    assert_eq!(bits(&run.races[2].frames), bits(&cm_solo.races[0].frames), "second cm");
}

/// The row applies to a same-text request prepared a second time (its own
/// request id, no clone sharing): both rows of one launch must carry the
/// same derived bytes, since the text is what the cache keys on.
#[test]
fn same_text_two_prepares_answer_alike() {
    let mut ctx = cm_context();
    let req = race_exec::multi_request(&request_line("examples/cm-10701-race.jsonl"), 10701, 1).expect("cm");
    // Warm the cache with one launch.
    let warm = ctx
        .run_requests(&[(req.clone(), 9i64)], Some(SlicePolicy::Off))
        .expect("warm run");
    let mut batch = vec![ctx.prepare(&req, 9).expect("p1"), ctx.prepare(&req, 9).expect("p2")];
    let (run, _) = ctx.run_prepared(&mut batch, None).expect("twin run");
    assert_eq!(bits(&run.races[0].frames), bits(&warm.races[0].frames), "p1 diverged");
    assert_eq!(bits(&run.races[1].frames), bits(&warm.races[0].frames), "p2 diverged");
}

/// A prepared-cache hit (F3) carries exactly what a fresh parse of the same
/// seed produces: the ri row, the pd row (including the reseeded solver pair
/// the later-stamina draw reads) and the compiled skill table, bit for bit.
#[test]
fn prepare_hit_carries_fresh_parse_bytes() {
    let mut ctx = cm_context();
    let mut other = cm_context();
    let req = race_exec::multi_request(&request_line("examples/cm-10701-race.jsonl"), 10701, 1).expect("cm");
    let (prepared, hits) = ctx.cache_probe();
    assert_eq!((prepared, hits), (0, 0), "fresh context holds caches");

    let cold = ctx.prepare(&req, 41).expect("cold prepare");
    assert_eq!(ctx.cache_probe(), (1, 0), "cold prepare fills the prepared cache");

    // The hit arrives at a different seed than the template was parsed at.
    let hit = ctx.prepare(&req, 77).expect("cache hit prepare");
    assert_eq!(ctx.cache_probe(), (1, 0), "a text hit adds no cache entry");
    assert_eq!(hit.host_input_bits(), other.prepare(&req, 77).expect("fresh parse").host_input_bits(), "hit differs from a fresh parse");

    // The plan toggle keeps its own entry: the compiled table carries the
    // marking per record, so the 159-skill request shows the difference.
    ctx.set_plan_enabled(false);
    let plan_off = ctx.prepare(&req, 41).expect("plan-off prepare");
    assert_eq!(ctx.cache_probe(), (2, 0), "plan-off marking is its own entry");
    assert_ne!(plan_off.host_input_bits().2, cold.host_input_bits().2, "plan marking missing from the table");
    ctx.set_plan_enabled(true);
    assert_eq!(
        ctx.prepare(&req, 41).expect("plan-on hit").host_input_bits(),
        cold.host_input_bits(),
        "plan-on hit differs from the plan-on parse"
    );
    let v77 = race_exec::multi_request(&request_line("examples/v77-race.jsonl"), 10504, 1).expect("v77");
    ctx.prepare(&v77, 7).expect("v77 prepare");
    assert_eq!(ctx.cache_probe(), (3, 0), "second text adds one entry");
}

/// Re-registering a course's geometry clears the prepared cache, so the next
/// prepare parses against the new registry content (the epoch rule).
#[test]
fn geometry_replacement_drops_prepared_entries() {
    let mut ctx = cm_context();
    let req = race_exec::multi_request(&request_line("examples/cm-10701-race.jsonl"), 10701, 1).expect("cm");
    ctx.prepare(&req, 1).expect("prepare fills the cache");
    assert_eq!(ctx.cache_probe().0, 1, "one prepared entry");
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    ctx.register_geometry(10701, &geometry).expect("re-register");
    assert_eq!(ctx.cache_probe().0, 0, "re-registered geometry must drop prepared entries");
    ctx.prepare(&req, 1).expect("prepare after re-register");
    assert_eq!(ctx.cache_probe().0, 1, "the next prepare parses fresh and caches");
}

/// A launched Prepared keeps its geometry (shared Arc, not drained), so the
/// same prepared race answers the same when launched a second time.
#[test]
fn launched_prepared_relaunches_identically() {
    let mut ctx = cm_context();
    let req = race_exec::multi_request(&request_line("examples/cm-10701-race.jsonl"), 10701, 1).expect("cm");
    let mut batch = vec![ctx.prepare(&req, 11).expect("prepare")];
    let (run, _) = ctx.run_prepared(&mut batch, None).expect("first launch");
    assert!(run.races[0].fallback.is_none(), "first launch fell back");
    let (again, _) = ctx.run_prepared(&mut batch, None).expect("relaunch");
    assert!(again.races[0].fallback.is_none(), "relaunch fell back");
    assert_eq!(bits(&run.races[0].frames), bits(&again.races[0].frames), "relaunch diverged");
}
