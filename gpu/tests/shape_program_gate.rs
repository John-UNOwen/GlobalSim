//! The launch-shape programs (handoff F4). The host builds one race program
//! per (effect bucket, tag bucket, record bucket, shape) define set. Shapes:
//! sampling (no reply channels, no journal), full (reply channels), diag
//! (journal), full diag. Compact row masks change frame memory only, so they
//! reuse the sampling program. This gate runs the same CM chain through the
//! sampling, diag and full entry points and requires identical frames, then
//! opens a private context to pin the program cache keys.

#![cfg(feature = "race")]

use globalsim_gpu::{
    dep_cpu,
    gpu::{race_bucket_build_options, RaceShape},
    race_exec,
};

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn cm_request() -> String {
    let text = std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("request");
    let base = text.lines().next().expect("first line");
    race_exec::multi_request(base, 10701, 1).expect("multi request")
}

fn register_geometry() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, geometry);
}

fn bits(frames: &[Vec<[f64; 6]>]) -> Vec<u64> {
    frames.iter().flatten().flatten().map(|v| v.to_bits()).collect()
}

#[test]
fn the_three_shape_programs_run_the_same_chain() {
    register_geometry();
    let request = cm_request();
    let cpu = dep_cpu::run_cpu_race(&request, 1).expect("cpu race");
    let wanted = bits(&cpu.frames);

    let sampling = race_exec::run(&[(request.clone(), 1)]).expect("sampling run");
    let diag = race_exec::run_with_skill_scratch(&[(request.clone(), 1)]).expect("diag run");
    let full = race_exec::run_full(&[(request.clone(), 1)]).expect("full run");

    let mut problems: Vec<String> = Vec::new();
    for (shape, run) in [("sampling", &sampling), ("diag", &diag), ("full", &full)] {
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{shape}: fallback {reason}"));
            continue;
        }
        if bits(&race.frames) != wanted {
            problems.push(format!("{shape}: frames differ from the CPU engine"));
        }
    }
    let journal = &diag.races[0].debug_sk.as_ref().expect("diag scratch").journal;
    if journal.is_empty() {
        problems.push("diag: the journal window stayed empty".into());
    }
    if full.races[0].reply.is_none() {
        problems.push("full: no reply channels came back".into());
    }
    if sampling.races[0].reply.is_some() {
        problems.push("sampling: a no-reply launch carried a reply stream".into());
    }
    if !problems.is_empty() {
        panic!("{} shape problems:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn program_binaries_reload_from_the_disk_cache() {
    let Some(dir) = std::env::var_os("GSIM_GPU_PROGRAM_CACHE") else {
        eprintln!("GSIM_GPU_PROGRAM_CACHE is not set, the disk cache test checks nothing");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let warm = race_bucket_build_options(256, 4096, 960, RaceShape::Sampling);
    let batch = race_bucket_build_options(256, 4096, 256, RaceShape::Sampling);
    register_geometry();
    let request = cm_request();
    let cpu = dep_cpu::run_cpu_race(&request, 1).expect("cpu race");

    // First process-equivalent: fresh builds, cached under the define set
    // each was compiled with.
    let mut ctx = race_exec::RaceContext::open().expect("first context");
    let template = ctx.prepare(&request, 1).expect("prepare");
    let mut prepared = vec![template.with_seed(1)];
    let (run, _) = ctx.run_prepared(&mut prepared, None).expect("first run");
    assert!(run.races[0].fallback.is_none(), "first run fell back");
    assert_eq!(bits(&run.races[0].frames), bits(&cpu.frames), "the fresh build raced wrong");
    drop(prepared);
    drop(ctx);
    for options in [&warm, &batch] {
        let path = globalsim_gpu::gpu::program_cache_path(&dir, options);
        assert!(path.is_file(), "no cached image for `{options}`");
    }

    // Second process-equivalent: the gate run log shows the load hit lines,
    // and a reload has to race the same chain bit for bit.
    let mut ctx2 = race_exec::RaceContext::open().expect("second context");
    let template2 = ctx2.prepare(&request, 1).expect("prepare");
    let mut prepared2 = vec![template2.with_seed(1)];
    let (run2, _) = ctx2.run_prepared(&mut prepared2, None).expect("second run");
    assert!(run2.races[0].fallback.is_none(), "second run fell back");
    assert_eq!(bits(&run2.races[0].frames), bits(&cpu.frames), "the cached image changed the race");
}

#[test]
fn the_program_cache_holds_one_entry_per_bucket_and_shape() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    let mut ctx = race_exec::RaceContext::open().expect("context");
    ctx.register_geometry(10701, &geometry).expect("geometry");
    let request = cm_request();
    let template = ctx.prepare(&request, 1).expect("prepare");
    assert_eq!(template.host_fallback_reason(), None);

    // One sampling race and one compact race in the same context. A launch
    // carries one frame row count, so they run as two launches sharing one
    // program: the compact race changes frame memory, not the image.
    let mut prepared = vec![template.with_seed(1)];
    let (run, _) = ctx.run_prepared(&mut prepared, None).expect("sampling run");
    assert_eq!(run.races.iter().filter(|r| r.fallback.is_some()).count(), 0, "a race fell back");
    let cpu = dep_cpu::run_cpu_race(&request, 1).expect("cpu race");
    assert_eq!(bits(&run.races[0].frames), bits(&cpu.frames), "sampling frames differ");
    drop(prepared);

    let mut compact = vec![template.with_seed(2)];
    assert!(compact[0].set_frame_rows(&[0, 1]), "the compact mask was refused");
    let (run, _) = ctx.run_prepared(&mut compact, None).expect("compact run");
    assert_eq!(run.races.iter().filter(|r| r.fallback.is_some()).count(), 0, "a race fell back");
    let cpu2 = dep_cpu::run_cpu_race(&request, 2).expect("cpu race seed 2");
    let kept = [0usize, 1];
    let compact_wanted: Vec<u64> = cpu2
        .frames
        .iter()
        .flat_map(|tick| kept.iter().flat_map(|&r| tick[r].map(|v| v.to_bits())))
        .collect();
    assert_eq!(bits(&run.races[0].frames), compact_wanted, "compact rows differ from the CPU engine");

    let keys = ctx.program_keys();
    let sampling: Vec<_> = keys.iter().filter(|(_, _, _, s)| *s == RaceShape::Sampling).collect();
    assert_eq!(sampling.len(), keys.len(), "a plain batch built a program of another shape");
    assert!(keys.contains(&(256, 4096, 960, RaceShape::Sampling)), "open's warm program is missing");
    // The CM field's 162 records land in the 192 rung of the record ladder.
    assert!(keys.contains(&(256, 4096, 192, RaceShape::Sampling)), "the batch's record bucket built no program");
    let rcap192 = keys.iter().filter(|(_, _, r, _)| *r == 192).count();
    assert_eq!(rcap192, 1, "the compact race built a second program at the same bucket");
}
