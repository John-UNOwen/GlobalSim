//! The plan-walk gate (handoff E2b). The host marks live records that carry
//! no sampled term with SK_FLAG_PLAN, and the kernel scan runs their arm
//! programs on the straight-line walk. Records that draw keep the general
//! stack walk, so the shared-stream draw order never changes.
//!
//! The gate pins three things on the CM workload, the scene with sampled
//! terms on both sides of the boundary: the marking boundary itself
//! (plan-eligible and draw-carrying records both exist, and no draw-carrying
//! record is marked), the toggle flips the marking without touching anything
//! else, and the kernel answers frame for frame the same with the plan walk
//! and with the general walk everywhere, both against the CPU oracle.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, race_exec, skills_compile};

const CM: &str = "examples/cm-10701-race.jsonl";
const SEED: i64 = 1;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn frames_equal(a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(fa, fb)| {
            fa.len() == fb.len()
                && fa.iter().zip(fb).all(|(ra, rb)| ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits()))
        })
}

/// Compares one GPU race against the CPU engine, tick by tick.
fn diff_against_cpu(tag: &str, problems: &mut Vec<String>, race: &race_exec::RaceGpu, seed: i64, request: &str) {
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        return;
    }
    'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} column {column}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// Prepares and runs the CM request once with the current plan marking, and
/// reports the record flag census.
fn run_once(
    ctx: &mut race_exec::RaceContext,
    problems: &mut Vec<String>,
    request: &str,
    plan_on: bool,
) -> Option<race_exec::RaceGpu> {
    let prepared = match ctx.prepare(request, SEED) {
        Ok(prepared) => prepared,
        Err(why) => {
            problems.push(format!("plan {plan_on}: prepare failed: {why}"));
            return None;
        }
    };
    if let Some(reason) = prepared.host_fallback_reason() {
        problems.push(format!("plan {plan_on}: fallback {reason}"));
        return None;
    }
    let mut marked = 0usize;
    let mut sampled = 0usize;
    let mut dead = 0usize;
    for (sid, flags) in prepared.sk_record_flags() {
        let plan = flags & skills_compile::SK_FLAG_PLAN != 0;
        let is_sampled = flags & skills_compile::SK_FLAG_SAMPLED != 0;
        if flags & skills_compile::SK_FLAG_DEAD != 0 {
            dead += 1;
        }
        if is_sampled {
            sampled += 1;
        }
        if plan {
            marked += 1;
            if is_sampled || flags & skills_compile::SK_FLAG_DEAD != 0 {
                problems.push(format!(
                    "plan {plan_on}: record {sid} carries draws or is dead but was marked, flags {flags:#x}"
                ));
            }
        }
    }
    println!(
        "plan {plan_on}: {} records, {marked} plan-marked, {sampled} sampled, {dead} dead",
        prepared.records()
    );
    if plan_on && marked == 0 {
        problems.push("plan on: no record carries the plan marking".into());
    }
    if !plan_on && marked > 0 {
        problems.push(format!("plan off: {marked} records still carry the plan marking"));
    }
    let (run, _) = match ctx.run_prepared(&mut [prepared], None) {
        Ok(run) => run,
        Err(why) => {
            problems.push(format!("plan {plan_on}: run failed: {why}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("plan {plan_on}: fallback {reason}"));
        return None;
    }
    Some((*race).clone())
}

#[test]
fn plan_walk_marking_is_draw_free_and_transparent() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, geometry.clone());
    let text = std::fs::read_to_string(repo_file(CM)).expect("cm request")
        .lines()
        .next()
        .expect("first line")
        .to_string();
    let request = race_exec::multi_request(&text, 10701, SEED).expect("multi request");

    let mut problems: Vec<String> = Vec::new();
    let mut ctx = race_exec::RaceContext::open().expect("context");
    ctx.register_geometry(10701, &geometry).expect("geometry register");

    ctx.set_plan_enabled(true);
    let on = match run_once(&mut ctx, &mut problems, &request, true) {
        Some(race) => race,
        None => panic!("plan-on run failed:\n{}", problems.join("\n")),
    };
    diff_against_cpu("plan on", &mut problems, &on, SEED, &request);

    ctx.set_plan_enabled(false);
    let off = match run_once(&mut ctx, &mut problems, &request, false) {
        Some(race) => race,
        None => panic!("plan-off run failed:\n{}", problems.join("\n")),
    };
    diff_against_cpu("plan off", &mut problems, &off, SEED, &request);

    if !frames_equal(&on.frames, &off.frames) {
        problems.push("the plan walk and the general walk differ frame for frame".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
