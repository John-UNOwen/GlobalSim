//! The app's default compare field: `examples/app-compare-race.jsonl` is the
//! exact request text the app's compare view sends (Aquarius Cup, course
//! 10611, the 9-runner default mob preset with the merged skill table, 1,155
//! ticks). The frame gate compares every value through to_bits() against the
//! CPU engine on seeds 1 to 3, and the non-vacuity check requires the field to
//! activate skills, which is what makes the per-tick skill runtime real here.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec};

const COURSE: i32 = 10611;
const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// Line 1 of the field re-pointed at one seed.
fn request_text(seed: i64) -> String {
    let path = repo_file("examples/app-compare-race.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()));
    let mut value = json::parse(text.lines().next().expect("field line")).expect("field parses");
    let json::Value::Object(map) = &mut value else { panic!("field is not a JSON object") };
    map.insert("seed".into(), json::Value::Number(seed as f64));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    out
}

fn register_geometry() {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
            .expect("geometry file");
    globalsim::register_course_geometry(COURSE, geometry);
}

#[test]
fn app_compare_field_replays_bit_exact() {
    register_geometry();
    let requests: Vec<(String, i64)> =
        SEEDS.iter().map(|&seed| (request_text(seed), seed)).collect();
    let out = race_exec::run(&requests).expect("kernel run");
    assert_eq!(out.races.len(), SEEDS.len());

    let mut problems: Vec<String> = Vec::new();
    let mut activations = 0usize;
    for ((request, seed), race) in requests.iter().zip(out.races.iter()) {
        let tag = format!("app compare field seed {seed}");
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.runners != 9 {
            problems.push(format!("{tag}: roster is {}, the app field is 9", cpu.runners));
        }
        if cpu.frames.len() < 1000 {
            problems.push(format!("{tag}: {} ticks is not the app's field", cpu.frames.len()));
        }
        activations += cpu.detail_events.len();
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!(
                "{tag}: tick count cpu {} gpu {}",
                cpu.frames.len(),
                race.frames.len()
            ));
            continue;
        }
        let mut mismatched = false;
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(gpu_frame.iter()).enumerate()
            {
                for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cv.to_bits() != gv.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                            COLUMNS[column],
                            cv.to_bits(),
                            gv.to_bits()
                        ));
                        mismatched = true;
                        break 'frames;
                    }
                }
            }
        }
        if mismatched {
            continue;
        }
        for (runner, cpu_time) in cpu.finish_times.iter().enumerate() {
            let gpu_time =
                race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
            if cpu_time.to_bits() != gpu_time.to_bits() {
                problems.push(format!(
                    "{tag} runner {runner} finishTimeRaw: cpu {cpu_time:?} ({}), gpu {gpu_time:?} ({})",
                    cpu_time.to_bits(),
                    gpu_time.to_bits()
                ));
            }
        }
    }
    if activations < 3 {
        problems.push(format!(
            "the app field activated {activations} skills over {} seeds; the frame gate is vacuous",
            SEEDS.len()
        ));
    }
    if !problems.is_empty() {
        panic!("{} app compare field problems:\n{}", problems.len(), problems.join("\n"));
    }
}
