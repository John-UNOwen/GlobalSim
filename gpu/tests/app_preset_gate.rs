//! The mob-preset parity gates over the two real app job payloads in testdata,
//! which carry the app's bare-string mob skill lists and the merged skill
//! definitions only as an out-of-band file. Courses 10504 and 10604, seeds 1 to
//! 3. No race may fall back to the CPU, and the identity gate compares every
//! frame value through to_bits().

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec};

const COURSES: [i32; 2] = [10504, 10604];
const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// The merged skill definitions the app ships alongside the job payload,
/// injected into each request as its skillData map.
fn merged_skill_data() -> json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

/// One testdata job re-pointed at a course seed with the merged skillData
/// attached, so the GPU and CPU read the same roster the app sent.
fn preset_request(job: &str, seed: i64, skill_data: &json::Value) -> String {
    let mut value = json::parse(job).expect("job parses");
    let json::Value::Object(map) = &mut value else { panic!("job is not a JSON object") };
    map.insert("seed".into(), json::Value::Number(seed as f64));
    map.insert("skillData".into(), skill_data.clone());
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    out
}

fn load_jobs(course: i32) -> Vec<String> {
    let path = repo_file(&format!("gpu/testdata/app_preset_jobs_{course}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()));
    json::parse(&text)
        .expect("jobs parse")
        .as_array()
        .expect("jobs array")
        .iter()
        .map(|job| {
            let request = job.get("request").expect("job has a request");
            let mut text = String::new();
            json::to_string_into(request, &mut text);
            text
        })
        .collect()
}

/// Build the request set once: each job over each course seed, with a tag.
fn preset_requests(skill_data: &json::Value) -> Vec<(String, i64, String)> {
    let mut requests: Vec<(String, i64, String)> = Vec::new();
    for &course in &COURSES {
        for job in load_jobs(course) {
            for &seed in &SEEDS {
                let request = preset_request(job.as_str(), seed, skill_data);
                requests.push((request, seed, format!("course {course} seed {seed}")));
            }
        }
    }
    requests
}

fn register_geometry() {
    for &course in &COURSES {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
    }
}

/// Every preset job must reach the GPU kernel: the compiler accepts the app's
/// bare-string skill lists, the merged definitions, and the mob rosters, with no
/// fallback. A preset set that activated no skill would make the identity gate
/// vacuous, so require at least one CPU-side activation across the set.
#[test]
fn mob_preset_jobs_compile_without_fallback() {
    let skill_data = merged_skill_data();
    register_geometry();
    let requests = preset_requests(&skill_data);
    let out = race_exec::run(&requests.iter().map(|(r, s, _)| (r.clone(), *s)).collect::<Vec<_>>())
        .expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    for (index, (request, seed, tag)) in requests.iter().enumerate() {
        let race = &out.races[index];
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
        assert_eq!(cpu.runners, 9, "{tag}: preset roster");
        if !cpu.detail_events.is_empty() {
            fired += 1;
        }
    }
    if fired == 0 {
        problems.push("no preset race activated a skill, the identity gate is vacuous".into());
    }
    if !problems.is_empty() {
        panic!("{} preset compile problems:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Frame-for-frame parity against the CPU engine over the preset rosters.
///
/// Active gate: the 12 preset races (both courses, both jobs, seeds 1 to 3)
/// replay bit-exact on frames and raw finish times. That needs the solver
/// constructor's activation pass for immediate records and the one-tick
/// acceleration impulse a natural-deceleration current-speed effect leaves when
/// it expires.
#[test]
fn mob_preset_jobs_replay_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry();
    let requests = preset_requests(&skill_data);
    let out = race_exec::run(&requests.iter().map(|(r, s, _)| (r.clone(), *s)).collect::<Vec<_>>())
        .expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed, tag)) in requests.iter().enumerate() {
        let race = &out.races[index];
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
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cv.to_bits() != gv.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                            COLUMNS[column],
                            cv.to_bits(),
                            gv.to_bits()
                        ));
                        break 'frames;
                    }
                }
            }
        }
        for (runner, cpu) in cpu.finish_times.iter().enumerate() {
            let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
            if cpu.to_bits() != gpu.to_bits() {
                problems.push(format!(
                    "{tag} runner {runner} finishTimeRaw: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                    cpu.to_bits(),
                    gpu.to_bits()
                ));
            }
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
