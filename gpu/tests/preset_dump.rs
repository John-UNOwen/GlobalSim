//! Diagnostics (ignored): read the kernel's merged sampled trigger windows for
//! the diverging app-preset races and the engine's own trigger sampling next to
//! them. Not a gate.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, race_exec, retail_initialization, skills_compile};

const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn merged_skill_data() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn preset_request(job: &str, seed: i64, skill_data: &Value) -> String {
    let mut value = json::parse(job).expect("job parses");
    let Value::Object(map) = &mut value else { panic!("job is not a JSON object") };
    map.insert("seed".into(), Value::Number(seed as f64));
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

fn register_geometry() {
    for course in [10504, 10604] {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
    }
}

/// The first frame where the CPU and the kernel disagree, as
/// (tick, runner, column, cpu, gpu), or None when every shared tick matches.
fn first_divergence(cpu: &dep_cpu::CpuRace, gpu: &race_exec::RaceGpu) -> Option<(usize, usize, &'static str, f64, f64)> {
    for (tick, (cf, gf)) in cpu.frames.iter().zip(gpu.frames.iter()).enumerate() {
        for (runner, (cr, gr)) in cf.iter().zip(gf.iter()).enumerate() {
            for (column, (cv, gv)) in cr.iter().zip(gr.iter()).enumerate() {
                if cv.to_bits() != gv.to_bits() {
                    return Some((tick, runner, COLUMNS[column], *cv, *gv));
                }
            }
        }
    }
    None
}

#[test]
#[ignore]
fn dump_preset_window_divergence() {
    let skill_data = merged_skill_data();
    register_geometry();
    let course_json =
        initialization_json().get("10504").expect("course 10504").clone();
    let mut rows: Vec<String> = Vec::new();
    for (job_index, job) in load_jobs(10504).iter().enumerate() {
        for seed in [1i64, 2, 3] {
            let request = preset_request(job.as_str(), seed, &skill_data);
            let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
            let gpu = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
            let race = &gpu.races[0];
            assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
            let head = format!("job {job_index} seed {seed}: cpu ticks {} gpu ticks {}", cpu.frames.len(), race.frames.len());
            match first_divergence(&cpu, race) {
                Some((tick, runner, column, cv, gv)) => {
                    rows.push(format!("{head} first divergence tick {tick} runner {runner} {column}: cpu {cv:.6} gpu {gv:.6}"));
                    let window = (tick.saturating_sub(3)..=(tick + 2).min(cpu.frames.len() - 1))
                        .map(|t| format!("t{t} pos cpu={:.6} gpu={:.6}", cpu.frames[t][runner][0], race.frames[t][runner][0]))
                        .collect::<Vec<_>>()
                        .join(" | ");
                    rows.push(format!("   runner {runner} {window}"));
                }
                None => rows.push(format!("{head} frames match")),
            }
        }
    }
    println!("{}", rows.join("\n"));
    let _ = course_json;
}

fn initialization_json() -> Value {
    globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data")
}

/// Give runner 0 only 200531 and blank every other roster. The tick range
/// where that one skill changes the speed names its fire tick and its expiry
/// tick on each side.
#[test]
#[ignore]
fn dump_effect_window_of_one_skill() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let parsed = json::parse(&base).expect("request parses");
    let build = |skill: Option<&str>| -> String {
        let mut value = parsed.clone();
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, row) in rows.iter_mut().enumerate() {
            let Value::Object(row) = row else { unreachable!() };
            let kept = match (index, skill) {
                (0, Some(name)) => vec![Value::String(name.to_string())],
                _ => Vec::new(),
            };
            row.insert("skills".into(), Value::Array(kept));
        }
        let mut request = String::new();
        json::to_string_into(&value, &mut request);
        request
    };
    let with = build(Some("200531"));
    let without = build(None);
    let cpu_with = dep_cpu::run_cpu_race(&with, seed).expect("cpu with");
    let cpu_without = dep_cpu::run_cpu_race(&without, seed).expect("cpu without");
    let mut gpu_run = race_exec::run(&[(with.clone(), seed)]).expect("gpu with");
    let gpu_with = gpu_run.races.remove(0);
    let mut gpu_run2 = race_exec::run(&[(without, seed)]).expect("gpu without");
    let gpu_without = gpu_run2.races.remove(0);
    let span = |a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>], label: &str| {
        let hits: Vec<usize> = a.iter().zip(b)
            .enumerate()
            .filter_map(|(tick, (ar, br))| {
                (ar[0][1].to_bits() != br[0][1].to_bits() || ar[0][0].to_bits() != br[0][0].to_bits())
                    .then_some(tick)
            })
            .collect();
        let first = hits.first().copied().unwrap_or(usize::MAX);
        let last = hits.last().copied().unwrap_or(usize::MIN);
        println!("{label}: speed differs on runner 0 from tick {first} to tick {last}");
    };
    span(&cpu_with.frames, &cpu_without.frames, "engine");
    span(&gpu_with.frames, &gpu_without.frames, "kernel");
    println!("tick engine-applied kernel-applied (speed increment above the skillless run)");
    for tick in 0..42 {
        let inc = |frames: &[Vec<[f64; 6]>]| -> f64 {
            frames[tick + 1][0][1] - frames[tick][0][1]
        };
        println!("tick {tick:>3} engine {:.6} kernel {:.6}",
            inc(&cpu_with.frames) - inc(&cpu_without.frames),
            inc(&gpu_with.frames) - inc(&gpu_without.frames));
    }
    match first_divergence(&cpu_with, &gpu_with) {
        Some((tick, runner, column, cv, gv)) =>
            println!("with 200531: engine and kernel diverge tick {tick} runner {runner} {column} {cv:.6}/{gv:.6}"),
        None => println!("with 200531: bit-exact"),
    }
    for tick in [16, 17, 18, 19, 20, 35, 36, 37, 38, 39] {
        println!("tick {tick:>3} engine {:.9} engine-no {:.9} kernel {:.9} kernel-no {:.9}",
            cpu_with.frames[tick][0][1], cpu_without.frames[tick][0][1],
            gpu_with.frames[tick][0][1], gpu_without.frames[tick][0][1]);
    }
}

/// Count fires per (runner, skill) on both sides, so a repeated or missing
/// activation of one record stands out without needing kernel fire ticks.
#[test]
#[ignore]
fn dump_fire_counts_per_runner_and_skill() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let request = preset_request(job.as_str(), seed, &skill_data);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let run = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
    let mut gpu: Vec<(usize, u32)> = run.races[0].events.clone();
    let mut cpu_events: Vec<(usize, u32)> = cpu
        .detail_events
        .iter()
        .map(|(horse, sid, _detail, _t)| (*horse, *sid as u32))
        .collect();
    gpu.sort_unstable();
    cpu_events.sort_unstable();
    let mut keys: Vec<(usize, u32)> = gpu.clone();
    keys.extend(cpu_events.iter().copied());
    keys.sort_unstable();
    keys.dedup();
    println!("runner skill engine kernel");
    for key in keys {
        let g = gpu.iter().filter(|k| **k == key).count();
        let c = cpu_events.iter().filter(|k| **k == key).count();
        if g != c {
            println!("  {key:?} engine {c} kernel {g}");
        }
    }
}

/// Strip skills from every horse, then from the two Nige runners only, to tell
/// a skill-driven fire apart from a plain movement difference.
#[test]
#[ignore]
fn bisect_divergence_by_field_skill_strip() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let report = |request: &str| -> String {
        let cpu = dep_cpu::run_cpu_race(request, seed).expect("cpu race");
        let run = race_exec::run(&[(request.to_string(), seed)]).expect("kernel run");
        match first_divergence(&cpu, &run.races[0]) {
            Some((tick, runner, column, cv, gv)) =>
                format!("diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => "bit-exact".to_string(),
        }
    };
    let stripped = |rows: &[usize]| -> String {
        let mut value = json::parse(&base).expect("request parses");
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for index in rows {
            if let Some(Value::Object(row)) = horses.get_mut(*index) {
                row.insert("skills".into(), json::arr(Vec::new()));
            }
        }
        let mut request = String::new();
        json::to_string_into(&value, &mut request);
        request
    };
    println!("full roster: {}", report(&base));
    println!("no skills anywhere: {}", report(&stripped(&[0, 1, 2, 3, 4, 5, 6, 7, 8])));
    println!("runners 0 and 1 skillless: {}", report(&stripped(&[0, 1])));
    println!("runners 2..8 skillless: {}", report(&stripped(&[2, 3, 4, 5, 6, 7, 8])));
    println!("runners 1..8 skillless: {}", report(&stripped(&[1, 2, 3, 4, 5, 6, 7, 8])));
}

/// The kernel's fire log against the engine's as multisets: an activation the
/// kernel has that the engine does not, or the other way round, is the culprit.
/// The kernel log also carries one EV_KAKARI entry per ended rushing episode,
/// which the engine's activation hook never emits, so those are dropped here.
#[test]
#[ignore]
fn dump_event_multiset_diff() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let request = preset_request(job.as_str(), seed, &skill_data);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let run = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
    let mut gpu: Vec<(usize, u32)> = run.races[0]
        .events
        .iter()
        .filter(|(_, id)| *id != 0xFFFF_FFFF)
        .copied()
        .collect();
    let kakari = run.races[0].events.len() - gpu.len();
    let mut cpu_events: Vec<(usize, u32)> = cpu
        .detail_events
        .iter()
        .map(|(horse, sid, _detail, _t)| (*horse, *sid as u32))
        .collect();
    gpu.sort();
    cpu_events.sort();
    println!("kernel fires {}, engine fires {}, kernel kakari entries {kakari}",
        gpu.len(), cpu_events.len());
    let mut left = gpu.iter().peekable();
    let mut right = cpu_events.iter().peekable();
    loop {
        match (left.peek(), right.peek()) {
            (Some(a), Some(b)) => match a.cmp(b) {
                std::cmp::Ordering::Less => { println!("  kernel only {a:?}"); left.next(); }
                std::cmp::Ordering::Greater => { println!("  engine only {b:?}"); right.next(); }
                std::cmp::Ordering::Equal => { left.next(); right.next(); }
            },
            (Some(_), None) => { println!("  kernel only {:?}", left.next().unwrap()); }
            (None, Some(_)) => { println!("  engine only {:?}", right.next().unwrap()); }
            (None, None) => break,
        }
    }
}

/// Every column for every runner across the diverging frames, so the shape of
/// the first difference (which runners, which fields) is visible at a glance.
#[test]
#[ignore]
fn dump_diverging_frames_all_runners() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let request = preset_request(job.as_str(), seed, &skill_data);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let run = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
    let race = &run.races[0];
    let n = cpu.frames[0].len();
    for tick in 0..46 {
        let mut lines = Vec::new();
        for r in 0..n {
            let mut marks = String::new();
            for (column, name) in COLUMNS.iter().enumerate() {
                let cv = cpu.frames[tick][r][column];
                let gv = race.frames[tick][r][column];
                if cv.to_bits() != gv.to_bits() {
                    marks.push_str(&format!(" {name} {cv:.6}/{gv:.6}"));
                }
            }
            if tick >= 34 && tick <= 40 {
                lines.push(format!("  t{tick} r{r} pos={:.6} spd={:.6} hp={:.6}{}",
                    cpu.frames[tick][r][0], cpu.frames[tick][r][1], cpu.frames[tick][r][2],
                    if marks.is_empty() { String::new() } else { format!(" DIFF:{marks}") }));
            } else if !marks.is_empty() {
                lines.push(format!("  t{tick} r{r} DIFF:{marks}"));
            }
        }
        if !lines.is_empty() {
            println!("tick {tick}");
            println!("{}", lines.join("\n"));
        }
    }
    println!("runner 0 early frames (tick pos speed):");
    println!("{}", (0..40).map(|t| format!("t{t} {:.6} {:.6}", cpu.frames[t][0][0], cpu.frames[t][0][1])).collect::<Vec<_>>().join(" | "));
}

/// its activations with the fire time. Put the two sequences side by side: the
/// first row where they disagree names the activation the kernel gets wrong.
#[test]
#[ignore]
fn dump_fire_event_diff() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let request = preset_request(job.as_str(), seed, &skill_data);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let run = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
    let mut cpu_events: Vec<(usize, i64, f64)> =
        cpu.detail_events.iter().map(|(horse, sid, _detail, t)| (*horse, *sid, *t)).collect();
    cpu_events.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let gpu_events = run.races[0].events.clone();
    println!("kernel events {}, cpu events {}", gpu_events.len(), cpu_events.len());
    let rows = gpu_events.len().max(cpu_events.len());
    for index in 0..rows {
        let gpu = gpu_events.get(index).map_or_else(String::new, |(runner, id)| format!("r{runner} {id}"));
        let cpu = cpu_events.get(index).map_or_else(String::new, |(horse, sid, t)| {
            format!("r{horse} {sid} t {t:.4} tick {:.3}", t / 0.0666)
        });
        let mark = if gpu.split_whitespace().ne(cpu.split_whitespace().take(2)) { "*" } else { " " };
        println!("{mark} #{index:<3} gpu {gpu:<16} cpu {cpu}");
    }
}

/// makes the race bit-exact is the one whose fire the kernel gets wrong. Only
/// the request changes, the kernel and the engine both run each variant.
#[test]
#[ignore]
fn bisect_divergence_by_runner0_skill() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let report = |request: &str| -> String {
        let cpu = dep_cpu::run_cpu_race(request, seed).expect("cpu race");
        let run = race_exec::run(&[(request.to_string(), seed)]).expect("kernel run");
        match first_divergence(&cpu, &run.races[0]) {
            Some((tick, runner, column, cv, gv)) =>
                format!("diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => "bit-exact".to_string(),
        }
    };
    println!("full roster: {}", report(&base));
    let root = json::parse(&base).expect("request parses");
    let horses = root.get_array("horses").expect("horses");
    let skills: Vec<String> = horses[0].get_array("skills").expect("skills").iter().map(|entry| {
        match entry.as_str() {
            Some(text) => text.to_string(),
            None => entry.get_f64("skillId").unwrap_or(0.).abs().to_string(),
        }
    }).collect();
    println!("runner 0 has {} skills", skills.len());
    let without = |dropped: Option<usize>| -> String {
        let mut value = json::parse(&base).expect("request parses");
        let Value::Object(map) = &mut value else { unreachable!() };
        let mut horses = match map.remove("horses").expect("horses") {
            Value::Array(rows) => rows,
            _ => unreachable!(),
        };
        let Value::Object(row) = &mut horses[0] else { unreachable!() };
        let existing = match row.remove("skills") {
            Some(Value::Array(list)) => list,
            _ => Vec::new(),
        };
        let kept: Vec<Value> = existing.into_iter().enumerate()
            .filter(|(i, _)| dropped.is_some_and(|d| *i != d))
            .map(|(_, v)| v)
            .collect();
        row.insert("skills".into(), json::arr(kept));
        map.insert("horses".into(), json::arr(horses));
        let mut request = String::new();
        json::to_string_into(&value, &mut request);
        request
    };
    for index in 0..skills.len() {
        let request = without(Some(index));
        println!("drop {index} {}: {}", skills[index], report(&request));
    }
    // When the runner fires nothing at all, the tick-37 divergence must be gone.
    let request = without(None);
    println!("runner 0 skillless: {}", report(&request));
    let cpu = dep_cpu::run_cpu_race(&base, seed).expect("cpu race");
    for (horse, sid, detail, t) in &cpu.detail_events {
        if *horse == 0 {
            println!("cpu event horse 0 skill {sid} detail {detail} t {t:.6} tick {:.4}", t / 0.0666);
        }
    }
}

/// Runners 2..8 carry no skills, and runners 0 and 1 keep all but one of their
/// own. The run that stops diverging names the skill the kernel fires wrong.
#[test]
#[ignore]
fn bisect_divergence_within_nige_pair() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let parsed = json::parse(&base).expect("request parses");
    let Value::Object(map) = &parsed else { unreachable!() };
    let Value::Array(horses) = map.get("horses").expect("horses") else { unreachable!() };
    let owned: Vec<Value> = horses[0]
        .get("skills")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    println!("runner 0 owns {} skills", owned.len());
    let report = |value: &Value| -> String {
        let mut request = String::new();
        json::to_string_into(value, &mut request);
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
        let run = race_exec::run(&[(request, seed)]).expect("kernel run");
        match first_divergence(&cpu, &run.races[0]) {
            Some((tick, runner, column, cv, gv)) =>
                format!("diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => "bit-exact".to_string(),
        }
    };
    let build = |drop: Option<usize>| -> Value {
        let mut value = parsed.clone();
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, row) in rows.iter_mut().enumerate() {
            let Value::Object(row) = row else { unreachable!() };
            let kept = match index {
                0 | 1 => match drop {
                    None => owned.clone(),
                    // usize::MAX is never a roster index, it means no skills.
                    Some(at) if at == usize::MAX => Vec::new(),
                    Some(at) => owned.iter().enumerate()
                        .filter(|(i, _)| *i != at)
                        .map(|(_, skill)| skill.clone())
                        .collect(),
                },
                _ => Vec::new(),
            };
            row.insert("skills".into(), Value::Array(kept));
        }
        value
    };
    println!("nige pair only: {}", report(&build(None)));
    for (index, skill) in owned.iter().enumerate() {
        let name = match skill {
            Value::String(text) => text.clone(),
            Value::Object(_) => match skill.get("skillId") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => format!("{other:?}"),
                None => "?".to_string(),
            },
            other => format!("{other:?}"),
        };
        println!("drop {index} {name}: {}", report(&build(Some(index))));
    }
    println!("nige pair with no skills: {}", report(&build(Some(usize::MAX))));
}

/// One roster blanked at a time, then one skill dropped from one roster at a
/// time, over the whole field. The runs that stop diverging name the rosters
/// and the skills the kernel still fires wrong.
#[test]
#[ignore]
fn bisect_divergence_by_runner_and_skill() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let parsed = json::parse(&base).expect("request parses");
    let name_of = |skill: &Value| -> String {
        match skill {
            Value::String(text) => text.clone(),
            Value::Object(_) => match skill.get("skillId") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => format!("{other:?}"),
                None => "?".to_string(),
            },
            other => format!("{other:?}"),
        }
    };
    let Value::Object(map) = &parsed else { unreachable!() };
    let Value::Array(rows) = map.get("horses").expect("horses") else { unreachable!() };
    let owned: Vec<Vec<Value>> = rows.iter().map(|row| {
        let Value::Object(row) = row else { unreachable!() };
        row.get("skills").and_then(Value::as_array).cloned().unwrap_or_default()
    }).collect();
    let report = |value: &Value| -> String {
        let mut request = String::new();
        json::to_string_into(value, &mut request);
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
        let run = race_exec::run(&[(request, seed)]).expect("kernel run");
        match first_divergence(&cpu, &run.races[0]) {
            Some((tick, runner, column, cv, gv)) =>
                format!("diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => "bit-exact".to_string(),
        }
    };
    let build = |keep: &dyn Fn(usize) -> Vec<Value>| -> Value {
        let mut value = parsed.clone();
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, row) in rows.iter_mut().enumerate() {
            let Value::Object(row) = row else { unreachable!() };
            row.insert("skills".into(), json::arr(keep(index)));
        }
        value
    };
    println!("full roster: {}", report(&build(&|index| owned[index].clone())));
    for runner in 0..owned.len() {
        let only = build(&|index| if index == runner { owned[index].clone() } else { Vec::new() });
        println!("only runner {runner}: {}", report(&only));
    }
    for runner in 0..owned.len() {
        let blanked = build(&|index| if index == runner { Vec::new() } else { owned[index].clone() });
        println!("blank runner {runner}: {}", report(&blanked));
    }
    for (runner, list) in owned.iter().enumerate() {
        for index in 0..list.len() {
            let value = build(&|row| {
                if row != runner {
                    return owned[row].clone();
                }
                list.iter().enumerate()
                    .filter(|(i, _)| *i != index)
                    .map(|(_, skill)| skill.clone())
                    .collect()
            });
            println!("runner {runner} drop {index} {}: {}", name_of(&list[index]), report(&value));
        }
    }
}

/// Runner 2 keeps only 200331 and 201311, the two target-speed records whose
/// joint presence starts the divergence. Each variant prints its speed-effect
/// span and the engine's own activation times for that runner.
#[test]
#[ignore]
fn dump_two_skill_overlap() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let parsed = json::parse(&base).expect("request parses");
    let build = |keep: &[&str]| -> String {
        let mut value = parsed.clone();
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, row) in rows.iter_mut().enumerate() {
            let Value::Object(row) = row else { unreachable!() };
            let kept: Vec<Value> = if index == 2 {
                keep.iter().map(|name| Value::String((*name).to_string())).collect()
            } else {
                Vec::new()
            };
            row.insert("skills".into(), Value::Array(kept));
        }
        let mut request = String::new();
        json::to_string_into(&value, &mut request);
        request
    };
    let report = |label: &str, keep: &[&str]| {
        let with = build(keep);
        let without = build(&[]);
        let cpu = dep_cpu::run_cpu_race(&with, seed).expect("cpu with");
        let cpu_off = dep_cpu::run_cpu_race(&without, seed).expect("cpu without");
        let mut gpu_run = race_exec::run(&[(with.clone(), seed)]).expect("gpu with");
        let gpu = gpu_run.races.remove(0);
        let mut gpu_run2 = race_exec::run(&[(without, seed)]).expect("gpu without");
        let gpu_off = gpu_run2.races.remove(0);
        let span = |a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>], side: &str| {
            let hits: Vec<usize> = a.iter().zip(b)
                .enumerate()
                .filter_map(|(tick, (ar, br))| {
                    (ar[2][1].to_bits() != br[2][1].to_bits() || ar[2][0].to_bits() != br[2][0].to_bits())
                        .then_some(tick)
                })
                .collect();
            let first = hits.first().copied().unwrap_or(usize::MAX);
            let last = hits.last().copied().unwrap_or(usize::MIN);
            println!("{label} {side}: speed differs on runner 2 from tick {first} to {last}");
        };
        span(&cpu.frames, &cpu_off.frames, "engine");
        span(&gpu.frames, &gpu_off.frames, "kernel");
        match first_divergence(&cpu, &gpu) {
            Some((tick, runner, column, cv, gv)) =>
                println!("{label}: diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => println!("{label}: bit-exact"),
        }
        for (horse, sid, detail, t) in &cpu.detail_events {
            if *horse == 2 {
                println!("{label} engine fire horse 2 skill {sid} detail {detail} t {t:.6} tick {:.2}", t / 0.0666);
            }
        }
    };
    report("both", &["200331", "201311"]);
    report("200331", &["200331"]);
    report("201311", &["201311"]);
}

/// Runner 2 keeps its whole roster and every other runner keeps none, the
/// smallest diverging setup. Dropping one of runner 2's skills at a time names
/// the records whose joint presence the kernel still gets wrong.
#[test]
#[ignore]
fn bisect_divergence_within_runner2() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let parsed = json::parse(&base).expect("request parses");
    let Value::Object(map) = &parsed else { unreachable!() };
    let Value::Array(rows) = map.get("horses").expect("horses") else { unreachable!() };
    let owned: Vec<Value> = rows[2]
        .get("skills")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let name_of = |skill: &Value| -> String {
        match skill {
            Value::String(text) => text.clone(),
            Value::Object(_) => match skill.get("skillId") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => format!("{other:?}"),
                None => "?".to_string(),
            },
            other => format!("{other:?}"),
        }
    };
    let build = |drop: Option<usize>| -> String {
        let mut value = parsed.clone();
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, row) in rows.iter_mut().enumerate() {
            let Value::Object(row) = row else { unreachable!() };
            let kept: Vec<Value> = match (index, drop) {
                (2, None) => owned.clone(),
                (2, Some(at)) => owned.iter().enumerate()
                    .filter(|(i, _)| *i != at)
                    .map(|(_, skill)| skill.clone())
                    .collect(),
                _ => Vec::new(),
            };
            row.insert("skills".into(), Value::Array(kept));
        }
        let mut request = String::new();
        json::to_string_into(&value, &mut request);
        request
    };
    let report = |request: &str| -> String {
        let cpu = dep_cpu::run_cpu_race(request, seed).expect("cpu race");
        let run = race_exec::run(&[(request.to_string(), seed)]).expect("kernel run");
        match first_divergence(&cpu, &run.races[0]) {
            Some((tick, runner, column, cv, gv)) =>
                format!("diverges tick {tick} runner {runner} {column} cpu {cv:.6} gpu {gv:.6}"),
            None => "bit-exact".to_string(),
        }
    };
    println!("runner 2 only: {}", report(&build(None)));
    for index in 0..owned.len() {
        println!("drop {index} {}: {}", name_of(&owned[index]), report(&build(Some(index))));
    }
}

/// Runner 2 keeps its whole roster, every other runner none. Prints the six
/// frame columns around the first divergence so the differing quantity is
/// visible before the position drift.
#[test]
#[ignore]
fn dump_runner2_frames_at_divergence() {
    let skill_data = merged_skill_data();
    register_geometry();
    let seed = 1i64;
    let job = load_jobs(10504).remove(0);
    let base = preset_request(job.as_str(), seed, &skill_data);
    let mut value = json::parse(&base).expect("request parses");
    let Value::Object(map) = &mut value else { unreachable!() };
    let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for (index, row) in rows.iter_mut().enumerate() {
        let Value::Object(row) = row else { unreachable!() };
        if index != 2 {
            row.insert("skills".into(), json::arr(Vec::new()));
        }
    }
    let mut request = String::new();
    json::to_string_into(&value, &mut request);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let run = race_exec::run(&[(request.clone(), seed)]).expect("kernel run");
    let gpu = &run.races[0];
    let off_value = {
        let mut value = json::parse(&base).expect("request parses");
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(rows) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for row in rows.iter_mut() {
            let Value::Object(row) = row else { unreachable!() };
            row.insert("skills".into(), json::arr(Vec::new()));
        }
        let mut text = String::new();
        json::to_string_into(&value, &mut text);
        text
    };
    let cpu_off = dep_cpu::run_cpu_race(&off_value, seed).expect("cpu without");
    let gpu_off_run = race_exec::run(&[(off_value, seed)]).expect("gpu without");
    let gpu_off = &gpu_off_run.races[0];
    println!("tick engine-delta kernel-delta (runner 2 speed increment above the skillless run)");
    for tick in 1180..1210 {
        let inc = |frames: &[Vec<[f64; 6]>], off: &[Vec<[f64; 6]>]| -> f64 {
            (frames[tick + 1][2][1] - frames[tick][2][1]) - (off[tick + 1][2][1] - off[tick][2][1])
        };
        println!("tick {tick:>4} engine {:>11.6} kernel {:>11.6}",
            inc(&cpu.frames, &cpu_off.frames), inc(&gpu.frames, &gpu_off.frames));
    }
    let at = match first_divergence(&cpu, gpu) {
        Some((tick, _, _, _, _)) => tick,
        None => { println!("bit-exact"); return; }
    };
    println!("first divergence tick {at}");
    for tick in at.saturating_sub(6)..=(at + 4).min(cpu.frames.len() - 1) {
        for (column, name) in COLUMNS.iter().enumerate() {
            let cv = cpu.frames[tick][2][column];
            let gv = gpu.frames[tick][2][column];
            let mark = if cv.to_bits() == gv.to_bits() { "=" } else { "!" };
            println!("tick {tick:>4} {name:<8} cpu {cv:.9} gpu {gv:.9} {mark}");
        }
    }
    for (horse, sid, detail, t) in &cpu.detail_events {
        if *horse == 2 && *t > 60.0 {
            println!("engine fire horse 2 skill {sid} detail {detail} t {t:.6} tick {:.2}", t / 0.0666);
        }
    }
}

#[test]
#[ignore]
fn dump_preset_windows_and_engine_triggers() {
    let skill_data = merged_skill_data();
    register_geometry();
    let course_id = 10504i32;
    let seed = 1i64;
    let job = load_jobs(course_id).remove(0);
    let request = preset_request(job.as_str(), seed, &skill_data);
    let gpu = race_exec::run_with_skill_scratch(&[(request.clone(), seed)]).expect("kernel run");
    let race = &gpu.races[0];
    assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
    let scratch = race.debug_sk.as_ref().expect("skill scratch readback");

    let root = json::parse(&request).expect("request parses");
    let course = initialization_json().get(&course_id.to_string()).expect("course").clone();
    let plan = skills_compile::plan(&root, &course);
    assert!(plan.reason.is_none(), "compile reason {:?}", plan.reason);
    let w = &plan.words;
    let nrec = w[skills_compile::SK_NREC] as usize;
    println!("records {nrec}, arms {} terms {}, regions {}", w[skills_compile::SK_NTERM], w[skills_compile::SK_NREG], w[skills_compile::SK_NREG]);
    let f64_at = |words: &[u32], at: usize| -> f64 {
        f64::from_bits(u64::from(words[at]) | (u64::from(words[at + 1]) << 32))
    };
    for rec in 0..nrec {
        let rd = &w[skills_compile::SK_RECORDS + skills_compile::SK_RSTRIDE * rec..];
        if rd[1] != 0 {
            continue;
        }
        let sid = rd[0];
        let arms_base = rd[21] as usize;
        let arms_len = rd[20] as usize;
        let dead = rd[2] & skills_compile::SK_FLAG_DEAD != 0;
        println!("rec {rec} skill {sid} flags {} dead {dead} priority {} arms {arms_len}", rd[2], rd[3]);
        for a in 0..arms_len {
            let ar = &w[skills_compile::SK_ARMS + skills_compile::SK_ASTRIDE * (arms_base + a)..];
            let sampled = ar[6] & 1 != 0;
            let arm_dead = ar[6] & 2 != 0;
            let term = if sampled {
                let tm = &w[skills_compile::SK_TERMS + skills_compile::SK_MSTRIDE * ar[7] as usize..];
                format!("term kind {} arg {} tag {:?}", tm[0], tm[1] as i32, &tm[2..4])
            } else {
                "no term".to_string()
            };
            println!("  arm {a} sampled {sampled} arm_dead {arm_dead} pre {}@{} post {}@{} {term}",
                ar[2], ar[3], ar[4], ar[5]);
            for (label, n, base) in [("pre", ar[2], ar[3]), ("post", ar[4], ar[5])] {
                for i in 0..n as usize {
                    let at = skills_compile::SK_REGIONS + skills_compile::SK_GSTRIDE * (base as usize + i);
                    println!("      {label}[{i}] [{:.6}, {:.6})", f64_at(w, at), f64_at(w, at + 2));
                }
            }
            let ad = &scratch.skd[140 * rec + 76 + 32 * a..];
            let nwin = scratch.sks[8 * rec + 4 + a] as usize;
            for i in 0..nwin {
                println!("      chain[{a}][{i}] [{:.6}, {:.6})", ad[2 * i], ad[2 * i + 1]);
            }
        }
        let mcount = scratch.sks[8 * rec + 1] as usize;
        let single = scratch.sks[8 * rec + 2];
        let state = scratch.sks[8 * rec];
        let wd = &scratch.skd[140 * rec..];
        println!("  merged {mcount} single {single} state {state} trigger [{:.6}, {:.6})", wd[10], wd[11]);
        for i in 0..mcount {
            println!("      merged[{i}] [{:.6}, {:.6})", wd[12 + 2 * i], wd[13 + 2 * i]);
        }
    }

    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    let gpu_race = &race_exec::run(&[(request.clone(), seed)]).expect("kernel run").races[0];
    match first_divergence(&cpu, gpu_race) {
        Some((tick, runner, column, cv, gv)) => {
            println!("first divergence tick {tick} runner {runner} {column}: cpu {cv:.6} gpu {gv:.6}")
        }
        None => println!("frames match"),
    }
    for tick in 33..42 {
        println!("tick {tick} runner 0 pos cpu={:.6} gpu={:.6} speed cpu={:.6} gpu={:.6}",
            cpu.frames[tick][0][0], gpu_race.frames[tick][0][0],
            cpu.frames[tick][0][1], gpu_race.frames[tick][0][1]);
    }

    // The engine's own trigger sampling for this race, in its own stream order.
    let horses: Vec<Value> = root.get_array("horses").expect("horses").iter().map(|horse| {
        let skills = horse.get_array("skills").expect("skills").iter().map(|entry| {
                    // The engine's own retail-input form: a numeric skill_id plus the
            // level. The roster spells skillId as a number or a string, like
            // skill_id_and_level in the request parser.
            let read_id = |value: &Value| -> Option<f64> {
                value.as_str().and_then(|text| text.parse::<f64>().ok())
                    .or_else(|| value.as_f64())
            };
            let (sid, level) = match entry.as_str() {
                Some(text) => (text.parse::<f64>().unwrap_or(0.), 1.),
                None => (
                    entry.get("skillId").and_then(read_id)
                        .or_else(|| entry.get("skill_id").and_then(read_id))
                        .unwrap_or(0.),
                    entry.get("level").and_then(Value::as_f64).unwrap_or(1.),
                ),
            };
            if sid == 0. {
                println!("roster entry without a readable skillId: {entry:?}");
            }
            json::obj(vec![("skill_id", Value::Number(sid)), ("level", Value::Number(level))])
        }).collect();
        let style = match horse.get_str("strategy").unwrap_or("Nige").to_ascii_uppercase().as_str() {
            "SENKOU" => 2., "SASI" | "SASHI" => 3., "OIKOMI" => 4., "OONIGE" => 1., _ => 1.,
        };
        json::obj(vec![
            ("skills", json::arr(skills)),
            ("base_wisdom", Value::Number(horse.get_f64("wisdom").unwrap_or(0.))),
            ("running_style", Value::Number(style)),
        ])
    }).collect();
    let parameters = globalsim_gpu::initialization::engine_parameters_value()
        .expect("retail parameters")
        .get("parameters").cloned().expect("parameters group");
    let init = match retail_initialization::initialize(
        seed as i32, &horses, &course, &skill_data, &parameters, false,
    ) {
        Ok(init) => init,
        Err(reason) => {
            println!("engine initialization refused: {reason}");
            return;
        }
    };
    println!("engine after_triggers draw count {}", init.stages.get("after_triggers").expect("stage"));
    for (runner, row) in init.horses.iter().enumerate() {
        if runner != 0 {
            continue;
        }
        for trigger in &row.triggers {
            let windows = match &trigger.windows {
                retail_initialization::Windows::Distances(list) => list
                    .iter()
                    .map(|(a, b)| format!("[{a:.6}, {b:.6})/{:08x}/{:08x}", a.to_bits(), b.to_bits()))
                    .collect::<Vec<_>>()
                    .join(" "),
                retail_initialization::Windows::Lottery(enabled) => format!("lottery {enabled}"),
            };
            println!("  trigger {} alt {} {} {} arg {} draws {}..{} {windows}",
                trigger.skill_id, trigger.alternative, trigger.field, trigger.name,
                trigger.argument, trigger.draw_start, trigger.draw_end);
        }
    }
}
