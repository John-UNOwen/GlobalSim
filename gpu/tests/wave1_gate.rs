//! Wave-1 app-parity gates: the six Phase A chart variant jobs (the exact
//! app requests from the fallback histogram dump, course 10504, seeds 1 to 3)
//! must compile with zero CPU fallback and replay frame-for-frame bit-exact
//! against the CPU engine. Plus a runtime exercise gate for the dynamically
//! ported constructs: the app jobs' late-race conditions never fire the three
//! behavioral golds (0/200 seeds measured), so a v77 roster race with
//! synthetic defs drives the any-skill observe, the band-cell read and the
//! dead-alternative wisdom carrier instead.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

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

/// The raw dumped Phase A variant requests: (variant label, request text).
fn variant_jobs() -> Vec<(String, String)> {
    let path = repo_file("gpu/testdata/wave1_variant_jobs.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()));
    json::parse(&text)
        .expect("jobs parse")
        .as_array()
        .expect("jobs array")
        .iter()
        .map(|job| {
            let label = job.get("label").and_then(json::Value::as_str).unwrap_or("job").to_string();
            let request = job.get("request").expect("job has a request");
            let mut text = String::new();
            json::to_string_into(request, &mut text);
            (label, text)
        })
        .collect()
}

/// One dumped request re-pointed at a seed with the merged skillData attached,
/// so the GPU and CPU read the same roster the app sent.
fn preset_request(job: &str, seed: i64, skill_data: &json::Value) -> String {
    let mut value = json::parse(job).expect("job parses");
    let json::Value::Object(map) = &mut value else { panic!("job is not an object") };
    map.insert("seed".into(), json::Value::Number(seed as f64));
    map.insert("skillData".into(), skill_data.clone());
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    out
}

/// The dumped Phase A variant requests, each label over each seed, with a tag.
fn variant_requests(skill_data: &json::Value) -> Vec<(String, i64, String)> {
    let mut requests: Vec<(String, i64, String)> = Vec::new();
    for (label, job) in variant_jobs() {
        for &seed in &SEEDS {
            let request = preset_request(job.as_str(), seed, skill_data);
            requests.push((request, seed, format!("{label} seed {seed}")));
        }
    }
    requests
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// Compares one GPU race against a fresh CPU run: same tick count, every
/// replay value through to_bits(), and every finish time.
fn compare_race(
    tag: &str,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
    problems: &mut Vec<String>,
) {
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{tag}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in
        cpu.frames.iter().zip(race.frames.iter()).enumerate()
    {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
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

/// Every dumped variant job must reach the GPU kernel after the wave-1 ports.
/// A variant set that activated no skill would make the identity gate
/// vacuous, so require at least one CPU-side activation across the set.
#[test]
fn ported_variant_jobs_compile_without_fallback() {
    let skill_data = merged_skill_data();
    register_geometry(10504);
    let requests = variant_requests(&skill_data);
    let out = race_exec::run(&requests.iter().map(|(r, s, _)| (r.clone(), *s)).collect::<Vec<_>>())
        .expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    for (index, (request, seed, tag)) in requests.iter().enumerate() {
        if let Some(reason) = &out.races[index].fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => {
                if !cpu.detail_events.is_empty() {
                    fired += 1;
                }
            }
            Err(reason) => problems.push(format!("{tag}: cpu race failed: {reason}")),
        }
    }
    if fired == 0 {
        problems.push("no variant race activated a skill, the identity gate is vacuous".into());
    }
    if !problems.is_empty() {
        panic!("{} variant compile problems:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Frame-for-frame parity against the CPU engine over the same variant set.
/// This is the active gate for the wave-1 constructs: the out-70 band cell in
/// 910611's precondition, 920011's shadowed alternative with the observe
/// flag, 910601's popularity-folded alternatives, and the static ground and
/// dirtgrade folds, all through to_bits() on every frame value.
#[test]
fn ported_variant_jobs_replay_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry(10504);
    let requests = variant_requests(&skill_data);
    let out = race_exec::run(&requests.iter().map(|(r, s, _)| (r.clone(), *s)).collect::<Vec<_>>())
        .expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed, tag)) in requests.iter().enumerate() {
        compare_race(&tag, request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Four synthetic defs built from 200361's real alternative (its effect,
/// cooldown and duration keys are engine-valid) ride on horse 0 of the v77
/// roster next to the six real skills the skills gate proves fire:
/// - 660101: an any-skill condition, exercising the observe flag end to end.
/// - 660102: that condition under a track-matching precondition plus a second
///   identical alternative, the shadowed-pair shape of 920011.
/// - 660103: the same condition under a track-failing first alternative (a
///   dead wisdom carrier) and a live second, the shared-roll shape.
/// - 660701: an out-70 or out-20 band condition, so the 70 arm is evaluated
///   at every gate visit and the skill fires through the 20 arm.
/// CPU activations of 660102 must all carry detail index 0 (a shadowed
/// alternative never fires) and of 660103 detail index 1 (a dead carrier
/// never fires). Frames stay bit-exact across the sweep.
#[test]
fn dynamic_constructs_fire_bit_exact() {
    let base = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string();
    register_geometry(10504);

    let root = json::parse(&base).expect("v77 parses");
    let template = root
        .get("skillData")
        .and_then(|sd| sd.get("200361"))
        .expect("v77 skillData has 200361");
    let template_alt = template
        .get("alternatives")
        .and_then(json::Value::as_array)
        .expect("200361 alternatives")[0]
        .clone();
    let make_alt = |pre: &str, cond: &str| -> Value {
        let mut alt = template_alt.clone();
        let json::Value::Object(map) = &mut alt else { unreachable!() };
        map.insert("precondition".into(), json::Value::String(pre.into()));
        map.insert("condition".into(), json::Value::String(cond.into()));
        alt
    };
    let make_def = |wisdom: i64, alts: Vec<Value>| -> Value {
        let mut def = template.clone();
        let json::Value::Object(map) = &mut def else { unreachable!() };
        map.insert("wisdomCheck".into(), json::Value::Number(wisdom as f64));
        map.insert("alternatives".into(), json::arr(alts));
        def
    };
    let anyskill = "is_activate_any_skill==1";
    let mut value = json::parse(&base).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        let json::Value::Object(sd) = map.get_mut("skillData").expect("skillData") else {
            unreachable!()
        };
        sd.insert("660101".into(), make_def(0, vec![make_alt("", anyskill)]));
        sd.insert(
            "660102".into(),
            make_def(0, vec![make_alt("track_id==10005", anyskill), make_alt("", anyskill)]),
        );
        sd.insert(
            "660103".into(),
            make_def(1, vec![make_alt("track_id==10006", anyskill), make_alt("", anyskill)]),
        );
        sd.insert(
            "660701".into(),
            make_def(
                0,
                vec![make_alt("", "order_rate_out70_continue==1@order_rate_out20_continue==1")],
            ),
        );
        let json::Value::Array(horses) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        let json::Value::Object(h0) = &mut horses[0] else { unreachable!() };
        h0.insert(
            "skills".into(),
            json::arr(
                [200361i64, 200341, 200333, 10071, 100401, 110591, 660101, 660102, 660103]
                    .iter()
                    .map(|&id| {
                        json::obj(vec![
                            ("skillId", json::Value::Number(id as f64)),
                            ("level", json::Value::Number(1.)),
                        ])
                    })
                    .collect(),
            ),
        );
        // The band cells latch from the rank held at t=5 s. Horse 0 leads
        // there, so the out-band skill rides the tail horse instead, whose
        // t=5 rank is in every out band.
        let json::Value::Object(h8) = &mut horses[8] else { unreachable!() };
        h8.insert(
            "skills".into(),
            json::arr(vec![json::obj(vec![
                ("skillId", json::Value::Number(660701.)),
                ("level", json::Value::Number(1.)),
            ])]),
        );
    }

    let mut requests: Vec<(String, i64)> = Vec::new();
    for &seed in &[1i64, 2, 3, 4] {
        let mut request = value.clone();
        let json::Value::Object(map) = &mut request else { unreachable!() };
        map.insert("seed".into(), json::Value::Number(seed as f64));
        let mut text = String::new();
        json::to_string_into(&request, &mut text);
        requests.push((text, seed));
    }
    let out = race_exec::run(&requests).expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut fired: Vec<i64> = Vec::new();
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = format!("v77 horse0 seed {seed}");
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
        for (_, sid, detail, _) in &cpu.detail_events {
            if !fired.contains(sid) {
                fired.push(*sid);
            }
            if *sid == 660102 && *detail != 0 {
                problems.push(format!(
                    "{tag}: 660102 fired on detail {detail}, the shadowed pair is wrong"
                ));
            }
            if *sid == 660103 && *detail != 1 {
                problems.push(format!(
                    "{tag}: 660103 fired on detail {detail}, a dead carrier fired"
                ));
            }
        }
        compare_race(&tag, request, *seed, race, &mut problems);
    }
    for sid in [660_101i64, 660_102, 660_103, 660_701] {
        if !fired.contains(&sid) {
            problems.push(format!("synthetic {sid} never fired on horse 0, port unexercised"));
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} mechanism problems, cpu fired {fired:?}:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}
