//! The startup-pass gates: the app's case (b) shape (the mob-preset jobs with
//! `experimentalRetailStartup: true`) replays bit-exact against the CPU engine
//! on both courses and seeds 1 to 3. The stage gate compares the kernel's
//! start delays with the in-process engine's `retail_initialization::initialize`
//! and pins the flag's invariance through the whole initialization chain.

#![cfg(feature = "race")]

use globalsim_gpu::initialization as ini;
use globalsim_gpu::retail_initialization;
use globalsim_gpu::{dep_cpu, json, race_exec};

const COURSES: [i32; 2] = [10504, 10604];
const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn merged_skill_data() -> json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

/// One testdata job re-pointed at a course seed with the merged skillData
/// attached, optionally carrying the startup flag.
fn preset_request(job: &str, seed: i64, skill_data: &json::Value, startup: bool) -> String {
    let mut value = json::parse(job).expect("job parses");
    let json::Value::Object(map) = &mut value else { panic!("job is not a JSON object") };
    map.insert("seed".into(), json::Value::Number(seed as f64));
    map.insert("skillData".into(), skill_data.clone());
    if startup {
        map.insert("experimentalRetailStartup".into(), json::Value::Bool(true));
    }
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

/// One case per (course, job, seed): the flagged and plain request texts and
/// a tag. The batch order of every kernel run in this file follows it.
struct Case {
    flagged: String,
    plain: String,
    seed: i64,
    tag: String,
}

fn cases(skill_data: &json::Value) -> Vec<Case> {
    let mut out = Vec::new();
    for &course in &COURSES {
        for job in load_jobs(course) {
            for &seed in &SEEDS {
                out.push(Case {
                    flagged: preset_request(&job, seed, skill_data, true),
                    plain: preset_request(&job, seed, skill_data, false),
                    seed,
                    tag: format!("course {course} seed {seed}"),
                });
            }
        }
    }
    out
}

fn register_geometry() {
    for &course in &COURSES {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
    }
}

/// Frame-for-frame parity of the startup-flagged jobs against the CPU engine.
/// No race may fall back, tick counts and all six replay columns must match
/// through to_bits(), and finishTimeRaw must match bit for bit.
#[test]
fn startup_preset_jobs_replay_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry();
    let cases = cases(&skill_data);
    let out = race_exec::run(
        &cases.iter().map(|c| (c.flagged.clone(), c.seed)).collect::<Vec<_>>()
    )
    .expect("kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut fired = 0usize;
    for (index, case) in cases.iter().enumerate() {
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{}: fallback {reason}", case.tag));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(&case.flagged, case.seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{}: cpu race failed: {reason}", case.tag));
                continue;
            }
        };
        assert_eq!(cpu.runners, 9, "{}: preset roster", case.tag);
        if !cpu.detail_events.is_empty() {
            fired += 1;
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!(
                "{}: tick count cpu {} gpu {}", case.tag, cpu.frames.len(), race.frames.len()
            ));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cv.to_bits() != gv.to_bits() {
                        problems.push(format!(
                            "{} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                            case.tag,
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
                    "{} runner {runner} finishTimeRaw: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                    case.tag,
                    cpu.to_bits(),
                    gpu.to_bits()
                ));
            }
        }
    }
    if fired == 0 {
        problems.push("no flagged race activated a skill, the identity gate is vacuous".into());
    }
    if !problems.is_empty() {
        panic!("{} flagged problems:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Whether the engine's own replies distinguish the flag on this shape. The
/// kernel matched both the flagged and the plain engine races (the identity
/// gate and the plain app-preset gate together cover every pair), so the
/// engine pair here decides what the flag means for these jobs. The test
/// pins the correspondence: every case where the engine replies differ must
/// be a case the kernel flagged race still equals, and where they agree the
/// kernel's flagged and plain runs must agree too.
#[test]
fn startup_flag_matches_the_engine_case_by_case() {
    let skill_data = merged_skill_data();
    register_geometry();
    let cases = cases(&skill_data);
    let on = race_exec::run(
        &cases.iter().map(|c| (c.flagged.clone(), c.seed)).collect::<Vec<_>>()
    )
    .expect("flagged kernel run");
    let off = race_exec::run(
        &cases.iter().map(|c| (c.plain.clone(), c.seed)).collect::<Vec<_>>()
    )
    .expect("plain kernel run");

    let mut problems: Vec<String> = Vec::new();
    let mut engine_differs = 0usize;
    for (index, case) in cases.iter().enumerate() {
        let (a, b) = (&on.races[index], &off.races[index]);
        if a.fallback.is_some() || b.fallback.is_some() {
            problems.push(format!(
                "{}: fallback {:?} {:?}", case.tag, a.fallback, b.fallback
            ));
            continue;
        }
        let on_reply = dep_cpu::run_cpu_binary(&case.flagged, case.seed)
            .expect("flagged cpu")
            .reply;
        let off_reply = dep_cpu::run_cpu_binary(&case.plain, case.seed)
            .expect("plain cpu")
            .reply;
        if on_reply == off_reply {
            if a.frames.len() != b.frames.len()
                || a.frames.iter().zip(&b.frames).any(|(fa, fb)| {
                    fa.iter().zip(fb).any(|(ra, rb)| {
                        ra.iter().zip(rb).any(|(x, y)| x.to_bits() != y.to_bits())
                    })
                })
            {
                problems.push(format!(
                    "{}: engine replies identical but kernel flagged run differs",
                    case.tag
                ));
            }
        } else {
            engine_differs += 1;
        }
    }
    if !problems.is_empty() {
        panic!("{} flag correspondence problems:\n{}", problems.len(), problems.join("\n"));
    }
    println!(
        "engine replies differ in {engine_differs} of {} cases",
        cases.len()
    );
}

// The stage gate mirrors the pure race_solver_builder.rs ranked_base_stat
// (source-pinned): adjust_overcap, then the team-rank bonus, then the
// vendored initial_base_wisdom.
fn ranked_base_stat(raw: f64, mood: i32, rank: i32) -> f32 {
    let overcap = if raw > 1200.0 { 1200.0 + (raw - 1200.0) / 2.0 } else { raw };
    let mut bonus = 0.0f32;
    if rank > 0 {
        bonus = [(6, 50.0f32), (11, 30.0), (16, 20.0), (21, 10.0)]
            .into_iter()
            .find(|(threshold, _)| rank < *threshold)
            .map_or(0.0, |(_, b)| b);
    }
    retail_initialization::initial_base_wisdom((overcap as f32 + bonus) as f64, mood)
}

// horse_types.rs base_running_style: Oonige keeps the Nige style byte.
fn base_style(strategy: &str) -> f64 {
    match strategy.to_ascii_uppercase().as_str() {
        "NIGE" => 1.0,
        "SENKOU" => 2.0,
        "SASI" | "SASHI" => 3.0,
        "OIKOMI" => 4.0,
        "OONIGE" => 1.0,
        other => panic!("unknown strategy {other}"),
    }
}

/// The request roster in the api.rs initialization form: numeric skill ids
/// with levels, the ranked base wisdom and the base running style.
fn api_form_horses(horses: &[json::Value]) -> Vec<json::Value> {
    horses
        .iter()
        .map(|horse| {
            let skills: Vec<json::Value> = horse
                .get_array("skills")
                .unwrap_or(&Vec::new())
                .iter()
                .map(|entry| {
                    let (id, level) = match entry {
                        json::Value::String(text) => (text.parse::<f64>().unwrap_or(0.), 1.0),
                        json::Value::Number(n) => (*n, 1.0),
                        _ => {
                            let id = match entry
                                .get("skillId")
                                .or_else(|| entry.get("skill_id"))
                            {
                                Some(json::Value::String(s)) => s.parse::<f64>().unwrap_or(0.),
                                Some(json::Value::Number(n)) => *n,
                                _ => 0.0,
                            };
                            let level = entry.get("level").and_then(json::Value::as_f64).unwrap_or(1.0);
                            (id, level)
                        }
                    };
                    json::obj(vec![
                        ("skill_id", json::Value::Number(id)),
                        ("level", json::Value::Number(level)),
                    ])
                })
                .collect();
            json::obj(vec![
                ("skills", json::Value::Array(skills)),
                (
                    "base_wisdom",
                    json::Value::Number(ranked_base_stat(
                        horse.get_f64("wisdom").unwrap_or(0.),
                        horse.get_i64("mood").unwrap_or(0) as i32,
                        horse.get_i64("teamRank").unwrap_or(0) as i32,
                    ) as f64),
                ),
                (
                    "running_style",
                    json::Value::Number(base_style(horse.get_str("strategy").unwrap_or("Nige"))),
                ),
            ])
        })
        .collect()
}

/// The kernel's exported start delays must equal the values the in-process
/// engine's own `retail_initialization::initialize` produced for the same
/// roster, flagged or not, and the reply must carry the same widened f32.
/// The stage draw counts from that initialize are pinned: the delay stage
/// adds exactly one draw per runner, and the flag never touches initialize
/// (api.rs:76-91), so the flagged and plain chains agree word for word.
#[test]
fn startup_stage_gate_vs_engine_initialize() {
    let skill_data = merged_skill_data();
    register_geometry();
    let cases = cases(&skill_data);
    let on = race_exec::run(
        &cases.iter().map(|c| (c.flagged.clone(), c.seed)).collect::<Vec<_>>()
    )
    .expect("flagged kernel run");
    let off = race_exec::run(
        &cases.iter().map(|c| (c.plain.clone(), c.seed)).collect::<Vec<_>>()
    )
    .expect("plain kernel run");

    let course_root = ini::engine_json("data/course_data.json").expect("course_data.json");
    let parameters = ini::engine_parameters_value()
        .expect("global_parameters.json")
        .get("parameters")
        .cloned()
        .expect("parameters group");

    let mut problems: Vec<String> = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let (a, b) = (&on.races[index], &off.races[index]);
        if a.fallback.is_some() || b.fallback.is_some() {
            problems.push(format!(
                "{}: fallback {:?} {:?}", case.tag, a.fallback, b.fallback
            ));
            continue;
        }
        let course_id = dep_cpu::course_id(&case.flagged).expect("course id");
        let course = course_root.get(&course_id.to_string()).expect("course data");
        let horses = api_form_horses(
            json::parse(&case.flagged)
                .expect("request")
                .get_array("horses")
                .expect("horses"),
        );
        let engine = retail_initialization::initialize(
            i32::try_from(case.seed).expect("seed fits i32"),
            &horses,
            course,
            &skill_data,
            &parameters,
            false,
        )
        .expect("engine initialize");

        let stage = |name: &str| -> usize { *engine.stages.get(name).expect("stage") };
        if stage("after_delays") - stage("after_wisdom") != horses.len() {
            problems.push(format!(
                "{}: delay stage drew {} for {} runners",
                case.tag,
                stage("after_delays") - stage("after_wisdom"),
                horses.len()
            ));
        }
        for name in ["after_triggers", "after_ai", "after_wisdom", "after_delays"] {
            if stage(name) == 0 && name != "after_triggers" {
                problems.push(format!("{}: stage {name} drew nothing", case.tag));
            }
        }

        let reply_text = dep_cpu::run_cpu_binary(&case.flagged, case.seed)
            .expect("cpu binary")
            .reply;
        let reply = json::parse(&reply_text).expect("reply parses");
        let results = reply.get_array("results").expect("results");

        for runner in 0..horses.len() {
            let want = engine.horses[runner].start_delay;
            let got_on = a.delay_all[runner];
            let got_off = b.delay_all[runner];
            if f64::from(want).to_bits() != got_on.to_bits() {
                problems.push(format!(
                    "{} runner {runner} start delay: engine initialize {want:?} ({}), kernel flagged {got_on:?} ({})",
                    case.tag,
                    f64::from(want).to_bits(),
                    got_on.to_bits()
                ));
            }
            if got_on.to_bits() != got_off.to_bits() {
                problems.push(format!(
                    "{} runner {runner} start delay moved with the startup flag",
                    case.tag
                ));
            }
            let reply_delay = results[runner].get_f64("startDelayTime").expect("startDelayTime");
            if reply_delay.to_bits() != got_on.to_bits() {
                problems.push(format!(
                    "{} runner {runner} startDelayTime: reply {reply_delay:?} ({}), kernel flagged {got_on:?} ({})",
                    case.tag,
                    reply_delay.to_bits(),
                    got_on.to_bits()
                ));
            }
        }
        println!(
            "{} engine init draws: {}",
            case.tag,
            ["after_triggers", "after_ai", "after_wisdom", "after_delays"]
                .into_iter()
                .map(|name| format!("{name} {}", stage(name)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if !problems.is_empty() {
        panic!("{} stage gate problems:\n{}", problems.len(), problems.join("\n"));
    }
}
