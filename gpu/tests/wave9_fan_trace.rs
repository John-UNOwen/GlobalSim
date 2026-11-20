//! Fan-churn divergence trace: decode the kernel's additionalActivate
//! journal against the engine's own `additionalTargetEvents` reply stream for
//! the ignored `fan_churn_tracks_the_engine` scene. The helpers mirror
//! `wave9_gate.rs` so the scene is the same field, seed 1. Diagnostics for
//! the ignored scene, printed to target/fan_journal_seed1.txt.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEED: i64 = 1;

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

fn v77_request() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

type HorsePatch<'a> = (usize, Option<&'a [i64]>, Option<&'a str>, Option<i64>, Option<i64>);

fn patched_requests(
    skill_data: &Value,
    horses: &[HorsePatch],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("skillData".into(), skill_data.clone());
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids, strategy, pop, team) in horses {
            let json::Value::Object(horse) = &mut entries[*index] else { unreachable!() };
            if let Some(ids) = ids {
                horse.insert(
                    "skills".into(),
                    json::arr(
                        ids.iter()
                            .map(|&id| {
                                json::obj(vec![
                                    ("skillId", json::Value::Number(id as f64)),
                                    ("level", json::Value::Number(1.)),
                                ])
                            })
                            .collect(),
                    ),
                );
            }
            if let Some(style) = strategy {
                horse.insert("strategy".into(), json::Value::String((*style).into()));
            }
            if let Some(popularity) = pop {
                horse.insert(
                    "popularity".into(),
                    json::Value::Number((*popularity) as f64),
                );
            }
            if let Some(team_id) = team {
                horse.insert("teamId".into(), json::Value::Number((*team_id) as f64));
            }
        }
    }
    let mut requests = Vec::new();
    for &seed in seeds {
        let mut request = value.clone();
        let json::Value::Object(map) = &mut request else { unreachable!() };
        map.insert("seed".into(), json::Value::Number(seed as f64));
        let mut text = String::new();
        json::to_string_into(&request, &mut text);
        requests.push((text, seed));
    }
    requests
}

fn with_battery(skill_data: &mut Value, defs: &[(i64, &str)]) {
    let json::Value::Object(sd) = skill_data else { unreachable!() };
    for (sid, def) in defs {
        sd.insert(sid.to_string(), json::parse(def).expect("battery def parses"));
    }
}

fn def(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

#[derive(Debug, Clone, Copy)]
struct Event {
    op: u32,
    tick: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e1: f64,
    e2: f64,
}

fn scene_requests(seeds: &[i64]) -> Vec<(String, i64)> {
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669147,
                &def(
                    "phase>=1",
                    9000,
                    9000,
                    r#"{"type":31,"target":9,"targetValue":2,"modifier":-350,"additionalActivate":1}"#,
                )
                .as_str(),
            ),
            (
                669141,
                &def("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669147i64, 669141][..]), None, None, None),
        (1, Some(&[669147, 669141][..]), None, None, None),
        (2, Some(&[669147, 669141][..]), None, None, None),
        (3, Some(&[669141][..]), None, None, None),
        (4, Some(&[669147, 669141][..]), None, None, None),
        (5, Some(&[669147, 669141][..]), None, None, None),
        (6, Some(&[669141][..]), None, None, None),
        (7, Some(&[669147, 669141][..]), None, None, None),
    ];
    patched_requests(&skill_data, &horses, seeds)
}

/// Ten seeds of the fan churn scene, frames compared bit for bit against the
/// engine. The ignored gate only samples three, and the original divergence
/// showed on four of five seeds, so the fix gets the wider sweep here.
#[test]
fn fan_churn_ten_seeds_bit_exact() {
    register_geometry(10504);
    let seeds: Vec<i64> = (1..=10).collect();
    let requests = scene_requests(&seeds);
    let run = race_exec::run(&requests).expect("gpu run");
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in requests.iter().enumerate() {
        let race = &run.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("seed {seed}: cpu race failed: {reason}"));
                continue;
            }
        };
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!(
                "seed {seed}: tick count cpu {} gpu {}",
                cpu.frames.len(),
                race.frames.len()
            ));
            continue;
        }
        'frames: for (tick, (cpu_frame, race_frame)) in
            cpu.frames.iter().zip(race.frames.iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in
                cpu_frame.iter().zip(race_frame.iter()).enumerate()
            {
                for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cv.to_bits() != gv.to_bits() {
                        problems.push(format!(
                            "seed {seed} tick {tick} runner {runner} column {column}: cpu {cv:?} gpu {gv:?}"
                        ));
                        break 'frames;
                    }
                }
            }
        }
    }
    if !problems.is_empty() {
        panic!("{} problems:\n{}", problems.len(), problems.join("\n"));
    }
}

fn decode(journal: &[u32]) -> Vec<Event> {
    journal
        .chunks_exact(10)
        .map(|r| Event {
            op: r[0],
            tick: r[1],
            a: r[2],
            b: r[3],
            c: r[4],
            d: r[5],
            e1: f64::from_bits(((r[7] as u64) << 32) | r[6] as u64),
            e2: f64::from_bits(((r[9] as u64) << 32) | r[8] as u64),
        })
        .collect()
}

#[test]
fn fan_journal_seed1_against_the_engine_stream() {
    register_geometry(10504);
    let requests = scene_requests(&[SEED]);
    let run = race_exec::run_with_skill_scratch(&requests).expect("gpu run");
    let race = &run.races[0];
    assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
    let journal = decode(&race.debug_sk.as_ref().expect("journal").journal);
    let cpu = dep_cpu::run_cpu_race(&requests[0].0, SEED).expect("cpu race");
    let mut engine: Vec<(usize, f64, Vec<usize>)> = cpu
        .additional_events
        .iter()
        .filter(|(_, id, _, _)| *id == 669147)
        .map(|(h, _, t, tg)| (*h, *t, tg.iter().flatten().copied().collect()))
        .collect();
    engine.sort_by(|x, y| x.1.total_cmp(&y.1));

    let name = |op: u32| match op {
        1 => "FIRE",
        2 => "FAN ",
        3 => "DIE ",
        4 => "RET ",
        5 => "FLIP",
        _ => "?",
    };
    let mut out = String::new();
    out.push_str("== kernel journal ==\n");
    for e in &journal {
        out.push_str(&format!(
            "{} t={:>5} a={:>3} b={:>3} c={:>3} d={:>3} e1={:.6} e2={:.6}\n",
            name(e.op), e.tick, e.a, e.b, e.c, e.d, e.e1, e.e2
        ));
    }
    out.push_str(&format!(
        "== engine additionalTargetEvents ({} fans) ==\n",
        engine.len()
    ));
    for (h, t, tg) in &engine {
        out.push_str(&format!("FAN? host={h} t={t:.6} recv={tg:?}\n"));
    }
    out.push_str(&format!(
        "counts: kernel FAN {} engine fans {} kernel FIRE {} engine fires {}\n",
        journal.iter().filter(|e| e.op == 2).count(),
        engine.len(),
        journal.iter().filter(|e| e.op == 1).count(),
        cpu.detail_events
            .iter()
            .filter(|(_, id, _, _)| *id == 669147)
            .count(),
    ));

    // Both sides emit one line per delivered receiver in the same order, so
    // compare the flat (host, receiver) streams position by position. The
    // pool timer of a copy now differs between receivers of one application
    // when their roster indices straddle the host, so grouping by timer
    // would split a single application.
    let kernel_pairs: Vec<(u32, u32)> = journal
        .iter()
        .filter(|e| e.op == 2)
        .map(|e| (e.b, e.c))
        .collect();
    let engine_pairs: Vec<(usize, usize)> = engine
        .iter()
        .flat_map(|(h, _, tg)| tg.iter().map(move |r| (*h, *r)))
        .collect();
    let mut first_bad = String::new();
    let n = kernel_pairs.len().max(engine_pairs.len());
    for i in 0..n {
        if kernel_pairs.get(i).copied()
            != engine_pairs.get(i).map(|(h, r)| (*h as u32, *r as u32))
        {
            first_bad = format!(
                "fan #{i} mismatch: engine {:?} kernel {:?}",
                engine_pairs.get(i),
                kernel_pairs.get(i),
            );
            break;
        }
    }

    std::fs::write(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/fan_journal_seed1.txt"),
        out,
    )
    .expect("write trace file");
    println!("trace written to gpu/target/fan_journal_seed1.txt");
    if !first_bad.is_empty() {
        panic!("{first_bad}");
    }
}
