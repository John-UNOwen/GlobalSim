//! Wave-10 full-reply gates: the batch facade's `run_batch_full` lane and
//! `full_reply::build` answer each race with the engine's whole non-compact
//! reply text, byte for byte, on seeds 1..3. Every scenario must produce a
//! GPU reply with no fallback, and the text must equal the engine's own
//! `write_json` output for the same request and seed, frames included.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{full_reply, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];

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

/// The engine's own reply text for one request and seed, through the same
/// public API the app's non-compact lane drives.
fn engine_reply(request: &str, seed: i64) -> String {
    let mut prepared = globalsim::PreparedRace::new(request).expect("engine accepts the request");
    let simulation = prepared.run(seed).expect("engine runs the race");
    let mut reply = String::new();
    simulation.write_json(&mut reply);
    reply
}

/// Every difference between two reply texts, as up to 12 byte windows.
fn first_difference(gpu: &str, engine: &str) -> String {
    let gb = gpu.as_bytes();
    let eb = engine.as_bytes();
    let mut out = format!(
        "gpu {} bytes, engine {} bytes\n",
        gb.len(),
        eb.len(),
    );
    let mut at = 0usize;
    let mut shown = 0usize;
    while at < gb.len().max(eb.len()) && shown < 12 {
        match gb.get(at).zip(eb.get(at)) {
            Some((a, b)) if a == b => {
                at += 1;
            }
            _ => {
                let start = at.saturating_sub(56);
                let gend = gb.len().min(at + 168);
                let eend = eb.len().min(at + 168);
                out.push_str(&format!(
                    "--- at byte {at}\ngpu: {:?}\nengine: {:?}\n",
                    &gpu[start..gend],
                    &engine[start..eend],
                ));
                shown += 1;
                // Skip past the mismatched region so the next window is new.
                let mut run = 0usize;
                while run < 512 && at < gb.len().max(eb.len()) {
                    match gb.get(at).zip(eb.get(at)) {
                        Some((a, b)) if a == b => {
                            run += 1;
                            at += 1;
                        }
                        _ => {
                            run = 0;
                            at += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

fn patched_requests(
    skill_data: &Value,
    extra: &[(&str, Value)],
    horses: &[(usize, Option<&[i64]>, Option<&str>, Option<i64>)],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        if !matches!(skill_data, Value::Null) {
            map.insert("skillData".into(), skill_data.clone());
        }
        for (key, item) in extra {
            map.insert((*key).into(), item.clone());
        }
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids, strategy, popularity) in horses {
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
            if let Some(popularity) = popularity {
                horse.insert(
                    "popularity".into(),
                    json::Value::Number((*popularity) as f64),
                );
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

/// One scenario, whole: every seed runs the GPU full lane and the engine,
/// and each pair of reply texts must be identical.
fn run_full_gate(label: &str, requests: &[(String, i64)]) {
    let run = race_exec::run_full(requests).expect("kernel run");
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = format!("{label} seed {seed}");
        let race = &run.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let gpu = match full_reply::build(request, *seed, race) {
            Ok(text) => text,
            Err(reason) => {
                problems.push(format!("{tag}: full reply assembly failed: {reason}"));
                continue;
            }
        };
        let engine = engine_reply(request, *seed);
        if gpu != engine {
            problems.push(format!("{tag}:\n{}", first_difference(&gpu, &engine)));
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} {label} problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// The plain retail lane, the app's default full request.
#[test]
fn retail_lane_full_reply_bit_exact() {
    register_geometry(10504);
    let requests = patched_requests(&Value::Null, &[], &[], &SEEDS);
    run_full_gate("retail lane", &requests);
}

/// Every runner stripped of skills, the empty-event-shape control.
#[test]
fn zero_skill_full_reply_bit_exact() {
    register_geometry(10504);
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for entry in entries.iter_mut() {
            let json::Value::Object(horse) = entry else { unreachable!() };
            horse.insert("skills".into(), json::arr(Vec::new()));
        }
    }
    let mut requests = Vec::new();
    for seed in SEEDS {
        let mut request = value.clone();
        let json::Value::Object(map) = &mut request else { unreachable!() };
        map.insert("seed".into(), json::Value::Number(seed as f64));
        let mut text = String::new();
        json::to_string_into(&request, &mut text);
        requests.push((text, seed));
    }
    run_full_gate("zero skill", &requests);
}

/// The merged app table with the wave-9 fan battery and the duration_usage 4
/// pair: skillEvents pending durations, skillDetailEvents, skillTargetEvents
/// with null and list selections, and additionalTargetEvents all ride the
/// same bytes.
#[test]
fn skill_battery_full_reply_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    let defs: [(i64, String); 6] = [
        (
            669201,
            json_battery(
                "phase>=2",
                5000000,
                20000,
                r#"{"type":27,"target":9,"targetValue":3,"modifier":-1000,"additionalActivate":1}"#,
            ),
        ),
        (
            669202,
            json_battery(
                "phase>=1",
                5000000,
                30000,
                r#"{"type":31,"target":10,"targetValue":4,"modifier":800,"additionalActivate":2}"#,
            ),
        ),
        (
            669203,
            json_battery(
                "phase>=2",
                5000000,
                25000,
                r#"{"type":28,"target":4,"targetValue":2,"modifier":-500,"additionalActivate":3}"#,
            ),
        ),
        (
            669205,
            json_battery(
                "phase>=2",
                5000000,
                20000,
                r#"{"type":21,"target":1,"modifier":500,"additionalActivate":2}"#,
            ),
        ),
        (
            669102,
            json_battery_du(
                "phase>=1",
                5000000,
                30000,
                4,
                concat!(
                    r#"{"type":31,"target":1,"modifier":700},"#,
                    r#"{"type":27,"target":1,"modifier":500,"additionalActivate":1}"#
                ),
            ),
        ),
        (
            669141,
            json_battery("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#),
        ),
    ];
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, def) in &defs {
            sd.insert(sid.to_string(), json::parse(def).expect("battery def parses"));
        }
    }
    let horses: Vec<(usize, Option<&[i64]>, Option<&str>, Option<i64>)> = vec![
        (0, Some(&[100531i64, 669141][..]), None, None),
        (2, Some(&[900531i64, 669141][..]), None, None),
        (1, Some(&[669102i64, 669141][..]), None, None),
        (3, Some(&[669201i64][..]), None, None),
        (4, Some(&[669202i64][..]), None, None),
        (5, Some(&[669203i64][..]), None, None),
        (6, Some(&[669205i64][..]), None, None),
        (7, Some(&[669201i64][..]), None, None),
    ];
    let requests = patched_requests(&skill_data, &[], &horses, &SEEDS);
    run_full_gate("skill battery", &requests);
}

/// The startup pass on top of the battery: startup-repeat ordinals, the
/// pre-tick competition batch and the repeat sort key all reach the bytes.
#[test]
fn startup_pass_full_reply_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        sd.insert(
            "669141".into(),
            json::parse(
                &json_battery("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#),
            )
            .expect("battery def parses"),
        );
        sd.insert(
            "669201".into(),
            json::parse(
                &json_battery(
                    "phase>=2",
                    5000000,
                    20000,
                    r#"{"type":27,"target":9,"targetValue":3,"modifier":-1000,"additionalActivate":1}"#,
                ),
            )
            .expect("battery def parses"),
        );
    }
    let horses: Vec<(usize, Option<&[i64]>, Option<&str>, Option<i64>)> = vec![
        (0, Some(&[669141i64, 669201][..]), None, None),
        (1, Some(&[669141][..]), None, None),
        (3, Some(&[669201][..]), None, None),
        (5, Some(&[669141][..]), None, None),
        (7, Some(&[669201][..]), None, None),
    ];
    let requests = patched_requests(
        &skill_data,
        &[("experimentalRetailStartup", Value::Bool(true))],
        &horses,
        &SEEDS,
    );
    run_full_gate("startup pass", &requests);
}

/// Paired streams and a data signature: the two wrapper-echoed keys must
/// appear with the request's values.
#[test]
fn paired_and_signature_full_reply_bit_exact() {
    register_geometry(10504);
    let signature = json::obj(vec![("lane", json::num(7.0)), ("name", Value::String("w10".into()))]);
    let requests = patched_requests(
        &Value::Null,
        &[
            ("pairedRandomStreams", Value::Bool(true)),
            ("dataSignature", signature),
        ],
        &[],
        &SEEDS,
    );
    run_full_gate("paired and signature", &requests);
}

/// World geometry on the retail lane, the app's geometry-enabled requests.
#[test]
fn geometry_full_reply_bit_exact() {
    register_geometry(10504);
    let requests = patched_requests(
        &Value::Null,
        &[("experimentalRetailGeometry", Value::Bool(true))],
        &[],
        &SEEDS,
    );
    run_full_gate("world geometry", &requests);
}

/// The binary-lane frame shape: every frame kept, no thinning keys.
#[test]
fn retain_all_full_reply_bit_exact() {
    register_geometry(10504);
    let requests =
        patched_requests(&Value::Null, &[("retainAllFrames", Value::Bool(true))], &[], &SEEDS);
    run_full_gate("retain all frames", &requests);
}

/// Nige and Oonige hosts on mixed popularity: the spot starts reorder under
/// the roster-rank key and the duel pairs push their own records, so the
/// competition bytes come out of the reorder, not the raw push order.
#[test]
fn competition_reorder_full_reply_bit_exact() {
    register_geometry(10504);
    let horses: Vec<(usize, Option<&[i64]>, Option<&str>, Option<i64>)> = vec![
        (0, Some(&[]), Some("OONIGE"), Some(3)),
        (1, Some(&[]), Some("NIGE"), Some(9)),
        (2, Some(&[]), Some("NIGE"), Some(1)),
        (3, Some(&[]), Some("SENKOU"), Some(5)),
        (4, Some(&[]), Some("OONIGE"), Some(7)),
        (5, Some(&[]), Some("OIKOMI"), Some(2)),
        (6, Some(&[]), Some("SASI"), Some(8)),
        (7, Some(&[]), Some("NIGE"), Some(4)),
    ];
    let requests = patched_requests(&Value::Null, &[], &horses, &SEEDS);
    run_full_gate("competition reorder", &requests);
}

/// The facade lane itself: `GpuRaces::run_batch_full` fills `reply` for
/// every race of a mixed batch, and a request outside the full lane keeps
/// a fallback with the other races answered.
#[test]
fn facade_full_batch_answers_every_race() {
    register_geometry(10504);
    let mut gpu = globalsim_gpu::facade::GpuRaces::new().expect("facade opens");
    let requests: Vec<String> = patched_requests(&Value::Null, &[], &[], &SEEDS)
        .into_iter()
        .map(|(request, _)| request)
        .collect();
    let seeds = [1i32, 2, 3];
    let outcomes = gpu.run_batch_full(&requests, &seeds).expect("facade batch");
    assert_eq!(outcomes.len(), 3);
    for (index, outcome) in outcomes.iter().enumerate() {
        let seed = seeds[index] as i64;
        let reply = outcome.reply.as_ref().unwrap_or_else(|| {
            panic!("race {index} fell back: {:?}", outcome.fallback)
        });
        let engine = engine_reply(&requests[index], seed);
        if reply != &engine {
            panic!("facade seed {seed}:\n{}", first_difference(reply, &engine));
        }
    }
    // A single-mode request is answered by the kernel (wave 13a) byte-equal
    // to the engine, and it joins the batch's single launch.
    let trace = patched_requests(
        &Value::Null,
        &[("singleMode", Value::Bool(true))],
        &[],
        &[1],
    );
    let mixed: Vec<String> = vec![requests[0].clone(), trace[0].0.clone(), requests[1].clone()];
    let outcomes = gpu.run_batch_full(&mixed, &[1, 1, 2]).expect("mixed batch");
    assert!(outcomes[1].fallback.is_none(), "singleMode answers on the GPU lane");
    let want = engine_reply(&trace[0].0, 1);
    assert_eq!(outcomes[1].reply.as_deref(), Some(want.as_str()), "singleMode answers byte-equal");
    assert!(outcomes[0].reply.is_some() && outcomes[2].reply.is_some());
}

fn json_battery(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

fn json_battery_du(
    cond: &str,
    cooldown: i64,
    base_duration: i64,
    duration_usage: i64,
    effects: &str,
) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"durationUsage\":{duration_usage},\"effects\":[{effects}]}}]}}"
    )
}
