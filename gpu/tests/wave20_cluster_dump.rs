//! Wave 20 follow-up cluster trace. For one scene it injects one skill on one
//! runner, runs both lanes, and prints the detail-event streams plus the first
//! differing frame word. Ignored diagnostics for the 11 sweep clusters.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, race_exec};

struct Scene {
    file: &'static str,
    course: i32,
    seed: i64,
    runner: usize,
    ids: &'static [&'static str],
}

const SCENES: [Scene; 5] = [
    Scene {
        file: "examples/cm-10808-race.jsonl",
        course: 10808,
        seed: 1817412797,
        runner: 4,
        ids: &["105102111", "109302211", "111902211"],
    },
    Scene {
        file: "examples/cm-10506-race.jsonl",
        course: 10506,
        seed: 382661117,
        runner: 5,
        ids: &["105102221", "108302111", "110502111"],
    },
    Scene {
        file: "examples/cm-10506-race.jsonl",
        course: 10506,
        seed: 382661117,
        runner: 0,
        ids: &["120111", "920111"],
    },
    Scene {
        file: "examples/cm-10903-race.jsonl",
        course: 10903,
        seed: 1969904124,
        runner: 0,
        ids: &["111091", "911091", "92111091"],
    },
    Scene {
        file: "examples/cm-10808-race.jsonl",
        course: 10808,
        seed: 1817412797,
        runner: 4,
        ids: &["10661"],
    },
];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn table(id: &str) -> Value {
    let text = std::fs::read_to_string(repo_file("data/skill_data.json")).expect("table");
    let table: Value = json::parse(&text).expect("table parses");
    table.get(id).expect("id in table").clone()
}

fn request_with(root: &Value, id: &str, runner: usize) -> String {
    let mut v = root.clone();
    let Value::Object(m) = &mut v else { unreachable!() };
    {
        let Value::Object(sd) = m
            .entry("skillData".to_string())
            .or_insert_with(|| Value::Object(Default::default()))
        else {
            unreachable!()
        };
        sd.entry(id.into()).or_insert_with(|| table(id));
    }
    let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
    let Value::Object(h) = &mut hs[runner] else { unreachable!() };
    let mut one = std::collections::BTreeMap::new();
    one.insert("skillId".to_string(), Value::String(id.into()));
    one.insert("level".to_string(), Value::Number(1.0));
    let Value::Array(s) = h
        .entry("skills".to_string())
        .or_insert_with(|| Value::Array(Vec::new()))
    else {
        unreachable!()
    };
    let present = s.iter().any(|x| x.get("skillId").and_then(Value::as_str) == Some(id));
    if !present {
        s.push(Value::Object(one));
    }
    let mut text = String::new();
    json::to_string_into(&v, &mut text);
    text
}

fn events(reply: &str, id: &str) -> Vec<String> {
    let parsed = json::parse(reply).expect("reply parses");
    let Some(Value::Array(rows)) = parsed.get("skillDetailEvents") else {
        unreachable!()
    };
    rows.iter()
        .filter(|r| r.get("skillId").and_then(Value::as_i64).map(|v| v.to_string()) == Some(id.into()))
        .map(|r| {
            format!(
                "h {} d {} t {}",
                r.get("horseIndex").and_then(Value::as_i64).unwrap_or(-1),
                r.get("detailIndex").and_then(Value::as_i64).unwrap_or(-1),
                r.get("t").and_then(Value::as_f64).unwrap_or(-1.0)
            )
        })
        .collect()
}

#[test]
#[ignore]
fn trace_clusters() {
    for scene in SCENES {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{}.json", scene.course)))
                .expect("geometry");
        globalsim::register_course_geometry(scene.course, geometry);
        let text = std::fs::read_to_string(repo_file(scene.file)).expect("field");
        let root = json::parse(text.lines().next().expect("first line")).expect("field parses");
        for id in scene.ids {
            let request = request_with(&root, id, scene.runner);
            let run = race_exec::run_full(&[(request.clone(), scene.seed)]).expect("gpu run");
            let race = &run.races[0];
            println!("== {} runner {} seed {}{}", id, scene.runner, scene.seed,
                match &race.fallback { Some(r) => format!(" FALLBACK {r}"), None => String::new() });
            if race.fallback.is_some() {
                continue;
            }
            let engine = dep_cpu::run_cpu_binary(&request, scene.seed).expect("cpu binary");
            let cpu = dep_cpu::run_cpu_race(&request, scene.seed).expect("cpu race");
            let gpu_reply = globalsim_gpu::full_reply::build(
                &dep_cpu::with_retain_all_frames(&request).expect("retain"),
                scene.seed,
                race,
            )
            .expect("reply");
            let cpu_events = events(&engine.reply, id);
            let gpu_events = events(&gpu_reply, id);
            println!("   engine: {cpu_events:?}");
            println!("   kernel: {gpu_events:?}");
            let mut first = String::from("frames equal");
            'outer: for (tick, (cf, gf)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
                for (r, (cr, gr)) in cf.iter().zip(gf.iter()).enumerate() {
                    for (c, (x, y)) in cr.iter().zip(gr.iter()).enumerate() {
                        if x.to_bits() != y.to_bits() {
                            first = format!("tick {tick} runner {r} column {c}: cpu {x:?} gpu {y:?}");
                            break 'outer;
                        }
                    }
                }
            }
            println!("   first: {first}");
        }
    }
}

/// Compile one skill per scene shape against a synthetic 2200 m turf course and
/// print the refusal reason, so the target-11 and precondition shapes show up.
#[test]
#[ignore]
fn plan_probe() {
    let course_text = concat!(
        r##"{"distance":2200.0,"raceTrackId":10008,"surface":1,"distanceType":3,"turn":1,"##,
        r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2200.0}],"##,
        r##""corners":[{"start":300.0,"length":100.0}],"slopes":[]}"##
    );
    let course = json::parse(course_text).expect("course parses");
    for id in ["105102111", "109302211", "111902211", "105102221", "108302111", "110502111", "111091", "120111", "10661"] {
        let mut sd = String::new();
        json::to_string_into(&table(id), &mut sd);
        let root_text = format!(
            "{{\"skillData\":{{\"{id}\":{sd}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"{id}\",\"level\":1}}]}}]}}"
        );
        let root = json::parse(&root_text).unwrap_or_else(|e| panic!("{id}: root: {e}"));
        let plan = globalsim_gpu::skills_compile::plan(&root, &course);
        let nrec = plan.words[globalsim_gpu::skills_compile::SK_NREC] as usize;
        println!("== plan {id}: reason {:?} nrec {nrec}", plan.reason);
    }
    let syn = ["{\"type\":27,\"target\":11,\"modifier\":1500}", "{\"type\":22,\"target\":11,\"modifier\":1500}"];
    for (i, eff) in syn.iter().enumerate() {
        let (cond, wis) = if i == 0 { ("distance_type==2&all_corner_random==1", 1) } else { ("phase>=2", 0) };
        let alt = format!(
            "{{\"rarity\":1,\"wisdomCheck\":{wis},\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":5000,\"baseDuration\":40000,\"effects\":[{eff}]}}]}}"
        );
        let root_text = format!(
            "{{\"skillData\":{{\"55500{i}\":{alt}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"55500{i}\",\"level\":1}}]}}]}}"
        );
        let root = json::parse(&root_text).expect("syn root");
        let plan = globalsim_gpu::skills_compile::plan(&root, &course);
        let nrec = plan.words[globalsim_gpu::skills_compile::SK_NREC] as usize;
        println!("== plan syn{i}: reason {:?} nrec {nrec}", plan.reason);
    }
    probe_field_plan("examples/cm-10808-race.jsonl", &["105102111", "109302211", "111902211", "10661"]);
    probe_field_plan("examples/cm-10506-race.jsonl", &["105102221", "108302111", "110502111"]);
}

/// Compile the real field with one cluster skill merged onto runner 4 the way
/// the sweep does, and print per-record flags and effect targets.
fn probe_field_plan(file: &str, ids: &[&str]) {
    let line = std::fs::read_to_string(repo_file(file))
        .unwrap_or_else(|e| panic!("{file}: {e}"))
        .lines()
        .next()
        .unwrap()
        .to_string();
    for id in ids {
        let mut root = json::parse(&line).expect("field parses");
        {
            let Value::Object(map) = &mut root else { unreachable!() };
            let Value::Object(sdmap) =
                map.entry("skillData".to_string()).or_insert_with(|| json::Value::Object(Default::default()))
            else {
                unreachable!()
            };
            let def = table(id);
            sdmap.insert((*id).to_string(), def);
            let Value::Array(horses) = map.get_mut("horses").unwrap() else { unreachable!() };
            for (r, h) in horses.iter_mut().enumerate() {
                let Value::Object(hm) = h else { unreachable!() };
                let mut present = false;
                if let Some(Value::Array(list)) = hm.get("skills") {
                    present = list.iter().any(|e| {
                        e.get("skillId").and_then(json::Value::as_str).unwrap_or("") == *id
                            || e.as_str().unwrap_or("") == *id
                    });
                }
                if r == 4 && !present {
                    let Value::Array(list) =
                        hm.entry("skills".to_string()).or_insert_with(|| json::Value::Array(Vec::new()))
                    else {
                        unreachable!()
                    };
                    list.push(json::Value::String((*id).to_string()));
                }
            }
        }
        let course_root = json::parse(
            &std::fs::read_to_string(repo_file("data/course_data.json")).expect("course_data.json"),
        )
        .expect("course_data parses");
        let cid = root.get("courseId").and_then(json::Value::as_i64).unwrap_or(0).to_string();
        let course = course_root.get(&cid).cloned().expect("course entry");
        let extras = globalsim_gpu::skills_compile::ClassicExtras {
            raw_power: Vec::new(),
            strategy: Vec::new(),
            later_stamina: false,
        };
        let plan =
            globalsim_gpu::skills_compile::plan_retail_with_plan(&root, &course, &extras, true);
        let nrec = plan.words[globalsim_gpu::skills_compile::SK_NREC] as usize;
        println!("== field {id}: reason {:?} nrec {nrec}", plan.reason);
        let want = id.parse::<u32>().unwrap_or(0);
        for rec in 0..nrec {
            let at = globalsim_gpu::skills_compile::SK_RECORDS
                + globalsim_gpu::skills_compile::SK_RSTRIDE * rec;
            let runner = plan.words[at + 1];
            if plan.words[at] != want || runner != 4 {
                continue;
            }
            let flags = plan.words[at + 2];
            let n_eff = plan.words[at + 5];
            let eff_base = plan.words[at + 6];
            let mut tgts = Vec::new();
            for e in 0..n_eff {
                let eat = globalsim_gpu::skills_compile::SK_EFFECTS
                    + globalsim_gpu::skills_compile::SK_ESTRIDE * (eff_base + e) as usize;
                tgts.push((plan.words[eat], plan.words[eat + 1]));
            }
            println!(
                "   rec {rec} dead {} sampled {} eff {tgts:?}",
                flags & globalsim_gpu::skills_compile::SK_FLAG_DEAD != 0,
                flags & globalsim_gpu::skills_compile::SK_FLAG_SAMPLED != 0,
            );
        }
    }
}

/// One-condition-per-line bisect on a synthetic self-target skill: which term
/// kills the kernel fire while the engine still fires.
#[test]
#[ignore]
fn bisect_fire_terms() {
    let pre111091 = "track_id==10009&course_distance==1600@track_id==10008&course_distance==2000@track_id==10006&course_distance==2400";
    let pre120111 = "distance_rate>=50&order_rate<=70&order_rate>=30&overtake_target_time>=1@distance_rate>=50&order_rate<=70&order_rate>=30&is_overtake==1";
    let cases: [(&str, i32, i64, i64, &[(&str, &str)]); 2] = [
        (
            "examples/cm-10903-race.jsonl",
            10903,
            1969904124,
            0,
            &[
                ("phase>=2&corner!=0", ""),
                ("phase>=2&corner!=0", pre111091),
                ("ground_type==1&phase>=2&corner!=0", pre111091),
                ("ground_type==1&phase>=2&corner!=0&slope==2", pre111091),
            ],
        ),
        (
            "examples/cm-10506-race.jsonl",
            10506,
            382661117,
            0,
            &[
                ("remain_distance<=401&remain_distance>=399", ""),
                (
                    "remain_distance<=401&remain_distance>=399",
                    pre120111,
                ),
                (
                    "remain_distance<=401&remain_distance>=399&is_basis_distance==0",
                    pre120111,
                ),
                (
                    "remain_distance<=401&remain_distance>=399&is_basis_distance==0",
                    "distance_rate>=50&order_rate<=70&order_rate>=30&is_overtake==1",
                ),
            ],
        ),
    ];
    for (file, course, seed, runner, conds) in cases {
        let geometry =
            std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
                .expect("geometry");
        globalsim::register_course_geometry(course, geometry);
        let text = std::fs::read_to_string(repo_file(file)).expect("field");
        let root = json::parse(text.lines().next().expect("first line")).expect("field parses");
        for (cond, pre) in conds {
            let mut v = root.clone();
            {
                let Value::Object(m) = &mut v else { unreachable!() };
                let Value::Object(sd) = m
                    .entry("skillData".to_string())
                    .or_insert_with(|| Value::Object(Default::default()))
                else {
                    unreachable!()
                };
                let def = format!(
                    "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"{pre}\",\"condition\":\"{cond}\",\"cooldown\":5000,\"baseDuration\":2000,\"effects\":[{{\"type\":27,\"target\":1,\"modifier\":1500}}]}}]}}"
                );
                sd.insert("669222".into(), json::parse(&def).expect("def parses"));
                let Some(Value::Array(hs)) = m.get_mut("horses") else { unreachable!() };
                let Value::Object(h) = &mut hs[runner as usize] else { unreachable!() };
                let mut one = std::collections::BTreeMap::new();
                one.insert("skillId".to_string(), Value::String("669222".into()));
                one.insert("level".to_string(), Value::Number(1.0));
                let Value::Array(s) = h
                    .entry("skills".to_string())
                    .or_insert_with(|| Value::Array(Vec::new()))
                else {
                    unreachable!()
                };
                s.push(Value::Object(one));
            }
            let mut request = String::new();
            json::to_string_into(&v, &mut request);
            let run = race_exec::run_full(&[(request.clone(), seed)]).expect("gpu run");
            let race = &run.races[0];
            let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary");
            let gpu_reply = globalsim_gpu::full_reply::build(
                &dep_cpu::with_retain_all_frames(&request).expect("retain"),
                seed,
                race,
            )
            .expect("reply");
            let mut first = String::from("frames equal");
            let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
            'outer: for (tick, (cf, gf)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
                for (r, (cr, gr)) in cf.iter().zip(gf.iter()).enumerate() {
                    for (c, (x, y)) in cr.iter().zip(gr.iter()).enumerate() {
                        if x.to_bits() != y.to_bits() {
                            first = format!("tick {tick} runner {r} column {c}");
                            break 'outer;
                        }
                    }
                }
            }
            println!(
                "== cond {cond:?} pre {pre:?} engine {:?} kernel {:?} first {first}",
                events(&engine.reply, "669222").len(),
                events(&gpu_reply, "669222").len()
            );
        }
    }
}
