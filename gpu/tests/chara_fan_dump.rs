//! Chara-fan event dump. Injects the crafted code-22 arm on cm-10701, runs
//! both lanes with the reply channels on, and prints the detail and target
//! event rows for the id plus the first differing reply byte. Ignored
//! diagnostics for the X14 chara fan.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value(file: &str) -> Value {
    let text = std::fs::read_to_string(repo_file(file)).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("base parses")
}

#[ignore]
#[test]
fn dump_chara_fan_events() {
    let geometry = std::fs::read_to_string(repo_file("data/course_geometry/10701.json")).expect("geometry");
    globalsim::register_course_geometry(10701, geometry);
    let roster = base_value("examples/cm-10701-race.jsonl");
    for horse in roster.get("horses").and_then(Value::as_array).expect("horses") {
        println!(
            "roster team {:?} chara {:?}",
            horse.get("teamId"),
            horse.get("charaId")
        );
    }
    for seed in [1i64, 2] {
        let mut value = base_value("examples/cm-10701-race.jsonl");
        {
            let Value::Object(map) = &mut value else { unreachable!() };
            map.insert("seed".into(), Value::Number(seed as f64));
            let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
            sd.insert(
                "920391".into(),
                json::parse(
                    r#"{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{"condition":"is_exist_chara_id==1010","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{"type":9,"target":22,"targetValue":1010,"additionalActivate":0,
"valueUsage":1,"modifier":550.0,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}]}],
"isGeneralSkill":true}"#,
                )
                .expect("crafted"),
            );
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(h) = &mut horses[0] else { unreachable!() };
            h.insert("skills".into(), json::arr(vec![Value::String("920391".into())]));
        }
        let mut text = String::new();
        json::to_string_into(&value, &mut text);
        let mut ctx = race_exec::RaceContext::open().expect("context");
        let prepared = ctx.prepare(&text, seed).expect("prepare");
        for (sid, flags) in prepared.sk_record_flags() {
            if sid == 920391 {
                println!("seed {seed} record flags {flags:#x}");
            }
        }
        let run = race_exec::run_full(&[(text.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        println!("seed {seed} fallback {:?}", race.fallback);
        let full = dep_cpu::with_retain_all_frames(&text).expect("retain");
        let gpu_reply = full_reply::build(&full, seed, race).expect("reply");
        let engine = dep_cpu::run_cpu_binary(&text, seed).expect("engine").reply;
        for (lane, reply) in [("engine", &engine), ("kernel", &gpu_reply)] {
            let parsed = json::parse(reply).expect("reply json");
            let rows = parsed.get("skillTargetEvents").and_then(Value::as_array).cloned().unwrap_or_default();
            let hits: Vec<String> = rows
                .iter()
                .filter(|r| {
                    r.get("skillId").and_then(Value::as_i64).map(|v| v.to_string())
                        == Some("920391".into())
                })
                .map(|r| format!("{r:?}"))
                .collect();
            println!("{lane} target events: {hits:?}");
            let rows = parsed.get("skillDetailEvents").and_then(Value::as_array).cloned().unwrap_or_default();
            let hits: Vec<String> = rows
                .iter()
                .filter(|r| {
                    r.get("skillId").and_then(Value::as_i64).map(|v| v.to_string())
                        == Some("920391".into())
                })
                .map(|r| format!("{r:?}"))
                .collect();
            println!("{lane} detail events: {hits:?}");
        }
        if gpu_reply != engine {
            let at = gpu_reply.bytes().zip(engine.bytes()).take_while(|(a, b)| a == b).count();
            let lo = at.saturating_sub(60);
            println!(
                "reply differs at byte {at}\nkernel: {:?}\nengine: {:?}",
                &gpu_reply[lo..at + 80.min(gpu_reply.len() - lo)],
                &engine[lo..at + 80.min(engine.len() - lo)]
            );
        } else {
            println!("reply equal");
        }
    }
}
