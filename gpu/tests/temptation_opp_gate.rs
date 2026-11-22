//! The team-filtered temptation opponent counts (handoff E1a, family 6), on
//! the retail lane.
//!
//! The engine counts rushing runners through is_opponent, which drops the
//! querying runner and every same-team runner when teams are set
//! (race_field.rs:913-915, 917-946; activation_conditions.rs:1341-1362). The
//! kernel scans the published order and the strategy cells with the same
//! team skip. Only the CM roster has teams, so every scene here runs on the
//! CM 10701 field.
//!
//! The roster's only Nige runner (horse 6) is the one that rushes early in
//! these seeds, and it is on team 1 with horses 0 and 2. That fixes the
//! filter pins: on teammate horse 2 the plain nige count fires while the
//! opponent nige count is structurally shut (its opponent group is empty),
//! on opponent horse 1 both opponent counts fire, and infront of horse 2 the
//! plain slice fires where the filtered slice stays shut. Every scene answers
//! bit-exact frames against the CPU engine.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, json, race_exec};
use json::Value;

const COURSE: i32 = 10701;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_cm() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    json::parse(text.lines().next().expect("first line")).expect("cm parses")
}

/// A crafted definition, the same shape the CM wire entries use.
fn crafted(condition: &str) -> Value {
    json::parse(&format!(
        r#"{{"rarity":5,"wisdomCheck":0,"priority":0,"tags":[401],
"alternatives":[{{"condition":"{condition}","precondition":"","baseDuration":50000,
"durationUsage":1,"cooldown":5000000,
"effects":[{{"type":27,"target":1,"targetValue":0,"additionalActivate":0,
"valueUsage":1,"modifier":4500.0,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}]}}],
"isGeneralSkill":true}}"#
    ))
    .expect("crafted skill json")
}

/// Horse carrier carries the named skill on top of its CM skill list; with an
/// empty id the field is the untouched CM race. A null def means the skill is
/// already defined in the CM skillData.
fn field_value(seed: i64, carrier: usize, id: &str, def: &Value) -> Value {
    let mut value = base_cm();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("seed".into(), Value::Number(seed as f64));
        if !id.is_empty() && def.get("alternatives").is_some() {
            let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
            sd.insert(id.into(), def.clone());
        }
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[carrier] else { unreachable!() };
        let mut ids: Vec<Value> = h
            .get("skills")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !id.is_empty() {
            ids.push(Value::String(id.into()));
        }
        h.insert("skills".into(), Value::Array(ids));
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn compare_frames(
    tag: &str,
    problems: &mut Vec<String>,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
) {
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

/// Runs the kernel, checks it against the CPU oracle, and returns the frames
/// so the caller can compare scenes against each other. Bit-exact frames are
/// the parity bar here; full replies are built for single-skill fields, so
/// the byte-equal reply check runs in the lastspurt gate on the v77 base.
fn run_kernel(tag: &str, problems: &mut Vec<String>, request: &str, seed: i64) -> Option<race_exec::RaceGpu> {
    let run = match race_exec::run_full(&[(request.to_string(), seed)]) {
        Ok(run) => run,
        Err(why) => {
            problems.push(format!("{tag}: kernel run failed: {why}"));
            return None;
        }
    };
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return None;
    }
    compare_frames(tag, problems, request, seed, race);
    Some((*race).clone())
}

fn frames_equal(a: &race_exec::RaceGpu, b: &race_exec::RaceGpu) -> bool {
    a.frames.len() == b.frames.len()
        && a.frames.iter().zip(b.frames.iter()).all(|(fa, fb)| {
            fa.len() == fb.len()
                && fa.iter().zip(fb).all(|(ra, rb)| {
                    ra.len() == rb.len()
                        && ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits())
                })
        })
}

#[test]
fn retail_temptation_opponent_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut plain_nige_teammate = 0usize;
    let mut opp_nige_opponent = 0usize;
    let mut opp_infront_opponent = 0usize;
    let mut infront_discriminated = 0usize;
    for &seed in &SEEDS {
        // Definition fidelity: the four CM running_style ids. The three
        // behind/infront ids attach effects on target 19 or 20, still
        // unported, so their condition side rides the crafted scenes.
        for id in ["200791", "200801", "200811", "200821"] {
            let req = request_text(&field_value(seed, 0, id, &Value::Null));
            let _ = run_kernel(&format!("{id} seed {seed}"), &mut problems, &req, seed);
        }
        let plain = match run_kernel(
            &format!("plain seed {seed}"),
            &mut problems,
            &request_text(&field_value(seed, 0, "", &Value::Null)),
            seed,
        ) {
            Some(race) => race,
            None => continue,
        };
        let scenes: [(usize, &str); 7] = [
            (0, "running_style_temptation_count_nige>=1"),
            (1, "running_style_temptation_opponent_count_nige>=1"),
            (2, "running_style_temptation_count_nige>=1"),
            (2, "running_style_temptation_opponent_count_nige>=1"),
            (2, "temptation_count_infront>=1"),
            (2, "temptation_opponent_count_infront>=1"),
            (1, "temptation_opponent_count_infront>=1"),
        ];
        let mut fired = [false; 7];
        for (i, &(carrier, cond)) in scenes.iter().enumerate() {
            let def = crafted(cond);
            let req = request_text(&field_value(seed, carrier, "920291", &def));
            let Some(scene) = run_kernel(&format!("{cond} seed {seed}"), &mut problems, &req, seed) else {
                continue;
            };
            fired[i] = !frames_equal(&scene, &plain);
        }
        // The only Nige runner shares horse 2's team, so its opponent group
        // is empty and the opponent count is provably zero on every seed.
        if fired[3] {
            problems.push(format!("seed {seed}: opponent nige count fired with an empty opponent group"));
        }
        if fired[2] {
            plain_nige_teammate += 1;
        }
        if fired[1] {
            opp_nige_opponent += 1;
        }
        if fired[6] {
            opp_infront_opponent += 1;
        }
        if fired[4] && !fired[5] {
            infront_discriminated += 1;
        }
    }
    if plain_nige_teammate == 0 {
        problems.push("the plain nige count never fired, the group scan is unproven".into());
    }
    if opp_nige_opponent == 0 {
        problems.push("the opponent nige count never fired for an opponent carrier".into());
    }
    if opp_infront_opponent == 0 {
        problems.push("the filtered infront slice never fired for an opponent carrier".into());
    }
    if infront_discriminated == 0 {
        problems.push("no seed showed the infront plain slice fire while the filtered slice stayed shut".into());
    }
    println!(
        "temptation pins: nige plain {plain_nige_teammate}/{} opp {opp_nige_opponent}/{} infront filtered {}/{} discriminated {}/{}",
        SEEDS.len(),
        SEEDS.len(),
        opp_infront_opponent,
        SEEDS.len(),
        infront_discriminated,
        SEEDS.len()
    );
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
