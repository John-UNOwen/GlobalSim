//! Target 18, the style-group debuff fan (handoff E1a), on the retail lane.
//!
//! The twelve wire ids are the Subdued / Flustered / Hesitant family from
//! the CM request's skillData, carried into the v77 roster verbatim: each
//! style (nige, senko, sashi, oikomi) has a phase 0 and a phase 1 variant of
//! the Recovery -100 debuff and a phase 2 variant of the CurrentSpeed -1500
//! debuff with a 3 s base duration. skill_targets resolves them to every
//! runner of the targetValue style except the owner (race_field.rs:439-445),
//! and the delivery lands per receiver as the rewritten self effect
//! (race_field.rs:465-477). Horse 0 owns the skill; the field gains the
//! style group it fans to, with an Oonige runner in the nige scenes to prove
//! the runaway counts as Front Runner (strategy_matches in horse_types.rs).
//!
//! Every frame compares through to_bits() against the CPU oracle, the full
//! reply must be byte-equal, and the carrying run must move the replay
//! against the same field without the skill. The last scene drops the random
//! phase term and delivers a Recovery debuff with no random term anywhere:
//! there the first frame change must be the single victim runner's hp column
//! and nothing else, which is the recipient path in isolation.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];
const HP_COL: usize = 2;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn base_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// The exact wire definition of one family id, read from the CM request.
fn cm_skill(id: &str) -> Value {
    let text =
        std::fs::read_to_string(repo_file("examples/cm-10701-race.jsonl")).expect("cm request");
    let root = json::parse(text.lines().next().expect("first line")).expect("cm parses");
    root.get("skillData")
        .and_then(|s| s.get(id))
        .cloned()
        .unwrap_or_else(|| panic!("skillData has no {id}"))
}

/// Horse 1 (and horse 2, as Oonige, in the nige scenes) joins the fan group.
fn field_value(seed: i64, id: &str, def: &Value, style: &str) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Object(sd) = map.get_mut("skillData").expect("skillData") else { unreachable!() };
        sd.insert(id.into(), def.clone());
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let strategies = match style {
            "Nige" => ["Nige", "Oonige"],
            other => [other, "Senkou"],
        };
        for (i, strategy) in strategies.into_iter().enumerate() {
            let Value::Object(h) = &mut horses[i] else { unreachable!() };
            h.insert("strategy".into(), Value::String(strategy.into()));
        }
        let Value::Object(h0) = &mut horses[0] else { unreachable!() };
        h0.insert("skills".into(), json::arr(vec![Value::String(id.into())]));
    }
    value
}

/// A field with the group members but no skill on any runner.
fn plain_value(seed: i64, style: &str) -> Value {
    let mut value = base_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(COURSE as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let strategies = match style {
            "Nige" => ["Nige", "Oonige"],
            other => [other, "Senkou"],
        };
        for (i, strategy) in strategies.into_iter().enumerate() {
            let Value::Object(h) = &mut horses[i] else { unreachable!() };
            h.insert("strategy".into(), Value::String(strategy.into()));
        }
    }
    value
}

fn request_text(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
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
            for (_column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
}

/// One family scene over the seeds. Returns the seeds whose carrying run
/// moved the replay against the same field without the skill.
fn run_scene(name: &str, id: &str, style: &str, problems: &mut Vec<String>) -> usize {
    let def = cm_skill(id);
    let mut changed = 0usize;
    for &seed in &SEEDS {
        let tag = format!("{name} seed {seed}");
        let request = request_text(&field_value(seed, id, &def, style));
        let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        compare_frames(&tag, problems, &request, seed, race);
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
        let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        if gpu_reply != engine {
            problems.push(format!("{tag}: full reply differs from the engine"));
        }
        let plain = race_exec::run(&[(request_text(&plain_value(seed, style)), seed)])
            .expect("plain kernel run");
        let plain_race = &plain.races[0];
        if plain_race.fallback.is_some() {
            problems.push(format!("{tag}: plain control refused"));
            continue;
        }
        let differs = plain_race.frames.len() != race.frames.len()
            || plain_race.frames.iter().zip(race.frames.iter()).any(|(a, b)| a != b);
        if differs {
            changed += 1;
        }
    }
    println!("scene {name} ({id}): the debuff moved the replay in {changed} of {} seeds", SEEDS.len());
    changed
}

#[test]
fn retail_target18_scenes_answer_bit_exact() {
    let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{COURSE}.json")))
        .expect("geometry");
    globalsim::register_course_geometry(COURSE, geometry);
    let mut problems: Vec<String> = Vec::new();
    let mut totals = 0usize;
    let scenes: [(&str, &str, &str); 12] = [
        ("nige phase0", "200831", "Nige"),
        ("nige phase1", "200841", "Nige"),
        ("nige phase2", "200851", "Nige"),
        ("senko phase0", "200861", "Senkou"),
        ("senko phase1", "200871", "Senkou"),
        ("senko phase2", "200881", "Senkou"),
        ("sashi phase0", "200891", "Sasi"),
        ("sashi phase1", "200901", "Sasi"),
        ("sashi phase2", "200911", "Sasi"),
        ("oikomi phase0", "200921", "Oikomi"),
        ("oikomi phase1", "200931", "Oikomi"),
        ("oikomi phase2", "200941", "Oikomi"),
    ];
    for (name, id, style) in scenes {
        let moved = run_scene(name, id, style, &mut problems);
        if moved == 0 {
            problems.push(format!("scene {name} ({id}): the debuff never fired on any seed, the scene is vacuous"));
        }
        totals += moved;
    }
    // The recipient path in isolation. No random term anywhere, so the fan
    // delivery is the only difference from the plain field, and its first
    // frame change must be the lone victim's hp column.
    let victim_seed = SEEDS[0];
    let victim_def = json::parse(
        r#"{"rarity":1,"wisdomCheck":0,"priority":0,"tags":[301],
"alternatives":[{"condition":"hp_per<=99","precondition":"","baseDuration":0,
"durationUsage":1,"cooldown":0,
"effects":[{"type":9,"target":18,"targetValue":1,"additionalActivate":0,
"valueUsage":1,"modifier":-100,"levelModifiers":[1.0,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}]}],
"isGeneralSkill":true}"#,
    )
    .expect("victim skill json");
    let tag = "victim path";
    let request = request_text(&field_value(victim_seed, "920081", &victim_def, "Nige"));
    let run = race_exec::run_full(&[(request.clone(), victim_seed)]).expect("kernel run");
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
    } else {
        compare_frames(tag, &mut problems, &request, victim_seed, race);
        let plain = race_exec::run(&[(request_text(&plain_value(victim_seed, "Nige")), victim_seed)])
            .expect("plain kernel run");
        let plain_race = &plain.races[0];
        if plain_race.fallback.is_some() {
            problems.push(format!("{tag}: plain control refused"));
        } else {
            let mut first: Option<(usize, usize, usize, f64, f64)> = None;
            let same_len = plain_race.frames.len() == race.frames.len();
            if !same_len {
                problems.push(format!(
                    "{tag}: tick count plain {} gpu {}",
                    plain_race.frames.len(),
                    race.frames.len()
                ));
            }
            if same_len {
                'diff: for (tick, (plain_frame, gpu_frame)) in
                    plain_race.frames.iter().zip(race.frames.iter()).enumerate()
                {
                    for (runner, (plain_row, gpu_row)) in
                        plain_frame.iter().zip(gpu_frame.iter()).enumerate()
                    {
                        for (column, (plain, gpu)) in
                            plain_row.iter().zip(gpu_row.iter()).enumerate()
                        {
                            if plain.to_bits() != gpu.to_bits() {
                                first = Some((tick, runner, column, *plain, *gpu));
                                break 'diff;
                            }
                        }
                    }
                }
            }
            match first {
                None => problems.push(format!("{tag}: the debuff never changed the replay, the scene is vacuous")),
                Some((tick, runner, column, plain_hp, gpu_hp)) => {
                    if runner != 1 || column != HP_COL {
                        problems.push(format!(
                            "{tag}: first change is runner {runner} column {column} at tick {tick}, \
                             expected the victim's hp column"
                        ));
                    }
                    if !(gpu_hp < plain_hp) {
                        problems.push(format!(
                            "{tag}: the victim hp moved the wrong way: {plain_hp} -> {gpu_hp}"
                        ));
                    }
                    println!("scene {tag}: first frame change at tick {tick}, runner {runner}, hp {plain_hp} -> {gpu_hp}");
                    totals += 1;
                }
            }
        }
    }
    if totals == 0 {
        problems.push("no scene changed any replay, no target-18 debuff fired".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
