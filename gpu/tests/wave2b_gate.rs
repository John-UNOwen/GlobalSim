//! Wave-2b identity gates: the effect families ported in this wave (cross
//! runner targets, kinds 4 5 6 8 14 29, per-record repeats, construction
//! bumps and the receiver instance entry) must compile from the real merged
//! skill set and replay frame-for-frame bit-exact, and each firing path must
//! run through synthetic defs so the gate is not vacuous.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

/// One real merged skill per ported family, the highest-count choice in each
/// family from the refusal histogram that activates on course 10504, with the
/// seats (horse index plus optional strategy) the alt condition needs. Skills
/// gated on other courses' distance_type (201012, 201222) fold dead here, so
/// the histogram runners-up stand in; 110301 needs a mid-pack seat that the
/// all-Senkou v77 field only produces for some strategies, so it spreads.
const REAL_SKILLS: [(i64, &str, &[(usize, Option<&str>)]); 10] = [
    (201502, "kind 8 vision self", &[(0, Some("Oikomi"))]),
    (201522, "kind 5 static bump plus vision", &[(0, Some("Nige"))]),
    (110581, "repeat 27 self plus 21 ahead and behind", &[(0, None)]),
    (910581, "kind 21 ahead and behind debuff pair", &[(0, None)]),
    (201442, "kind 9 visible-target damage", &[(0, Some("Sashi"))]),
    (202051, "kind 6 runaway static", &[(0, Some("Nige"))]),
    (202141, "kind 14 plus kind 5 static bumps", &[(0, None)]),
    (202511, "kind 29 kakari bonus bump", &[(0, None)]),
    (202181, "kind 4 plus kind 5 static bumps", &[(0, None)]),
    (
        110301,
        "kind 9 repeat self and ahead",
        &[(0, Some("Sashi")), (4, Some("Oikomi")), (8, Some("Oonige"))],
    ),
];

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

/// The v77 base with merged skillData, the given horses' skill lists (and
/// optional strategies) replaced.
fn patched_requests(
    skill_data: &Value,
    horse_skills: &[(usize, Vec<i64>, Option<&str>)],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("skillData".into(), skill_data.clone());
        let json::Value::Array(horses) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids, strategy) in horse_skills {
            let json::Value::Object(horse) = &mut horses[*index] else { unreachable!() };
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
            if let Some(style) = strategy {
                horse.insert("strategy".into(), json::Value::String((*style).into()));
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

fn fired_ids(results: &[(String, i64)], out: &race_exec::RaceRun) -> (Vec<i64>, Vec<String>) {
    let mut fired: Vec<i64> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in results.iter().enumerate() {
        if let Some(reason) = &out.races[index].fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => {
                for (_, sid, _, _) in &cpu.detail_events {
                    if !fired.contains(sid) {
                        fired.push(*sid);
                    }
                }
            }
            Err(reason) => problems.push(format!("seed {seed}: cpu race failed: {reason}")),
        }
    }
    fired.sort_unstable();
    (fired, problems)
}

/// One real merged skill per ported family on horse 0 of the v77 roster:
/// zero CPU fallback, frame-for-frame identity, and each skill confirmed
/// active in the CPU detail events over the seeds.
#[test]
fn wave2b_real_skills_replay_bit_exact() {
    let skill_data = merged_skill_data();
    register_geometry(10504);

    let mut problems: Vec<String> = Vec::new();
    let mut fired_report: Vec<String> = Vec::new();
    for (sid, what, seats) in REAL_SKILLS {
        let horse_skills: Vec<(usize, Vec<i64>, Option<&str>)> = seats
            .iter()
            .map(|(index, strategy)| (*index, vec![sid], *strategy))
            .collect();
        let results = patched_requests(&skill_data, &horse_skills, &SEEDS);
        let out = race_exec::run(&results).expect("kernel run");
        let (fired, mut fires) = fired_ids(&results, &out);
        problems.append(&mut fires);
        if !fired.contains(&sid) {
            problems.push(format!("{sid} ({what}) never activated on seeds 1..3"));
        }
        fired_report.push(format!("{sid} ({what}) fired {fired:?}"));
        for (index, (request, seed)) in results.iter().enumerate() {
            let tag = format!("{sid} ({what}) seed {seed}");
            compare_race(&tag, request, *seed, &out.races[index], &mut problems);
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} wave2b real-skill problems, firing report:\n{}\nproblems:\n{}",
            problems.len(),
            fired_report.join("\n"),
            problems.join("\n")
        );
    }
}

/// Each new firing path on synthetic defs (663xxx) over three v77 horses:
/// cross targets 9 10 4, repeats, cross recovery, runaway, the static bumps,
/// and one cross-skill id carried at three seats so a receiver can also be a
/// caster of the same id.
#[test]
fn wave2b_effects_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();

    let make_def = |cond: &str, cooldown: i64, effects: &str| -> Value {
        json::parse(&format!(
            r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":{cooldown},"baseDuration":30000,"effects":[{effects}]}}]}}"#
        ))
        .expect("def parses")
    };
    let battery: [(i64, &str); 12] = [
        (663001, r#"{"type":27,"target":10,"modifier":1000,"targetValue":3}"#),
        (663002, r#"{"type":21,"target":9,"modifier":-1000,"targetValue":2}"#),
        (663003, r#"{"type":8,"target":4,"modifier":-2000,"targetValue":2}"#),
        (663004, r#"{"type":9,"target":9,"modifier":200,"targetValue":2}"#),
        (
            663005,
            r#"{"type":27,"target":1,"modifier":1000},{"type":27,"target":9,"modifier":500,"targetValue":2}"#,
        ),
        (663006, r#"{"type":6,"target":10,"modifier":1000,"targetValue":2}"#),
        (663007, r#"{"type":14,"target":1,"modifier":850},{"type":5,"target":1,"modifier":-400000}"#),
        (663008, r#"{"type":29,"target":1,"modifier":100000}"#),
        (663009, r#"{"type":4,"target":1,"modifier":4000}"#),
        (663010, r#"{"type":21,"target":9,"modifier":1000,"targetValue":2}"#),
        (663011, r#"{"type":31,"target":4,"modifier":500,"targetValue":3}"#),
        (663012, r#"{"type":22,"target":9,"modifier":800,"targetValue":1}"#),
    ];
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, effects) in battery {
            let cond = if matches!(sid, 663007 | 663008 | 663009) { "always>=0" } else { "order>=2" };
            let cooldown = if matches!(sid, 663007 | 663008 | 663009) { 0 } else { 60000 };
            sd.insert(sid.to_string(), make_def(cond, cooldown, effects));
        }
    }
    let carriers: Vec<i64> = battery.iter().map(|(sid, _)| *sid).collect();
    let mut horse0 = carriers.clone();
    horse0.retain(|sid| *sid != 663010);
    // Horse 0 runs every battery skill, horses 4 and 8 carry 663010 at two
    // more seats so its deliveries cross three casters and receivers.
    let results = patched_requests(
        &skill_data,
        &[(0, horse0, None), (4, vec![663010, 663002], None), (8, vec![663010], None)],
        &SEEDS,
    );
    let out = race_exec::run(&results).expect("kernel run");

    let (fired, mut problems) = fired_ids(&results, &out);
    for (sid, _) in battery {
        if !fired.contains(&sid) {
            problems.push(format!("{sid} never activated"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(&format!("battery seed {seed}"), request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} wave2b battery problems, fired {:?}, problems:\n{}",
            problems.len(),
            fired,
            problems.join("\n")
        );
    }
}
