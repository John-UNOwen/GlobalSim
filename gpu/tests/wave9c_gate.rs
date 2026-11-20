//! Wave-9c real-id gates: the merged table's grade (100471, 900471),
//! post_number (200251), time (202231), base_power (202331), motivation
//! (210141) and corner_random (200351) ids replay frame-for-frame bit-exact
//! against the CPU engine on seeds 1..3 with zero fallback. The merged table
//! reaches neither every fold direction nor the StaminaUp kind or the viewer
//! row, so a 6694xx battery rides alongside and its mid-race rows must
//! actually activate.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

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

fn run_gate(label: &str, skill_data: &Value, horses: &[HorsePatch], expect_fired: &[i64]) {
    let results = patched_requests(skill_data, horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");
    let (fired, mut problems) = fired_ids(&results, &out);
    for sid in expect_fired {
        if !fired.contains(sid) {
            problems.push(format!("{sid} never activated on seeds 1..3"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(
            &format!("{label} seed {seed}"),
            request,
            *seed,
            &out.races[index],
            &mut problems,
        );
    }
    if !problems.is_empty() {
        panic!(
            "{} {label} problems, fired {fired:?}, problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// Diagnostic: one real id at a time, to locate a cross-skill interaction.
#[test]
#[ignore]
fn wave9c_bisect_one_skill_at_a_time() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    for (index, sid) in [200251i64, 202231, 202331, 210141, 200351, 100471, 900471]
        .into_iter()
        .enumerate()
    {
        let ids = [sid];
        let horses = vec![(index, Some(&ids[..]), None, None, None)];
        run_gate(&format!("wave9c single {sid}"), &skill_data, &horses, &[]);
    }
}

/// Diagnostic: for the 210141 horse, print the last hp column the two engines
/// agree on, then the first divergence, plus the finish. hp drives race length.
#[test]
#[ignore]
fn wave9c_tail_dump() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (669401, &def("remain_distance_viewer_id<=1500", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669411, &def("phase>=1&post_number>=2", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669412, &def("phase>=1&post_number<=3", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669413, &def("phase>=1&post_number>=3", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669421, &def("phase>=1&time==2", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669422, &def("phase>=1&grade==100", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669423, &def("phase>=1&motivation>=4", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669424, &def("phase>=1&base_power>=800", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669431, &def("corner_random==1@corner_random==99", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669432, &def("corner_random==1@corner_random==2@corner_random==3@corner_random==4", 5000000, 0, r#"{"type":9,"target":1,"modifier":400000}"#)),
            (669441, &def("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000}"#)),
            (669442, &def("phase>=1", 5000000, 30000, r#"{"type":2,"target":1,"modifier":200000}"#)),
            (669443, &def("phase>=1", 5000000, 30000, r#"{"type":2,"target":9,"targetValue":3,"modifier":300000}"#)),
            (669444, &def("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
            (669445, &def_wis("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000}"#)),
            (669446, &def_wis("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
            (669447, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000}"#)),
            (669448, &def_wis("grade>=0", 0, -1, r#"{"type":5,"target":1,"modifier":400000}"#)),
            (669449, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669411i64, 669424, 669447]), None, None, None),
        (1, Some(&[669412, 669448]), None, None, None),
        (2, Some(&[669413, 669449]), None, None, None),
        (3, Some(&[669421]), None, None, None),
        (4, Some(&[669422]), None, None, None),
        (5, Some(&[669423]), None, None, None),
        (6, Some(&[669401]), None, None, None),
        (7, Some(&[669431, 669432, 669442]), None, None, None),
        (8, Some(&[669441, 669443, 669444, 669445, 669446]), None, None, None),
    ];
    let results = patched_requests(&skill_data, &horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");
    for (index, (request, seed)) in results.iter().enumerate() {
        let race = &out.races[index];
        let cpu = dep_cpu::run_cpu_race(request, *seed).expect("cpu");
        println!("tail seed {seed}: cpu {} gpu {} frames", cpu.frames.len(), race.frames.len());
        let common = cpu.frames.len().min(race.frames.len());
        let mut first = None;
        for tick in 0..common {
            for runner in 0..cpu.frames[tick].len() {
                for column in 0..6 {
                    let cv = cpu.frames[tick][runner][column];
                    let gv = race.frames[tick][runner][column];
                    if cv.to_bits() != gv.to_bits() {
                        first = Some((tick, runner, column, cv, gv));
                        break;
                    }
                }
                if first.is_some() { break; }
            }
            if first.is_some() { break; }
        }
        if let Some((tick, runner, column, cv, gv)) = first {
            println!(
                "seed {seed} first divergence tick {tick} runner {runner} {}: cpu {cv:?} gpu {gv:?}",
                COLUMNS[column]
            );
            for t in (tick.saturating_sub(12))..=tick {
                println!("  tick {t} cpu {:?}", cpu.frames[t][runner]);
                println!("  tick {t} gpu {:?}", race.frames[t][runner]);
            }
        } else {
            println!("seed {seed} no cell divergence in {common} common ticks");
        }
    }
}


/// Adds battery definitions to the merged table.
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

fn def_wis(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":1,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

/// The merged table's wave-9c ids, one per host horse. grade folds true at
/// the g1 default, so 100471 fires its mid-race acceleration when the
/// remaining-distance and order band align, and 900471 rides the wisdom
/// roll. 200251 bumps the first three gates and folds dead on the rest;
/// 202231, 202331, 210141 and 200351 fold false, false, true (behind the
/// wisdom roll) and a live four-arm sampler.
#[test]
fn wave9c_real_ids_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[200251i64]), None, None, None),
        (1, Some(&[202231i64]), None, None, None),
        (2, Some(&[202331i64]), None, None, None),
        (3, Some(&[210141i64]), None, None, None),
        (4, Some(&[200351i64]), None, None, None),
        (5, Some(&[100471i64]), None, None, None),
        (6, Some(&[900471i64]), None, None, None),
    ];
    run_gate("wave9c real ids", &skill_data, &horses, &[]);
}

/// The directions the real ids cannot reach on this roster: the false-side
/// post_number fold (669411), the true sides of time, grade, motivation,
/// base_power and the viewer-row distance as activating mid-race fires, the
/// corner sampler with a missing row and with four arms (669431, 669432),
/// and the StaminaUp kind static (669441, mixed 669444, wisdom-gated
/// 669445), mid-race (669442) and fan-delivered (669443). The kind-2 rows
/// rebuild max hp where the engine's refresh_stat_effects does
/// (race_solver.rs:2556-2558, 2646).
#[test]
fn wave9c_battery_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (669401, &def("remain_distance_viewer_id<=1500", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669411, &def("phase>=1&post_number>=2", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669412, &def("phase>=1&post_number<=3", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669413, &def("phase>=1&post_number>=3", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669421, &def("phase>=1&time==2", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669422, &def("phase>=1&grade==100", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669423, &def("phase>=1&motivation>=4", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669424, &def("phase>=1&base_power>=800", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669431, &def("corner_random==1@corner_random==99", 5000000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#)),
            (669432, &def("corner_random==1@corner_random==2@corner_random==3@corner_random==4", 5000000, 0, r#"{"type":9,"target":1,"modifier":400000}"#)),
            (669441, &def("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000}"#)),
            (669442, &def("phase>=1", 5000000, 30000, r#"{"type":2,"target":1,"modifier":200000}"#)),
            (669443, &def("phase>=1", 5000000, 30000, r#"{"type":2,"target":9,"targetValue":3,"modifier":300000}"#)),
            (669444, &def("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
            (669445, &def_wis("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000}"#)),
            (669446, &def_wis("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
            (669447, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000}"#)),
            (669448, &def_wis("grade>=0", 0, -1, r#"{"type":5,"target":1,"modifier":400000}"#)),
            (669449, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
        ],
    );
    // Horse 0 hosts the false-side post_number fold (post 1 of 9), horse 1
    // and horse 2 the activating post_number sides, and the StaminaUp rows
    // sit on horses 7 and 8 with the corner battery.
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669411i64, 669424, 669447]), None, None, None),
        (1, Some(&[669412, 669448]), None, None, None),
        (2, Some(&[669413, 669449]), None, None, None),
        (3, Some(&[669421]), None, None, None),
        (4, Some(&[669422]), None, None, None),
        (5, Some(&[669423]), None, None, None),
        (6, Some(&[669401]), None, None, None),
        (7, Some(&[669431, 669432, 669442]), None, None, None),
        (8, Some(&[669441, 669443, 669444, 669445, 669446]), None, None, None),
    ];
    run_gate(
        "wave9c battery",
        &skill_data,
        &horses,
        &[
            669401, 669412, 669413, 669421, 669422, 669423, 669424, 669431, 669432, 669442,
            669443,
        ],
    );
}

/// Diagnostic: compile the gated battery rows and print the record flags and
/// bumps so the INIT_BUMP/BUMPED bits and host bump push are visible.
#[test]
#[ignore]
fn wave9c_flag_dump() {
    use globalsim_gpu::skills_compile;
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (669445, &def_wis("grade>=0", 0, -1, r#"{"type":2,"target":1,"modifier":300000}"#)),
            (669447, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000}"#)),
            (669448, &def_wis("grade>=0", 0, -1, r#"{"type":5,"target":1,"modifier":400000}"#)),
            (669449, &def_wis("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#)),
        ],
    );
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").unwrap();
    let course = courses.get("10504").unwrap();
    let value = json::parse(&v77_request()).unwrap();
    let json::Value::Object(map) = &value else { unreachable!() };
    let mut req = json::Value::Object(map.clone());
    {
        let json::Value::Object(m) = &mut req else { unreachable!() };
        m.insert("skillData".into(), skill_data);
        let json::Value::Array(entries) = m.get_mut("horses").unwrap() else { unreachable!() };
        for (i, sid) in [(1i64, 669445i64), (2, 669447), (3, 669448), (4, 669449)] {
            let json::Value::Object(h) = &mut entries[i as usize] else { unreachable!() };
            h.insert("skills".into(), json::arr(vec![json::obj(vec![
                ("skillId", json::Value::Number(sid as f64)),
                ("level", json::Value::Number(1.)),
            ])]));
        }
    }
    let plan = skills_compile::plan(&req, course);
    println!("reason {:?} bumps {:?} live_guts {}", plan.reason, plan.bumps, plan.live_guts);
    let nrec = plan.words[0] as usize;
    for rec in 0..nrec {
        let at = skills_compile::SK_RECORDS + skills_compile::SK_RSTRIDE * rec;
        let sid = plan.words[at];
        if !(669444..=669449).contains(&sid) { continue; }
        println!("rec {rec} sid {sid} runner {} flags {:#x}", plan.words[at + 1], plan.words[at + 2]);
    }
    let _ = (669446i64, 0);
}
