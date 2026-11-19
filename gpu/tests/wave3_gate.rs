//! Wave-3 identity gates: the stateful time machines, activation-history
//! counters, live in-band cells, the bashin-family siblings and the roster
//! folds must compile from the real merged skill set and replay frame-for-frame
//! bit-exact, and each new leaf must run through synthetic defs so the gate is
//! not vacuous.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

/// One merged-table skill per ported construct, the highest-count choice the
/// histogram or a table scan yields, with the seats (horse index plus optional
/// strategy) the condition needs. Some conditions only hold behind a helper
/// skill that produces the earlier fires, so the row carries the full horse 0
/// list. Horse 0's popularity can be raised for the favorite fold.
struct Row {
    sid: i64,
    what: &'static str,
    skills: &'static [i64],
    host: usize,
    /// Every horse carries the row skills, so whichever seat the condition
    /// fits fires.
    spread: bool,
    seats: &'static [(usize, Option<&'static str>)],
    pop: Option<i64>,
    /// The row replays identity but the condition is too narrow to demand a
    /// firing on three seeds.
    fires: bool,
}

/// The merged table blocks the wave-3 roster-rate and temptation-opponent
/// constructs on their effect side (910151 two alternatives, 200281 effect
/// type 3, 200771/200781/200801 effect target 20/19 or type 13), so those
/// leaves ride the battery and fold gates only. The mid-pack and back markers
/// ride horses 2, 4, 5 and 8 so rank windows, front blockers and final-corner
/// passes can hold.
const REAL_SKILLS: [Row; 14] = [
    Row { sid: 900561, what: "blocked_front binary", skills: &[900561], host: 2, spread: true, seats: &[], pop: None, fires: true },
    Row { sid: 200622, what: "distance_diff_rate", skills: &[200622], host: 0, spread: false, seats: &[(0, Some("Oikomi"))], pop: None, fires: true },
    Row { sid: 910231, what: "order_rate_in50 cell", skills: &[910231], host: 4, spread: false, seats: &[], pop: None, fires: true },
    Row { sid: 900641, what: "order_rate_in20 cell", skills: &[900641], host: 0, spread: false, seats: &[], pop: None, fires: true },
    Row { sid: 210131, what: "activate_count_later_half", skills: &[210131, 664100], host: 0, spread: false, seats: &[], pop: None, fires: true },
    Row { sid: 201621, what: "activate_count_end_after", skills: &[201621, 664100], host: 0, spread: false, seats: &[], pop: None, fires: true },
    Row { sid: 910371, what: "change_order_up_finalcorner_after", skills: &[910371], host: 4, spread: true, seats: &[], pop: None, fires: true },
    Row { sid: 201641, what: "same_skill_horse_count", skills: &[201641], host: 0, spread: false, seats: &[], pop: None, fires: true },
    Row { sid: 201643, what: "PROBE never-fire", skills: &[201643], host: 0, spread: false, seats: &[], pop: None, fires: false },
    Row { sid: 201644, what: "PROBE always-fire", skills: &[201644], host: 0, spread: false, seats: &[], pop: None, fires: false },
    Row { sid: 200271, what: "running_style_count_same", skills: &[200271], host: 0, spread: false, seats: &[(0, Some("Nige"))], pop: None, fires: true },
    Row { sid: 200291, what: "running_style_equal_popularity_one", skills: &[200291], host: 0, spread: false, seats: &[], pop: Some(1), fires: true },
    Row { sid: 910031, what: "overtake_target_no_order_up_time", skills: &[910031], host: 0, spread: false, seats: &[], pop: None, fires: false },
    Row { sid: 200401, what: "is_surrounded", skills: &[200401], host: 5, spread: false, seats: &[], pop: None, fires: true },
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

/// One horse patch: skill list (or None to keep the request's), strategy and
/// popularity overrides (or None to keep the request's).
type HorsePatch<'a> = (usize, Option<&'a [i64]>, Option<&'a str>, Option<i64>);

/// The v77 base with merged skillData and the given horse patches applied.
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
        for (index, ids, strategy, pop) in horses {
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

/// One real merged skill per ported construct on the v77 roster: zero CPU
/// fallback, frame-for-frame identity, and each skill confirmed active in the
/// CPU detail events over the seeds (the one narrow exception marked so).
#[test]
fn wave3_real_skills_replay_bit_exact() {
    let mut skill_data = merged_skill_data();
    // Probe defs: the 201641 shape (kind 1, cooldown 0, baseDuration -1) with
    // a never-true and an always-true condition, separating fold, firing and
    // effect as divergence sources.
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        let source = sd
            .get("201641")
            .expect("201641 present")
            .clone();
        for (id, cond) in [("201643", "order==99"), ("201644", "always>=0")] {
            let mut probe = source.clone();
            let json::Value::Object(p) = &mut probe else { unreachable!() };
            let json::Value::Array(alts) = p.get_mut("alternatives").expect("alts") else {
                unreachable!()
            };
            let json::Value::Object(alt) = &mut alts[0] else { unreachable!() };
            alt.insert("condition".into(), json::Value::String(cond.into()));
            sd.insert(id.into(), probe);
        }
        // A self-firing helper so the two counter conditions can reach their
        // thresholds inside the replay window.
        sd.insert(
            "664100".into(),
            json::parse(
                r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{"precondition":"","condition":"always>=0","cooldown":8000,"baseDuration":30000,"effects":[{"type":10,"target":1,"modifier":0}]}]}"#,
            )
            .expect("helper parses"),
        );
    }
    register_geometry(10504);

    let mut problems: Vec<String> = Vec::new();
    let mut fired_report: Vec<String> = Vec::new();
    for row in REAL_SKILLS {
        let mut horses: Vec<HorsePatch> = if row.spread {
            (0..9).map(|i| (i, Some(row.skills), None, None)).collect()
        } else {
            vec![(row.host, Some(row.skills), None, row.pop)]
        };
        for (index, strategy) in row.seats {
            if *index == row.host {
                horses[0] = (row.host, Some(row.skills), *strategy, row.pop);
            } else {
                horses.push((*index, None, *strategy, None));
            }
        }
        let results = patched_requests(&skill_data, &horses, &SEEDS);
        let out = race_exec::run(&results).expect("kernel run");
        let (fired, mut fires) = fired_ids(&results, &out);
        problems.append(&mut fires);
        if row.fires && !fired.contains(&row.sid) {
            problems.push(format!("{} ({}) never activated on seeds 1..3", row.sid, row.what));
        }
        fired_report.push(format!("{} ({}) fired {fired:?}", row.sid, row.what));
        for (index, (request, seed)) in results.iter().enumerate() {
            let tag = format!("{} ({}) seed {seed}", row.sid, row.what);
            compare_race(&tag, request, *seed, &out.races[index], &mut problems);
        }
    }
    if !problems.is_empty() {
        panic!(
            "{} wave3 real-skill problems, firing report:\n{}\nproblems:\n{}",
            problems.len(),
            fired_report.join("\n"),
            problems.join("\n")
        );
    }
}

/// Every new leaf fires through synthetic 664xxx defs so the identity gate is
/// not vacuous. The self-referential activation gate is the one def that must
/// never fire: the CPU itself proves the record dead.
#[test]
fn wave3_constructs_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();

    let make_def = |cond: &str, cooldown: i64, effects: &str| -> Value {
        json::parse(&format!(
            r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":{cooldown},"baseDuration":30000,"effects":[{effects}]}}]}}"#
        ))
        .expect("def parses")
    };
    let heal = r#"{"type":9,"target":1,"modifier":200}"#;
    let inert = r#"{"type":10,"target":1,"modifier":0}"#;
    // Wire order is list order: each paired def follows its provider, and the
    // bounded-count def precedes the fires that would push the count past it.
    let battery: [(i64, &str, i64, &str); 24] = [
        (664010, "activate_count_all<=3", 60000, inert),
        (664006, "always>=0", 60000, inert),
        (664005, "is_used_skill_id==664006", 60000, inert),
        (664008, "always>=0", 60000, heal),
        (664007, "is_activate_heal_skill==1", 60000, inert),
        (664009, "activate_count_all>=1", 60000, inert),
        (664001, "blocked_front==1", 60000, inert),
        (664002, "is_surrounded==1", 100000, inert),
        (664003, "near_infront_count>=1", 60000, inert),
        (664004, "visiblehorse>=1", 60000, inert),
        (664011, "activate_count_later_half>=0", 60000, inert),
        (664012, "activate_count_end_after>=0", 60000, inert),
        (664013, "overtake_target_no_order_up_time>=0", 60000, inert),
        (664014, "overtake_target_no_order_down_time>=0", 60000, inert),
        (664015, "change_order_up_middle>=0", 60000, inert),
        (664016, "change_order_up_finalcorner_after>=0", 60000, inert),
        (664017, "blocked_all_continuetime>=0", 60000, inert),
        (664018, "distance_diff_rate>=0", 60000, inert),
        (664019, "distance_diff_top_float>=0", 60000, inert),
        (664020, "temptation_count_behind>=0", 60000, inert),
        (664021, "temptation_count_infront>=0", 60000, inert),
        (664022, "running_style_temptation_count_sashi>=0", 60000, inert),
        (664023, "running_style_temptation_count_nige>=0", 60000, inert),
        (664025, "is_activate_other_skill_detail==1", 60000, inert),
    ];
    let band_cells: [i64; 4] = [664024, 664026, 664027, 664028];
    // The leader seat (horse 0) owns only the top-band cell; the back seat
    // (horse 5) runs everything else so the front-blocker and surrounded
    // conditions can actually hold.
    let lead: Vec<i64> = vec![664024];
    let back: Vec<i64> = battery
        .iter()
        .map(|(sid, ..)| *sid)
        .chain(band_cells[1..].iter().copied())
        .collect();
    {
        let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
        for (sid, cond, cd, effects) in battery {
            sd.insert(sid.to_string(), make_def(cond, cd, effects));
        }
        for (sid, rate) in band_cells.iter().zip([20i64, 40, 50, 80]) {
            let cond = format!("order_rate_in{rate}_continue==1");
            sd.insert((*sid).to_string(), make_def(&cond, 60000, inert));
        }
    }

    let results = patched_requests(
        &skill_data,
        &[(0, Some(&lead), None, None), (5, Some(&back), None, None)],
        &SEEDS,
    );
    let out = race_exec::run(&results).expect("kernel run");

    let (fired, mut problems) = fired_ids(&results, &out);
    // blocked_front and is_surrounded on the back seat: the 200401 real row
    // shows a surrounded pack in this replay, and a mid-pack horse is blocked
    // from the opening phase on.
    for sid in [664001, 664002] {
        if !fired.contains(&sid) {
            problems.push(format!("{sid} never activated on the back seat"));
        }
    }
    for sid in 664003..=664028 {
        if sid == 664025 {
            continue;
        }
        if sid == 664024 {
            continue;
        }
        if !fired.contains(&sid) {
            problems.push(format!("{sid} never activated"));
        }
    }
    // The used-id leaf only reaches its scan branch through a firing record,
    // and 664005 is such a record: is_used_skill_id saw the sibling's bit.
    if !fired.contains(&664005) {
        problems.push("664005 is_used_skill_id did not run a firing record".into());
    }
    // The bounded band cell needs the horse's rank inside the top quintile,
    // which the leader seat satisfies.
    if !fired.contains(&664024) {
        problems.push("664024 order_rate_in20_continue did not reach the leader seat".into());
    }
    if fired.contains(&664025) {
        problems.push("664025 is_activate_other_skill_detail==1 fired on the CPU".into());
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(&format!("battery seed {seed}"), request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} wave3 battery problems, fired {:?}, problems:\n{}",
            problems.len(),
            fired,
            problems.join("\n")
        );
    }
}

