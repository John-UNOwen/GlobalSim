//! Engine parity audit. The wave gates pin individual capabilities one scene
//! at a time. This file sweeps the whole surface the app can send: the real
//! comparison traffic (the app-preset mob jobs and the mob/uma roster shapes)
//! on both lanes, compact and full; every skill id in the merged table
//! through the retail and classic compilers; and the exact set of fallback
//! reasons `facade::gate_reasons` can still name.

#![cfg(feature = "race")]

use globalsim_gpu::{
    classic_exec, dep_cpu, facade, json, race_exec, skills_compile,
};
use json::Value;
use std::collections::BTreeSet;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

fn merged_skill_data() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn v77_base() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

fn accepted() -> facade::DeviceFacts {
    facade::DeviceFacts {
        extensions: "cl_khr_fp64".into(),
        double_fp_config: 0x81,
        single_fp_config: 0x81,
    }
}

fn engine_oracle(request: &str, seed: i64) -> String {
    let mut prepared = globalsim::PreparedRace::new(request).expect("engine accepts the request");
    let simulation = prepared.run(seed).expect("engine runs the race");
    let mut reply = String::new();
    simulation.write_json(&mut reply);
    reply
}

fn encode(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// An app-preset mob job turned into a runnable retail request.
fn preset_request(job: &str, seed: i64, skill_data: &Value) -> String {
    let mut value = json::parse(job).expect("job parses");
    let Value::Object(map) = &mut value else { panic!("job is not a JSON object") };
    map.insert("seed".into(), Value::Number(seed as f64));
    map.insert("skillData".into(), skill_data.clone());
    encode(&value)
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
        .map(|job| encode(job.get("request").expect("job has a request")))
        .collect()
}

/// The v77 field with the four retail flags forced off (the classic lane).
fn classic_request(seed: i64, course: i32, horses: Option<Vec<Value>>) -> String {
    let mut value = v77_base();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), Value::Bool(false));
        }
        map.insert("courseId".into(), Value::Number(course as f64));
        if let Some(horses) = horses {
            map.insert("horses".into(), json::arr(horses));
        }
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    encode(&value)
}

/// One runner in the app adapter's shape, mirroring zero_skill_gate.rs.
fn horse(strategy: &str, stats: [f64; 5], dist: &str, mood: i64, gate: usize) -> Value {
    json::obj(vec![
        ("speed", Value::Number(stats[0])),
        ("stamina", Value::Number(stats[1])),
        ("power", Value::Number(stats[2])),
        ("guts", Value::Number(stats[3])),
        ("wisdom", Value::Number(stats[4])),
        ("strategy", Value::String(strategy.into())),
        ("distanceAptitude", Value::String(dist.into())),
        ("surfaceAptitude", Value::String("A".into())),
        ("strategyAptitude", Value::String("A".into())),
        ("mood", Value::Number(mood as f64)),
        ("skills", json::arr(vec![])),
        ("gateIndex", Value::Number(gate as f64)),
    ])
}

fn mob(strategy: &str, gate: usize, spread: (f64, f64)) -> Value {
    horse(strategy, [1600. + spread.0, 1200., 1200., 600., 1200. + spread.1], "S", 0, gate)
}

fn uma(strategy: &str, gate: usize) -> Value {
    horse(strategy, [1000., 1000., 900., 800., 900.], "A", 2, gate)
}

fn mob8() -> Vec<Value> {
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let spreads = [(0., 0.), (-60., 0.), (0., -80.), (0., 0.), (-120., 40.), (0., 0.), (60., 0.), (-60., -80.)];
    let mut out = Vec::new();
    let mut gate = 0usize;
    for (i, style) in styles.iter().enumerate() {
        if gate == 3 {
            gate += 1;
        }
        out.push(mob(style, gate, spreads[i]));
        gate += 1;
    }
    out
}

fn mob18() -> Vec<Value> {
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let mut out = Vec::new();
    let mut gate = 0usize;
    for round in 0..2 {
        for i in 0..8 {
            if gate == 3 || gate == 12 {
                out.push(uma("Senkou", gate));
                gate += 1;
            }
            let spread = if round == 0 { (0., 0.) } else { (-120., 40.) };
            out.push(mob(styles[i], gate, spread));
            gate += 1;
        }
    }
    if gate < 18 {
        out.push(uma("Senkou", gate));
    }
    out.truncate(18);
    out
}

/// Frame-for-frame bit equality against the in-process engine, reported as a
/// problem list so one scene's failure cannot hide another's. The engine's
/// six columns are compared; wider kernel rows carry extra channels.
fn compare_frames<const N: usize>(
    problems: &mut Vec<String>,
    tag: &str,
    request: &str,
    seed: i64,
    gpu_frames: &[Vec<[f64; N]>],
) {
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: engine race failed: {reason}"));
            return;
        }
    };
    if cpu.frames.len() != gpu_frames.len() {
        problems.push(format!("{tag}: tick count engine {} gpu {}", cpu.frames.len(), gpu_frames.len()));
        return;
    }
    for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(gpu_frames).enumerate() {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{tag}: tick {tick} runner {runner} column {column} engine {cpu:?} gpu {gpu:?}"
                    ));
                    return;
                }
            }
        }
    }
}

/// The app's whole comparison traffic compact: the mob jobs with their uma
/// skill sets on the retail lane, the v77, mob8 and mob18 rosters on the
/// classic lane, courses 10504 and 10604, seeds 1 to 3. Zero fallbacks, and
/// every frame bit-equal to the in-process engine.
#[test]
fn app_traffic_answers_compact_bit_exact_on_both_lanes() {
    for course in [10504, 10604] {
        register_geometry(course);
    }
    let skill_data = merged_skill_data();
    let mut problems: Vec<String> = Vec::new();
    let mut scenes: Vec<(String, String, i64)> = Vec::new();
    for course in [10504, 10604] {
        for (job, text) in load_jobs(course).into_iter().enumerate() {
            for seed in [1i64, 2, 3] {
                let request = preset_request(&text, seed, &skill_data);
                scenes.push((format!("mobjob {course}#{job} seed {seed} (retail)"), request, seed));
            }
        }
    }
    for (roster_name, roster) in [
        ("v77", None),
        ("mob8", Some(mob8())),
        ("mob18", Some(mob18())),
    ] {
        for course in [10504, 10604] {
            for seed in [1i64, 2, 3] {
                let request = classic_request(seed, course, roster.clone());
                scenes.push((format!("{roster_name} seed {seed} course {course} (classic)"), request, seed));
            }
        }
    }
    let (classic, retail): (Vec<&(String, String, i64)>, Vec<&(String, String, i64)>) =
        scenes.iter().partition(|(_, request, _)| {
            json::parse(request).map_or(false, |root| {
                ![
                    "experimentalRetailInitialization",
                    "experimentalRetailSections",
                    "experimentalRetailRuntime",
                    "experimentalRetailGeometry",
                ]
                .iter()
                .any(|key| root.get(*key).and_then(Value::as_bool) == Some(true))
            })
        });
    let requests: Vec<(String, i64)> = retail.iter().map(|(_, r, s)| (r.clone(), *s)).collect();
    if !requests.is_empty() {
        let run = race_exec::run(&requests).expect("retail batch");
        for ((tag, request, seed), race) in retail.iter().zip(&run.races) {
            if let Some(reason) = &race.fallback {
                problems.push(format!("{tag}: fallback {reason}"));
                continue;
            }
            compare_frames(&mut problems, tag, request, *seed, &race.frames);
        }
    }
    let requests: Vec<(String, i64)> = classic.iter().map(|(_, r, s)| (r.clone(), *s)).collect();
    if !requests.is_empty() {
        let run = classic_exec::run(&requests).expect("classic batch");
        for ((tag, request, seed), race) in classic.iter().zip(&run.races) {
            if let Some(reason) = &race.fallback {
                problems.push(format!("{tag}: fallback {reason}"));
                continue;
            }
            compare_frames(&mut problems, tag, request, *seed, &race.frames);
        }
    }
    assert!(problems.is_empty(), "{} scene(s) failed:\n{}", problems.len(), problems.join("\n"));
}

/// The same traffic on the full reply lane: one mixed-lane batch through
/// `facade::GpuRaces::run_batch_full`, every reply byte-equal to the
/// in-process engine's reply, zero fallbacks.
#[test]
fn app_traffic_answers_full_replies_byte_equal() {
    for course in [10504, 10604] {
        register_geometry(course);
    }
    let skill_data = merged_skill_data();
    let mut requests: Vec<(String, String, i64)> = Vec::new();
    for course in [10504, 10604] {
        for (job, text) in load_jobs(course).into_iter().enumerate() {
            for seed in [1i64, 2, 3] {
                requests.push((format!("mobjob {course}#{job} seed {seed} (retail full)"), preset_request(&text, seed, &skill_data), seed));
            }
        }
    }
    for (roster_name, roster) in [
        ("v77", None),
        ("mob8", Some(mob8())),
        ("mob18", Some(mob18())),
    ] {
        for course in [10504, 10604] {
            for seed in [1i64, 2, 3] {
                requests.push((format!("{roster_name} seed {seed} course {course} (classic full)"), classic_request(seed, course, roster.clone()), seed));
            }
        }
    }
    let mut gpu = facade::GpuRaces::new().expect("device");
    let texts: Vec<String> = requests.iter().map(|(_, r, _)| r.clone()).collect();
    let seeds: Vec<i32> = requests.iter().map(|(_, _, s)| *s as i32).collect();
    let outcomes = gpu.run_batch_full(&texts, &seeds).expect("full batch");
    let mut problems: Vec<String> = Vec::new();
    for ((tag, request, seed), outcome) in requests.iter().zip(&outcomes) {
        if let Some(reason) = &outcome.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let reply = outcome.reply.as_ref().expect("a launchable race carries a reply");
        let engine = engine_oracle(request, *seed);
        if *reply != engine {
            let at = reply.bytes().zip(engine.bytes()).take_while(|(a, b)| a == b).count();
            problems.push(format!("{tag}: reply differs at byte {at}: engine {:?} gpu {:?}",
                &engine[at..at + 40.min(engine.len() - at)], &reply[at..at + 40.min(reply.len() - at)]));
        }
    }
    assert!(problems.is_empty(), "{} scene(s) failed:\n{}", problems.len(), problems.join("\n"));
}

/// Skill ids the retail compiler refuses, normalized to the family the
/// refusal names: the skill id is dropped, and a condition refusal keeps
/// only the missing leaf behind the colon. 688 of the 751 ids compile.
const RETAIL_REFUSALS: [&str; 17] = [
    "'is_badstart'==0 is not ported",
    "'is_basis_distance'==0 is not ported",
    "'is_basis_distance'==1 is not ported",
    "'is_exist_chara_id'==1002 is not ported",
    "'is_other_character_activate_advantage_skill' needs the field's team-filtered previous-tick advantage pair set",
    "'last_straight_random'==1 is not ported",
    "'lastspurt'==2 is not ported",
    "'random_lot'==15 is not ported",
    "'random_lot'==30 is not ported",
    "'random_lot'==50 is not ported",
    "'straight_front_type'==2 is not ported",
    "PowerUp is not ported to the retail kernel",
    "condition has more than 4 arms",
    "effect target 18 is not ported",
    "effect target 19 is not ported",
    "effect target 20 is not ported",
    "has no alternatives",
];

/// Drop the per-id prefix and the condition text so a refusal maps to its
/// documented family: "skill 200131 condition '...': 'x'==1 is not ported"
/// and "skill 200141 condition '...': 'x'==0 is not ported" both normalize
/// to their leaf clause, and "skill 200151 PowerUp is not ported..." to the
/// bare sentence.
fn refusal_family(id: &str, reason: &str) -> String {
    let mut family = reason.to_string();
    let prefix = format!("skill {id} ");
    if let Some(stripped) = family.strip_prefix(&prefix) {
        family = stripped.to_string();
    }
    if let Some((_, tail)) = family.rsplit_once("': ") {
        family = tail.to_string();
    }
    family
}

/// The retail compiler over every id in the merged table: a v77 roster with
/// the id on runner 0 must either compile into words or refuse with one of
/// the documented reasons. Any other reason, and any panic, fails the audit.
#[test]
fn every_merged_skill_compiles_or_refuses_by_name_retail() {
    let skill_data = merged_skill_data();
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    let ids: Vec<String> = skill_data
        .as_object()
        .expect("skills.json is an object")
        .keys()
        .cloned()
        .collect();
    assert_eq!(ids.len(), 751, "the merged table's id count");
    let allow: BTreeSet<&str> = RETAIL_REFUSALS.iter().copied().collect();
    let mut compiled = 0usize;
    let mut refusals: BTreeSet<String> = BTreeSet::new();
    let mut unexpected: Vec<String> = Vec::new();
    for id in &ids {
        let mut root = v77_base();
        {
            let Value::Object(map) = &mut root else { unreachable!() };
            map.insert("skillData".into(), skill_data.clone());
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(row) = &mut horses[0] else { unreachable!() };
            row.insert("skills".into(), json::arr(vec![Value::String(id.clone())]));
        }
        let plan = skills_compile::plan(&root, &course);
        match plan.reason {
            None => compiled += 1,
            Some(reason) => {
                let family = refusal_family(id, &reason);
                if !allow.contains(family.as_str()) {
                    unexpected.push(format!("skill {id}: {reason}"));
                }
                refusals.insert(family);
            }
        }
    }
    println!("retail sweep: {compiled} compiled, {} distinct refusals", refusals.len());
    for reason in &refusals {
        println!("  refused: {reason}");
    }
    assert!(unexpected.is_empty(), "undocumented retail refusals:\n{}", unexpected.join("\n"));
}

/// Classic counterparts of RETAIL_REFUSALS, same normalization. 445 of the
/// 751 ids compile.
const CLASSIC_REFUSALS: [&str; 54] = [
    "'is_badstart'==0 is not ported",
    "'is_basis_distance'==0 is not ported",
    "'is_basis_distance'==1 is not ported",
    "'is_exist_chara_id'==1002 is not ported",
    "'is_other_character_activate_advantage_skill' needs the field's team-filtered previous-tick advantage pair set",
    "'lastspurt'==2 is not ported",
    "'random_lot'==15 is not ported",
    "'random_lot'==30 is not ported",
    "'random_lot'==50 is not ported",
    "'straight_front_type'==2 is not ported",
    "condition has more than 4 arms",
    "condition term 'activate_count_all' reads live runtime state with no reader in the classic kernel",
    "condition term 'activate_count_end_after' reads live runtime state with no reader in the classic kernel",
    "condition term 'activate_count_heal' reads live runtime state with no reader in the classic kernel",
    "condition term 'activate_count_later_half' reads live runtime state with no reader in the classic kernel",
    "condition term 'activate_count_middle' reads live runtime state with no reader in the classic kernel",
    "condition term 'activate_count_start' reads live runtime state with no reader in the classic kernel",
    "condition term 'bashin_diff_behind' reads live runtime state with no reader in the classic kernel",
    "condition term 'bashin_diff_infront' reads live runtime state with no reader in the classic kernel",
    "condition term 'behind_near_lane_time' reads live runtime state with no reader in the classic kernel",
    "condition term 'blocked_front' reads live runtime state with no reader in the classic kernel",
    "condition term 'blocked_front_continuetime' reads live runtime state with no reader in the classic kernel",
    "condition term 'change_order_onetime' reads live runtime state with no reader in the classic kernel",
    "condition term 'change_order_up_end_after' reads live runtime state with no reader in the classic kernel",
    "condition term 'change_order_up_finalcorner_after' reads live runtime state with no reader in the classic kernel",
    "condition term 'compete_fight_count' reads live runtime state with no reader in the classic kernel",
    "condition term 'distance_diff_rate' reads live runtime state with no reader in the classic kernel",
    "condition term 'distance_diff_top' reads live runtime state with no reader in the classic kernel",
    "condition term 'infront_near_lane_time' reads live runtime state with no reader in the classic kernel",
    "condition term 'is_activate_any_skill' reads live runtime state with no reader in the classic kernel",
    "condition term 'is_behind_in' reads live runtime state with no reader in the classic kernel",
    "condition term 'is_lastspurt' reads live runtime state with no reader in the classic kernel",
    "condition term 'is_move_lane' reads live runtime state with no reader in the classic kernel",
    "condition term 'is_overtake' reads live runtime state with no reader in the classic kernel",
    "condition term 'lane_type' reads live runtime state with no reader in the classic kernel",
    "condition term 'order' reads live runtime state with no reader in the classic kernel",
    "condition term 'order_rate' reads live runtime state with no reader in the classic kernel",
    "condition term 'order_rate_in20_continue' reads live runtime state with no reader in the classic kernel",
    "condition term 'order_rate_in50_continue' reads live runtime state with no reader in the classic kernel",
    "condition term 'order_rate_out50_continue' reads live runtime state with no reader in the classic kernel",
    "condition term 'order_rate_out70_continue' reads live runtime state with no reader in the classic kernel",
    "condition term 'overtake_target_time' reads live runtime state with no reader in the classic kernel",
    "condition term 'running_style_temptation_opponent_count_nige' reads live runtime state with no reader in the classic kernel",
    "condition term 'running_style_temptation_opponent_count_oikomi' reads live runtime state with no reader in the classic kernel",
    "condition term 'running_style_temptation_opponent_count_sashi' reads live runtime state with no reader in the classic kernel",
    "condition term 'running_style_temptation_opponent_count_senko' reads live runtime state with no reader in the classic kernel",
    "condition term 'temptation_count' reads live runtime state with no reader in the classic kernel",
    "condition term 'temptation_opponent_count_behind' reads live runtime state with no reader in the classic kernel",
    "condition term 'temptation_opponent_count_infront' reads live runtime state with no reader in the classic kernel",
    "effect target 18 is not ported",
    "has no alternatives",
    "type 37 is not ported to the classic runtime",
    "value_usage 14 draws live state at activation, which the classic runtime does not model",
    "value_usage 8 draws live state at activation, which the classic runtime does not model",
];

/// The classic compiler over every id in the merged table, with the host
/// extras a classic race carries (adjusted raw power and resolved strategy per
/// runner). Same rule as the retail sweep: compile or refuse by a documented
/// name.
#[test]
fn every_merged_skill_compiles_or_refuses_by_name_classic() {
    let skill_data = merged_skill_data();
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").expect("course data");
    let course = courses.get("10504").expect("course 10504").clone();
    let ids: Vec<String> = skill_data
        .as_object()
        .expect("skills.json is an object")
        .keys()
        .cloned()
        .collect();
    let allow: BTreeSet<&str> = CLASSIC_REFUSALS.iter().copied().collect();
    let extras = skills_compile::ClassicExtras {
        raw_power: vec![1200.0; 9],
        strategy: vec![1; 9],
    };
    let mut compiled = 0usize;
    let mut refusals: BTreeSet<String> = BTreeSet::new();
    let mut unexpected: Vec<String> = Vec::new();
    for id in &ids {
        let mut root = v77_base();
        {
            let Value::Object(map) = &mut root else { unreachable!() };
            for key in [
                "experimentalRetailInitialization",
                "experimentalRetailSections",
                "experimentalRetailRuntime",
                "experimentalRetailGeometry",
            ] {
                map.insert(key.into(), Value::Bool(false));
            }
            map.insert("skillData".into(), skill_data.clone());
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let Value::Object(row) = &mut horses[0] else { unreachable!() };
            row.insert("skills".into(), json::arr(vec![Value::String(id.clone())]));
        }
        let plan = skills_compile::plan_classic_with(&root, &course, &extras);
        match plan.reason {
            None => compiled += 1,
            Some(reason) => {
                let family = refusal_family(id, &reason);
                if !allow.contains(family.as_str()) {
                    unexpected.push(format!("skill {id}: {reason}"));
                }
                refusals.insert(family);
            }
        }
    }
    println!("classic sweep: {compiled} compiled, {} distinct refusals", refusals.len());
    for reason in &refusals {
        println!("  refused: {reason}");
    }
    assert!(unexpected.is_empty(), "undocumented classic refusals:\n{}", unexpected.join("\n"));
}

/// The exact fallback set the app-facing gate can still name for wave 13b.
/// Every documented reason must be reachable by one probe, and the probes
/// must name nothing else.
#[test]
fn gate_reasons_enumerates_exactly_the_documented_fallbacks() {
    let no_fp = facade::DeviceFacts {
        extensions: String::new(),
        single_fp_config: 0,
        double_fp_config: 0,
    };
    let grid20 = r#"{"columns":20,"rowSpacing":3}"#;
    let twenty_one = || vec!["{}"; 21].join(",");
    let classic_min = r#"{"courseId":10504,"horses":[{}]}"#;
    let probes: Vec<(&str, String, Option<facade::DeviceFacts>)> = vec![
        ("no device", classic_min.into(), None),
        ("device without fp64", classic_min.into(), Some(no_fp)),
        (
            "grid shape",
            format!(r#"{{"courseId":10504,"experimentalGrid":{grid20},"horses":[{{}}]}}"#),
            Some(accepted()),
        ),
        (
            "later-stamina not a boolean",
            format!(r#"{{"courseId":10504,"horses":[{{}}],"experimentalLaterStaminaMechanics":1}}"#),
            Some(accepted()),
        ),
        (
            "trace not a boolean",
            format!(r#"{{"courseId":10504,"horses":[{{}}],"traceConservePower":"yes"}}"#),
            Some(accepted()),
        ),
        (
            "startup not a boolean",
            format!(r#"{{"courseId":10504,"horses":[{{}}],"experimentalRetailStartup":1}}"#),
            Some(accepted()),
        ),
        (
            "classic retail runtime",
            r#"{"courseId":10504,"horses":[{}],"retailRuntime":true}"#.into(),
            Some(accepted()),
        ),
        (
            "classic trace",
            r#"{"courseId":10504,"traceConservePower":true,"horses":[{}]}"#.into(),
            Some(accepted()),
        ),
        (
            "classic later stamina above 2100 m",
            r#"{"courseId":10108,"experimentalLaterStaminaMechanics":true,"horses":[{}]}"#.into(),
            Some(accepted()),
        ),
        (
            "classic unknown skill",
            {
                let mut request = json::parse(&classic_request(1, 10504, None)).expect("classic v77");
                let Value::Object(map) = &mut request else { unreachable!() };
                map.insert("skillData".into(), merged_skill_data());
                let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
                let Value::Object(row) = &mut horses[0] else { unreachable!() };
                row.insert("skills".into(), json::arr(vec![Value::String("999999999".into())]));
                encode(&request)
            },
            Some(accepted()),
        ),
        (
            "classic roster past 120",
            format!(r#"{{"courseId":10504,"horses":[{}]}}"#, vec!["{}"; 121].join(",")),
            Some(accepted()),
        ),
        (
            "retail lane non-retail",
            r#"{"courseId":10504,"horses":[{}],"experimentalRetailInitialization":true}"#.into(),
            Some(accepted()),
        ),
        (
            "retail grid with geometry",
            format!(r#"{{"courseId":10504,"experimentalGrid":{grid20},"experimentalRetailRuntime":true,"experimentalRetailGeometry":true,"horses":[{}]}}"#, twenty_one()),
            Some(accepted()),
        ),
        (
            "retail later stamina",
            format!(r#"{{"courseId":10504,"experimentalRetailRuntime":true,"experimentalLaterStaminaMechanics":true,"horses":[{}]}}"#, twenty_one()),
            Some(accepted()),
        ),
        (
            "retail roster past 144",
            format!(r#"{{"courseId":10504,"experimentalRetailRuntime":true,"horses":[{}]}}"#, vec!["{}"; 145].join(",")),
            Some(accepted()),
        ),
    ];
    let documented: BTreeSet<String> = [
        "no GPU passes the exactness gate",
        "missing cl_khr_fp64, missing CL_FP_DENORM in single_fp_config, missing CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, missing CL_FP_DENORM in double_fp_config",
        "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0.",
        "experimentalLaterStaminaMechanics must be a boolean.",
        "traceConservePower must be a boolean",
        "experimentalRetailStartup must be a boolean",
        "retail runtime belongs to the retail lane",
        "traceConservePower requires experimentalRetailRuntime",
        "staminasyoubu, injected above 2100 m, stays outside the classic kernel",
        "skillData has no entry for skill 999999999",
        "roster size outside the kernel's range",
        "non-retail runtime",
        "World geometry requires the experimental runtime and standard starting grid",
        "experimentalLaterStaminaMechanics draws from a live solver stream the retail kernel resolves host-side",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    let mut named: BTreeSet<String> = BTreeSet::new();
    let mut problems: Vec<String> = Vec::new();
    for (label, request, facts) in &probes {
        match facade::gate_reasons(request, facts.as_ref()) {
            None => problems.push(format!("{label}: the gate answered None, every probe must fall back")),
            Some(reason) => {
                if !documented.contains(&reason) {
                    problems.push(format!("{label}: undocumented reason {reason:?}"));
                }
                named.insert(reason);
            }
        }
    }
    for reason in &documented {
        if !named.contains(reason) {
            problems.push(format!("documented fallback never reached by a probe: {reason:?}"));
        }
    }
    assert!(problems.is_empty(), "gate_reasons enumeration:\n{}", problems.join("\n"));
}

/// The two prepare-only classic fallbacks that the real course table cannot
/// trigger stay reachable through the injected course text.
#[test]
fn classic_prepare_only_fallbacks_stay_reachable() {
    let request = classic_request(1, 99999, None);
    let many_slopes: Vec<String> = (0..=classic_exec::CLS_MAX_SLOPES)
        .map(|i| format!(r#"{{"start":{:.1},"length":5.0,"slope":0.0}}"#, i as f64 * 10.))
        .collect();
    let courses = format!(
        r#"{{"99999":{{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"straights":[{{"start":0.0,"end":2000.0}}],"corners":[],"slopes":[{}]}}}}"#,
        many_slopes.join(",")
    );
    let run = classic_exec::run_with_courses_text(&[(request.clone(), 1i64)], &courses)
        .expect("classic run");
    assert_eq!(
        run.races[0].fallback.as_deref(),
        Some("more slopes than the classic kernel carries"),
        "a course over the slope cap falls back by name"
    );
}
