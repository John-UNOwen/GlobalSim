//! Wave-10 full-reply folds: the gate proves the bytes equal the engine's,
//! these folds prove the bytes come from the kernel. Mutating one channel of
//! a decoded race moves exactly the fields decoded from that channel, and
//! every channel carries live records in one of the gated scenarios.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::race_exec::RaceGpu;
use globalsim_gpu::{full_reply, json, race_exec};

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn merged_skill_data() -> Value {
    let path = repo_file("../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string();
    json::parse(&text).expect("v77 parses")
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
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

fn with_seed(mut value: Value, extras: &[(&str, Value)]) -> String {
    let json::Value::Object(map) = &mut value else { unreachable!() };
    for (key, item) in extras {
        map.insert((*key).into(), item.clone());
    }
    map.insert("seed".into(), json::Value::Number(1.));
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn set_skills(value: &mut Value, index: usize, ids: &[i64]) {
    let json::Value::Object(map) = value else { unreachable!() };
    let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
        unreachable!()
    };
    let json::Value::Object(horse) = &mut entries[index] else { unreachable!() };
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

/// The wave-9 fan battery on the merged table, the gate's scenario.
fn battery_value() -> Value {
    let mut value = v77_value();
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        let mut skill_data = merged_skill_data();
        {
            let json::Value::Object(sd) = &mut skill_data else { unreachable!() };
            for (sid, def) in [
                (
                    "669201",
                    json_battery(
                        "phase>=2",
                        5000000,
                        20000,
                        r#"{"type":27,"target":9,"targetValue":3,"modifier":-1000,"additionalActivate":1}"#,
                    ),
                ),
                (
                    "669202",
                    json_battery(
                        "phase>=1",
                        5000000,
                        30000,
                        r#"{"type":31,"target":10,"targetValue":4,"modifier":800,"additionalActivate":2}"#,
                    ),
                ),
                (
                    "669203",
                    json_battery(
                        "phase>=2",
                        5000000,
                        25000,
                        r#"{"type":28,"target":4,"targetValue":2,"modifier":-500,"additionalActivate":3}"#,
                    ),
                ),
                (
                    "669205",
                    json_battery(
                        "phase>=2",
                        5000000,
                        20000,
                        r#"{"type":21,"target":1,"modifier":500,"additionalActivate":2}"#,
                    ),
                ),
                (
                    "669102",
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
                    "669141",
                    json_battery("phase>=1", 5000000, 20000, r#"{"type":31,"target":1,"modifier":600}"#),
                ),
                (
                    "669206",
                    json_battery(
                        "phase>=1",
                        5000000,
                        25000,
                        concat!(
                            r#"{"type":31,"target":1,"modifier":500},"#,
                            r#"{"type":27,"target":4,"targetValue":2,"modifier":-500}"#
                        ),
                    ),
                ),
            ] {
                sd.insert(sid.into(), json::parse(&def).expect("def parses"));
            }
        }
        map.insert("skillData".into(), skill_data);
    }
    for (index, ids) in [
        (0usize, &[100531i64, 669141][..]),
        (2, &[900531, 669141, 669206][..]),
        (1, &[669102, 669141][..]),
        (3, &[669201][..]),
        (4, &[669202][..]),
        (5, &[669203][..]),
        (6, &[669205][..]),
        (7, &[669201][..]),
    ] {
        set_skills(&mut value, index, ids);
    }
    value
}

/// The competition-reorder roster of the gate, seeds aside.
fn reorder_value() -> Value {
    let mut value = v77_value();
    for (index, ids, strategy, popularity) in [
        (0usize, &[][..], "OONIGE", 3i64),
        (1, &[][..], "NIGE", 9),
        (2, &[][..], "NIGE", 1),
        (3, &[][..], "SENKOU", 5),
        (4, &[][..], "OONIGE", 7),
        (5, &[][..], "OIKOMI", 2),
        (6, &[][..], "SASI", 8),
        (7, &[][..], "NIGE", 4),
    ] {
        set_skills(&mut value, index, ids);
        let json::Value::Object(map) = &mut value else { unreachable!() };
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        let json::Value::Object(horse) = &mut entries[index] else { unreachable!() };
        horse.insert("strategy".into(), Value::String(strategy.into()));
        horse.insert("popularity".into(), json::Value::Number(popularity as f64));
    }
    value
}

/// Horse 0 above the 1200 asitame power gate, nothing else changed.
fn asitame_value() -> Value {
    let mut value = v77_value();
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        let json::Value::Object(horse) = &mut entries[0] else { unreachable!() };
        horse.insert("power".into(), json::Value::Number(1500.));
    }
    value
}

fn run_race(request: &str) -> RaceGpu {
    let run = race_exec::run_full(&[(request.to_string(), 1i64)]).expect("kernel run");
    let race = run.races.into_iter().next().expect("one race");
    if let Some(reason) = &race.fallback {
        panic!("race fell back: {reason}");
    }
    race
}

fn gpu_value(request: &str, race: &RaceGpu) -> Value {
    let text = full_reply::build(request, 1, race).expect("full reply builds");
    json::parse(&text).expect("gpu reply parses")
}

fn array<'a>(root: &'a Value, key: &str) -> &'a [Value] {
    root.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_else(|| panic!("{key} is an array"))
}

fn array_text(root: &Value, key: &str) -> String {
    json::to_string(root.get(key).unwrap_or_else(|| panic!("{key} present")))
}

fn f64_field(item: &Value, key: &str) -> f64 {
    item.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{key} is a number"))
}

#[test]
fn activation_and_kakari_streams_answer_skill_events() {
    register_geometry(10504);
    let request = with_seed(battery_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert!(raw.act.len() >= 20 && raw.act.len() % 10 == 0, "ten-word activations");
    assert!(raw.kak.len() >= 8 && raw.kak.len() % 4 == 0, "start and end rush edges");
    let base = gpu_value(&request, &race);
    let events = array(&base, "skillEvents");
    let pending = events.iter().filter(|e| f64_field(e, "duration") > 0.0).count();
    let immediate = events.iter().filter(|e| f64_field(e, "duration") == 0.0).count();
    let kakari = events
        .iter()
        .filter(|e| e.get("skillId").and_then(Value::as_str) == Some("kakari"))
        .count();
    assert!(pending > 0 && immediate > 0, "both duration classes present");
    assert!(kakari > 0, "finished rush episodes reach the events");

    // Without the activation stream only the kakari episodes survive.
    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").act.clear();
    let without = gpu_value(&request, &patched);
    let kept = array(&without, "skillEvents");
    assert_eq!(kept.len(), kakari, "the act stream answers the rest");

    // Without the rush edges only the activation events survive.
    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").kak.clear();
    let without = gpu_value(&request, &patched);
    let kept = array(&without, "skillEvents");
    assert_eq!(kept.len(), events.len() - kakari, "the kak stream answers the rushes");
}

#[test]
fn target_and_fan_streams_split_their_arrays() {
    register_geometry(10504);
    let request = with_seed(battery_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert!(!raw.tgt.is_empty() && !raw.fan.is_empty(), "both streams carry records");
    let base = gpu_value(&request, &race);
    let targets = array(&base, "skillTargetEvents");
    // The engine leaves only unselected Noop and Runaway effects null
    // (race_solver.rs:2479-2483), so a selected list is the live shape here.
    let mut nulls = 0usize;
    let mut lists = 0usize;
    for event in targets {
        if let Some(Value::Array(selections)) = event.get("abilityTargets") {
            nulls += selections.iter().filter(|s| matches!(**s, Value::Null)).count();
            lists += selections.iter().filter(|s| matches!(**s, Value::Array(_))).count();
        }
    }
    assert!(lists > 0, "target events carry selections (null {nulls} list {lists})");
    assert!(!array(&base, "skillDetailEvents").is_empty(), "detail events publish");
    let additional = array(&base, "additionalTargetEvents");
    assert!(additional.iter().all(|e| e
        .get("abilityIndexScope")
        .and_then(Value::as_str)
        == Some("runtime_subset")), "fan events carry their scope tag");
    assert!(!additional.is_empty(), "the fan stream publishes");

    // The detail and target arrays share the tgt stream and only it.
    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").tgt.clear();
    let without = gpu_value(&request, &patched);
    assert!(array(&without, "skillDetailEvents").is_empty(), "detail events come from tgt");
    assert!(array(&without, "skillTargetEvents").is_empty(), "target events come from tgt");
    assert_eq!(
        array(&without, "additionalTargetEvents").len(),
        additional.len(),
        "the fan stream is independent of tgt"
    );

    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").fan.clear();
    let without = gpu_value(&request, &patched);
    assert!(array(&without, "additionalTargetEvents").is_empty(), "fan events come from fan");
    assert_eq!(array(&without, "skillTargetEvents").len(), targets.len(), "tgt is independent");
}

#[test]
fn conserve_stream_follows_the_asitame_power_gate() {
    register_geometry(10504);
    // The default roster sits under the 1200 power gate and stays silent.
    let request = with_seed(v77_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert!(raw.cons.is_empty(), "no release under the power gate");
    assert!(array(&gpu_value(&request, &race), "conservePowerEvents").is_empty());

    // One horse above the gate and the release reaches the array.
    let request = with_seed(asitame_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert_eq!(raw.cons.len() % 3, 0, "runner plus clock word pairs");
    let base = gpu_value(&request, &race);
    let events = array(&base, "conservePowerEvents");
    assert_eq!(events.len(), raw.cons.len() / 3, "one record per release");
    assert_eq!(f64_field(&events[0], "horseIndex"), 0.0, "the patched horse releases");

    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").cons.clear();
    let without = gpu_value(&request, &patched);
    assert!(array(&without, "conservePowerEvents").is_empty(), "the cons stream answers it");
}

#[test]
fn competition_stream_answers_the_reordered_array() {
    register_geometry(10504);
    let request = with_seed(reorder_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert!(raw.comp.len() % 3 == 0 && !raw.comp.is_empty(), "tick kind runner triples");
    let mut kinds: Vec<u32> = raw.comp.chunks(3).map(|rec| rec[1]).collect();
    kinds.sort_unstable();
    kinds.dedup();
    assert!(kinds.contains(&4) && kinds.contains(&5), "spot and duel kinds both fire");
    let base = gpu_value(&request, &race);
    assert_eq!(array(&base, "competitionEvents").len(), raw.comp.len() / 3);

    // A runner word of a spot record, fed a roster index the stream never
    // used, appears as a horseIndex: the array answers the stream words.
    let has_horse = |root: &Value, horse: f64| {
        array(root, "competitionEvents")
            .iter()
            .any(|e| f64_field(e, "horseIndex") == horse)
    };
    assert!(!has_horse(&base, 3.0));
    let mut patched = race.clone();
    {
        let raw = patched.reply.as_mut().expect("raw");
        let first = raw
            .comp
            .chunks(3)
            .position(|rec| rec[1] == 4)
            .expect("a spot record");
        raw.comp[3 * first + 2] = 3;
    }
    let moved = gpu_value(&request, &patched);
    assert!(has_horse(&moved, 3.0), "the runner word answers the horseIndex column");

    // Duel records pass through the write order, so swapping one same-tick
    // duel pair flips the array. Emptying the stream empties the array.
    let swapped = |race: &RaceGpu, kind: u32| -> Option<String> {
        let mut patched = race.clone();
        let raw = patched.reply.as_mut().expect("raw");
        let mut at = 0usize;
        while at + 3 <= raw.comp.len() {
            let tick = raw.comp[at];
            let mut end = at;
            while end + 3 <= raw.comp.len()
                && (if raw.comp[end] == u32::MAX { 0 } else { raw.comp[end] }) == tick
            {
                end += 3;
            }
            let picks: Vec<usize> =
                (at..end).step_by(3).filter(|&s| raw.comp[s + 1] == kind).collect();
            if picks.len() >= 2 {
                for word in 0..3 {
                    raw.comp.swap(picks[0] + word, picks[1] + word);
                }
                return Some(array_text(&gpu_value(&request, &patched), "competitionEvents"));
            }
            at = end;
        }
        None
    };
    let base_text = array_text(&base, "competitionEvents");
    let duels = swapped(&race, 5).expect("a same-tick duel pair");
    assert_ne!(duels, base_text, "duel records keep the write order");

    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").comp.clear();
    let without = gpu_value(&request, &patched);
    assert!(array(&without, "competitionEvents").is_empty(), "the comp stream answers it");
}

#[test]
fn positioning_stream_gates_on_the_style_leader_bit() {
    register_geometry(10504);
    let request = with_seed(reorder_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    assert!(raw.phr.len() % 4 == 0 && !raw.phr.is_empty(), "runner phase tick lead");
    let leads = raw.phr.chunks(4).filter(|rec| rec[3] != 0).count();
    let quiet = raw.phr.chunks(4).filter(|rec| rec[3] == 0).count();
    assert!(leads > 0 && quiet > 0, "leader and non-leader phase changes both record");
    let base = gpu_value(&request, &race);
    let results = array(&base, "results");
    let published: usize = results
        .iter()
        .map(|r| array(r, "positioningPhases").len())
        .sum();
    assert_eq!(published, leads, "only the leader records publish");

    // Flattening the lead bit keeps every record and publishes none of them.
    let mut patched = race.clone();
    {
        let raw = patched.reply.as_mut().expect("raw");
        for rec in raw.phr.chunks_mut(4) {
            rec[3] = 0;
        }
    }
    let without = gpu_value(&request, &patched);
    let emptied: usize = array(&without, "results")
        .iter()
        .map(|r| array(r, "positioningPhases").len())
        .sum();
    assert_eq!(emptied, 0, "the lead bit is the publish gate");

    // Raising it publishes the whole stream.
    let mut patched = race.clone();
    {
        let raw = patched.reply.as_mut().expect("raw");
        for rec in raw.phr.chunks_mut(4) {
            rec[3] = 1;
        }
    }
    let all = gpu_value(&request, &patched);
    let every: usize = array(&all, "results")
        .iter()
        .map(|r| array(r, "positioningPhases").len())
        .sum();
    assert_eq!(every, raw.phr.len() / 4, "phase values come through unchanged");
}

#[test]
fn snapshot_and_spurt_columns_answer_the_result_fields() {
    register_geometry(10504);
    let request = with_seed(reorder_value(), &[]);
    let race = run_race(&request);
    let raw = race.reply.as_ref().expect("full channels");
    let n = raw.snap_guts.len();
    assert!(raw.spfirst.iter().all(|&v| v != 0), "every runner enters the last spurt");
    assert!(raw.spurt_time.iter().all(|v| v.is_some()), "every spurt start pair captured");
    assert!(raw.spurt_dist.iter().all(|v| v.is_some()));

    // A spread snapshot order re-ranks guts while wizOrder stays put. The
    // values sit inside the engine's bounded_stat window, all ties broken.
    let mut patched = race.clone();
    {
        let raw = patched.reply.as_mut().expect("raw");
        for i in 0..n {
            raw.snap_guts[i] = 100. * (i as f64 + 1.0);
        }
    }
    let moved = gpu_value(&request, &patched);
    let results = array(&moved, "results");
    let base = gpu_value(&request, &race);
    let got: Vec<f64> = results.iter().map(|r| f64_field(r, "gutsOrder")).collect();
    let was: Vec<f64> = array(&base, "results").iter().map(|r| f64_field(r, "gutsOrder")).collect();
    let want: Vec<f64> = (0..n).map(|i| (n - 1 - i) as f64).collect();
    assert_eq!(got, want, "the snapshot order re-ranks every runner");
    assert_ne!(got, was, "the mutation moves the order");
    for i in 0..n {
        assert_eq!(
            f64_field(&results[i], "wizOrder"),
            f64_field(&array(&base, "results")[i], "wizOrder"),
            "wizOrder keeps its own snapshot"
        );
    }

    // Dropping a start pair nulls exactly that runner's two columns.
    let mut patched = race.clone();
    {
        let raw = patched.reply.as_mut().expect("raw");
        raw.spurt_time[0] = None;
        raw.spurt_dist[0] = None;
    }
    let moved = gpu_value(&request, &patched);
    let results = array(&moved, "results");
    assert!(matches!(results[0].get("actualLastSpurtStartTime"), Some(Value::Null)));
    assert!(matches!(results[0].get("actualLastSpurtStartDistance"), Some(Value::Null)));
    assert!(results[1].get("actualLastSpurtStartTime").and_then(Value::as_f64).is_some());

    // Clearing the first last-spurt frame sends the player question to the
    // finish-time fallback the engine uses for a runner that never spurts.
    let mut patched = race.clone();
    patched.reply.as_mut().expect("raw").spfirst[0] = 0;
    let moved = gpu_value(&request, &patched);
    let fallback = f64_field(&array(&base, "results")[0], "finishTime");
    assert_eq!(f64_field(&moved, "playerLastSpurtTime"), fallback);
    assert_ne!(
        f64_field(&base, "playerLastSpurtTime"),
        fallback,
        "the captured frame answers while the edge stands"
    );
}

#[test]
fn requests_outside_the_full_lane_stay_refused() {
    register_geometry(10504);
    for (key, expect) in [
        ("traceConservePower", "traceConservePower"),
    ] {
        let request = with_seed(v77_value(), &[(key, Value::Bool(true))]);
        let run = race_exec::run_full(&[(request.clone(), 1i64)]).expect("kernel run");
        let race = &run.races[0];
        let reason = match &race.fallback {
            Some(reason) => reason.clone(),
            None => full_reply::build(&request, 1, race)
                .err()
                .unwrap_or_else(|| panic!("{key}: the full lane answered a refused request")),
        };
        assert!(reason.contains(expect), "{key}: refusal names it, got {reason}");
    }
    // Wave 18 moved experimentalLaterStaminaMechanics from this set to
    // answered on the retail lane (wave18_gate proves the answer).
    let request = with_seed(
        v77_value(),
        &[("experimentalLaterStaminaMechanics", Value::Bool(true))],
    );
    let run = race_exec::run_full(&[(request, 1i64)]).expect("kernel run");
    assert!(
        run.races[0].fallback.is_none(),
        "the retail lane answers the later-stamina flag"
    );
}
