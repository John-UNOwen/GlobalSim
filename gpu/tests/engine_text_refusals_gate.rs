//! Engine-error refusals on the retail lane.
//!
//! A skill whose shape makes the CPU engine fail the whole race stays a GPU
//! refusal, but the refusal text is the engine's own error text byte for byte:
//! the untraced retail lottery names (retail_initialization.rs:193), the
//! unregistered comparator body (activation_conditions.rs:48), and the
//! builder's pre-parse token scan (race_solver_builder.rs:501-509). The
//! fallback then hands the race to the CPU engine, which answers with the
//! same error the engine raises for the request.

#![cfg(feature = "race")]

use globalsim_gpu::{json, race_exec, skills_compile};
use json::Value;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request file");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// One alternative of the given condition and precondition, one negative
/// Recovery to self, no wisdom gate.
fn alt(condition: &str, precondition: &str) -> Value {
    json::obj(vec![
        ("precondition", Value::String(precondition.into())),
        ("condition", Value::String(condition.into())),
        ("cooldown", json::num(5000000.0)),
        ("baseDuration", json::num(20000.0)),
        (
            "effects",
            json::arr(vec![json::obj(vec![
                ("type", json::num(9.0)),
                ("target", json::num(1.0)),
                ("modifier", json::num(-500.0)),
            ])]),
        ),
    ])
}

/// v77 with one synthetic skill on runner 0. The alternatives are given per
/// shape so the offending token can sit on any alternative and column.
fn battery_request(sid: &str, alts: &[(&str, &str)]) -> String {
    let mut root = v77_value();
    {
        let Value::Object(map) = &mut root else { unreachable!() };
        let Value::Object(sd) = map
            .entry("skillData".to_string())
            .or_insert_with(|| Value::Object(Default::default()))
        else {
            unreachable!()
        };
        sd.insert(
            sid.into(),
            json::obj(vec![
                ("rarity", json::num(1.0)),
                ("wisdomCheck", json::num(0.0)),
                ("priority", json::num(0.0)),
                (
                    "alternatives",
                    Value::Array(alts.iter().map(|(c, p)| alt(c, p)).collect()),
                ),
            ]),
        );
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(h) = &mut horses[0] else { unreachable!() };
        let mut held: Vec<Value> = h
            .get("skills")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        held.push(Value::String(sid.into()));
        h.insert("skills".into(), Value::Array(held));
    }
    let mut text = String::new();
    json::to_string_into(&root, &mut text);
    text
}

/// The engine's prepare-or-run error text for one request.
fn engine_error(request: &str) -> String {
    let mut prepared = match globalsim::PreparedRace::new(request) {
        Err(e) => return e,
        Ok(p) => p,
    };
    match prepared.run(1) {
        Ok(_) => panic!("the engine accepted a request the gate expects it to reject"),
        Err(e) => e,
    }
}

/// The kernel's compact and full-lane refusals for one request.
fn kernel_refusals(request: &str) -> (String, String) {
    let run = race_exec::run(&[(request.to_string(), 1i64)]).expect("kernel run");
    let compact = run.races[0]
        .fallback
        .clone()
        .unwrap_or_else(|| panic!("the compact lane answered a request meant to be refused"));
    let full = race_exec::run_full(&[(request.to_string(), 1i64)]).expect("kernel full run");
    let full = full.races[0]
        .fallback
        .clone()
        .unwrap_or_else(|| panic!("the full lane answered a request meant to be refused"));
    (compact, full)
}

fn register_10504() {
    globalsim::register_course_geometry(
        10504,
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry"),
    );
}

/// The lottery names sample_trigger has no sampler for
/// (retail_initialization.rs:93-193) fail the race wherever the lottery
/// finds them.
#[test]
fn untraced_retail_lottery_names_quote_the_engine_error() {
    register_10504();
    let shapes = [
        "phase_firstquarter_random==2",
        "phase_corner_random==1",
        "phase_straight_random==1",
        "phase_first_half_straight_random==2",
        "phase_latter_half_straight_random==1",
        "up_slope_random_later_half==1",
        "run_at_full_speed_random==1",
    ];
    for (index, condition) in shapes.iter().enumerate() {
        let sid = format!("6694{}0", index);
        let request = battery_request(&sid, &[(condition, "")]);
        let name = condition.split("==").next().expect("name");
        let expected = format!("Untraced retail random trigger: {name}");
        assert_eq!(engine_error(&request), expected, "engine text for {condition}");
        let (compact, full) = kernel_refusals(&request);
        assert_eq!(compact, expected, "compact refusal for {condition}");
        assert_eq!(full, expected, "full refusal for {condition}");
    }
}

/// A sampled term with a non-'==' operator trips the lottery's operator
/// check before the name dispatch (retail_initialization.rs:226), and the
/// refusal quotes that text instead.
#[test]
fn an_untraced_operator_shapes_the_lotterys_operator_error() {
    register_10504();
    let request = battery_request("669490", &[("phase_corner_random>=2", "")]);
    let expected = "Untraced retail random operator: phase_corner_random>=";
    assert_eq!(engine_error(&request), expected, "engine text for the operator form");
    let (compact, full) = kernel_refusals(&request);
    assert_eq!(compact, expected, "compact refusal for the operator form");
    assert_eq!(full, expected, "full refusal for the operator form");
}

/// activate_count_middle registers the gte filter only
/// (activation_conditions.rs:1518); every other comparator runs the default
/// error body (:48), failing the race with that bare text.
#[test]
fn an_unregistered_comparator_quotes_unsupported_comparison() {
    register_10504();
    let request = battery_request("669491", &[("activate_count_middle==7", "")]);
    let expected = "unsupported comparison";
    assert_eq!(engine_error(&request), expected, "engine text for the eq form");
    let (compact, full) = kernel_refusals(&request);
    assert_eq!(compact, expected, "compact refusal for the eq form");
    assert_eq!(full, expected, "full refusal for the eq form");

    let request = battery_request("669492", &[("activate_count_middle<=3", "")]);
    assert_eq!(engine_error(&request), expected, "engine text for the le form");
    let (compact, _) = kernel_refusals(&request);
    assert_eq!(compact, expected, "compact refusal for the le form");
}

/// The retail builder scans every alternative's columns for
/// is_exist_skill_id before parsing (race_solver_builder.rs:501-509). The
/// refusal names the skill, the alternative index, and the column, exactly
/// as the engine does.
#[test]
fn the_is_exist_skill_id_token_quotes_the_builders_token_error() {
    register_10504();
    let shapes: Vec<(&str, Vec<(&str, &str)>, &str)> = vec![
        ("669493", vec![("always==1", "is_exist_skill_id==201631")], "precondition"),
        ("669494", vec![("is_exist_skill_id==201631", "always==1")], "condition"),
        ("669495", vec![("always==1", ""), ("is_exist_skill_id==201631", "")], "condition"),
    ];
    for (sid, alts, column) in shapes {
        let request = battery_request(sid, &alts);
        let detail = alts.len() - 1;
        let expected =
            format!("skill {sid} detail {detail}: unsupported {column} token is_exist_skill_id");
        assert_eq!(engine_error(&request), expected, "engine text for {sid}");
        let (compact, full) = kernel_refusals(&request);
        assert_eq!(compact, expected, "compact refusal for {sid}");
        assert_eq!(full, expected, "full refusal for {sid}");
    }
}

/// The classic lane registers is_exist_skill_id as a pass-through no-op
/// (activation_conditions.rs:1574): the term is always true and the skill
/// compiles. An unregistered activate_count_middle comparator is a classic
/// build error, so the compiler drops only that skill, as the engine does
/// (lib.rs:728-739), and answers the race.
fn skill_table(id: &str, skill: Value) -> Value {
    let mut table = json::obj(vec![]);
    let Value::Object(map) = &mut table else { unreachable!() };
    map.insert(id.into(), skill);
    table
}

/// A classic-lane root: one runner carrying one synthetic skill.
fn classic_root(sid: &str, skill: Value) -> Value {
    json::obj(vec![
        ("skillData", skill_table(sid, skill)),
        (
            "horses",
            json::arr(vec![json::obj(vec![(
                "skills",
                json::arr(vec![Value::String(sid.into())]),
            )])]),
        ),
    ])
}

#[test]
fn the_classic_lane_folds_the_noop_name_and_drops_the_build_error() {
    let course = json::parse(
        r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,
             "straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],
             "corners":[],"slopes":[]}"##,
    )
    .expect("course parses");

    let skill = json::parse(
        "{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{\"precondition\":\"\",\"condition\":\"is_exist_skill_id==201631&phase>=2\",\"cooldown\":5000000,\"baseDuration\":20000,\"effects\":[{\"type\":9,\"target\":1,\"modifier\":-500}]}]}",
    )
    .expect("skill parses");
    let plan = skills_compile::plan_classic(&classic_root("669496", skill), &course);
    assert_eq!(plan.reason, None, "the no-op name compiles on classic");
    assert!(plan.words[skills_compile::SK_NREC] >= 1, "the record survives");

    let skill = json::parse(
        "{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{\"precondition\":\"\",\"condition\":\"activate_count_middle==7\",\"cooldown\":5000000,\"baseDuration\":20000,\"effects\":[{\"type\":9,\"target\":1,\"modifier\":-500}]}]}",
    )
    .expect("skill parses");
    let plan = skills_compile::plan_classic(&classic_root("669497", skill), &course);
    assert_eq!(plan.reason, None, "a classic build error drops only the skill");
    assert_eq!(plan.words[skills_compile::SK_NREC], 0, "the dropped skill leaves no record");
}
