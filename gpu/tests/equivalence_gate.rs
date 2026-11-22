//! Wave-11 equivalence gates: every construct the GPU lane refuses comes
//! with a quote of what the CPU engine does on the same request. Effect kinds
//! the engine has no SkillType variant for, request shapes the engine itself
//! errors on, the kernel-capacity rosters past the cap, and the four wave-10
//! named full-lane refusals each get their engine behavior pinned here, error
//! text verbatim.

#![cfg(feature = "race")]

use globalsim_gpu::facade::{self, DeviceFacts, GpuRaces};
use globalsim_gpu::{classic_exec, full_reply, json, race_exec};
use json::Value;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request file");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

fn merged_skill_data() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The engine's own reply text for one request and seed, through the same
/// public API the app's non-compact lane drives.
fn engine_reply(request: &str, seed: i64) -> String {
    let mut prepared = globalsim::PreparedRace::new(request).expect("engine accepts the request");
    let simulation = prepared.run(seed).expect("engine runs the race");
    let mut reply = String::new();
    simulation.write_json(&mut reply);
    reply
}

/// The engine's prepare-or-run error text for one request.
fn engine_error(request: &str) -> String {
    match globalsim::PreparedRace::new(request) {
        Err(e) => e,
        Ok(mut prepared) => match prepared.run(1) {
            Ok(_) => panic!("the engine accepted a request the gate expects it to reject"),
            Err(e) => e,
        },
    }
}

/// One request patched onto v77, seed rewritten.
fn patched(extra: &[(&str, Value)], horse_count: Option<usize>, seed: i64) -> String {
    text_of(&value_patched(extra, horse_count, seed))
}

fn value_patched(extra: &[(&str, Value)], horse_count: Option<usize>, seed: i64) -> Value {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        for (key, item) in extra {
            map.insert((*key).into(), item.clone());
        }
        if let Some(n) = horse_count {
            let Value::Array(entries) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let base = entries[1].clone();
            while entries.len() < n {
                entries.push(base.clone());
            }
            entries.truncate(n);
        }
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    value
}

fn text_of(value: &Value) -> String {
    let mut text = String::new();
    json::to_string_into(value, &mut text);
    text
}

/// The kernel's compact-lane refusal for one request, and the same refusal
/// from the app's pure pre-filter.
fn kernel_refusals(request: &str) -> (String, Option<String>) {
    let run = race_exec::run(&[(request.to_string(), 1i64)]).expect("kernel run");
    let race = &run.races[0];
    let lane = match &race.fallback {
        Some(reason) => reason.clone(),
        None => panic!("the kernel answered a request the gate expects it to refuse"),
    };
    let facts = DeviceFacts {
        extensions: "cl_khr_fp64".into(),
        single_fp_config: 0x81,
        double_fp_config: 0x81,
    };
    (lane, facade::gate_reasons(request, Some(&facts)))
}

/// One engine-refused shape: the engine's error text, the kernel's refusal,
/// and the app pre-filter's reason must all be the same string.
fn shape_quotes_the_engine(label: &str, extra: &[(&str, Value)], horse_count: Option<usize>) {
    let request = patched(extra, horse_count, 1);
    let engine = engine_error(&request);
    let (kernel, gate) = kernel_refusals(&request);
    assert_eq!(kernel, engine, "{label}: the kernel names the engine's text");
    assert_eq!(gate.as_deref(), Some(engine.as_str()), "{label}: gate_reasons matches");
}

fn battery(cond: &str, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":5000000,\"baseDuration\":20000,\"effects\":[{effects}]}}]}}"
    )
}

/// A request whose horse 0 carries one battery skill of the given effect type.
fn battery_request(kind: i64) -> String {
    let sid = if kind == 33 { 669331 } else { 669332 };
    let mut skill_data = merged_skill_data();
    {
        let Value::Object(sd) = &mut skill_data else { unreachable!() };
        sd.insert(
            sid.to_string(),
            json::parse(&battery("phase>=2", &format!(r#"{{"type":{kind},"target":1,"modifier":500}}"#)))
                .expect("battery def"),
        );
    }
    let mut value = value_patched(&[("skillData", skill_data)], None, 1);
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(entries) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(horse) = &mut entries[0] else { unreachable!() };
        horse.insert(
            "skills".into(),
            json::arr(vec![json::obj(vec![
                ("skillId", Value::Number(sid as f64)),
                ("level", Value::Number(1.)),
            ])]),
        );
    }
    text_of(&value)
}

/// Kinds 33 and 34 have no SkillType variant, so the engine's retail builder
/// errors on them (race_solver_builder.rs:513) and the kernel refuses with a
/// reason naming that same missing variant (skills_compile.rs:1018-1021).
#[test]
fn effect_kinds_without_a_skilltype_variant_quote_the_engine() {
    register_geometry(10504);
    for kind in [33i64, 34] {
        let sid = if kind == 33 { 669331 } else { 669332 };
        let request = battery_request(kind);
        let engine = engine_error(&request);
        assert_eq!(
            engine,
            format!("skill {sid} detail 0: unsupported effect type {kind}"),
            "engine error text for kind {kind}"
        );
        let (kernel, _) = kernel_refusals(&request);
        assert_eq!(
            kernel,
            format!(
                "skill {sid} effect has no SkillType variant (retail builder errors, no-ffi degrades to Noop)"
            ),
            "kernel refusal for kind {kind}"
        );
        let run = race_exec::run_full(&[(request, 1i64)]).expect("kernel run");
        assert_eq!(run.races[0].fallback.as_deref(), Some(kernel.as_str()), "the full lane refuses kind {kind} too");
    }
}

/// The request shapes the engine itself errors on: non-boolean flag values
/// and invalid grid keys. Each error text is quoted byte-equal by the kernel.
#[test]
fn engine_rejected_request_shapes_quote_their_error_texts() {
    register_geometry(10504);
    let grid = |columns: f64, spacing: f64| {
        Value::Object(
            [("columns".to_string(), json::num(columns)), ("rowSpacing".to_string(), json::num(spacing))]
                .into_iter()
                .collect(),
        )
    };
    shape_quotes_the_engine("non-boolean startup", &[("experimentalRetailStartup", json::num(1.0))], None);
    shape_quotes_the_engine(
        "string startup",
        &[("experimentalRetailStartup", Value::String("yes".into()))],
        None,
    );
    shape_quotes_the_engine(
        "non-boolean later-stamina",
        &[("experimentalLaterStaminaMechanics", json::num(1.0))],
        None,
    );
    shape_quotes_the_engine("non-boolean trace", &[("traceConservePower", json::num(1.0))], None);
    shape_quotes_the_engine("grid columns 19", &[("experimentalGrid", grid(19.0, 3.0))], None);
    shape_quotes_the_engine("grid with an 8-horse roster", &[("experimentalGrid", grid(20.0, 3.0))], None);
    shape_quotes_the_engine("grid with a 121 roster", &[("experimentalGrid", grid(20.0, 3.0))], Some(121));
}

/// A skill id the request's skillData has no entry for is an engine build
/// error. The retail config errors first at the init lookup
/// (retail_initialization.rs:222) and answers the request with that text;
/// the retail kernel refusal quotes it. The classic lane hits the builder
/// error (race_solver_builder.rs:496) and retries after dropping the skill
/// (lib.rs:726-739), so the classic compiler drops it and answers too,
/// pinned by `classic_folds::a_classic_build_error_drops_only_that_skill`
/// and `wave12c_classic_gate::a_missing_skill_data_entry_drops_and_answers_like_the_engine`.
#[test]
fn a_missing_skill_data_entry_quotes_the_engine_error() {
    register_geometry(10504);
    let mut root = v77_value();
    {
        let Value::Object(map) = &mut root else { unreachable!() };
        map.insert("skillData".into(), merged_skill_data());
        let Value::Array(entries) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let Value::Object(horse) = &mut entries[0] else { unreachable!() };
        horse.insert("skills".into(), json::arr(vec![Value::String("999999999".into())]));
    }
    let request = text_of(&root);
    let engine = engine_error(&request);
    assert_eq!(engine, "Missing retail initialization field 999999999", "the engine's error text");
    let (kernel, _) = kernel_refusals(&request);
    assert_eq!(kernel, engine, "the retail kernel quotes the engine text");
}

/// The kernel cap is `gpu::ROSTER_CAP`, the game's 18-horse field limit.
/// Rosters past it do not run: they refuse by size from the run gate and
/// `gate_reasons` alike, compact and full, while the engine answers them all.
#[test]
fn rosters_beyond_the_kernel_cap_refuse_by_size() {
    register_geometry(10504);
    let cap = globalsim_gpu::gpu::ROSTER_CAP;
    assert_eq!(race_exec::RACE_MAXR, cap, "the kernel's roster cap");
    assert_eq!(classic_exec::CLS_MAXR, cap, "the classic kernel's roster cap");
    let counts = [cap + 1, 22, 24, 25, 121];
    for horse_count in counts {
        let request = text_of(&large_roster(1, horse_count));
        let engine = engine_reply(&request, 1);
        let parsed = json::parse(&engine).expect("reply");
        let results = parsed.get_array("results").expect("results");
        assert_eq!(results.len(), horse_count, "the engine answers all {horse_count} runners");
        let (kernel, gate) = kernel_refusals(&request);
        assert_eq!(kernel, "roster size outside the kernel's range", "the kernel refuses past its cap of {cap}");
        assert_eq!(gate.as_deref(), Some("roster size outside the kernel's range"), "gate_reasons matches");
        let mut gpu = facade::GpuRaces::new().expect("facade");
        let batch = gpu.run_batch(&[request.clone()], &[1]).expect("batch runs");
        assert_eq!(
            batch[0].fallback.as_deref(),
            Some("roster size outside the kernel's range"),
            "n={horse_count}: run_batch falls back"
        );
        let full = gpu.run_batch_full(&[request.clone()], &[1]).expect("full batch runs");
        assert_eq!(
            full[0].fallback.as_deref(),
            Some("roster size outside the kernel's range"),
            "n={horse_count}: run_batch_full falls back"
        );
    }
}

/// v77 grown to the given size with dense gate indices and a deterministic
/// spread of stats and strategies so no two runners shadow each other, the
/// shape a real app roster has. The clones carry no skills, so a 144-horse
/// roster stays inside the 960-record skill table the compiler enforces.
fn large_roster(seed: i64, horse_count: usize) -> Value {
    let mut value = value_patched(&[], Some(horse_count), seed);
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(entries) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oonige"];
        for (index, entry) in entries.iter_mut().enumerate() {
            let Value::Object(horse) = entry else { unreachable!() };
            horse.insert("gateIndex".into(), Value::Number(index as f64));
            horse.insert("strategy".into(), Value::String(styles[index % styles.len()].into()));
            let mix = |offset: u64| -> f64 {
                let mut x = (index as u64 + 1)
                    .wrapping_mul(2654435761)
                    .wrapping_add(offset * 40503)
                    & 0xFFFF_FFFF;
                x ^= x >> 15;
                x = x.wrapping_mul(2246822519);
                780.0 + (x % 420) as f64
            };
            horse.insert("speed".into(), Value::Number(mix(1)));
            horse.insert("stamina".into(), Value::Number(mix(2)));
            horse.insert("power".into(), Value::Number(mix(3)));
            horse.insert("guts".into(), Value::Number(mix(4)));
            horse.insert("wisdom".into(), Value::Number(mix(5)));
            if index >= 9 {
                horse.remove("skills");
            }
        }
    }
    value
}

/// singleMode is a real engine mode: the adjusted stats gain +400
/// (race_solver_builder.rs:304-320) and the reply carries singleModeVersion
/// (lib.rs:1125-1128). Both lanes port the fold: the retail gate here pins
/// the full reply byte-equal with the tag, and the classic lane pins frames
/// and reply in `wave12c_modes`. A non-boolean singleMode is ignored by the
/// engine (lib.rs:342), and that ignore is ported: the GPU full reply
/// answers it byte-equal to the engine on seeds 1 through 3.
#[test]
fn single_mode_is_an_engine_mode_and_the_non_boolean_form_is_ignored() {
    register_geometry(10504);
    let base = patched(&[], None, 1);
    let on = patched(&[("singleMode", Value::Bool(true))], None, 1);

    let engine_on = engine_reply(&on, 1);
    assert_eq!(
        json::parse(&engine_on).expect("reply").get_f64("singleModeVersion").expect("singleModeVersion"),
        1.0,
        "the engine tags a single-mode reply"
    );
    let engine_off = engine_reply(&base, 1);
    assert_ne!(engine_on, engine_off, "single mode changes the engine's answer");

    // The retail full lane answers the true form byte-equal with the tag.
    let run = race_exec::run_full(&[(on.clone(), 1i64)]).expect("kernel run");
    let race = &run.races[0];
    assert!(race.fallback.is_none(), "the retail full lane accepts singleMode");
    let gpu = full_reply::build(&on, 1, race).expect("full reply");
    assert_eq!(gpu, engine_on, "singleMode answers byte-equal");
    assert_eq!(
        json::parse(&gpu).expect("reply").get_f64("singleModeVersion").expect("singleModeVersion"),
        1.0,
        "the GPU reply carries the engine's tag"
    );

    let non_bool = patched(&[("singleMode", json::num(1.0))], None, 1);
    assert_eq!(engine_reply(&non_bool, 1), engine_off, "the engine ignores a non-boolean singleMode");
    for seed in [1i64, 2, 3] {
        let request = patched(&[("singleMode", json::num(1.0))], None, seed);
        let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
        let race = &run.races[0];
        assert!(race.fallback.is_none(), "the GPU full lane accepts the ignored form");
        let gpu = full_reply::build(&request, seed, race).expect("full reply");
        assert!(!gpu.contains("singleModeVersion"), "the ignored form must not tag the reply");
        assert_eq!(gpu, engine_reply(&request, seed), "seed {seed} answers byte-equal");
    }
}

/// traceConservePower is a real engine mode: it requires the retail runtime
/// (api.rs:58-59, quoted) and adds the conservePowerDiagnosticVersion plus
/// conservePowerDiagnostics keys to the reply (lib.rs:1115-1123) with the
/// physics unchanged. The full lane refuses it by name; the compact lane
/// accepts it as an ignore port because those keys never reach the compact
/// output, and the gate proves the output equals the flag-off output byte for
/// byte.
#[test]
fn trace_conserve_power_is_reply_diagnostics_only() {
    register_geometry(10504);
    let base = patched(&[], None, 1);
    let on = patched(&[("traceConservePower", Value::Bool(true))], None, 1);
    let engine_on = engine_reply(&on, 1);
    let mut parsed = json::parse(&engine_on).expect("reply");
    {
        let Value::Object(map) = &mut parsed else { unreachable!() };
        assert_eq!(map.get("conservePowerDiagnosticVersion").and_then(Value::as_f64), Some(1.0));
        assert!(matches!(map.get("conservePowerDiagnostics"), Some(Value::Array(_))));
        map.remove("conservePowerDiagnosticVersion");
        map.remove("conservePowerDiagnostics");
    }
    assert_eq!(
        text_of(&parsed),
        engine_reply(&base, 1),
        "the diagnostics keys are the engine's whole delta"
    );

    let run = race_exec::run_full(&[(on.clone(), 1i64)]).expect("kernel run");
    let race = &run.races[0];
    assert!(race.fallback.is_none(), "the kernel runs the race, only the reply lane refuses");
    let reason = full_reply::build(&on, 1, race).expect_err("the full lane refuses traceConservePower");
    assert_eq!(reason, "traceConservePower is outside the full GPU lane");

    let mut gpu = GpuRaces::new().expect("device");
    gpu.register_geometry(
        10504,
        &std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry"),
    )
    .expect("inject geometry");
    let outcomes = gpu.run_batch(&[base.clone(), on.clone()], &[1, 1]).expect("batch");
    for (index, outcome) in outcomes.iter().enumerate() {
        assert!(outcome.fallback.is_none(), "the compact lane accepts trace {index}");
    }
    assert_eq!(outcomes[0].meta, outcomes[1].meta, "compact meta equals the flag-off output");
    assert_eq!(outcomes[0].packet, outcomes[1].packet, "the packet equals the flag-off output");

    let non_retail = patched(
        &[("traceConservePower", Value::Bool(true)), ("experimentalRetailRuntime", Value::Bool(false))],
        None,
        1,
    );
    assert_eq!(engine_error(&non_retail), "traceConservePower requires experimentalRetailRuntime");
    let (kernel, _) = kernel_refusals(&non_retail);
    assert_eq!(kernel, "non-retail runtime", "the kernel's non-retail refusal stands first");
}

/// experimentalLaterStaminaMechanics is a real engine mode: it adds the
/// stamina keep, compete-before-spurt, and secure-lead per-tick systems
/// (race_solver.rs:2751-2756) and the staminasyoubu injection
/// (race_solver_builder.rs:817-848), and the reply echoes the flag
/// (lib.rs:1071). Both lanes port them; the wave 18 port adds the retail
/// lane, and the activation evidence lives in wave18_gate.
#[test]
fn later_stamina_mechanics_is_an_engine_mode_answered_by_both_lanes() {
    register_geometry(10504);
    let on = patched(&[("experimentalLaterStaminaMechanics", Value::Bool(true))], None, 1);
    let reply = json::parse(&engine_reply(&on, 1)).expect("reply");
    assert_eq!(reply.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool), Some(true));
    let run = race_exec::run_full(&[(on.clone(), 1i64)]).expect("kernel run");
    let race = &run.races[0];
    assert!(race.fallback.is_none(), "the retail lane ports the flag");
    assert_eq!(
        full_reply::build(&on, 1, race).expect("full reply builds"),
        engine_reply(&on, 1),
        "the full reply equals the engine with the mode on"
    );
}
