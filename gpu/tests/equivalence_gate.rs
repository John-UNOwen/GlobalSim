//! Wave-11 equivalence gates: every construct the GPU lane refuses comes
//! with a quote of what the CPU engine does on the same request. Effect kinds
//! the engine has no SkillType variant for, request shapes the engine itself
//! errors on, the kernel-capacity rosters past the cap, the wave-15 non-grid
//! band refusal, and the four wave-10 named full-lane refusals each get their
//! engine behavior pinned here, error text verbatim.

#![cfg(feature = "race")]

use globalsim_gpu::facade::{self, DeviceFacts, GpuRaces};
use globalsim_gpu::{classic_exec, dep_cpu, full_reply, json, race_exec};
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

/// Wave 15 puts the kernel cap back at the grid-mode 120. Rosters past it
/// no longer run: 121, 128, 144 and 145 non-grid refuse by size from the
/// run gate and `gate_reasons` alike, compact and full, while the engine
/// answers them all. The band refusal covers non-grid 25 through 120 and
/// is pinned by `non_grid_crowds_above_24_refuse_the_kernel_band`.
#[test]
fn rosters_beyond_the_kernel_cap_refuse_by_size() {
    register_geometry(10504);
    assert_eq!(race_exec::RACE_MAXR, 120, "the kernel's roster cap");
    assert_eq!(classic_exec::CLS_MAXR, 120, "the classic kernel's roster cap");
    for horse_count in [121usize, 128, 144, 145] {
        let request = text_of(&large_roster(1, horse_count));
        let engine = engine_reply(&request, 1);
        let parsed = json::parse(&engine).expect("reply");
        let results = parsed.get_array("results").expect("results");
        assert_eq!(results.len(), horse_count, "the engine answers all {horse_count} runners");
        let (kernel, gate) = kernel_refusals(&request);
        assert_eq!(kernel, "roster size outside the kernel's range", "the kernel refuses past its cap of 120");
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

/// Wave-15 band gate, the user decision of 2026-11-20: a non-grid field
/// above 24 runners refuses the kernel on both lanes and stays on the
/// host. The band stops at 24 because dense non-grid crowds stack at the
/// inside rail (laneMax 12000, rail 1.2, one 0.4/18 overlap push away),
/// and there the kernel's resolve_overlaps parks runners at rail+0.4/18
/// unclamped while the engine's settles them at the rail. The wave-14
/// experiment made 25 to 144 byte-exact; the user kept the game's and the
/// app's 18-horse ceiling plus margin as the validated band instead. 22,
/// 23 and 24 stay byte-exact here, and 25 and up refuse by name, compact
/// and full, with the engine answering each one. Grid mode is unaffected
/// at any size it admits because the staggered starts keep every lane
/// under the rail. Rosters past the 120-horse kernel cap refuse by size,
/// pinned by `rosters_beyond_the_kernel_cap_refuse_by_size`.
#[test]
fn non_grid_crowds_above_24_refuse_the_kernel_band() {
    register_geometry(10504);
    assert_eq!(race_exec::NON_GRID_MAX_RUNNERS, 24, "the validated non-grid band");
    // 22 through 24 stay bit-exact end to end.
    for horse_count in [22usize, 23, 24] {
        let request = text_of(&large_roster(1, horse_count));
        let run = race_exec::run_full(&[(request.clone(), 1i64)]).expect("kernel run");
        let race = &run.races[0];
        assert!(race.fallback.is_none(), "n={horse_count}: {:?}", race.fallback);
        let cpu = dep_cpu::run_cpu_race(&request, 1).expect("cpu race");
        assert_eq!(cpu.frames.len(), race.frames.len(), "n={horse_count}: tick count");
        for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    assert_eq!(
                        cpu.to_bits(),
                        gpu.to_bits(),
                        "n={horse_count}: tick {tick} runner {runner} column {column}"
                    );
                }
            }
        }
    }
    // 25, 28 and 40 refuse with the band reason, compact and full, and
    // the engine answers each one.
    for horse_count in [25usize, 28, 40] {
        let request = text_of(&large_roster(1, horse_count));
        let (kernel, gate) = kernel_refusals(&request);
        assert_eq!(kernel, race_exec::NON_GRID_BAND_REFUSAL, "n={horse_count}: the run gate flags with the band reason");
        assert_eq!(
            gate.as_deref(),
            Some(race_exec::NON_GRID_BAND_REFUSAL),
            "n={horse_count}: the app prefilter names the band reason"
        );
        let mut gpu = facade::GpuRaces::new().expect("facade");
        let batch = gpu.run_batch(&[request.clone()], &[1]).expect("batch runs");
        assert_eq!(
            batch[0].fallback.as_deref(),
            Some(race_exec::NON_GRID_BAND_REFUSAL),
            "n={horse_count}: run_batch falls back"
        );
        let full = gpu.run_batch_full(&[request.clone()], &[1]).expect("full batch runs");
        assert_eq!(
            full[0].fallback.as_deref(),
            Some(race_exec::NON_GRID_BAND_REFUSAL),
            "n={horse_count}: run_batch_full falls back"
        );
        let parsed = json::parse(&engine_reply(&request, 1)).expect("reply");
        let results = parsed.get_array("results").expect("results");
        assert_eq!(results.len(), horse_count, "the engine answers all {horse_count} runners");
    }
    // The classic lane carries the same band (it was only ever validated
    // to 18 runners and the gate band is uniform), and its measured
    // 25/26/28 crowds were bit-exact at HEAD f86a15d.
    for horse_count in [25usize, 28] {
        let request = classic_large_roster(horse_count);
        let run = classic_exec::run(&[(request.clone(), 1i64)]).expect("classic run");
        let kernel = run.races[0]
            .fallback
            .clone()
            .unwrap_or_else(|| panic!("classic n={horse_count}: the classic lane answered a banded request"));
        assert_eq!(kernel, race_exec::NON_GRID_BAND_REFUSAL, "classic n={horse_count}: the run gate flags with the band reason");
        let facts = DeviceFacts {
            extensions: "cl_khr_fp64".into(),
            single_fp_config: 0x81,
            double_fp_config: 0x81,
        };
        assert_eq!(
            facade::gate_reasons(&request, Some(&facts)).as_deref(),
            Some(race_exec::NON_GRID_BAND_REFUSAL),
            "classic n={horse_count}: the app prefilter names the band reason"
        );
        let parsed = json::parse(&engine_reply(&request, 1)).expect("reply");
        let results = parsed.get_array("results").expect("results");
        assert_eq!(results.len(), horse_count, "the engine answers the classic field of {horse_count}");
    }
}

/// The v77 shared fields with the four retail flags off and the dense
/// large_roster field: a non-grid classic crowd.
fn classic_large_roster(horse_count: usize) -> String {
    let mut value = large_roster(1, horse_count);
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
    }
    text_of(&value)
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
/// (race_solver.rs:2751-2756) and the reply echoes the flag (lib.rs:1071).
/// The classic lane ports them (gpu/tests/wave13a_later_stamina.rs); the
/// retail lane refuses by name because its kernel resolves randomness on the
/// host.
#[test]
fn later_stamina_mechanics_is_an_engine_mode_refused_by_name() {
    register_geometry(10504);
    let on = patched(&[("experimentalLaterStaminaMechanics", Value::Bool(true))], None, 1);
    let reply = json::parse(&engine_reply(&on, 1)).expect("reply");
    assert_eq!(reply.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool), Some(true));
    let want = "experimentalLaterStaminaMechanics draws from a live solver stream \
                the retail kernel resolves host-side";
    let (kernel, gate) = kernel_refusals(&on);
    assert_eq!(kernel, want);
    assert_eq!(gate.as_deref(), Some(want));
    let run = race_exec::run_full(&[(on, 1i64)]).expect("kernel run");
    assert_eq!(run.races[0].fallback.as_deref(), Some(want));
}

/// Grid mode is a fully engine-supported race shape. The GPU lane runs grid
/// compact bit-exactly (gpu/tests/grid_gate.rs) and answers the full reply
/// byte-equal too: the staggered roster ranks the engine's competition
/// reorder sorts by are rebuilt from the grid starts (src/race_field.rs:118-125)
/// and the reply carries the engine's experimentalGridVersion tag
/// (src/lib.rs:1135-1138).
#[test]
fn grid_full_replies_are_byte_equal_to_the_engine() {
    register_geometry(10504);
    let mut kind4_total = 0usize;
    for horse_count in [21usize, 120] {
        for seed in [1i64, 2, 3] {
            let request = text_of(&grid_value(seed, horse_count));
            let engine = engine_reply(&request, seed);
            let reply = json::parse(&engine).expect("engine reply");
            assert_eq!(
                reply.get_f64("experimentalGridVersion"),
                Ok(1.0),
                "the engine tags the grid reply, n={horse_count} seed {seed}"
            );
            kind4_total += reply
                .get_array("competitionEvents")
                .expect("competitionEvents")
                .iter()
                .filter(|event| event.get_f64("type") == Ok(4.0))
                .count();
            let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
            let race = &run.races[0];
            assert!(race.fallback.is_none(), "n={horse_count} seed {seed}: the kernel runs grid full");
            let gpu = full_reply::build(&request, seed, race).expect("grid full reply");
            assert_eq!(gpu, engine, "n={horse_count} seed {seed}: grid full reply is byte-equal");
        }
    }
    assert!(kind4_total >= 2, "the grid replies must carry kind-4 competition events, the reordered ones, found {kind4_total}");
}

/// The grid variant of the patched v77 request: geometry off (the engine
/// refuses geometry with grid starts, api.rs:68-71), a valid columns/rowSpacing
/// shape, dense gate indices 0..n-1, and cycled styles so the recording set
/// (Nige and Oonige) spans grid rows, the case the staggered ranks decide.
fn grid_value(seed: i64, horse_count: usize) -> Value {
    let mut value = value_patched(
        &[
            ("experimentalRetailGeometry", Value::Bool(false)),
            ("experimentalGrid", Value::Object(
                [("columns".to_string(), json::num(20.0)), ("rowSpacing".to_string(), json::num(3.0))]
                    .into_iter()
                    .collect(),
            )),
        ],
        Some(horse_count),
        seed,
    );
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(entries) = map.get_mut("horses").expect("horses") else { unreachable!() };
        let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oonige"];
        for (index, entry) in entries.iter_mut().enumerate() {
            let Value::Object(horse) = entry else { unreachable!() };
            horse.insert("gateIndex".into(), Value::Number(index as f64));
            horse.insert("strategy".into(), Value::String(styles[index % styles.len()].into()));
        }
    }
    value
}
