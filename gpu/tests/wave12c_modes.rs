//! Wave 12c Front C: the classic request modes. The engine's own oracle
//! classifies each flag under the classic lane (retail flags off): shapes it
//! errors on (invalid grid, traceConservePower) must be quoted byte-for-byte
//! by the GPU refusal; modes it answers (singleMode, later stamina up to
//! 2100 m, grid starts) must be answered by the classic kernel bit-exact,
//! and the named host fallbacks (later stamina above 2100 m) must carry
//! their exact reason so the app hands those to the host instead of
//! approximating.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, facade, full_reply, json, race_exec};
use json::Value;

const COURSE: i32 = 10504;
const SEEDS: [i64; 3] = [1, 2, 3];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The v77 field on the classic lane: the four retail flags forced off, the
/// roster optionally grown to `n` horses, and the listed mode flags overlaid.
fn classic_request(seed: i64, extra: &[(&str, Value)], n: Option<usize>) -> String {
    let mut value = {
        let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
        json::parse(text.lines().next().expect("first line")).expect("v77 parses")
    };
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
        if let Some(n) = n {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            let base = horses[1].clone();
            while horses.len() < n {
                horses.push(base.clone());
            }
        }
        for (k, v) in extra {
            map.insert((*k).into(), v.clone());
        }
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn engine_oracle(request: &str, seed: i64) -> Result<String, String> {
    match globalsim::PreparedRace::new(request) {
        Err(e) => Err(e),
        Ok(mut prepared) => match prepared.run(seed) {
            Err(e) => Err(e),
            Ok(simulation) => {
                let mut reply = String::new();
                simulation.write_json(&mut reply);
                Ok(reply)
            }
        },
    }
}

fn device_facts() -> facade::DeviceFacts {
    facade::DeviceFacts {
        extensions: "cl_khr_fp64".into(),
        double_fp_config: 0x81,
        single_fp_config: 0x81,
    }
}

/// An engine error under classic: the classic kernel's fallback reason and
/// the app gate both quote the engine's text verbatim.
#[test]
fn classic_engine_errors_quote_the_engine_text() {
    register_geometry(COURSE);
    let grid = Value::Object(
        [("columns".to_string(), json::num(20.0)), ("rowSpacing".to_string(), json::num(3.0))]
            .into_iter()
            .collect(),
    );
    for (label, extra, want) in [
        ("traceConservePower", vec![("traceConservePower", Value::Bool(true))],
         "traceConservePower requires experimentalRetailRuntime"),
        // The v77 field is nine horses: the grid shape check (lib.rs:262-272)
        // errors before the trace check can even apply.
        ("grid with a nine-horse field", vec![("experimentalGrid", grid.clone())],
         "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0."),
        // Both at once: the parse-time grid check wins, matching the engine.
        ("grid error outranks the trace error",
         vec![("experimentalGrid", grid.clone()), ("traceConservePower", Value::Bool(true))],
         "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0."),
    ] {
        let request = classic_request(1, &extra, None);
        let engine = engine_oracle(&request, 1).expect_err("{label}: the engine errors");
        assert_eq!(engine, want, "{label}: engine text");
        let out = classic_exec::run(&[(request.clone(), 1i64)]).expect("classic run");
        assert_eq!(out.races[0].fallback.as_deref(), Some(want), "{label}: classic kernel quotes it");
        assert_eq!(
            facade::gate_reasons(&request, Some(&device_facts())).as_deref(),
            Some(want),
            "{label}: the app gate quotes it"
        );
    }
}

/// A mode the engine answers: the named host fallback carries the exact
/// reason on both the classic kernel and the app gate, and the engine oracle
/// proves it really answers (so the fallback is a GPU gap, never a mask over
/// an engine error).
#[test]
fn classic_engine_answers_carry_named_host_fallbacks() {
    register_geometry(COURSE);
    // Later stamina above 2100 m is the only named classic fallback left:
    // the engine answers, the injection stays outside the kernel.
    for (label, extra, want) in [
        (
            "later stamina above 2100 m",
            vec![
                ("experimentalLaterStaminaMechanics", Value::Bool(true)),
                ("courseId", json::num(10108.0)),
            ],
            "staminasyoubu, injected above 2100 m, stays outside the classic kernel",
        ),
    ] {
        let request = classic_request(1, &extra, None);
        assert!(engine_oracle(&request, 1).is_ok(), "{label}: the engine answers");
        let out = classic_exec::run(&[(request.clone(), 1i64)]).expect("classic run");
        assert_eq!(out.races[0].fallback.as_deref(), Some(want), "{label}: named fallback");
        assert_eq!(
            facade::gate_reasons(&request, Some(&device_facts())).as_deref(),
            Some(want),
            "{label}: the app gate names it"
        );
    }
}

/// singleMode is ported to the classic kernel: the +400 fold lands in the
/// adjusted stats (race_solver_builder.rs:304-321) and the reply carries
/// singleModeVersion (lib.rs:1125-1129). Frames bit-exact against the CPU
/// engine and the full reply byte-equal on every seed, and the fold really
/// moved the physics: frames and reply differ from the flag-off field.
#[test]
fn classic_single_mode_answers_bit_exact() {
    register_geometry(COURSE);
    let on = classic_request(1, &[("singleMode", Value::Bool(true))], None);
    let off = classic_request(1, &[], None);
    assert_ne!(
        engine_oracle(&on, 1).expect("engine answers singleMode"),
        engine_oracle(&off, 1).expect("engine answers the field"),
        "singleMode changes the engine's answer"
    );
    let mut problems: Vec<String> = Vec::new();
    let mut any_frame_diff = false;
    for seed in SEEDS {
        let request = classic_request(seed, &[("singleMode", Value::Bool(true))], None);
        assert!(engine_oracle(&request, seed).is_ok(), "seed {seed}: the engine answers");
        let out = classic_exec::run(&[(request.clone(), seed)]).expect("classic kernel run");
        let race = &out.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: kernel fallback {reason}"));
            continue;
        }
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("seed {seed}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        let plain = classic_request(seed, &[], None);
        let plain_race = classic_exec::run(&[(plain.clone(), seed)]).expect("plain run")
            .races.into_iter().next().expect("one race");
        for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "seed {seed}: tick {tick} runner {runner} column {column} cpu {} gpu {}",
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                    }
                }
            }
        }
        // Non-vacuity: the +400 must actually move the physics.
        if let (Some(a), Some(b)) = (plain_race.frames.get(100), race.frames.get(100)) {
            if a[0] == b[0] {
                problems.push(format!("seed {seed}: tick 100 runner 0 pos identical to flag-off: the +400 did nothing"));
            } else {
                any_frame_diff = true;
            }
        }
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = classic_reply::build(&full, seed, race).expect("full reply build");
        let engine_reply = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        if gpu_reply != engine_reply {
            problems.push(format!("seed {seed}: singleMode full reply differs from the engine"));
        }
    }
    assert!(any_frame_diff, "the +400 fold never changed a frame");
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// A non-boolean singleMode is ignored by the engine (lib.rs:342): classic
/// answers the flag-off race, and the classic kernel answers it bit-exact.
#[test]
fn classic_non_boolean_single_mode_is_ignored() {
    register_geometry(COURSE);
    let on = classic_request(1, &[("singleMode", json::num(1.0))], None);
    let off = classic_request(1, &[], None);
    assert_eq!(
        engine_oracle(&on, 1).expect("engine answers"),
        engine_oracle(&off, 1).expect("engine answers the field"),
        "the engine ignores a non-boolean singleMode"
    );
    let out = classic_exec::run(&[(on.clone(), 1i64)]).expect("classic run");
    let race = &out.races[0];
    assert!(race.fallback.is_none(), "the classic kernel answers the ignored form");
    let full = dep_cpu::with_retain_all_frames(&on).expect("retain request");
    let gpu_reply = classic_reply::build(&full, 1, race).expect("full reply build");
    assert!(!gpu_reply.contains("singleModeVersion"), "the ignored form must not tag the reply");
    let engine_reply = dep_cpu::run_cpu_binary(&on, 1).expect("cpu binary").reply;
    assert_eq!(gpu_reply, engine_reply, "the ignored form answers byte-equal");
}

/// The v77 field on the retail lane, the listed mode flags overlaid.
fn retail_request(seed: i64, extra: &[(&str, Value)]) -> String {
    let mut value = {
        let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
        json::parse(text.lines().next().expect("first line")).expect("v77 parses")
    };
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        for (k, v) in extra {
            map.insert((*k).into(), v.clone());
        }
        map.insert("seed".into(), Value::Number(seed as f64));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// singleMode is ported to the retail kernel too: the +400 fold lands inside
/// the im_stats clamps (race_solver_builder.rs:304-321) and the reply carries
/// singleModeVersion (lib.rs:1125-1129). Frames bit-exact against the CPU
/// engine and the full reply byte-equal on every seed; the fold really moved
/// the physics: frames differ from the flag-off field, and the flag-off
/// frames equal the untouched v77.
#[test]
fn retail_single_mode_answers_bit_exact() {
    register_geometry(COURSE);
    let on = retail_request(1, &[("singleMode", Value::Bool(true))]);
    let off = retail_request(1, &[]);
    assert_ne!(
        engine_oracle(&on, 1).expect("engine answers singleMode"),
        engine_oracle(&off, 1).expect("engine answers the field"),
        "singleMode changes the engine's answer"
    );
    let mut problems: Vec<String> = Vec::new();
    let mut any_frame_diff = false;
    for seed in SEEDS {
        let request = retail_request(seed, &[("singleMode", Value::Bool(true))]);
        assert!(engine_oracle(&request, seed).is_ok(), "seed {seed}: the engine answers");
        let out = race_exec::run(&[(request.clone(), seed)]).expect("retail kernel run");
        let race = &out.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: kernel fallback {reason}"));
            continue;
        }
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("seed {seed}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "seed {seed}: tick {tick} runner {runner} column {column} cpu {} gpu {}",
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                    }
                }
            }
        }
        let full = race_exec::run_full(&[(request.clone(), seed)]).expect("retail full run");
        let full_race = &full.races[0];
        let gpu_reply = full_reply::build(&request, seed, full_race).expect("full reply build");
        if gpu_reply != engine_oracle(&request, seed).expect("engine reply") {
            problems.push(format!("seed {seed}: singleMode retail full reply differs from the engine"));
        }
        // Non-vacuity: the +400 must move the physics against flag off.
        let plain_race = race_exec::run(&[(retail_request(seed, &[]), seed)])
            .expect("flag-off run")
            .races
            .into_iter()
            .next()
            .expect("one race");
        if let (Some(a), Some(b)) = (plain_race.frames.get(100), race.frames.get(100)) {
            if a[0] == b[0] {
                problems.push(format!("seed {seed}: tick 100 runner 0 pos identical to flag-off: the +400 did nothing"));
            } else {
                any_frame_diff = true;
            }
        }
    }
    assert!(any_frame_diff, "the +400 fold never changed a retail frame");
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// The classic v77 roster grown to 21 horses with dense gate indices, so
/// grid-mode starts stagger across two rows, with the listed mode flags on.
fn classic_gated_request(seed: i64, extra: &[(&str, Value)]) -> String {
    let mut value = json::parse(&classic_request(seed, extra, Some(21))).expect("request parses");
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for (index, entry) in horses.iter_mut().enumerate() {
            let Value::Object(horse) = entry else { unreachable!() };
            horse.insert("gateIndex".into(), Value::Number(index as f64));
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

fn grid_shape() -> Value {
    Value::Object(
        [("columns".to_string(), json::num(20.0)), ("rowSpacing".to_string(), json::num(3.0))]
            .into_iter()
            .collect(),
    )
}

/// Wave 13b: grid starts on the classic lane. The engine answers with the
/// pure-f64 row-lane pair (lib.rs:607, :274-279; the f32 round-trip at
/// lib.rs:610 is retail-runtime only). The 21-horse grid race replays
/// bit-exact compact and byte-equal full on every seed, carries the engine's
/// experimentalGridVersion tag, and the staggered starts really move the
/// physics against the same gate-order flag-off field.
#[test]
fn classic_grid_starts_answer_bit_exact() {
    register_geometry(COURSE);
    let mut problems: Vec<String> = Vec::new();
    let mut any_start_diff = false;
    for seed in SEEDS {
        let request = classic_gated_request(seed, &[("experimentalGrid", grid_shape())]);
        let engine = engine_oracle(&request, seed).expect("the engine answers classic grid");
        assert_eq!(
            json::parse(&engine).expect("engine reply").get_f64("experimentalGridVersion"),
            Ok(1.0),
            "seed {seed}: the engine tags the classic grid reply"
        );
        assert_eq!(
            facade::gate_reasons(&request, Some(&device_facts())),
            None,
            "seed {seed}: the app gate accepts classic grid"
        );
        let out = classic_exec::run(&[(request.clone(), seed)]).expect("classic kernel run");
        let race = &out.races[0];
        if let Some(reason) = &race.fallback {
            problems.push(format!("seed {seed}: kernel fallback {reason}"));
            continue;
        }
        assert!(race.grid, "seed {seed}: the kernel records grid starts");
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("seed {seed}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "seed {seed}: tick {tick} runner {runner} column {column} cpu {} gpu {}",
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                    }
                }
            }
        }
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = classic_reply::build(&full, seed, race).expect("classic grid full reply");
        assert!(
            gpu_reply.contains("\"experimentalGridVersion\""),
            "seed {seed}: the GPU reply carries the engine's grid tag"
        );
        let engine_reply = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        if gpu_reply != engine_reply {
            problems.push(format!("seed {seed}: classic grid full reply differs from the engine"));
        }
        // Non-vacuity: the staggered starts must move the physics against the
        // same roster with the grid flag off. Runner 0 sits at rank 0, where
        // both modes start identically; runner 20 starts a row back.
        let plain = classic_gated_request(seed, &[]);
        let plain_race = classic_exec::run(&[(plain, seed)])
            .expect("flag-off run")
            .races
            .into_iter()
            .next()
            .expect("one race");
        if let (Some(a), Some(b)) = (plain_race.frames.get(100), race.frames.get(100)) {
            if a[20] == b[20] {
                problems.push(format!("seed {seed}: tick 100 runner 20 state identical to flag-off: the grid starts did nothing"));
            } else {
                any_start_diff = true;
            }
        }
    }
    assert!(any_start_diff, "the grid starts never changed a frame");
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
