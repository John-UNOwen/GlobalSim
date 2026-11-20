//! The batch facade gate (handoff step 8): the compact meta plus the
//! GSIMRP01 packet the app's run_compact lane consumes. Every field
//! compact_record reads is compared against the engine's own `write_json`
//! reply on the same request, and the packet's frame bytes against the
//! engine's `into_binary` packet. A startup request inside the same batch
//! must come back as a fallback without disturbing its siblings,
//! `gate_reasons` is checked as a pure function on synthetic device facts,
//! and `register_geometry` injection outranks the geometry file copy. A
//! non-retail zero-skill request answers from the classic kernel lane, in the
//! same batch as a retail request.

#![cfg(feature = "race")]

use globalsim_gpu::facade::{self, DeviceFacts, GpuRaces};
use globalsim_gpu::{dep_cpu, json, race_exec, replay_packet};

const SKILLS: [i64; 6] = [200361, 200341, 200333, 10071, 100401, 110591];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_line() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

/// The skillEvents multiset the way compact_record reads it: (horseIndex, id)
/// pairs with skillId first then id, entries lacking both keys ignored,
/// numbers rendered as their integer digits.
fn events_multiset(events: &json::Value) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = events
        .as_array()
        .map_or_else(Vec::new, |items| {
            items
                .iter()
                .filter_map(|entry| {
                    let horse = entry.get_f64("horseIndex").ok()? as i64;
                    let id = entry.get("skillId").or_else(|| entry.get("id"))?;
                    let id = match id {
                        json::Value::Number(n) => (*n as i64).to_string(),
                        json::Value::String(s) => s.clone(),
                        _ => return None,
                    };
                    Some((horse, id))
                })
                .collect()
        });
    out.sort();
    out
}

fn check_num(
    problems: &mut Vec<String>,
    tag: &str,
    field: &str,
    cpu_row: &json::Value,
    gpu_row: &json::Value,
) {
    let (cpu, gpu) = match (cpu_row.get(field), gpu_row.get(field)) {
        (Some(cpu), Some(gpu)) => (cpu, gpu),
        _ => {
            problems.push(format!("{tag} {field}: missing on one side"));
            return;
        }
    };
    match (cpu.as_f64(), gpu.as_f64()) {
        (Some(a), Some(b)) if a.to_bits() == b.to_bits() => {}
        (Some(a), Some(b)) => problems.push(format!(
            "{tag} {field}: cpu {a:?} ({}), gpu {b:?} ({})",
            a.to_bits(),
            b.to_bits()
        )),
        _ => problems.push(format!("{tag} {field}: number expected, cpu {cpu:?}, gpu {gpu:?}")),
    }
}

fn check_bool(
    problems: &mut Vec<String>,
    tag: &str,
    field: &str,
    cpu_row: &json::Value,
    gpu_row: &json::Value,
) {
    let (cpu, gpu) = match (cpu_row.get(field), gpu_row.get(field)) {
        (Some(cpu), Some(gpu)) => (cpu, gpu),
        _ => {
            problems.push(format!("{tag} {field}: missing on one side"));
            return;
        }
    };
    match (cpu.as_bool(), gpu.as_bool()) {
        (Some(a), Some(b)) if a == b => {}
        _ => problems.push(format!("{tag} {field}: cpu {cpu:?}, gpu {gpu:?}")),
    }
}

/// One batch: four paired seeds and the six-skill request, every compact
/// field against the engine, every packet byte against the engine.
#[test]
fn compact_meta_and_packet_match_the_engine_lane() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);

    let mut requests: Vec<String> = Vec::new();
    for seed in 1..=4i64 {
        requests.push(race_exec::paired_request(&base, 10504, seed).expect("paired request"));
    }
    requests.push(race_exec::skills_request(&base, 10504, 1, 0, &SKILLS, None).expect("skills request"));
    let seeds = [1i32, 2, 3, 4, 1];

    let mut races = GpuRaces::new().expect("device gate");
    let outcomes = races.run_batch(&requests, &seeds).expect("batch");
    assert_eq!(outcomes.len(), requests.len());

    let mut problems: Vec<String> = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let seed = seeds[index];
        let tag = format!("request {index} seed {seed}");
        let outcome = &outcomes[index];
        let (Some(meta), Some(packet)) = (&outcome.meta, &outcome.packet) else {
            problems.push(format!("{tag}: expected a compact answer, got fallback {:?}", outcome.fallback));
            continue;
        };
        let cpu = match dep_cpu::run_cpu_binary(request, seed as i64) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: engine run failed: {reason}"));
                continue;
            }
        };
        let cpu_reply = json::parse(&cpu.reply).expect("engine reply parses");
        let gpu_meta = json::parse(meta).expect("facade meta parses");

        check_num(&mut problems, &tag, "playerHorseIndex", &cpu_reply, &gpu_meta);
        let cpu_results = cpu_reply.get_array("results").expect("engine results");
        let gpu_results = gpu_meta.get_array("results").expect("facade results");
        if cpu_results.len() != gpu_results.len() {
            problems.push(format!("{tag}: results len cpu {} gpu {}", cpu_results.len(), gpu_results.len()));
            continue;
        }
        for (runner, (cpu_row, gpu_row)) in cpu_results.iter().zip(gpu_results).enumerate() {
            let tag = format!("{tag} runner {runner}");
            check_num(&mut problems, &tag, "finishTimeRaw", cpu_row, gpu_row);
            check_num(&mut problems, &tag, "finishOrder", cpu_row, gpu_row);
            check_bool(&mut problems, &tag, "hpEmptyOnRace", cpu_row, gpu_row);
            check_num(&mut problems, &tag, "rushingStartCount", cpu_row, gpu_row);
            check_num(&mut problems, &tag, "rushingPenaltyCount", cpu_row, gpu_row);
            check_num(&mut problems, &tag, "lastSpurtStartDistance", cpu_row, gpu_row);
            check_num(&mut problems, &tag, "startDelayTime", cpu_row, gpu_row);
        }
        let cpu_events = events_multiset(cpu_reply.get("skillEvents").expect("engine skillEvents"));
        let gpu_events = events_multiset(gpu_meta.get("skillEvents").expect("facade skillEvents"));
        if cpu_events != gpu_events {
            problems.push(format!("{tag}: skillEvents multiset cpu {cpu_events:?}, gpu {gpu_events:?}"));
        }

        assert_eq!(&packet[..8], replay_packet::MAGIC, "{tag}: packet magic");
        let (meta_len, frames, horses) = replay_packet::decode_header(packet).expect("header");
        let (cpu_frames, cpu_horses) = {
            let (_, frames, horses) = replay_packet::decode_header(&cpu.packet).expect("engine header");
            (frames, horses)
        };
        if (frames, horses) != (cpu_frames, cpu_horses) {
            problems.push(format!(
                "{tag}: packet counts cpu ({cpu_frames}, {cpu_horses}), gpu ({frames}, {horses})"
            ));
        }
        match (replay_packet::frame_bytes(packet), replay_packet::frame_bytes(&cpu.packet)) {
            (Some(gpu_frames), Some(cpu_frames)) if gpu_frames == cpu_frames => {}
            (Some(gpu_frames), Some(cpu_frames)) => problems.push(format!(
                "{tag}: packet frame bytes differ (gpu {} bytes, engine {} bytes)",
                gpu_frames.len(),
                cpu_frames.len()
            )),
            _ => problems.push(format!("{tag}: a packet lacks its frame payload")),
        }
        if meta_len != meta.len()
            || packet.get(32..32 + meta_len) != Some(meta.as_bytes())
        {
            problems.push(format!("{tag}: the packet's embedded meta bytes differ from the returned meta string"));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// The startup flag is a ported domain: a flagged request inside a batch
/// answers with its own meta and packet, and its siblings stay whole.
#[test]
fn a_startup_request_runs_on_gpu_and_its_siblings_stay_whole() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);

    let first = race_exec::paired_request(&base, 10504, 1).expect("first request");
    let mut startup_value = json::parse(&first).expect("parses");
    if let json::Value::Object(map) = &mut startup_value {
        map.insert("experimentalRetailStartup".into(), json::Value::Bool(true));
    }
    let mut startup = String::new();
    json::to_string_into(&startup_value, &mut startup);
    let second = race_exec::paired_request(&base, 10504, 2).expect("second request");

    let mut races = GpuRaces::new().expect("device gate");
    let outcomes = races.run_batch(&[first, startup, second], &[1, 1, 2]).expect("batch");
    assert_eq!(outcomes.len(), 3);

    for (index, outcome) in outcomes.iter().enumerate() {
        assert!(
            outcome.meta.is_some() && outcome.packet.is_some() && outcome.fallback.is_none(),
            "race {index} should carry a compact answer, got fallback {:?}",
            outcome.fallback
        );
    }
}

/// `run_repeated` against `run_batch`, the same request and seeds: every
/// outcome field byte-identical. This is the hard contract of the
/// prepare-once lane (the seed only enters the prepared inputs through
/// `RI_SEED`, so the clone path must equal a per-seed prepare).
#[test]
fn run_repeated_matches_run_batch_over_64_seeds() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);

    let request = race_exec::multi_request(&base, 10504, 1).expect("request");
    let seeds: Vec<i32> = (1..=64).collect();
    let requests: Vec<String> = seeds.iter().map(|_| request.clone()).collect();

    let mut races = GpuRaces::new().expect("device gate");
    let batched = races.run_batch(&requests, &seeds).expect("run_batch");
    let repeated = races.run_repeated(&request, &seeds).expect("run_repeated");
    assert_eq!(batched.len(), seeds.len());
    assert_eq!(repeated.len(), seeds.len());
    for (index, (batched, repeated)) in batched.iter().zip(&repeated).enumerate() {
        let tag = format!("seed {}", seeds[index]);
        assert_eq!(&batched.fallback, &repeated.fallback, "{tag}: fallback");
        assert_eq!(&batched.meta, &repeated.meta, "{tag}: meta");
        assert_eq!(&batched.packet, &repeated.packet, "{tag}: packet");
        assert!(batched.meta.is_some(), "{tag}: the fixture should launch, got fallback {:?}", batched.fallback);
    }
}

/// The shape `run_batch` supports and `run_repeated` splits: two courses in
/// one mixed call. Each course answered on its own must byte-equal its slice
/// of the mixed batch. A startup-flagged request and an unparseable one ride
/// along: startup now runs, the broken text pins the fallback arm in both
/// lanes.
#[test]
fn run_repeated_matches_a_mixed_two_course_batch() {
    let base = v77_line();
    for course in [10504, 10604] {
        let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
        globalsim::register_course_geometry(course, geometry);
    }

    let first = race_exec::multi_request(&base, 10504, 1).expect("first request");
    let second = race_exec::multi_request(&base, 10604, 1).expect("second request");
    let mut startup_value = json::parse(&first).expect("parses");
    if let json::Value::Object(map) = &mut startup_value {
        map.insert("experimentalRetailStartup".into(), json::Value::Bool(true));
    }
    let mut startup = String::new();
    json::to_string_into(&startup_value, &mut startup);

    let seeds = [1i32, 2, 3, 4];
    let mut requests: Vec<String> = Vec::new();
    requests.extend(std::iter::repeat(first.clone()).take(4));
    requests.extend(std::iter::repeat(second.clone()).take(4));
    requests.extend(std::iter::repeat(startup.clone()).take(2));
    requests.extend(std::iter::repeat("not json".into()).take(2));
    let all_seeds: Vec<i32> = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4].iter().copied().collect();

    let mut races = GpuRaces::new().expect("device gate");
    let batched = races.run_batch(&requests, &all_seeds).expect("run_batch");
    let repeated_first = races.run_repeated(&first, &seeds).expect("run_repeated first");
    let repeated_second = races.run_repeated(&second, &seeds).expect("run_repeated second");
    let repeated_startup = races.run_repeated(&startup, &seeds).expect("run_repeated startup");
    let repeated_broken = races.run_repeated("not json", &seeds).expect("run_repeated broken");

    let groups = [
        (&batched[0..4], repeated_first.as_slice()),
        (&batched[4..8], repeated_second.as_slice()),
        (&batched[8..10], &repeated_startup[0..2]),
        (&batched[10..12], &repeated_broken[0..2]),
    ];
    for (course, group) in [("10504", groups[0]), ("10604", groups[1]), ("startup", groups[2]), ("broken", groups[3])] {
        assert_eq!(group.0.len(), group.1.len(), "{course}: len");
        for (index, (batched, repeated)) in group.0.iter().zip(group.1).enumerate() {
            let tag = format!("{course} seed {}", seeds[index]);
            assert_eq!(&batched.fallback, &repeated.fallback, "{tag}: fallback");
            assert_eq!(&batched.meta, &repeated.meta, "{tag}: meta");
            assert_eq!(&batched.packet, &repeated.packet, "{tag}: packet");
        }
    }
    assert!(
        batched[8].fallback.is_none() && batched[8].meta.is_some(),
        "the startup slice must run on the GPU now that startup is ported"
    );
    assert!(batched[10].fallback.is_some(), "the unparseable slice must carry the fallback arm");
}

/// The app's registered geometry is the kernel's authority. An injected copy
/// that differs observably from the file copy changes the kernel's answer,
/// and re-registering the file text restores the baseline byte for byte, so
/// injection replaces (never merges) and the file is the fallback only while
/// nothing was registered.
#[test]
fn injected_geometry_outranks_the_file_copy() {
    let base = v77_line();
    let file =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    let request = race_exec::multi_request(&base, 10504, 1).expect("request");

    // Shift the nearest kind-0 corner event's distance by +30 m. No verified
    // identity field is touched, so the flattener accepts it, and the corner
    // table and the derived first_move_lane_point both move.
    let mut value = json::parse(&file).expect("geometry parses");
    let json::Value::Object(root) = &mut value else { panic!("geometry is an object") };
    let json::Value::Object(diagnostics) =
        root.get_mut("course_event_diagnostics").expect("course_event_diagnostics")
    else {
        panic!("course_event_diagnostics is an object")
    };
    let json::Value::Array(rows) = diagnostics.get_mut("rows").expect("rows") else {
        panic!("rows is an array")
    };
    let nearest = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let row = row.as_object()?;
            if row.get("kind")?.as_f64()? != 0. {
                return None;
            }
            Some((row.get("distance")?.as_f64()?, index))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("the fixture carries a kind-0 corner event");
    let json::Value::Object(row) = &mut rows[nearest.1] else {
        panic!("corner row is an object")
    };
    let json::Value::Number(distance) = row.get_mut("distance").expect("distance") else {
        panic!("distance is a number")
    };
    *distance += 30.;
    let mut perturbed = String::new();
    json::to_string_into(&value, &mut perturbed);

    let answer = |races: &mut GpuRaces| {
        let outcome = races.run_repeated(&request, &[7]).expect("run").remove(0);
        assert!(outcome.meta.is_some(), "the race should launch, got fallback {:?}", outcome.fallback);
        (outcome.meta.expect("meta"), outcome.packet.expect("packet"))
    };

    let mut races = GpuRaces::new().expect("device gate");
    let (baseline_meta, baseline_packet) = answer(&mut races);
    races.register_geometry(10504, &perturbed).expect("perturbed geometry accepted");
    let (injected_meta, injected_packet) = answer(&mut races);
    assert_ne!(
        injected_meta, baseline_meta,
        "the kernel answered identically to the file baseline, the injection did not take effect"
    );
    assert_ne!(injected_packet, baseline_packet, "the kernel's frames answered identically to the file baseline");

    races.register_geometry(10504, &file).expect("file text accepted");
    let (restored_meta, restored_packet) = answer(&mut races);
    assert_eq!(restored_meta, baseline_meta, "re-registering the file text must restore the baseline exactly");
    assert_eq!(restored_packet, baseline_packet, "restored frames must equal the baseline frames");
}

/// The app's vendor copy of 10504 is byte-equal to the engine copy, and with
/// that vendor text injected the v77 line-1 race stays bit-exact against the
/// CPU engine (frames compared byte for byte). Reads the app checkout through
/// `GLOBALSIM_APP_VENDOR` (a `vendor/GlobalSim` dir), default to the sibling
/// main app checkout.
#[test]
fn injected_app_vendor_geometry_keeps_the_race_bit_exact() {
    let vendor_dir = std::env::var("GLOBALSIM_APP_VENDOR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../Yet-Another-Umalator-Fork/vendor/GlobalSim")
        });
    let vendor_file = vendor_dir.join("data/course_geometry/10504.json");
    let vendor = std::fs::read_to_string(&vendor_file)
        .unwrap_or_else(|e| panic!("{} unreadable: {e}", vendor_file.display()));
    let engine =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("engine geometry file");
    assert_eq!(vendor, engine, "the app vendor geometry must equal the engine copy before the identity run");

    // The CPU reference lane keeps its registration flow, same text.
    globalsim::register_course_geometry(10504, vendor.clone());
    let base = v77_line();
    let request = race_exec::multi_request(&base, 10504, 1).expect("request");

    let mut races = GpuRaces::new().expect("device gate");
    races.register_geometry(10504, &vendor).expect("vendor geometry accepted");
    let outcome = races.run_repeated(&request, &[1]).expect("run").remove(0);
    let (Some(_meta), Some(packet)) = (outcome.meta, outcome.packet) else {
        panic!("the injected vendor race should launch, got fallback {:?}", outcome.fallback);
    };
    let cpu = dep_cpu::run_cpu_binary(&request, 1).expect("engine run");
    match (replay_packet::frame_bytes(&packet), replay_packet::frame_bytes(&cpu.packet)) {
        (Some(gpu), Some(cpu)) if gpu == cpu => {}
        (Some(gpu), Some(cpu)) => panic!(
            "frame bytes differ (gpu {} bytes, engine {} bytes)",
            gpu.len(),
            cpu.len()
        ),
        _ => panic!("a packet lacks its frame payload"),
    }
}

fn facts(extensions: &str, single_fp_config: u64, double_fp_config: u64) -> DeviceFacts {
    DeviceFacts { extensions: extensions.into(), single_fp_config, double_fp_config }
}

/// The pure refusal gate on synthetic device metadata, no device opened.
#[test]
fn gate_reasons_names_every_refusal() {
    let accepted = facts("cl_khr_int64_base_atomics cl_khr_fp64", 0xbf, 0x3f);
    let request = r#"{"courseId":10504,"seed":1,"experimentalRetailRuntime":true,"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(request, Some(&accepted)), None);

    let no_gpu = r#"cl_khr_fp16"#;
    assert_eq!(
        facade::gate_reasons(request, Some(&facts(no_gpu, 0xbf, 0x3f))).as_deref(),
        Some("missing cl_khr_fp64")
    );
    assert_eq!(
        facade::gate_reasons(request, None).as_deref(),
        Some("no GPU passes the exactness gate")
    );
    // 0x3f carries no correctly-rounded divide/sqrt bit, 0xbe no denormal bit.
    assert_eq!(
        facade::gate_reasons(request, Some(&facts("cl_khr_fp64", 0x3f, 0x3f))).as_deref(),
        Some("missing CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT")
    );
    assert_eq!(
        facade::gate_reasons(request, Some(&facts("cl_khr_fp64", 0xbe, 0x3f))).as_deref(),
        Some("missing CL_FP_DENORM in single_fp_config")
    );
    assert_eq!(
        facade::gate_reasons(request, Some(&facts("cl_khr_fp64", 0xbf, 0x3e))).as_deref(),
        Some("missing CL_FP_DENORM in double_fp_config")
    );

    // Request refusals, in the order the facade's domain gates ask them.
    // The startup flag is ported, so it no longer refuses, and a valid grid
    // key runs, so only the invalid grid shape, runtime and roster gates stay.
    let startup = r#"{"horses":[{}],"experimentalRetailRuntime":true,"experimentalRetailStartup":true}"#;
    assert_eq!(facade::gate_reasons(startup, Some(&accepted)), None);
    // An invalid grid key (too few keys, too few horses) quotes the engine's
    // message, matching prepare() and src/lib.rs:262-272.
    let grid = r#"{"horses":[{}],"experimentalRetailRuntime":true,"experimentalGrid":{}}"#;
    assert_eq!(
        facade::gate_reasons(grid, Some(&accepted)).as_deref(),
        Some("experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0.")
    );
    // A valid grid key on a 21-horse retail roster is now runnable.
    let grid_ok = format!(
        "{{\"experimentalRetailRuntime\":true,\"experimentalGrid\":{{\"columns\":20,\"rowSpacing\":3}},\"horses\":[{}]}}",
        vec!["{}"; 21].join(",")
    );
    assert_eq!(facade::gate_reasons(&grid_ok, Some(&accepted)), None);
    // A request without a retail flag routes to the classic lane, so the old
    // blanket non-retail refusal is gone. The classic refusals name what the
    // classic kernel does not port.
    let classic = r#"{"courseId":10504,"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(classic, Some(&accepted)), None);
    let legacy = r#"{"courseId":10504,"retailRuntime":true,"horses":[{}]}"#;
    assert_eq!(
        facade::gate_reasons(legacy, Some(&accepted)).as_deref(),
        Some("retail runtime belongs to the retail lane")
    );
    // A skilled classic race is no longer blanket-refused. The gate answers
    // with the same compiler prepare() uses, so the refusal names the skill
    // the compiler cannot lower.
    let classic_skills = classic_v77_with_skills(&["999999999"]);
    assert_eq!(
        facade::gate_reasons(&classic_skills, Some(&accepted)).as_deref(),
        Some("skillData has no entry for skill 999999999")
    );
    let classic_grid = format!(
        "{{\"courseId\":10504,\"experimentalGrid\":{{\"columns\":20,\"rowSpacing\":3}},\"horses\":[{}]}}",
        vec!["{}"; 21].join(",")
    );
    assert_eq!(
        facade::gate_reasons(&classic_grid, Some(&accepted)).as_deref(),
        Some("experimentalGrid belongs to the retail lane")
    );
    let classic_trace = r#"{"courseId":10504,"traceConservePower":true,"horses":[{}]}"#;
    assert_eq!(
        facade::gate_reasons(classic_trace, Some(&accepted)).as_deref(),
        Some("traceConservePower requires experimentalRetailRuntime")
    );
    // The flag-only systems are ported to the classic lane up to 2100 m, so
    // the gate accepts; above 2100 m the staminasyoubu injection falls back.
    let classic_stamina = r#"{"courseId":10504,"experimentalLaterStaminaMechanics":true,"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(classic_stamina, Some(&accepted)), None);
    let classic_stamina_far =
        r#"{"courseId":10108,"experimentalLaterStaminaMechanics":true,"horses":[{}]}"#;
    assert_eq!(
        facade::gate_reasons(classic_stamina_far, Some(&accepted)).as_deref(),
        Some("staminasyoubu, injected above 2100 m, stays outside the classic kernel")
    );
    // singleMode is ported on the classic lane: the +400 fold runs in the
    // adjusted stats, so the gate accepts it.
    let classic_single = r#"{"courseId":10504,"singleMode":true,"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(classic_single, Some(&accepted)), None);
    let flag_off = r#"{"courseId":10504,"experimentalRetailRuntime":false,"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(flag_off, Some(&accepted)), None);
    let empty = r#"{"horses":[],"experimentalRetailRuntime":true}"#;
    assert_eq!(
        facade::gate_reasons(empty, Some(&accepted)).as_deref(),
        Some("roster size outside the kernel's range")
    );
    let oversized = format!(
        "{{\"experimentalRetailRuntime\":true,\"horses\":[{}]}}",
        vec!["{}"; race_exec::RACE_MAXR + 1].join(",")
    );
    assert_eq!(
        facade::gate_reasons(&oversized, Some(&accepted)).as_deref(),
        Some("roster size outside the kernel's range")
    );
    let broken = "not json";
    assert!(facade::gate_reasons(broken, Some(&accepted)).is_some(), "a broken request is refused");
}

/// The v77 roster with the retail flags off and every skill slot removed, the
/// zero-skill shape the classic kernel lane answers.
fn classic_v77() -> String {
    let mut value = json::parse(&v77_line()).expect("v77 parses");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
            "experimentalRetailStartup",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
        if let Some(json::Value::Array(horses)) = map.get("horses") {
            let cleared: Vec<json::Value> = horses
                .iter()
                .map(|horse| {
                    let mut horse = horse.clone();
                    if let json::Value::Object(fields) = &mut horse {
                        fields.remove("skills");
                    }
                    horse
                })
                .collect();
            map.insert("horses".into(), json::arr(cleared));
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The zero-skill classic v77 roster with the given skill ids on the first
/// runner.
fn classic_v77_with_skills(ids: &[&str]) -> String {
    let mut value = json::parse(&classic_v77()).expect("classic v77 parses");
    let json::Value::Object(root) = &mut value else {
        panic!("classic v77 is an object");
    };
    let horses = match root.remove("horses") {
        Some(json::Value::Array(list)) => list,
        _ => panic!("classic v77 has a horses array"),
    };
    let skills = json::arr(
        ids.iter().map(|id| json::Value::String((*id).to_string())).collect(),
    );
    let updated: Vec<json::Value> = horses
        .into_iter()
        .enumerate()
        .map(|(i, horse)| {
            if i != 0 {
                return horse;
            }
            let json::Value::Object(mut fields) = horse else {
                panic!("horse is an object");
            };
            fields.insert("skills".to_string(), skills.clone());
            json::Value::Object(fields)
        })
        .collect();
    root.insert("horses".to_string(), json::arr(updated));
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The facade's lane flip: a non-retail zero-skill request runs on the classic
/// kernel lane in the same batch as a retail request, and its compact meta,
/// packet and full reply match the engine.
#[test]
fn a_classic_request_answers_from_the_gpu_lane() {
    let classic = classic_v77();
    let retail =
        race_exec::skills_request(&v77_line(), 10504, 1, 0, &SKILLS, None).expect("skills request");
    let requests = [classic.clone(), retail];
    let mut gpu = GpuRaces::new().expect("device gate");
    let outcomes = gpu.run_batch(&requests, &[1, 1]).expect("batch");
    assert_eq!(outcomes.len(), requests.len());

    let mut problems: Vec<String> = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let tag = format!("request {index}");
        let outcome = &outcomes[index];
        let (Some(meta), Some(packet)) = (&outcome.meta, &outcome.packet) else {
            problems
                .push(format!("{tag}: expected a compact answer, got fallback {:?}", outcome.fallback));
            continue;
        };
        let cpu = match dep_cpu::run_cpu_binary(request, 1) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: engine run failed: {reason}"));
                continue;
            }
        };
        let cpu_reply = json::parse(&cpu.reply).expect("engine reply parses");
        let gpu_meta = json::parse(meta).expect("facade meta parses");
        check_num(&mut problems, &tag, "playerHorseIndex", &cpu_reply, &gpu_meta);
        let cpu_results = cpu_reply.get_array("results").expect("engine results");
        let gpu_results = gpu_meta.get_array("results").expect("facade results");
        if cpu_results.len() != gpu_results.len() {
            problems.push(format!(
                "{tag}: results len cpu {}, gpu {}",
                cpu_results.len(),
                gpu_results.len()
            ));
            continue;
        }
        for (runner, (cpu_row, gpu_row)) in cpu_results.iter().zip(gpu_results).enumerate() {
            let tag = format!("{tag} runner {runner}");
            for field in [
                "finishOrder",
                "finishTimeRaw",
                "rushingStartCount",
                "rushingPenaltyCount",
                "lastSpurtStartDistance",
                "startDelayTime",
            ] {
                check_num(&mut problems, &tag, field, cpu_row, gpu_row);
            }
            // hpEmptyOnRace is a bool where the retail config ran and null
            // where it did not (src/lib.rs:994).
            let hp_matches = match (cpu_row.get("hpEmptyOnRace"), gpu_row.get("hpEmptyOnRace")) {
                (Some(json::Value::Null), Some(json::Value::Null)) => true,
                (Some(json::Value::Bool(cpu)), Some(json::Value::Bool(gpu))) => cpu == gpu,
                _ => false,
            };
            if !hp_matches {
                problems.push(format!(
                    "{tag} hpEmptyOnRace: cpu {:?}, gpu {:?}",
                    cpu_row.get("hpEmptyOnRace"),
                    gpu_row.get("hpEmptyOnRace")
                ));
            }
        }
        let cpu_events = events_multiset(cpu_reply.get("skillEvents").expect("engine skillEvents"));
        let gpu_events = events_multiset(gpu_meta.get("skillEvents").expect("facade skillEvents"));
        if cpu_events != gpu_events {
            problems.push(format!("{tag}: skillEvents multiset cpu {cpu_events:?}, gpu {gpu_events:?}"));
        }
        match (replay_packet::frame_bytes(packet), replay_packet::frame_bytes(&cpu.packet)) {
            (Some(gpu_frames), Some(cpu_frames)) if gpu_frames == cpu_frames => {}
            (Some(gpu_frames), Some(cpu_frames)) => problems.push(format!(
                "{tag}: packet frame bytes differ (gpu {} bytes, engine {} bytes)",
                gpu_frames.len(),
                cpu_frames.len()
            )),
            _ => problems.push(format!("{tag}: a packet lacks its frame payload")),
        }
    }

    let full = gpu.run_batch_full(&[classic.clone()], &[7]).expect("full batch");
    let cpu = dep_cpu::run_cpu_binary(&classic, 7).expect("engine full run");
    if full[0].reply.as_deref() != Some(cpu.reply.as_str()) {
        let gpu_text = full[0]
            .reply
            .clone()
            .unwrap_or_else(|| format!("fallback {:?}", full[0].fallback));
        let common = gpu_text
            .as_bytes()
            .iter()
            .zip(cpu.reply.as_bytes())
            .take_while(|(gpu, cpu)| gpu == cpu)
            .count();
        let gpu_tail = &gpu_text[common..(common + 80).min(gpu_text.len())];
        let cpu_tail = &cpu.reply[common..(common + 80).min(cpu.reply.len())];
        problems.push(format!(
            "classic full reply differs at byte {common}: gpu {gpu_tail:?}, engine {cpu_tail:?}"
        ));
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
