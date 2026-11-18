//! The batch facade gate (handoff step 8): the compact meta plus the
//! GSIMRP01 packet the app's run_compact lane consumes. Every field
//! compact_record reads is compared against the engine's own `write_json`
//! reply on the same request, and the packet's frame bytes against the
//! engine's `into_binary` packet. A startup request inside the same batch
//! must come back as a fallback without disturbing its siblings, and
//! `gate_reasons` is checked as a pure function on synthetic device facts.

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

/// The batch contract's fallback arm: an unsupported request inside a batch
/// answers with the reason only, and its siblings still get meta and packet.
#[test]
fn a_startup_request_falls_back_and_its_siblings_stay_whole() {
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
        if index == 1 {
            continue;
        }
        assert!(
            outcome.meta.is_some() && outcome.packet.is_some() && outcome.fallback.is_none(),
            "sibling {index} should carry a compact answer, got fallback {:?}",
            outcome.fallback
        );
    }
    let middle = &outcomes[1];
    let reason = middle.fallback.as_deref().unwrap_or_default();
    assert!(reason.contains("startup"), "the fallback should name the domain, got {reason:?}");
    assert!(middle.meta.is_none() && middle.packet.is_none(), "a fallback race carries no meta or packet");
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
/// of the mixed batch. A flagged and an unparseable request ride along to
/// pin the fallback arms in both lanes.
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
    assert!(batched[8].fallback.is_some() && batched[8].meta.is_none(), "the startup slice must carry the fallback arm");
    assert!(batched[10].fallback.is_some(), "the unparseable slice must carry the fallback arm");
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
    let startup = r#"{"horses":[{}],"experimentalRetailRuntime":true,"experimentalRetailStartup":true}"#;
    assert_eq!(facade::gate_reasons(startup, Some(&accepted)).as_deref(), Some("startup mode"));
    let grid = r#"{"horses":[{}],"experimentalRetailRuntime":true,"experimentalGrid":{}}"#;
    assert_eq!(facade::gate_reasons(grid, Some(&accepted)).as_deref(), Some("grid mode"));
    let legacy = r#"{"horses":[{}]}"#;
    assert_eq!(facade::gate_reasons(legacy, Some(&accepted)).as_deref(), Some("non-retail runtime"));
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
