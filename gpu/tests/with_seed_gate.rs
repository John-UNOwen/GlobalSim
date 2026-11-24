//! `with_seed` gate: a prepared clone must carry exactly what `prepare` of
//! that seed produces, including the per-runner solver pair the later-stamina
//! systems spend. The seed enters the prepared inputs only through `RI_SEED`
//! and the pair, so a clone that skips the pair answers with the template
//! seed's draws.
//!
//! `run_repeated` builds every entry past the first with `with_seed`, while
//! `run_batch` prepares each seed alone. The two must answer byte for byte,
//! and the CPU engine is the oracle for both.

#![cfg(feature = "race")]

use globalsim_gpu::facade::GpuRaces;
use globalsim_gpu::{dep_cpu, json, race_exec, replay_packet};

use json::Value;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request file");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// Register course 10504's file geometry on the engine side and return its
/// text, so the device side can register the same copy.
fn register_geometry(course: i32) -> String {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry file");
    globalsim::register_course_geometry(course, geometry.clone());
    geometry
}

/// The v77 field with experimentalLaterStaminaMechanics on, plus the wave 12b
/// mid-race recovery (kind 9 at distance_rate 55). Its activation is the
/// engine's one in-window clear of the keep latch, so the compete and secure
/// rolls draw from the solver stream and the seed's pair decides their ticks.
fn later_stamina_request() -> String {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("experimentalLaterStaminaMechanics".into(), Value::Bool(true));
        let def = "{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[\
            {\"precondition\":\"\",\"condition\":\"distance_rate>=55\",\"cooldown\":5000000,\
            \"baseDuration\":30000,\"effects\":[{\"type\":9,\"target\":1,\"targetValue\":0,\
            \"additionalActivate\":0,\"valueUsage\":1,\"durationUsage\":1,\"targetIsDebuff\":false,\
            \"modifier\":80000}]}]}";
        let skill_data = map.entry("skillData".to_string()).or_insert_with(|| json::obj(vec![]));
        if let Value::Object(skill_data) = skill_data {
            skill_data.insert("669606".to_string(), json::parse(def).expect("recovery def"));
        }
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for horse in horses.iter_mut() {
            let Value::Object(horse) = horse else { unreachable!() };
            horse.insert(
                "skills".into(),
                json::arr(vec![json::obj(vec![
                    ("skillId", Value::Number(669606.)),
                    ("level", Value::Number(1.)),
                ])]),
            );
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The compact answer bits of one facade outcome, for a short failure message.
fn outcome_bits(outcome: &globalsim_gpu::facade::BatchOutcome) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(outcome.meta.as_deref().unwrap_or("").as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(outcome.packet.as_deref().unwrap_or(&[]));
    bytes
}

fn first_byte_diff(left: &[u8], right: &[u8]) -> Option<usize> {
    left.iter().zip(right).position(|(a, b)| a != b)
}

/// The only seed-dependent words are `RI_SEED` and the per-runner solver pair,
/// both in the per-race row, so the clone path and the fresh-parse path must
/// land on the same bytes at every seed. Non-vacuity rides on the pair actually
/// differing for every seed while the shared pd row does not move.
#[test]
fn with_seed_clone_carries_a_fresh_parses_bytes() {
    register_geometry(10504);
    let request = later_stamina_request();
    let mut ctx = race_exec::RaceContext::open().expect("context");
    let template = ctx.prepare(&request, 1).expect("template prepare");
    assert!(template.host_fallback_reason().is_none(), "the template must be a parsed prepare");

    let seeds: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut prepared_bits: Vec<(Vec<u32>, Vec<u64>, Vec<u32>)> = Vec::new();
    for &seed in &seeds {
        let clone = template.with_seed(seed);
        let fresh = ctx.prepare(&request, seed as i64).expect("prepare");
        assert!(fresh.host_fallback_reason().is_none(), "seed {seed}: the parse flagged");
        let (clone_bits, fresh_bits) = (clone.host_input_bits(), fresh.host_input_bits());
        assert!(
            clone_bits == fresh_bits,
            "seed {seed}: the with_seed clone differs from prepare of that seed\n{}",
            bit_diff_text(&clone_bits, &fresh_bits)
        );
        prepared_bits.push(fresh_bits);
    }

    // The pair is the whole seed-dependent delta: exactly RI_SEED and the two
    // pair words per runner differ between neighbouring seeds, and the shared
    // pd row stays byte-identical.
    let runners = template.runners();
    let mut pair_slots: Vec<usize> = vec![race_exec::RI_SEED];
    for runner in 0..runners {
        let base = race_exec::RI_RUN0 + race_exec::RI_RUN_STRIDE * runner;
        pair_slots.push(base + race_exec::RR_R30_HI);
        pair_slots.push(base + race_exec::RR_R30_LO);
    }
    for (index, pair) in prepared_bits.windows(2).enumerate() {
        let (left, right) = (&pair[0].0, &pair[1].0);
        let moved: Vec<usize> = left
            .iter()
            .zip(right)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(slot, _)| slot)
            .collect();
        assert_eq!(
            moved, pair_slots,
            "seeds {} and {}: the ri delta is not RI_SEED plus the solver pair",
            seeds[index], seeds[index + 1]
        );
        assert_eq!(
            pair[0].1, pair[1].1,
            "seeds {} and {}: the shared pd row moved with the seed",
            seeds[index], seeds[index + 1]
        );
    }
}

fn bit_diff_text(
    left: &(Vec<u32>, Vec<u64>, Vec<u32>),
    right: &(Vec<u32>, Vec<u64>, Vec<u32>),
) -> String {
    let ri: Vec<usize> = left
        .0
        .iter()
        .zip(&right.0)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(slot, _)| slot)
        .collect();
    let pd: Vec<usize> = left
        .1
        .iter()
        .zip(&right.1)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(slot, _)| slot)
        .collect();
    let sk = left.2.iter().zip(&right.2).filter(|(a, b)| a != b).count();
    format!("ri slots {ri:?}, pd slots {pd:?}, sk words {sk}")
}

/// The hard contract of the prepare-once lane on a later-stamina field:
/// `run_repeated` (with_seed clones) answers byte-identically to `run_batch`
/// (one prepare per seed), and the kernel is bit-exact against the CPU
/// engine. The engine check compares the full reply bytes (v77 retains every
/// frame) and the compact packet's frame payload.
#[test]
fn later_stamina_repeated_matches_batch_and_engine() {
    let geometry = register_geometry(10504);
    let request = later_stamina_request();
    let seeds: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let requests: Vec<String> = seeds.iter().map(|_| request.clone()).collect();

    let mut races = GpuRaces::new().expect("device gate");
    races.register_geometry(10504, &geometry).expect("geometry");
    let batched = races.run_batch(&requests, &seeds).expect("run_batch");
    let repeated = races.run_repeated(&request, &seeds).expect("run_repeated");
    assert_eq!(batched.len(), seeds.len());
    assert_eq!(repeated.len(), seeds.len());

    let mut problems: Vec<String> = Vec::new();
    for (index, &seed) in seeds.iter().enumerate() {
        let (batched, repeated) = (&batched[index], &repeated[index]);
        if batched.fallback != repeated.fallback {
            problems.push(format!(
                "seed {seed}: fallback differs, run_batch {:?}, run_repeated {:?}",
                batched.fallback, repeated.fallback
            ));
            continue;
        }
        if batched.meta != repeated.meta {
            let (left, right) = (outcome_bits(batched), outcome_bits(repeated));
            problems.push(format!(
                "seed {seed}: meta differs, first byte {}",
                first_byte_diff(&left, &right).map_or("end".to_string(), |at| at.to_string())
            ));
        }
        if batched.packet != repeated.packet {
            let (left, right) = (
                batched.packet.as_deref().unwrap_or(&[]),
                repeated.packet.as_deref().unwrap_or(&[]),
            );
            problems.push(format!(
                "seed {seed}: packet differs, first byte {}",
                first_byte_diff(left, right).map_or("end".to_string(), |at| at.to_string())
            ));
        }
        if batched.fallback.is_some() || batched.meta.is_none() || batched.packet.is_none() {
            problems.push(format!(
                "seed {seed}: the fixture should launch, got fallback {:?}",
                batched.fallback
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "run_repeated differs from run_batch on {} seeds:\n{}",
        problems.len(),
        problems.join("\n")
    );

    // The full lane against the CPU engine, then the compact packet's frames
    // against the engine's own packet.
    let full = races.run_batch_full(&requests, &seeds).expect("full batch");
    for (index, &seed) in seeds.iter().enumerate() {
        let cpu = dep_cpu::run_cpu_binary(&request, seed as i64).expect("engine run");
        match &full[index].reply {
            Some(reply) if *reply == cpu.reply => {}
            Some(reply) => problems.push(format!(
                "seed {seed}: full reply differs from the engine ({} vs {} bytes)",
                reply.len(),
                cpu.reply.len()
            )),
            None => problems.push(format!(
                "seed {seed}: full lane fell back: {:?}",
                full[index].fallback
            )),
        }
        let packet = batched[index].packet.as_ref().expect("compact packet");
        match (
            replay_packet::frame_bytes(packet),
            replay_packet::frame_bytes(&cpu.packet),
        ) {
            (Some(gpu), Some(engine)) if gpu == engine => {}
            (Some(gpu), Some(engine)) => problems.push(format!(
                "seed {seed}: packet frames differ (gpu {} bytes, engine {} bytes)",
                gpu.len(),
                engine.len()
            )),
            _ => problems.push(format!("seed {seed}: a packet lacks its frame payload")),
        }
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
