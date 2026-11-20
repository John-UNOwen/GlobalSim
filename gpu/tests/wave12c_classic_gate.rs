//! Wave 12c: classic-lane parity for condition leaves. Front A drives the
//! sampled leaves: a 6697xx battery sends every classic sampler name through
//! the host replay of the engine's builder sampler in one-runner fields on
//! course 10504, seeds 1..3, with one scene sharing one builder stream in the
//! engine's skilldata order, duplicate-entry window reuse, and the engine's
//! drop semantics for condition filter errors. Front B drives the live-state
//! leaves: the same battery ids in twelve-runner fields, where the kernel's
//! arm programs read order, order_rate, hp_per, the race clock, near and
//! surrounded counts, and the vision cone, fired and never-fired pairs on
//! every reader, plus the merged table's own ids whose conditions read live
//! state. Frames bit-exact against the CPU engine, full replies byte-equal,
//! and every intended skill seen activating in the kernel's event stream.
//! Course 10504 carries four corners, two uphill and one downhill slope band
//! and three straights, so every sampled chain is nonempty and every sampled
//! window is traversed.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COURSE: i32 = 10504;

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

/// A battery skill: SpeedUp +3.0 on self for six seconds, one-shot cooldown.
/// A fired window moves every later frame, so a mistimed sample cannot hide.
fn def(cond: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":5000000,\"baseDuration\":60000,\"effects\":[{{\"type\":1,\"target\":1,\"targetValue\":0,\"additionalActivate\":0,\"valueUsage\":1,\"durationUsage\":1,\"targetIsDebuff\":false,\"modifier\":30000}}]}}]}}"
    )
}

/// The v77 solo request for one seed with battery definitions overlaid on the
/// v77 skillData and horse 0 re-rostered onto the given (sid, condition)s.
fn solo_battery(seed: i64, entries: &[(i64, &str)]) -> String {
    let mut value = json::parse(&v77_line()).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
        if let Some(json::Value::Object(skill_data)) = map.get_mut("skillData") {
            for (sid, cond) in entries {
                let parsed = json::parse(&def(cond)).expect("battery def parses");
                skill_data.insert(sid.to_string(), parsed);
            }
        }
        if let Some(json::Value::Array(horses)) = map.get_mut("horses") {
            if let Some(json::Value::Object(first)) = horses.get_mut(0) {
                first.insert(
                    "skills".into(),
                    json::arr(
                        entries
                            .iter()
                            .map(|(sid, _)| {
                                json::obj(vec![
                                    ("skillId", json::Value::Number(*sid as f64)),
                                    ("level", json::Value::Number(1.)),
                                ])
                            })
                            .collect(),
                    ),
                );
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::solo_request(&text, COURSE, seed).expect("solo request")
}

/// The kernel's unified skillEvents stream, numeric activation rows only.
fn decode_skill_events(words: &[u32]) -> Vec<(i64, i64, i64, f64)> {
    let mut fired = Vec::new();
    let mut i = 0;
    while i < words.len() {
        match words[i] {
            0 if i + 9 <= words.len() => {
                let t = f64::from_bits(u64::from(words[i + 6]) << 32 | u64::from(words[i + 5]));
                fired.push((
                    words[i + 1] as i64,
                    words[i + 2] as i64,
                    words[i + 3] as i64,
                    t,
                ));
                i += 9;
            }
            1 if i + 6 <= words.len() => i += 6,
            _ => break,
        }
    }
    fired
}

/// One solo scene: the kernel answers without a fallback, frames are bit-exact,
/// the full reply is byte-equal to the engine binary, and every expected skill
/// id is seen activating. Problems collect instead of stopping the sweep.
fn gate(label: &str, request: &str, seed: i64, expect: &[i64], problems: &mut Vec<String>) {
    gate_field(label, request, seed, expect, &[], problems);
}

/// The same gate with a forbidden list: ids that must never activate, so a
/// reader that fires when the engine's comparison does not shows up here.
fn gate_field(
    label: &str,
    request: &str,
    seed: i64,
    expect: &[i64],
    forbid: &[i64],
    problems: &mut Vec<String>,
) {
    let fired = gate_scene(label, request, seed, problems);
    for sid in expect {
        if !fired.iter().any(|&(_, _, s, _)| s == *sid) {
            problems.push(format!("{label} seed {seed}: skill {sid} never activated"));
        }
    }
    for sid in forbid {
        if fired.iter().any(|&(_, _, s, _)| s == *sid) {
            problems.push(format!("{label} seed {seed}: skill {sid} activated but must not"));
        }
    }
}

/// The same checks without fired-list expectations, returning the activation
/// rows for callers whose expectation spans several seeds.
fn gate_scene(
    label: &str,
    request: &str,
    seed: i64,
    problems: &mut Vec<String>,
) -> Vec<(i64, i64, i64, f64)> {
    let out = classic_exec::run(&[(request.to_string(), seed)]).expect("classic kernel run");
    let race = &out.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("{label} seed {seed}: kernel fallback {reason}"));
        return Vec::new();
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(r) => r,
        Err(reason) => {
            problems.push(format!("{label} seed {seed}: cpu race failed: {reason}"));
            return Vec::new();
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{label} seed {seed}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
        return Vec::new();
    }
    'frames: for (tick, (cpu_frame, gpu_frame)) in
        cpu.frames.iter().zip(race.frames.iter()).enumerate()
    {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
            for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cpu.to_bits() != gpu.to_bits() {
                    problems.push(format!(
                        "{label} seed {seed}: tick {tick} runner {runner} column {column} cpu {} gpu {}",
                        cpu.to_bits(),
                        gpu.to_bits()
                    ));
                    break 'frames;
                }
            }
        }
    }
    let full = match dep_cpu::with_retain_all_frames(request) {
        Ok(r) => r,
        Err(reason) => {
            problems.push(format!("{label} seed {seed}: retain request failed: {reason}"));
            return Vec::new();
        }
    };
    let gpu_reply = match classic_reply::build(&full, seed, race) {
        Ok(r) => r,
        Err(reason) => {
            problems.push(format!("{label} seed {seed}: full reply build failed: {reason}"));
            return Vec::new();
        }
    };
    let engine_reply = match dep_cpu::run_cpu_binary(request, seed) {
        Ok(r) => r.reply,
        Err(reason) => {
            problems.push(format!("{label} seed {seed}: cpu binary failed: {reason}"));
            return Vec::new();
        }
    };
    if gpu_reply != engine_reply {
        problems.push(format!("{label} seed {seed}: full reply differs from the engine"));
        return Vec::new();
    }
    decode_skill_events(&race.skev)
}

/// The full twelve-horse v77 field for one seed: battery definitions overlaid
/// on the v77 skillData, the listed horses re-rostered onto (sid, condition)
/// entries. Live-state readers need a field: ranks, near counts and the
/// vision cone all read other runners.
fn field_battery(seed: i64, hosts: &[(usize, &[(i64, &str)])]) -> String {
    let mut value = json::parse(&v77_line()).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
        if let Some(json::Value::Object(skill_data)) = map.get_mut("skillData") {
            for (_, entries) in hosts {
                for &(sid, cond) in entries.iter() {
                    let parsed = json::parse(&def(cond)).expect("battery def parses");
                    skill_data.insert(sid.to_string(), parsed);
                }
            }
        }
        if let Some(json::Value::Array(horses)) = map.get_mut("horses") {
            for (index, entries) in hosts {
                if let Some(json::Value::Object(horse)) = horses.get_mut(*index) {
                    horse.insert(
                        "skills".into(),
                        json::arr(
                            entries
                                .iter()
                                .map(|(sid, _)| {
                                    json::obj(vec![
                                        ("skillId", json::Value::Number(*sid as f64)),
                                        ("level", json::Value::Number(1.)),
                                    ])
                                })
                                .collect(),
                        ),
                    );
                }
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::multi_request(&text, COURSE, seed).expect("field request")
}

/// The full field with chosen horses re-rostered onto real skillData ids,
/// no definition overlay: the merged table's own conditions drive the kernel.
fn field_real(seed: i64, hosts: &[(usize, &[i64])]) -> String {
    let mut value = json::parse(&v77_line()).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
        if let Some(json::Value::Array(horses)) = map.get_mut("horses") {
            for (index, ids) in hosts {
                if let Some(json::Value::Object(horse)) = horses.get_mut(*index) {
                    horse.insert(
                        "skills".into(),
                        json::arr(
                            ids.iter()
                                .map(|sid| {
                                    json::obj(vec![
                                        ("skillId", json::Value::Number(*sid as f64)),
                                        ("level", json::Value::Number(1.)),
                                    ])
                                })
                                .collect(),
                        ),
                    );
                }
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::multi_request(&text, COURSE, seed).expect("field request")
}

/// The eighteen classic sampler names, one synthetic skill each. The args are
/// the engine filter bodies' accepted ones, and course 10504's geometry keeps
/// every chain nonempty: corners at 375, 625, 1200 and 1450, uphill bands at
/// 125, 325 and 1825, a downhill band at 625, straights closing at 375, 1200
/// and 2000.
const SAMPLED: [(i64, &str); 18] = [
    (669701, "all_corner_random==1"),
    (669702, "straight_random==1"),
    (669703, "corner_random==1"),
    (669704, "is_finalcorner_random==1"),
    (669705, "phase_random==1"),
    (669706, "phase_firsthalf_random==1"),
    (669707, "phase_firstquarter_random==1"),
    (669708, "phase_laterhalf_random==1"),
    (669709, "phase_corner_random==2"),
    (669710, "phase_straight_random==1"),
    (669711, "phase_first_half_straight_random==1"),
    (669712, "phase_latter_half_straight_random==1"),
    (669713, "down_slope_random==1"),
    (669714, "up_slope_random==1"),
    (669715, "up_slope_random_later_half==1"),
    (669716, "distance_rate_after_random==20"),
    (669717, "run_at_full_speed_random==1"),
    (669718, "last_straight_random==1"),
];

/// One sampled name per solo field, every seed. A bit-exact race that never
/// activates the skill would prove nothing, so the activation is a gate too.
#[test]
fn every_sampled_name_answers_bit_exact_one_at_a_time() {
    let mut problems: Vec<String> = Vec::new();
    for (sid, cond) in SAMPLED {
        for seed in SEEDS {
            let label = format!("sid {sid} {cond}");
            let request = solo_battery(seed, &[(sid, cond)]);
            gate(&label, &request, seed, &[sid], &mut problems);
        }
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// All eighteen sampled skills on one horse: one skilldata vector, so the
/// builder sampler runs eighteen draws in the engine's skilldata order and a
/// wrong draw order, key or fold moves a later window and breaks the frames.
#[test]
fn all_sampled_names_share_one_builder_stream_in_wire_order() {
    let mut problems: Vec<String> = Vec::new();
    let entries: Vec<(i64, &str)> = SAMPLED.to_vec();
    let expect: Vec<i64> = SAMPLED.iter().map(|(sid, _)| *sid).collect();
    for seed in SEEDS {
        gate("eighteen on one horse", &solo_battery(seed, &entries), seed, &expect, &mut problems);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// Duplicate entries of the same skill id share the sampled windows and the
/// engine raises no second draw for them, while the following skill still
/// draws from the shared stream. A double draw would move the third window and
/// break the frames. The pair of different ids on the same condition samples
/// twice, because the engine's reuse key carries the cooldown key.
#[test]
fn duplicate_entries_reuse_windows_and_same_chain_other_ids_do_not() {
    let mut problems: Vec<String> = Vec::new();
    for seed in SEEDS {
        let request = solo_battery(
            seed,
            &[
                (669701, "all_corner_random==1"),
                (669701, "all_corner_random==1"),
                (669702, "straight_random==1"),
            ],
        );
        gate("duplicate reuse", &request, seed, &[669701, 669702], &mut problems);
        let request = solo_battery(
            seed,
            &[(669701, "all_corner_random==1"), (669719, "all_corner_random==1")],
        );
        gate("same chain other id", &request, seed, &[669701, 669719], &mut problems);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// A classic condition build error does not error the engine's race: the
/// classic path drops just that skill and answers without it
/// (lib.rs:728-739). The kernel does the same: it answers bit-exact, the
/// skill is absent from the event stream, and the engine's reply for the
/// broken request is byte-identical to the same race without the skill. The
/// mixed scene proves the drop leaves the surviving skills' sampler draws
/// untouched: the sibling fires at the engine's sampled window.
#[test]
fn sampled_condition_build_errors_drop_the_skill_like_the_engine() {
    register_geometry(COURSE);
    let cases: &[(&str, &str)] = &[
        ("669731", "straight_random>=1"),
        ("669732", "phase_random!=1"),
        ("669733", "all_corner_random==0"),
        ("669734", "is_finalcorner_random==2"),
        ("669735", "run_at_full_speed_random==0"),
        ("669736", "up_slope_random_later_half<=1"),
        ("669737", "straight_random==1&all_corner_random==1"),
        ("669738", "all_corner_random==1@straight_random==1"),
    ];
    let mut problems: Vec<String> = Vec::new();
    for (sid_text, cond) in cases {
        let sid: i64 = sid_text.parse().expect("sid");
        let broken = solo_battery(1, &[(sid, *cond)]);
        let clean = solo_battery(1, &[]);
        let engine_broken = match dep_cpu::run_cpu_binary(&broken, 1) {
            Ok(r) => r.reply,
            Err(reason) => {
                problems.push(format!("{sid} {cond}: cpu binary failed: {reason}"));
                continue;
            }
        };
        let engine_clean = match dep_cpu::run_cpu_binary(&clean, 1) {
            Ok(r) => r.reply,
            Err(reason) => {
                problems.push(format!("{sid} {cond}: clean race failed: {reason}"));
                continue;
            }
        };
        if engine_broken != engine_clean {
            problems.push(format!("{sid} {cond}: the engine did not drop the skill cleanly"));
            continue;
        }
        gate(&format!("{sid} {cond}"), &broken, 1, &[], &mut problems);
        let out = classic_exec::run(&[(broken, 1i64)]).expect("classic kernel run");
        let fired = decode_skill_events(&out.races[0].skev);
        if fired.iter().any(|&(_, _, s, _)| s == sid) {
            problems.push(format!("{sid} {cond}: the kernel ran a skill the engine dropped"));
        }
    }
    let mixed = solo_battery(
        1,
        &[(669731, "straight_random>=1"), (669702, "straight_random==1")],
    );
    gate("drop beside live skill", &mixed, 1, &[669702], &mut problems);
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

// Front B battery: live-state leaves in a twelve-runner field. The fired side
// uses thresholds the readers must satisfy (the engine's constructor pass
// fires the ungated hp_per and clock leaves at t=0, the field-guarded ones
// from the first rank tick, the band leaf mid-race); the never side uses
// thresholds no reader value can reach (rank beyond the field, counts beyond
// the runner count, ratios beyond one), so a reader that misfires shows up in
// the event stream even where a frame difference would not.
const LIVE_FIRED: [(i64, usize, &str); 10] = [
    (669750, 0, "accumulatetime>=0"),
    (669751, 0, "hp_per>=0"),
    (669752, 0, "order>=1"),
    (669753, 0, "order_rate>=1"),
    (669754, 1, "visiblehorse>=0"),
    (669755, 1, "near_count>=0"),
    (669756, 1, "near_infront_count>=0"),
    (669757, 1, "is_surrounded!=2"),
    (669758, 2, "accumulatetime>=5&order<=8&hp_per>=50"),
    (669759, 2, "visiblehorse>=0@hp_per>=1000"),
];
const LIVE_NEVER: [(i64, usize, &str); 9] = [
    (669760, 0, "accumulatetime>=1000000"),
    (669761, 0, "hp_per>=101"),
    (669762, 0, "order>=13"),
    (669763, 0, "order_rate>=1000"),
    (669764, 1, "visiblehorse>=12"),
    (669765, 1, "near_count>=12"),
    (669766, 1, "near_infront_count>=12"),
    (669767, 2, "is_surrounded==2"),
    (669768, 2, "near_count>=99@order>=13"),
];

/// The fired and never hosts grouped per horse, one entry per horse.
fn live_hosts() -> Vec<(usize, Vec<(i64, &'static str)>)> {
    let mut hosts: Vec<(usize, Vec<(i64, &'static str)>)> = Vec::new();
    for (sid, horse, cond) in LIVE_FIRED.iter().chain(LIVE_NEVER.iter()) {
        match hosts.iter_mut().find(|(h, _)| *h == *horse) {
            Some(g) => g.1.push((*sid, *cond)),
            None => hosts.push((*horse, vec![(*sid, *cond)])),
        }
    }
    hosts
}

/// Every ported live reader, fired and never side, in one twelve-runner
/// scene per seed. Frames and replies pin the kernel's reader formulas to the
/// engine's; the fired and forbid lists pin the activation times.
#[test]
fn live_state_leaves_answer_both_directions_in_a_field() {
    register_geometry(COURSE);
    let hosts = live_hosts();
    let host_refs: Vec<(usize, &[(i64, &str)])> =
        hosts.iter().map(|(h, e)| (*h, e.as_slice())).collect();
    let expect: Vec<i64> = LIVE_FIRED.iter().map(|(sid, _, _)| *sid).collect();
    let forbid: Vec<i64> = LIVE_NEVER.iter().map(|(sid, _, _)| *sid).collect();
    let mut problems: Vec<String> = Vec::new();
    for seed in SEEDS {
        let request = field_battery(seed, &host_refs);
        gate_field("live-state battery", &request, seed, &expect, &forbid, &mut problems);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

// The merged table's own ids whose conditions read live state, one white-copy
// family per band plus the gold and start-up copies: order and order_rate
// bands at the final corner and the closing straights, a near-count id, the
// surrounded and clock ids, and the near-count-with-clock pack id. The v77
// field is 9 horses; every id is hosted. wisdomCheck ids (all but the gold
// copies and the clock ids) lose the per-(horse, seed) lottery on some seeds,
// so each id must appear in the fired set on at least one of the six seeds.
const REAL_HOSTS: [(usize, &[i64]); 9] = [
    (0, &[900041, 900061]),
    (1, &[100041, 100281]),
    (2, &[900201, 900311]),
    (3, &[900521, 100201, 100521]),
    (4, &[200401, 200481]),
    (5, &[201591, 201592]),
    (6, &[900071, 900251]),
    (7, &[100311, 100061]),
    (8, &[100071]),
];

const REAL_SEEDS: [i64; 6] = [1, 2, 3, 4, 5, 6];

/// Real ids, no definition overlay: the kernel programs the merged table's
/// conditions and answers bit-exact on every seed. The reply check forces the
/// kernel's fired set to equal the engine's, and every listed id must appear
/// in that set on at least one seed: wisdomCheck ids lose a per-(horse, seed)
/// lottery sometimes, so a seed with a suppressed id is not a failure.
#[test]
fn real_live_leaf_ids_answer_bit_exact() {
    register_geometry(COURSE);
    let expect: Vec<i64> = REAL_HOSTS
        .iter()
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect();
    let mut problems: Vec<String> = Vec::new();
    let mut seen: Vec<i64> = Vec::new();
    for seed in REAL_SEEDS {
        let request = field_real(seed, &REAL_HOSTS);
        let fired = gate_scene("real live-leaf ids", &request, seed, &mut problems);
        for &(_, _, sid, _) in &fired {
            if expect.contains(&sid) && !seen.contains(&sid) {
                seen.push(sid);
            }
        }
    }
    for sid in &expect {
        if !seen.contains(sid) {
            problems.push(format!("real live-leaf ids: skill {sid} never activated on any seed"));
        }
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
