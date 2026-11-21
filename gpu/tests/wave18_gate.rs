//! Wave 18: the injected asitame release under high-power rosters on the
//! classic lane, and experimentalLaterStaminaMechanics on the retail lane.
//! The classic asitame ships from wave 12a (retail from wave 9); these rows
//! deepen the roster (every runner past the 1200 raw-power arm) and put it
//! beside the staminasyoubu injection on 2400 m. The retail rows port the
//! flag: the three per-tick systems (race_solver.rs:2751-2756) draw from the
//! solver stream the host replays at prepare, and staminasyoubu above 2100 m
//! (race_solver_builder.rs:817-848) fires through the record scan. The CPU
//! engine is the oracle for frames and the full reply, and the kernel's
//! activation counters prove the systems and the injection ran.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, full_reply, json, race_exec};
use json::Value;

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

fn v77_value() -> Value {
    let text = std::fs::read_to_string(repo_file("examples/v77-race.jsonl")).expect("request");
    json::parse(text.lines().next().expect("first line")).expect("v77 parses")
}

/// One request from the v77 roster: classic or retail physics, one course
/// and seed, listed mode flags overlaid, and every horse rewritten to fixed
/// wisdom, stamina and power with skills cleared (classic scenes).
fn race_request(
    classic: bool,
    course: i32,
    seed: i64,
    extra: &[(&str, Value)],
    rewrite: Option<(f64, f64, f64)>,
) -> String {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        if classic {
            for key in [
                "experimentalRetailInitialization",
                "experimentalRetailSections",
                "experimentalRetailRuntime",
                "experimentalRetailGeometry",
            ] {
                map.insert(key.into(), Value::Bool(false));
            }
        }
        map.insert("courseId".into(), Value::Number(course as f64));
        for (k, v) in extra {
            map.insert((*k).into(), v.clone());
        }
        map.insert("seed".into(), Value::Number(seed as f64));
        if let Some((wisdom, stamina, power)) = rewrite {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            for h in horses.iter_mut() {
                let Value::Object(hm) = h else { unreachable!() };
                if classic {
                    hm.insert("skills".into(), json::arr(vec![]));
                    if let Some(Value::Array(sd)) = hm.get_mut("skillData") {
                        sd.clear();
                    }
                }
                for key in ["wisdom", "stamina", "power", "guts", "speed"] {
                    let want = match key {
                        "wisdom" => wisdom,
                        "stamina" => stamina,
                        "power" => power,
                        _ => 600.0,
                    };
                    if hm.contains_key(key) {
                        hm.insert(key.into(), json::num(want));
                    }
                }
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The engine captures six columns per runner; the kernels mirror the same
/// values in their wider frame rows.
fn first_frame_diff<const N: usize, const M: usize>(
    a: &[Vec<[f64; N]>],
    b: &[Vec<[f64; M]>],
) -> Option<usize> {
    if a.len() != b.len() {
        return Some(0);
    }
    a.iter().zip(b).zip(0..).find_map(|((fa, fb), tick)| {
        (fa.len() != fb.len()
            || fa.iter().zip(fb).any(|(ra, rb)| {
                ra.iter().zip(rb).any(|(x, y)| x.to_bits() != y.to_bits())
            }))
        .then_some(tick)
    })
}

fn frame_diff_text<const N: usize, const M: usize>(
    label: &str,
    problems: &mut Vec<String>,
    a: &[Vec<[f64; N]>],
    b: &[Vec<[f64; M]>],
) {
    if let Some(tick) = first_frame_diff(a, b) {
        let probe = if tick < a.len() && tick < b.len()
            && !a[tick].is_empty() && !b[tick].is_empty()
            && !a[tick][0].is_empty() && !b[tick][0].is_empty()
        {
            let (x, y) = (a[tick][0][0], b[tick][0][0]);
            format!(" len {} vs {}, runner 0 col 0: {x} vs {y}", a.len(), b.len())
        } else {
            format!(" len {} vs {}", a.len(), b.len())
        };
        problems.push(format!(
            "{label}: frames differ from the engine, first at tick {tick}{probe}"
        ));
    }
}

fn engine_full(request: &str, seed: i64) -> String {
    dep_cpu::run_cpu_binary(request, seed).expect("cpu binary").reply
}

/// Job A. One scene: classic physics, every runner's power past the raw
/// 1200 asitame arm, one course and seed. The race is not refused, the
/// frames match the CPU engine bit for bit, the full reply is byte-equal,
/// and the kernel's release channel count equals the engine's
/// conservePowerEvents count and is never zero.
fn assert_asitame_seed(
    course: i32,
    seed: i64,
    extra: &[(&str, Value)],
    rewrite: Option<(f64, f64, f64)>,
    problems: &mut Vec<String>,
) -> (u32, usize) {
    let request = race_request(true, course, seed, extra, rewrite);
    let out = classic_exec::run(&[(request.clone(), seed)]).expect("classic kernel run");
    let race = &out.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("course {course} seed {seed}: refused {reason}"));
        return (0, 0);
    }
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    frame_diff_text(&format!("course {course} seed {seed}"), problems, &cpu.frames, &race.frames);
    let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
    let gpu_reply = classic_reply::build(&full, seed, race).expect("full reply build");
    let engine = engine_full(&request, seed);
    if gpu_reply != engine {
        problems.push(format!("course {course} seed {seed}: full reply differs from the engine"));
    }
    let engine_events = json::parse(&engine).ok().and_then(|v| {
        v.get("conservePowerEvents")
            .and_then(json::Value::as_array)
            .map(|a| a.len())
    }).unwrap_or(usize::MAX);
    if engine_events != race.skcns.len() / 3 {
        problems.push(format!(
            "course {course} seed {seed}: conserve release count engine {engine_events} kernel {}",
            race.skcns.len() / 3
        ));
    }
    if engine_events < 1 {
        problems.push(format!("course {course} seed {seed}: the asitame release never fired"));
    }
    (race.later_stats[4], engine_events)
}

/// Job A: a full-field roster with every runner past the raw-power arm (the
/// wave 12a single-runner row keeps the 1206 boundary case). Seeds 1..3 on
/// 10504, and the same roster with the later-stamina systems and the
/// staminasyoubu injection armed on 2400 m 10108, where raw stamina above
/// 1200 coexists with raw power above 1200 without interference.
#[test]
fn asitame_high_power_rosters_answer_bit_exact() {
    register_geometry(10504);
    register_geometry(10108);
    let mut problems: Vec<String> = Vec::new();
    let power_on = || [("experimentalLaterStaminaMechanics".to_string(), Value::Bool(true))];
    let mut fired_total = 0usize;
    for seed in SEEDS {
        let (_, events) = assert_asitame_seed(
            10504, seed, &[], Some((1000.0, 30000.0, 30000.0)), &mut problems);
        fired_total += events;
        // The 2400 m co-presence scene: high stamina arms staminasyoubu,
        // high power arms asitame, the flag arms the three systems.
        let owned = power_on();
        let extra: Vec<(&str, Value)> =
            owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let (syb, _) = assert_asitame_seed(
            10108, seed, &extra, Some((1000.0, 30000.0, 30000.0)), &mut problems);
        if syb < 1 {
            problems.push(format!(
                "10108 seed {seed}: staminasyoubu never fired beside the asitame releases"
            ));
        }
    }
    assert!(fired_total >= SEEDS.len(), "the high-power roster never released conserve power");
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// The wave 12b mid-race recovery row (kind 9 at distance_rate 55): its
/// activation is the engine's one in-window clear of the keep latch, which
/// lets the compete and secure rolls fire beside the gated ones.
fn arm_recovery(map: &mut std::collections::BTreeMap<String, Value>) {
    let def = "{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[\
        {\"precondition\":\"\",\"condition\":\"distance_rate>=55\",\"cooldown\":5000000,\
        \"baseDuration\":30000,\"effects\":[{\"type\":9,\"target\":1,\"targetValue\":0,\
        \"additionalActivate\":0,\"valueUsage\":1,\"durationUsage\":1,\"targetIsDebuff\":false,\
        \"modifier\":80000}]}]}";
    let sd = map
        .entry("skillData".to_string())
        .or_insert_with(|| json::obj(vec![]));
    if let Value::Object(sd) = sd {
        sd.insert("669606".to_string(), json::parse(def).expect("recovery def parses"));
    }
    let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
    for h in horses.iter_mut() {
        let Value::Object(hm) = h else { unreachable!() };
        hm.insert(
            "skills".into(),
            json::arr(vec![json::obj(vec![
                ("skillId", Value::Number(669606.)),
                ("level", Value::Number(1.)),
            ])]),
        );
    }
}

/// Job B. One retail scene: no fallback, frames bit-exact against the CPU
/// engine, full reply byte-equal, and the kernel's five activation counters
/// returned (keep, compete, secure, gated rolls, staminasyoubu). The flag-off
/// control on the same request must show all counters zero.
fn assert_retail_seed(
    course: i32,
    seed: i64,
    rewrite: Option<(f64, f64, f64)>,
    recovery: bool,
    problems: &mut Vec<String>,
) -> [u32; 5] {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        map.insert(
            "experimentalLaterStaminaMechanics".into(),
            Value::Bool(true),
        );
        if let Some((wisdom, stamina, power)) = rewrite {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            for h in horses.iter_mut() {
                let Value::Object(hm) = h else { unreachable!() };
                for key in ["wisdom", "stamina", "power"] {
                    let want = match key {
                        "wisdom" => wisdom,
                        "stamina" => stamina,
                        _ => power,
                    };
                    if hm.contains_key(key) {
                        hm.insert(key.into(), json::num(want));
                    }
                }
            }
        }
        if recovery {
            arm_recovery(map);
        }
    }
    let mut request = String::new();
    json::to_string_into(&value, &mut request);
    assert!(
        globalsim::PreparedRace::new(&request).is_ok(),
        "course {course} seed {seed}: the engine answers the flagged request"
    );
    let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("course {course} seed {seed}: kernel fallback {reason}"));
        return [0; 5];
    }
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    frame_diff_text(
        &format!("course {course} seed {seed}"),
        problems,
        &cpu.frames,
        &race.frames,
    );
    let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
    let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
    let engine = engine_full(&request, seed);
    if gpu_reply != engine {
        problems.push(format!("course {course} seed {seed}: flagged full reply differs from the engine"));
    }
    if let Value::Object(map) = &mut value {
        map.insert(
            "experimentalLaterStaminaMechanics".into(),
            Value::Bool(false),
        );
    }
    let mut off = String::new();
    json::to_string_into(&value, &mut off);
    let off_run = race_exec::run_full(&[(off, seed)]).expect("control kernel run");
    if off_run.races[0].later_stats != [0, 0, 0, 0, 0] {
        problems.push(format!("course {course} seed {seed}: flag-off run activated the systems"));
    }
    race.later_stats
}

/// Job B: the retail lane ports experimentalLaterStaminaMechanics. Scene one
/// is the plain v77 roster at 2000 m: the keep latch fires and gates every
/// compete and secure roll, and the injection stays absent (the engine never
/// injects at or below 2100 m). Scene two adds the mid-race recovery, whose
/// keep clear lets compete and secure fire in the same window, so the whole
/// battery covers all three systems live and gated. Scene three rewrites
/// stamina past 1200 on the 2400 m course 10108, where the staminasyoubu
/// injection (race_solver_builder.rs:817-848) fires through the record scan.
/// Seeds 1..3, frames bit-exact and full replies byte-equal per scene.
#[test]
fn retail_later_stamina_answers_bit_exact() {
    register_geometry(10504);
    register_geometry(10108);
    let mut problems: Vec<String> = Vec::new();
    let mut totals = [0u32; 5];
    for seed in SEEDS {
        let stats = assert_retail_seed(10504, seed, None, false, &mut problems);
        for (slot, value) in stats.iter().enumerate() {
            totals[slot] += value;
        }
        if stats[4] != 0 {
            problems.push(format!("10504 seed {seed}: staminasyoubu fired at 2000 m, where the engine never injects it"));
        }
        if stats[0] < 1 || stats[3] < 1 {
            problems.push(format!("10504 seed {seed}: keep or its gating never activated"));
        }
    }
    for seed in SEEDS {
        let stats = assert_retail_seed(10504, seed, None, true, &mut problems);
        for (slot, value) in stats.iter().enumerate() {
            totals[slot] += value;
        }
        if stats[1] < 1 || stats[2] < 1 {
            problems.push(format!(
                "10504 seed {seed} recovery: compete {} secure {} fired",
                stats[1], stats[2]
            ));
        }
    }
    for seed in SEEDS {
        let stats =
            assert_retail_seed(10108, seed, Some((600.0, 30000.0, 600.0)), false, &mut problems);
        for (slot, value) in stats.iter().enumerate() {
            totals[slot] += value;
        }
        if stats[4] < 1 {
            problems.push(format!("10108 seed {seed}: staminasyoubu never fired on the 2400 m course"));
        }
    }
    let [keep, cmp, sec, gate, syb] = totals;
    println!(
        "retail activation totals: keep {keep}, compete {cmp}, secure {sec}, gate stops {gate}, staminasyoubu {syb}"
    );
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    assert!(keep >= 1, "stamina keep never activated across the battery");
    assert!(cmp >= 1, "compete before spurt never activated across the battery");
    assert!(sec >= 1, "secure lead never activated across the battery");
    assert!(gate >= 1, "stamina keep never gated a compete or secure roll");
    assert!(syb >= 1, "staminasyoubu never activated across the battery");
}
