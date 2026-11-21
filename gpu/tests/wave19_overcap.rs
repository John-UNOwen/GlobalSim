//! Wave 19: the retail overcap raw-power family. A wire power above 1200
//! arms the injected asitame, and the release modifier is sqrt over the raw
//! power, which carries the overcap half (race_solver_builder.rs:295). The
//! kernel's SD_RAWPW column was fed the base power (initialization output 2)
//! instead of the raw power (output 6). The two only differ once the wire
//! power passes 1200, so the whole overcap band fired the release with a
//! smaller accel modifier and the diverging scene lost a runner's speed at
//! the last-spurt transition. The gates pin the reported 10504 scene, the
//! same rewrite on 10108 with the recovery skill present, a mild overcap
//! band representative and the flagged variant with both injections live.

#![cfg(feature = "race")]

use globalsim_gpu::{dep_cpu, full_reply, json, race_exec};
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

/// The wave 12b mid-race recovery row (kind 9 at distance_rate 55), the same
/// retail-legal record the wave 18 gate arms. It exercises the record scan
/// and the HP recovery arm beside the overcap releases.
fn recovery_skill_data() -> Value {
    let def = "{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[\
        {\"precondition\":\"\",\"condition\":\"distance_rate>=55\",\"cooldown\":5000000,\
        \"baseDuration\":30000,\"effects\":[{\"type\":9,\"target\":1,\"targetValue\":0,\
        \"additionalActivate\":0,\"valueUsage\":1,\"durationUsage\":1,\"targetIsDebuff\":false,\
        \"modifier\":80000}]}]}";
    let mut sd = json::obj(vec![]);
    if let Value::Object(sd) = &mut sd {
        sd.insert("669606".to_string(), json::parse(def).expect("recovery def parses"));
    }
    sd
}

/// One v77 retail request: one course and seed, the whole roster rewritten
/// to fixed wisdom, stamina, power, guts and speed, the later-stamina flag
/// overlaid, and the recovery skill armed on every runner when asked.
fn overcap_request(
    course: i32,
    seed: i64,
    stamina: f64,
    power: f64,
    flag: bool,
    skills: bool,
) -> String {
    let mut value = v77_value();
    {
        let Value::Object(map) = &mut value else { unreachable!() };
        map.insert("courseId".into(), Value::Number(course as f64));
        map.insert("seed".into(), Value::Number(seed as f64));
        map.insert(
            "experimentalLaterStaminaMechanics".into(),
            Value::Bool(flag),
        );
        if skills {
            let sd = map
                .entry("skillData".to_string())
                .or_insert_with(|| json::obj(vec![]));
            if let (Value::Object(sd), Value::Object(defs)) =
                (sd, recovery_skill_data())
            {
                for (id, def) in defs {
                    sd.insert(id, def);
                }
            }
        }
        let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
        for h in horses.iter_mut() {
            let Value::Object(hm) = h else { unreachable!() };
            for key in ["wisdom", "stamina", "power", "guts", "speed"] {
                let want = match key {
                    "wisdom" => 1000.0,
                    "stamina" => stamina,
                    "power" => power,
                    _ => 600.0,
                };
                if hm.contains_key(key) {
                    hm.insert(key.into(), json::num(want));
                }
            }
            if skills {
                hm.insert(
                    "skills".into(),
                    json::arr(vec![json::obj(vec![
                        ("skillId", Value::Number(669606.)),
                        ("level", Value::Number(1.)),
                    ])]),
                );
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The engine captures six columns per runner, and the kernel mirrors the
/// same values in its wider frame rows.
fn first_frame_diff(a: &[Vec<[f64; 6]>], b: &[Vec<[f64; 6]>]) -> Option<usize> {
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

/// One scene: no fallback, every frame column `to_bits`-equal against the
/// CPU engine, the full reply byte-equal, and the later-stamina counters
/// zero on a flag-off request. Returns the counters of the flagged run.
fn assert_overcap_seed(
    course: i32,
    seed: i64,
    stamina: f64,
    power: f64,
    flag: bool,
    skills: bool,
    problems: &mut Vec<String>,
) -> [u32; 5] {
    let request = overcap_request(course, seed, stamina, power, flag, skills);
    assert!(
        globalsim::PreparedRace::new(&request).is_ok(),
        "course {course} seed {seed}: the engine answers the request"
    );
    let run = race_exec::run_full(&[(request.clone(), seed)]).expect("kernel run");
    let race = &run.races[0];
    if let Some(reason) = &race.fallback {
        problems.push(format!("course {course} seed {seed} st{stamina} pw{power} flag{flag}: refused {reason}"));
        return [0; 5];
    }
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    if let Some(tick) = first_frame_diff(&cpu.frames, &race.frames) {
        let probe = if tick < cpu.frames.len() && tick < race.frames.len()
            && !cpu.frames[tick].is_empty() && !race.frames[tick].is_empty()
        {
            let (x, y) = (cpu.frames[tick][0][0], race.frames[tick][0][0]);
            format!(" len {} vs {}, runner 0 col 0: {x} vs {y}", cpu.frames.len(), race.frames.len())
        } else {
            format!(" len {} vs {}", cpu.frames.len(), race.frames.len())
        };
        problems.push(format!(
            "course {course} seed {seed} st{stamina} pw{power} flag{flag} skill{skills}: \
             frames differ from the engine, first at tick {tick}{probe}"
        ));
    }
    let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
    let gpu_reply = full_reply::build(&full, seed, race).expect("full reply build");
    let engine = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
    if gpu_reply != engine {
        problems.push(format!(
            "course {course} seed {seed} st{stamina} pw{power} flag{flag} skill{skills}: full reply differs"
        ));
    }
    if !flag && race.later_stats != [0, 0, 0, 0, 0] {
        problems.push(format!(
            "course {course} seed {seed}: flag-off run activated the later-stamina systems"
        ));
    }
    race.later_stats
}

/// The scene the wave 18 README left open: the extreme overcap rewrite on
/// the retail lane at 10504, flag off. Before the fix the v77 leader lost
/// the asitame overcap accel at the last-spurt transition, first visible as
/// the position column at tick 1020 (seed 1), 1021 (seed 2) and 1022 (seed 3).
#[test]
fn retail_extreme_overcap_answers_bit_exact() {
    register_geometry(10504);
    let mut problems: Vec<String> = Vec::new();
    for seed in SEEDS {
        assert_overcap_seed(10504, seed, 30000.0, 30000.0, false, false, &mut problems);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// The family representatives: the mild overcap band (wire power 2000, raw
/// power base + 400) at 10504 flag off, and the same extreme rewrite one
/// course over with the recovery record armed, so the record scan and the
/// HP recovery arm run beside every overcap asitame release.
#[test]
fn retail_overcap_bands_and_skills_answer_bit_exact() {
    register_geometry(10504);
    register_geometry(10108);
    let mut problems: Vec<String> = Vec::new();
    for seed in SEEDS {
        assert_overcap_seed(10504, seed, 2000.0, 2000.0, false, false, &mut problems);
        assert_overcap_seed(10108, seed, 30000.0, 30000.0, false, true, &mut problems);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// The flagged variant: the extreme rewrite on the 2400 m course with the
/// recovery armed, so the staminasyoubu injection (raw stamina 30000 past
/// 1200, above 2100 m) and the asitame releases (raw power 16400) are live
/// in the same last-spurt window, on top of the keep, compete and secure
/// rolls. Counters prove every one of them ran.
#[test]
fn retail_overcap_flagged_injections_answer_bit_exact() {
    register_geometry(10108);
    let mut problems: Vec<String> = Vec::new();
    let mut totals = [0u32; 5];
    for seed in SEEDS {
        let stats = assert_overcap_seed(10108, seed, 30000.0, 30000.0, true, true, &mut problems);
        if stats[4] < 1 {
            problems.push(format!(
                "10108 seed {seed}: staminasyoubu never fired beside the overcap releases"
            ));
        }
        for (slot, value) in totals.iter_mut().zip(stats) {
            *slot += value;
        }
    }
    let [keep, cmp, sec, _gate, syb] = totals;
    println!("overcap flagged activation totals: keep {keep}, compete {cmp}, secure {sec}, staminasyoubu {syb}");
    assert!(keep >= 1, "the stamina keep never activated across the battery");
    assert!(cmp + sec >= 1, "neither compete nor secure ever activated across the battery");
    assert!(syb >= 3, "the staminasyoubu injection did not fire on every seed");
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
