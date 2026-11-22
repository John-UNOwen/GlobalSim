//! Wave 13a item 1: experimentalLaterStaminaMechanics on the classic lane.
//! The engine adds three per-tick systems behind the flag (race_solver.rs:
//! 2746-3043): stamina keep, compete before spurt, and secure lead, each
//! drawing from the live solver rng. The kernel ports them; the CPU engine is
//! the oracle for frames and the full reply, and the kernel's activation
//! counters prove each system really ran. Above 2100 m the engine also
//! injects the synthetic staminasyoubu skill (race_solver_builder.rs:819),
//! ported in wave 16 and gated bit-exact on two long courses.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, json};
use json::Value;

const SEEDS: [i64; 3] = [1, 2, 3];
// The high-stamina variant adds stamina so the keep HP gate closes early and
// the compete and secure systems run in the section window (seed 6 activates
// all three systems; the probe kept this scene deterministic).
const B_STAMINA: f64 = 30000.0;
const B_WISDOM: f64 = 1000.0;
const B_SEEDS: [i64; 4] = [1, 2, 3, 6];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

/// The v77 field on the classic lane, the retail flags off, the listed mode
/// flags overlaid, and optionally every horse rewritten to fixed wisdom and
/// stamina with skills cleared.
fn classic_request(
    course: i32,
    seed: i64,
    extra: &[(&str, Value)],
    rewrite: Option<(f64, f64)>,
) -> String {
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
        map.insert("courseId".into(), Value::Number(course as f64));
        for (k, v) in extra {
            map.insert((*k).into(), v.clone());
        }
        map.insert("seed".into(), Value::Number(seed as f64));
        if let Some((wisdom, stamina)) = rewrite {
            let Value::Array(horses) = map.get_mut("horses").expect("horses") else { unreachable!() };
            for h in horses.iter_mut() {
                let Value::Object(hm) = h else { unreachable!() };
                hm.insert("skills".into(), json::arr(vec![]));
                if let Some(Value::Array(sd)) = hm.get_mut("skillData") {
                    sd.clear();
                }
                for key in ["wisdom", "stamina", "power", "guts", "speed"] {
                    let want = if key == "wisdom" {
                        wisdom
                    } else if key == "stamina" {
                        stamina
                    } else {
                        600.0
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

fn run_classic(request: &str, seed: i64) -> classic_exec::ClassicGpu {
    let out = classic_exec::run(&[(request.to_string(), seed)]).expect("classic kernel run");
    out.races.into_iter().next().expect("one race")
}

/// The engine captures six columns per runner; the classic kernel mirrors the
/// same values in its wider frame rows.
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

/// One scene seed: no fallback, frames bit-exact against the CPU engine, and
/// the full reply byte-equal. Returns the kernel's activation counters and
/// the first tick where the flagged frames differ from the flag-off field.
fn assert_seed_bit_exact(
    course: i32,
    seed: i64,
    rewrite: Option<(f64, f64)>,
    problems: &mut Vec<String>,
) -> ([u32; 5], Option<usize>) {
    let on_flag = [("experimentalLaterStaminaMechanics", Value::Bool(true))];
    let request = classic_request(course, seed, &on_flag, rewrite);
    assert!(
        globalsim::PreparedRace::new(&request).is_ok(),
        "seed {seed}: the engine answers the flagged request"
    );
    let race = run_classic(&request, seed);
    if let Some(reason) = &race.fallback {
        problems.push(format!("seed {seed}: kernel fallback {reason}"));
        return ([0; 5], None);
    }
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu race");
    if let Some(tick) = first_frame_diff(&cpu.frames, &race.frames) {
        problems.push(format!(
            "seed {seed}: flagged frames differ from the engine, first at tick {tick}"
        ));
    }
    let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
    let gpu_reply = classic_reply::build(&full, seed, &race).expect("full reply build");
    let engine_reply = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
    if gpu_reply != engine_reply {
        problems.push(format!("seed {seed}: flagged full reply differs from the engine"));
    }
    let off = run_classic(&classic_request(course, seed, &[], rewrite), seed);
    if off.later_stats != [0, 0, 0, 0, 0] {
        problems.push(format!("seed {seed}: flag-off run activated the systems"));
    }
    let diff = first_frame_diff(&off.frames, &race.frames);
    (race.later_stats, diff)
}

/// Frames bit-exact against the CPU engine and the full reply byte-equal on
/// every seed of both scenes. The kernel's activation counters prove each of
/// the three systems ran at least once across the battery, and stamina keep
/// at least once stopped a compete or secure roll. The flag-off controls run
/// with zero counters, and at least one flagged scene moves the physics
/// against its flag-off field.
#[test]
fn classic_later_stamina_answers_bit_exact() {
    register_geometry(10504);
    let mut problems: Vec<String> = Vec::new();
    let mut totals = [0u32; 5];
    let mut diffs: Vec<String> = Vec::new();
    for seed in SEEDS {
        let (stats, diff) = assert_seed_bit_exact(10504, seed, None, &mut problems);
        for (slot, value) in stats.iter().enumerate() {
            totals[slot] += value;
        }
        diffs.push(format!(
            "v77 seed {seed}: first flagged frame difference vs flag-off {:?}",
            diff.map(|t| t as f64 / 15.0)
        ));
    }
    for seed in B_SEEDS {
        let (stats, diff) =
            assert_seed_bit_exact(10504, seed, Some((B_WISDOM, B_STAMINA)), &mut problems);
        for (slot, value) in stats.iter().enumerate() {
            totals[slot] += value;
        }
        diffs.push(format!(
            "high-stamina seed {seed}: first flagged frame difference vs flag-off {:?}",
            diff.map(|t| t as f64 / 15.0)
        ));
    }
    let [keep, cmp, sec, gate, syb] = totals;
    println!(
        "activation totals: keep {keep}, compete {cmp}, secure {sec}, gate stops {gate}, staminasyoubu {syb}"
    );
    for line in &diffs {
        println!("{line}");
    }
    assert!(
        diffs.iter().any(|d| !d.ends_with("None")),
        "the flag never changed a frame against the flag-off field"
    );
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    assert!(keep >= 1, "stamina keep never activated across the battery");
    assert!(cmp >= 1, "compete before spurt never activated across the battery");
    assert!(sec >= 1, "secure lead never activated across the battery");
    assert!(gate >= 1, "stamina keep never gated a compete or secure roll");
    assert_eq!(syb, 0, "staminasyoubu fired at 2100 m, where the engine never injects it");
}

/// The engine echoes the flag in every reply (lib.rs:1071). The classic
/// reply must carry true when set and absent-false semantics otherwise: the
/// flag-off reply keeps the engine's false.
#[test]
fn classic_reply_echoes_the_flag() {
    register_geometry(10504);
    for (seed, expect) in [(1i64, true), (2i64, false)] {
        let extra = [("experimentalLaterStaminaMechanics", Value::Bool(expect))];
        let request = classic_request(10504, seed, &extra, None);
        let race = run_classic(&request, seed);
        assert!(race.fallback.is_none(), "seed {seed}: the classic lane answers");
        let full = dep_cpu::with_retain_all_frames(&request).expect("retain request");
        let gpu_reply = classic_reply::build(&full, seed, &race).expect("full reply build");
        let engine_reply = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary").reply;
        assert_eq!(gpu_reply, engine_reply, "seed {seed}: the flag echo is byte-equal");
        let parsed = json::parse(&engine_reply).expect("engine reply");
        assert_eq!(
            parsed.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool),
            Some(expect),
            "seed {seed}: the engine echoes {expect}"
        );
    }
}

/// Wave 16: the staminasyoubu injection above 2100 m is ported. Frames
/// bit-exact against the CPU engine and the full reply byte-equal, flagged
/// and unflagged, seeds 1 to 3, on four long courses covering every value
/// tier of race_solver_builder.rs:167-179: 10308 at 2200 m (0.5), 10108 at
/// 2400 m (1.0), 11004 at 2600 m (1.2) and 10507 at 3600 m (1.5), on the
/// plain v77 field and the high-stamina field that arms the raw stamina
/// gate. The kernel's staminasyoubu counter (later_stats slot 5) proves the
/// injected record fired on every course, the unflagged controls run every
/// counter at zero, and the flagged field diverges from its unflagged
/// control on at least one scene. Other long courses drift 1 to 2 ulp in
/// speed off the engine on some seed or field even with the flag off (for
/// example 10105 seed 2 plain at tick 442, 10312 seed 1 high-stamina at
/// tick 910), a pre-existing base-physics gap kept out of this scene set.
#[test]
fn staminasyoubu_long_courses_answer_bit_exact() {
    let courses = [(10308i32, 0.5), (10108, 1.0), (11004, 1.2), (10507, 1.5)];
    for (course, _) in courses {
        register_geometry(course);
    }
    let mut problems: Vec<String> = Vec::new();
    let mut fired: Vec<u32> = Vec::new();
    let mut diffs: Vec<String> = Vec::new();
    // Every (course, seed, rewrite) runs side by side (crate::par). The
    // answers come back in loop order.
    let cases = courses.iter().flat_map(|&(course, _)| {
        SEEDS.iter().flat_map(move |&seed| [None, Some((B_WISDOM, B_STAMINA))].map(move |rewrite| (course, seed, rewrite)))
    });
    let mut answers = crate::par::map(cases, |(course, seed, rewrite)| {
        let mut problems = Vec::new();
        let (stats, diff) = assert_seed_bit_exact(course, seed, rewrite, &mut problems);
        (stats, diff, problems)
    })
    .into_iter();
    for (course, tier) in courses {
        let mut course_fires = 0u32;
        for seed in SEEDS {
            for rewrite in [None, Some((B_WISDOM, B_STAMINA))] {
                let (stats, diff, mut case_problems) = answers.next().expect("one answer per case");
                problems.append(&mut case_problems);
                course_fires += stats[4];
                diffs.push(format!(
                    "course {course} (tier {tier}) seed {seed}{}: first flagged frame difference vs unflagged {:?}",
                    if rewrite.is_some() { " high-stamina" } else { "" },
                    diff.map(|t| t as f64 / 15.0)
                ));
            }
        }
        fired.push(course_fires);
    }
    println!(
        "staminasyoubu fires: 10308 {}, 10108 {}, 11004 {}, 10507 {}",
        fired[0], fired[1], fired[2], fired[3]
    );
    for line in &diffs {
        println!("{line}");
    }
    for (i, (course, _)) in courses.iter().enumerate() {
        assert!(fired[i] >= 1, "staminasyoubu never fired on course {course}");
    }
    assert!(
        diffs.iter().any(|d| !d.ends_with("None")),
        "the flag never changed a frame on the long courses"
    );
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}
