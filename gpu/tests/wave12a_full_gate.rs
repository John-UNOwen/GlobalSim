//! Wave 12a step 4: the classic lane answers the engine's whole non-compact
//! reply, byte for byte. Solo style scenes, the v77 nine-horse field, and the
//! app-shaped mob8 and mob18 rosters, seeds 1..3, courses 10504 and 10508.
//! The oracle is `dep_cpu::run_cpu_binary`, which forces `retainAllFrames`,
//! so every scene compares the full-frame reply form. A mismatch reports the
//! first differing byte offset with 80 characters of context from each side.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, classic_reply, dep_cpu, json, race_exec};

const COURSES: [i32; 2] = [10504, 10508];
const SEEDS: [i64; 3] = [1, 2, 3];
const STYLES: [&str; 5] = ["NIGE", "SENKOU", "SASI", "OIKOMI", "OONIGE"];

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

/// Clear the four retail flags so the request runs classic physics.
fn classic_base(base: &str) -> json::Value {
    let mut value = json::parse(base).expect("parse v77 line");
    if let json::Value::Object(map) = &mut value {
        for key in [
            "experimentalRetailInitialization",
            "experimentalRetailSections",
            "experimentalRetailRuntime",
            "experimentalRetailGeometry",
        ] {
            map.insert(key.into(), json::Value::Bool(false));
        }
    }
    value
}

/// The v77 line with horse 0 forced to one running style, solo on one course.
fn style_request(base: &str, course: i32, seed: i64, style: &str) -> String {
    let mut value = classic_base(base);
    if let json::Value::Object(map) = &mut value {
        if let Some(json::Value::Array(horses)) = map.get_mut("horses") {
            if let Some(json::Value::Object(first)) = horses.get_mut(0) {
                first.insert("strategy".into(), json::Value::String(style.into()));
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::solo_request(&text, course, seed).expect("solo request")
}

/// The whole v77 roster kept in place, classic physics, one course and seed.
fn field_request(base: &str, course: i32, seed: i64) -> String {
    let mut text = String::new();
    json::to_string_into(&classic_base(base), &mut text);
    race_exec::multi_request(&text, course, seed).expect("field request")
}

/// One runner in the app adapter's shape, mirroring zero_skill_gate.rs: raw
/// stats, canonical aptitude spellings, zero skills, its own gate.
fn horse(strategy: &str, stats: [f64; 5], dist: &str, mood: i64, gate: usize) -> json::Value {
    json::obj(vec![
        ("speed", json::Value::Number(stats[0])),
        ("stamina", json::Value::Number(stats[1])),
        ("power", json::Value::Number(stats[2])),
        ("guts", json::Value::Number(stats[3])),
        ("wisdom", json::Value::Number(stats[4])),
        ("strategy", json::Value::String(strategy.into())),
        ("distanceAptitude", json::Value::String(dist.into())),
        ("surfaceAptitude", json::Value::String("A".into())),
        ("strategyAptitude", json::Value::String("A".into())),
        ("mood", json::Value::Number(mood as f64)),
        ("skills", json::arr(vec![])),
        ("gateIndex", json::Value::Number(gate as f64)),
    ])
}

/// The mob-preset stat block (1600/1200/1200/600/1200, Distance S, mood 0),
/// optionally with a stat spread on speed and wisdom.
fn mob(strategy: &str, gate: usize, spread: (f64, f64)) -> json::Value {
    horse(strategy, [1600. + spread.0, 1200., 1200., 600., 1200. + spread.1], "S", 0, gate)
}

/// The tested-uma stat block (1000/1000/900/800/900, Distance A, mood 2).
fn uma(strategy: &str, gate: usize) -> json::Value {
    horse(strategy, [1000., 1000., 900., 800., 900.], "A", 2, gate)
}

/// The eight mob rows of zero_skill_gate's field9(3) without the uma row,
/// mixed styles and spreads, gates 0,1,2,4,5,6,7,8.
fn mob8() -> Vec<json::Value> {
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let spreads = [(0., 0.), (-60., 0.), (0., -80.), (0., 0.), (-120., 40.), (0., 0.), (60., 0.), (-60., -80.)];
    let mut out = Vec::new();
    let mut gate = 0usize;
    for (i, style) in styles.iter().enumerate() {
        if gate == 3 {
            gate += 1;
        }
        out.push(mob(style, gate, spreads[i]));
        gate += 1;
    }
    out
}

/// zero_skill_gate's field18: two mob blocks and two umas, gates 0..17, the
/// ranks at 9 and beyond take the extra lane.
fn mob18() -> Vec<json::Value> {
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let mut out = Vec::new();
    let mut gate = 0usize;
    for round in 0..2 {
        for i in 0..8 {
            if gate == 3 || gate == 12 {
                out.push(uma("Senkou", gate));
                gate += 1;
            }
            let spread = if round == 0 { (0., 0.) } else { (-120., 40.) };
            out.push(mob(styles[i], gate, spread));
            gate += 1;
        }
    }
    if gate < 18 {
        out.push(uma("Senkou", gate));
    }
    out.truncate(18);
    out
}

/// The v77 template with the retail flags cleared, horses replaced, courseId
/// and seed set, the app-shaped request.
fn app_request(base: &str, horses: Vec<json::Value>, course: i32, seed: i64) -> String {
    let mut value = classic_base(base);
    if let json::Value::Object(map) = &mut value {
        map.insert("horses".into(), json::arr(horses));
        map.insert("courseId".into(), json::Value::Number(f64::from(course)));
        map.insert("seed".into(), json::Value::Number(seed as f64));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// The first byte where two reply texts differ, with 80 characters of
/// context from each side.
fn first_difference(gpu: &str, engine: &str) -> String {
    let gb = gpu.as_bytes();
    let eb = engine.as_bytes();
    let at = gb.iter().zip(eb.iter()).take_while(|(a, b)| a == b).count();
    let window = |bytes: &[u8]| {
        let start = at.saturating_sub(40);
        String::from_utf8_lossy(&bytes[start..bytes.len().min(at + 40)]).into_owned()
    };
    format!(
        "first differing byte {at} (gpu {} bytes, engine {} bytes)\ngpu:    {}\nengine: {}",
        gb.len(),
        eb.len(),
        window(gb),
        window(eb)
    )
}

struct Scene {
    tag: String,
    request: String,
    seed: i64,
}

#[test]
fn wave12a_full_bit_exact() {
    let base = v77_line();
    let mut scenes: Vec<Scene> = Vec::new();
    for &course in &COURSES {
        for &seed in &SEEDS {
            for &style in &STYLES {
                scenes.push(Scene {
                    tag: format!("solo {style} course {course} seed {seed}"),
                    request: style_request(&base, course, seed, style),
                    seed,
                });
            }
            scenes.push(Scene {
                tag: format!("v77 course {course} seed {seed}"),
                request: field_request(&base, course, seed),
                seed,
            });
        }
    }
    for (name, horses) in [("mob8", mob8()), ("mob18", mob18())] {
        for &course in &COURSES {
            for &seed in &SEEDS {
                scenes.push(Scene {
                    tag: format!("{name} course {course} seed {seed}"),
                    request: app_request(&base, horses.clone(), course, seed),
                    seed,
                });
            }
        }
    }

    let requests: Vec<(String, i64)> = scenes.iter().map(|s| (s.request.clone(), s.seed)).collect();
    let out = classic_exec::run(&requests).expect("classic kernel run");
    println!("device {} build {:.1} ms, {} races", out.device.name, out.build_ms, out.races.len());

    let mut problems: Vec<String> = Vec::new();
    let mut spot_total = 0usize;
    let mut duel_total = 0usize;
    let mut nonempty = 0usize;
    for (index, scene) in scenes.iter().enumerate() {
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{}: fallback {reason}", scene.tag));
            continue;
        }
        let spots = race.comp.chunks(3).filter(|rec| rec[1] == 4).count();
        let duels = race.comp.chunks(3).filter(|rec| rec[1] == 5).count();
        spot_total += spots;
        duel_total += duels;
        if spots + duels > 0 {
            nonempty += 1;
        }
        println!("{}: competitionEvents {} spot {} duel {}", scene.tag, spots + duels, spots, duels);
        let full = match dep_cpu::with_retain_all_frames(&scene.request) {
            Ok(text) => text,
            Err(reason) => {
                problems.push(format!("{}: retain request failed: {reason}", scene.tag));
                continue;
            }
        };
        let gpu = match classic_reply::build(&full, scene.seed, race) {
            Ok(text) => text,
            Err(reason) => {
                problems.push(format!("{}: full reply build failed: {reason}", scene.tag));
                continue;
            }
        };
        let engine = match dep_cpu::run_cpu_binary(&scene.request, scene.seed) {
            Ok(binary) => binary.reply,
            Err(reason) => {
                problems.push(format!("{}: cpu binary lane failed: {reason}", scene.tag));
                continue;
            }
        };
        if gpu != engine {
            problems.push(format!("{}:\n{}", scene.tag, first_difference(&gpu, &engine)));
        }
    }
    println!(
        "competitionEvents non-vacuity: {nonempty} of {} replies carry events, {spot_total} spot starts, {duel_total} duel starts",
        scenes.len()
    );
    if nonempty == 0 {
        problems.push("no reply carried a competitionEvents entry".into());
    }
    if spot_total == 0 {
        problems.push("no scene fired a spot start (kind 4)".into());
    }
    if duel_total == 0 {
        problems.push("no scene fired a duel start (kind 5)".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
