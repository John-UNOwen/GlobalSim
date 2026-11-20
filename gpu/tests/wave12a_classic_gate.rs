//! Wave 12a: the classic (non-retail) physics kernel replays the CPU engine
//! bit for bit. Solo first: single-runner scenes over the five running
//! styles, seeds 1..3, courses 10504 (2000 m turf, four slopes) and 10508
//! (1200 m dirt, emptyInMove). Then full-field scenes, the empty-slopes pair
//! that proves the hill path answers, and the app-shaped 8 and 18 runner
//! rosters. Every frame value compares through `to_bits()`, the tick count
//! and finishTimeRaw too. Debug frame columns 6..13 (target speed, lane
//! target, pos-keep coef and mode, spot, duel, active slope, downhill mode)
//! carry the non-vacuity evidence the six replay columns hide.

#![cfg(feature = "race")]

use globalsim_gpu::{classic_exec, dep_cpu, json, race_exec};

const COURSES: [i32; 2] = [10504, 10508];
const SEEDS: [i64; 3] = [1, 2, 3];
const STYLES: [&str; 5] = ["NIGE", "SENKOU", "SASI", "OIKOMI", "OONIGE"];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

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

/// The v77 line with horse 0 forced to one running style, solo on one course.
/// The four retail flags are cleared so the request runs the classic physics.
fn style_request(base: &str, course: i32, seed: i64, style: &str) -> String {
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
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    race_exec::multi_request(&text, course, seed).expect("field request")
}

/// Mood lifts power into the range where the injected asitame arms
/// (`race_solver_builder.rs:802`), and that release is wave 12b work, so the
/// lane refuses the race instead of answering without it.
#[test]
fn adjusted_raw_power_above_1200_is_refused() {
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
            if let Some(json::Value::Object(first)) = horses.get_mut(0) {
                first.insert("power".into(), json::Value::Number(1160.));
                first.insert("mood".into(), json::Value::Number(2.));
            }
        }
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    let request = race_exec::solo_request(&text, 10504, 1).expect("solo request");
    let out = classic_exec::run(&[(request, 1)]).expect("classic kernel run");
    let reason = out.races[0].fallback.clone().expect("refusal expected");
    assert!(reason.contains("asitame"), "{reason}");
}

#[test]
fn classic_full_field_scenes_are_bit_exact() {
    let base = v77_line();
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for &course in &COURSES {
        for &seed in &SEEDS {
            tags.push(format!("course {course} seed {seed} field"));
            requests.push((field_request(&base, course, seed), seed));
        }
    }

    let out = classic_exec::run(&requests).expect("classic kernel run");
    let mut problems: Vec<String> = Vec::new();
    let mut evidence: Vec<String> = Vec::new();
    let mut fired: [usize; 5] = [0; 5];
    for (index, ((request, seed), tag)) in requests.iter().zip(tags.iter()).enumerate() {
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let checks: [(usize, &str, usize, fn(f64) -> bool); 5] = [
            (8, "pos-keep coef", 0, |v| v != 1.0),
            (10, "spot", 1, |v| v != 0.0),
            (11, "duel", 2, |v| v != 0.0),
            (12, "active slope", 3, |v| v != 0.0),
            (13, "downhill", 4, |v| v != 0.0),
        ];
        for (column, label, slot, pred) in checks {
            if let Some((tick, runner, value)) = first_where(race, column, pred) {
                fired[slot] += 1;
                evidence.push(format!("{tag}: first {label} tick {tick} runner {runner} value {value}"));
            }
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        if race.ticks == 0 || race.ticks > classic_exec::CLS_MAX_TICKS - 2 {
            problems.push(format!("{tag}: implausible tick count {}", race.ticks));
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        let mut first_bad: Option<(usize, String)> = None;
        'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        first_bad = Some((tick, format!(
                            "{tag} tick {tick} runner {runner} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({}), gpu target {:?} lane_target {:?}",
                            COLUMNS[column],
                            cpu.to_bits(),
                            gpu.to_bits(),
                            gpu_frame[runner][6],
                            gpu_frame[runner][7]
                        )));
                        break 'frames;
                    }
                }
            }
        }
        if let Some((tick, msg)) = first_bad {
            problems.push(msg);
            if problems.len() == 1 {
                // Window around the first divergence, all runners, for triage.
                let start = tick.saturating_sub(6);
                let mut dump = String::new();
                for t in start..(start + 9).min(cpu.frames.len()) {
                    for runner in 0..cpu.runners {
                        dump.push_str(&format!(
                            "t{t} r{runner} cpu {:?} gpu {:?}\n",
                            cpu.frames[t][runner], race.frames[t][runner]
                        ));
                    }
                }
                problems.push(dump);
            }
            continue;
        }
        for (runner, finish) in cpu.finish_times.iter().enumerate() {
            let gpu_finish = race.finish_raw_all.get(runner).copied().flatten();
            match gpu_finish {
                Some(gpu) if finish.to_bits() == gpu.to_bits() => {}
                gpu => problems.push(format!(
                    "{tag}: finishTimeRaw runner {runner} cpu {} gpu {gpu:?}",
                    finish.to_bits()
                )),
            }
        }
    }
    println!("full-field debug columns: pos-keep {} spot {} duel {} slope {} downhill {} (of {} races)",
        fired[0], fired[1], fired[2], fired[3], fired[4], out.races.len());
    for line in evidence.iter().take(18) {
        println!("{line}");
    }
    if fired[0] == 0 {
        problems.push("no full-field frame carried a nonzero pos-keep coefficient (column 8)".into());
    }
    if fired[3] == 0 {
        problems.push("no full-field frame put a runner on an active slope (column 12)".into());
    }
    if fired[4] == 0 {
        problems.push("no full-field frame reached downhill mode (column 13)".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

#[test]
fn classic_solo_scenes_are_bit_exact() {
    let base = v77_line();
    let mut requests: Vec<(String, i64)> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for &course in &COURSES {
        for &seed in &SEEDS {
            for &style in &STYLES {
                tags.push(format!("course {course} seed {seed} style {style}"));
                requests.push((style_request(&base, course, seed, style), seed));
            }
        }
    }

    let out = classic_exec::run(&requests).expect("classic kernel run");
    println!("device {} build {:.1} ms, {} races", out.device.name, out.build_ms, out.races.len());

    let mut problems: Vec<String> = Vec::new();
    let mut evidence: Vec<String> = Vec::new();
    let mut fired: [usize; 2] = [0; 2];
    for (index, ((request, seed), tag)) in requests.iter().zip(tags.iter()).enumerate() {
        let race = &out.races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        for (column, label, slot) in [(12, "active slope", 0usize), (13, "downhill", 1usize)] {
            if let Some((tick, runner, value)) = first_where(race, column, |v| v != 0.0) {
                fired[slot] += 1;
                evidence.push(format!("{tag}: first {label} tick {tick} runner {runner} value {value}"));
            }
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        assert_eq!(cpu.runners, 1, "{tag}: solo field");
        if race.ticks == 0 || race.ticks > classic_exec::CLS_MAX_TICKS - 2 {
            problems.push(format!("{tag}: implausible tick count {}", race.ticks));
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        let mut first_bad: Option<String> = None;
        'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (cpu_row, gpu_row) in cpu_frame.iter().zip(gpu_frame.iter()) {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        let msg = format!(
                            "{tag} tick {tick} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                            COLUMNS[column],
                            cpu.to_bits(),
                            gpu.to_bits()
                        );
                        first_bad = Some(msg);
                        break 'frames;
                    }
                }
            }
        }
        if let Some(msg) = first_bad {
            problems.push(msg);
            continue;
        }
        match (cpu.finish_times.first(), race.finish_raw_all.first().copied().flatten()) {
            (Some(cpu), Some(gpu)) if cpu.to_bits() == gpu.to_bits() => {}
            (cpu, gpu) => problems.push(format!(
                "{tag}: finishTimeRaw cpu {:?} gpu {:?}",
                cpu.map(|v| v.to_bits()),
                gpu.map(f64::to_bits)
            )),
        }
    }
    println!("solo debug columns: slope {} downhill {} (of {} races)", fired[0], fired[1], out.races.len());
    for line in evidence.iter().take(14) {
        println!("{line}");
    }
    if fired[0] == 0 {
        problems.push("no solo frame put a runner on an active slope (column 12)".into());
    }
    if fired[1] == 0 {
        problems.push("no solo frame reached downhill mode (column 13)".into());
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}

/// First frame where one debug column passes the predicate, (tick, runner, value).
fn first_where(race: &classic_exec::ClassicGpu, column: usize, pred: impl Fn(f64) -> bool) -> Option<(usize, usize, f64)> {
    for (tick, frame) in race.frames.iter().enumerate() {
        for (runner, row) in frame.iter().enumerate() {
            if pred(row[column]) {
                return Some((tick, runner, row[column]));
            }
        }
    }
    None
}

/// The CPU oracle cannot run a course table the file does not carry: the
/// engine embeds `data/course_data.json` at compile time (src/lib.rs:66
/// `include_str!`), initializes the table once behind a `OnceLock`
/// (src/api.rs:4-10, src/course_data.rs:102) and a request only names a
/// courseId. So the empty-slopes pair runs GPU-side: course 10504 against
/// the same table with the entry's `slopes` emptied, same roster and seed.
/// The with-slopes trace must still equal its own CPU race, the slopeless
/// trace must be hill-free by construction, and the two must differ.
#[test]
fn classic_slopes_change_the_trace() {
    let base = v77_line();
    let requests: Vec<(String, i64)> = SEEDS
        .iter()
        .map(|&seed| (field_request(&base, 10504, seed), seed))
        .collect();
    let tags: Vec<String> = SEEDS.iter().map(|&seed| format!("10504 seed {seed}")).collect();
    let slopeless = {
        let path = repo_file("data/course_data.json");
        let mut table = json::parse(&std::fs::read_to_string(path).expect("course table")).expect("course table parses");
        let mut emptied = false;
        if let json::Value::Object(map) = &mut table {
            if let Some(json::Value::Object(course)) = map.get_mut("10504") {
                course.insert("slopes".into(), json::arr(Vec::new()));
                emptied = true;
            }
        }
        assert!(emptied, "course_data.json has no 10504 entry");
        let mut text = String::new();
        json::to_string_into(&table, &mut text);
        text
    };
    let with = classic_exec::run(&requests).expect("run with the real table");
    let without = classic_exec::run_with_courses_text(&requests, &slopeless).expect("run with 10504's slopes emptied");
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = &tags[index];
        let a = &with.races[index];
        let b = &without.races[index];
        if let Some(reason) = &a.fallback {
            problems.push(format!("{tag} with slopes: fallback {reason}"));
            continue;
        }
        if let Some(reason) = &b.fallback {
            problems.push(format!("{tag} slopeless: fallback {reason}"));
            continue;
        }
        problems.extend(compare_batch(&[tag.clone()], &[(request.clone(), *seed)], &with.races[index..index + 1]));
        let mut hill_seen = false;
        for (tick, frame) in b.frames.iter().enumerate() {
            for (runner, row) in frame.iter().enumerate() {
                if row[12] != 0.0 || row[13] != 0.0 {
                    hill_seen = true;
                    problems.push(format!("{tag} slopeless tick {tick} runner {runner}: hill columns ({}, {})", row[12], row[13]));
                }
            }
        }
        println!("{tag}: slopeless run hill-free {}", !hill_seen);
        let mut first_diff: Option<String> = None;
        if a.ticks != b.ticks {
            first_diff = Some(format!("tick count {}/{}", a.ticks, b.ticks));
        } else {
            'diff: for (tick, (fa, fb)) in a.frames.iter().zip(b.frames.iter()).enumerate() {
                for (runner, (ra, rb)) in fa.iter().zip(fb).enumerate() {
                    for column in 0..6 {
                        if ra[column].to_bits() != rb[column].to_bits() {
                            first_diff = Some(format!("tick {tick} runner {runner} {}: {}/{}", COLUMNS[column], ra[column], rb[column]));
                            break 'diff;
                        }
                    }
                }
            }
        }
        match &first_diff {
            Some(msg) => println!("{tag}: emptying the slopes changes the trace, first at {msg}"),
            None => problems.push(format!("{tag}: emptying the slopes changed nothing")),
        }
    }
    if !problems.is_empty() {
        panic!("{} problems:\n{}", problems.len(), problems.join("\n"));
    }
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
        map.insert("horses".into(), json::arr(horses));
        map.insert("courseId".into(), json::Value::Number(f64::from(course)));
        map.insert("seed".into(), json::Value::Number(seed as f64));
    }
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    text
}

/// Frame-for-frame and finishTimeRaw comparison of the matching race slices.
fn compare_batch(tags: &[String], requests: &[(String, i64)], races: &[classic_exec::ClassicGpu]) -> Vec<String> {
    let mut problems: Vec<String> = Vec::new();
    for (index, ((request, seed), tag)) in requests.iter().zip(tags.iter()).enumerate() {
        let race = &races[index];
        if let Some(reason) = &race.fallback {
            problems.push(format!("{tag}: fallback {reason}"));
            continue;
        }
        let cpu = match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => cpu,
            Err(reason) => {
                problems.push(format!("{tag}: cpu race failed: {reason}"));
                continue;
            }
        };
        if race.ticks == 0 || race.ticks > classic_exec::CLS_MAX_TICKS - 2 {
            problems.push(format!("{tag}: implausible tick count {}", race.ticks));
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
            continue;
        }
        'frames: for (tick, (cpu_frame, gpu_frame)) in cpu.frames.iter().zip(race.frames.iter()).enumerate() {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} runner {runner} {}: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                            COLUMNS[column],
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                        break 'frames;
                    }
                }
            }
        }
        for (runner, finish) in cpu.finish_times.iter().enumerate() {
            let gpu_finish = race.finish_raw_all.get(runner).copied().flatten();
            match gpu_finish {
                Some(gpu) if finish.to_bits() == gpu.to_bits() => {}
                gpu => problems.push(format!(
                    "{tag}: finishTimeRaw runner {runner} cpu {} gpu {gpu:?}",
                    finish.to_bits()
                )),
            }
        }
    }
    problems
}

/// The app-shaped zero-skill rosters, mixed styles, the 18-runner field with
/// ranks past 8 on the extra lane, seeds 1..3, both courses.
#[test]
fn classic_app_shaped_rosters_are_bit_exact() {
    let base = v77_line();
    let mut problems: Vec<String> = Vec::new();
    for (name, horses) in [("mob8", mob8()), ("mob18", mob18())] {
        for &course in &COURSES {
            let requests: Vec<(String, i64)> = SEEDS
                .iter()
                .map(|&seed| (app_request(&base, horses.clone(), course, seed), seed))
                .collect();
            let tags: Vec<String> = SEEDS.iter().map(|&seed| format!("{name} course {course} seed {seed}")).collect();
            let out = match classic_exec::run(&requests) {
                Ok(out) => out,
                Err(e) => {
                    problems.push(format!("{name} course {course}: kernel run failed: {e}"));
                    continue;
                }
            };
            let mut spot = 0usize;
            let mut duel = 0usize;
            for race in &out.races {
                if race.fallback.is_some() {
                    continue;
                }
                spot += usize::from(first_where(race, 10, |v| v != 0.0).is_some());
                duel += usize::from(first_where(race, 11, |v| v != 0.0).is_some());
            }
            println!("{name} course {course}: spot fired in {spot} of {} races, duel in {duel}", requests.len());
            problems.extend(compare_batch(&tags, &requests, &out.races));
        }
    }
    if !problems.is_empty() {
        panic!("{} mismatching races:\n{}", problems.len(), problems.join("\n"));
    }
}
