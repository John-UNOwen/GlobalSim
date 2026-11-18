//! Zero-skill app-shaped rosters (mixed running styles, mob-preset stats,
//! varied roster sizes) must stay bit-exact in the race kernel. The v77
//! roster is all-Senkou at gates 0..8. This suite reaches the mixed-style
//! field logic, mixed-style temptations and the >=9-gate extra lane.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, race_exec};

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

/// One runner in the app adapter's shape: raw stats, canonical aptitude
/// spellings, zero skills, its own gate.
fn horse(strategy: &str, stats: [f64; 5], dist: &str, mood: i64, gate: usize) -> Value {
    json::obj(vec![
        ("speed", Value::Number(stats[0])),
        ("stamina", Value::Number(stats[1])),
        ("power", Value::Number(stats[2])),
        ("guts", Value::Number(stats[3])),
        ("wisdom", Value::Number(stats[4])),
        ("strategy", Value::String(strategy.into())),
        ("distanceAptitude", Value::String(dist.into())),
        ("surfaceAptitude", Value::String("A".into())),
        ("strategyAptitude", Value::String("A".into())),
        ("mood", Value::Number(mood as f64)),
        ("skills", json::arr(vec![])),
        ("gateIndex", Value::Number(gate as f64)),
    ])
}

/// The mob-preset stat block (1600/1200/1200/600/1200, Distance S, mood 0),
/// optionally with a stat spread on speed and wisdom.
fn mob(strategy: &str, gate: usize, spread: (f64, f64)) -> Value {
    horse(strategy, [1600. + spread.0, 1200., 1200., 600., 1200. + spread.1], "S", 0, gate)
}

/// The tested-uma stat block (1000/1000/900/800/900, Distance A, mood 2).
fn uma(strategy: &str, gate: usize) -> Value {
    horse(strategy, [1000., 1000., 900., 800., 900.], "A", 2, gate)
}

/// The v77 line-1 template with its horses replaced, courseId and seed set.
fn request_with(base: &str, horses: Vec<Value>, course: i32, seed: i64) -> String {
    let mut value = json::parse(base).expect("template parses");
    let Value::Object(map) = &mut value else { unreachable!() };
    map.insert("horses".into(), json::arr(horses));
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    out
}

/// The mob-layout 9-runner field: the Default preset's eight mob rows
/// ascending, the uma at gate 3, zero skills throughout.
fn field9(uma_gate: usize) -> Vec<Value> {
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let spreads = [(0., 0.), (-60., 0.), (0., -80.), (0., 0.), (-120., 40.), (0., 0.), (60., 0.), (-60., -80.)];
    let mut out = Vec::new();
    let mut gate = 0usize;
    for (i, style) in styles.iter().enumerate() {
        if gate == uma_gate {
            out.push(uma("Senkou", gate));
            gate += 1;
        }
        out.push(mob(style, gate, spreads[i]));
        gate += 1;
    }
    out
}

/// An 18-runner field: two mob blocks (gates 9..17 cross the extra-lane
/// boundary) plus the two umas, zero skills.
fn field18() -> Vec<Value> {
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

/// One shape's races (3 seeds) through the kernel and the CPU engine,
/// returning every problem: fallbacks, tick-count mismatches, the first
/// frame-for-frame divergence with both bit patterns, and finishTimeRaw.
fn check_shape(name: &str, base: &str, course: i32, horses: Vec<Value>) -> Vec<String> {
    let n = horses.len();
    let mut problems: Vec<String> = Vec::new();
    let mut requests: Vec<(String, i64)> = Vec::new();
    for seed in 1..=3i64 {
        requests.push((request_with(base, horses.clone(), course, seed), seed));
    }
    let out = match race_exec::run(&requests) {
        Ok(out) => out,
        Err(e) => {
            problems.push(format!("{name} n{n}: kernel run failed: {e}"));
            return problems;
        }
    };
    for (index, (request, seed)) in requests.iter().enumerate() {
        let tag = format!("{name} n{n} course {course} seed {seed}");
        let race = &out.races[index];
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
        if cpu.runners != n {
            problems.push(format!("{tag}: cpu field size {} (expected {n})", cpu.runners));
            continue;
        }
        if cpu.frames.len() != race.frames.len() {
            problems.push(format!("{tag}: tick count cpu {} gpu {}", cpu.frames.len(), race.frames.len()));
        }
        let common = cpu.frames.len().min(race.frames.len());
        'frames: for (tick, (cpu_frame, gpu_frame)) in
            cpu.frames[..common].iter().zip(race.frames[..common].iter()).enumerate()
        {
            for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(gpu_frame.iter()).enumerate() {
                for (column, (cpu, gpu)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                    if cpu.to_bits() != gpu.to_bits() {
                        problems.push(format!(
                            "{tag} tick {tick} runner {runner} field {}: cpu bits {:#018x} gpu bits {:#018x}",
                            COLUMNS[column],
                            cpu.to_bits(),
                            gpu.to_bits()
                        ));
                        break 'frames;
                    }
                }
            }
        }
        for (runner, cpu) in cpu.finish_times.iter().enumerate() {
            let gpu = race.finish_raw_all.get(runner).copied().flatten().unwrap_or(race.loop_end);
            if cpu.to_bits() != gpu.to_bits() {
                problems.push(format!(
                    "{tag} runner {runner} finishTimeRaw: cpu {cpu:?} ({}), gpu {gpu:?} ({})",
                    cpu.to_bits(),
                    gpu.to_bits()
                ));
            }
        }
    }
    problems
}

/// Prints the first divergence (or a clean summary) per roster shape, one
/// seed, course 10504. Diagnostic for the bisect, not a gate.
fn report(name: &str, base: &str, horses: Vec<Value>) {
    let problems = check_shape(name, base, 10504, horses);
    if problems.is_empty() {
        println!("{name}: clean");
    } else {
        println!("{name}:\n{}", problems.join("\n"));
    }
}

#[test]
#[ignore]
fn report_first_divergences() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);

    // One Oikomi in the field, at different runner positions.
    report("3r-one-oikomi-last", &base, vec![uma("Senkou", 0), mob("Sasi", 1, (0., 0.)), mob("Oikomi", 2, (0., 0.))]);
    report("3r-one-oikomi-mid", &base, vec![uma("Senkou", 0), mob("Oikomi", 1, (0., 0.)), mob("Sasi", 2, (0., 0.))]);
    report("4r-one-oikomi", &base, vec![mob("Nige", 0, (0., 0.)), uma("Senkou", 1), mob("Sasi", 2, (0., 0.)), mob("Oikomi", 3, (0., 0.))]);
    // Two Oikomi (the mob preset carries a pair).
    report("4r-two-oikomi", &base, vec![uma("Senkou", 0), mob("Sasi", 1, (0., 0.)), mob("Oikomi", 2, (0., 0.)), mob("Oikomi", 3, (0., 0.))]);
    // Mixed styles without any Oikomi.
    report("4r-no-oikomi", &base, vec![mob("Nige", 0, (0., 0.)), uma("Senkou", 1), mob("Sasi", 2, (0., 0.)), mob("Sasi", 3, (0., 0.))]);
    // Oonige probe.
    report("4r-oonige", &base, vec![uma("Senkou", 0), mob("Sasi", 1, (0., 0.)), mob("Nige", 2, (0., 0.)), horse("Oonige", [1600., 1200., 1200., 600., 1200.], "S", 0, 3)]);
    // The app field with the uma gate moved.
    report("mob9-gate0", &base, field9(0));
    report("mob9-gate8", &base, field9(8));
    // Uniform mob stats, uniform style, 9 runners: the stat-block control.
    report("uniform-senkou9", &base, (0..9).map(|g| mob("Senkou", g, (0., 0.))).collect());
    // The v77 shapes with mob stats: style-only control at 3 and 9 runners.
    report("senkou-spread9", &base, (0..9).map(|g| mob("Senkou", g, (0., (g as f64) * -10.))).collect());
    // Mixed styles, no stat spread: isolation of style alone.
    report("mixed-no-spread9", &base, field9(3).into_iter().enumerate().map(|(g, _)| mob(["Nige", "Nige", "Senkou", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"][g], g, (0., 0.))).collect());
}

#[test]
fn zero_skill_mixed_style_fields_are_bit_exact() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);
    let geometry2 =
        std::fs::read_to_string(repo_file("data/course_geometry/10604.json")).expect("geometry file");
    globalsim::register_course_geometry(10604, geometry2);

    let mut all: Vec<String> = Vec::new();

    // Two-runner controls: the smallest field that can block.
    let n2_uniform = vec![uma("Senkou", 0), mob("Senkou", 1, (0., 0.))];
    let n2_mixed = vec![uma("Senkou", 0), mob("Oikomi", 1, (0., 0.))];
    // Three-runner fields, the hypothesis grid: style combinations over the
    // mob stat block, one uniform-style control with the same spread.
    let n3_senkou = vec![uma("Senkou", 0), mob("Senkou", 1, (0., 0.)), mob("Senkou", 2, (0., -80.))];
    let n3_nige_senkou_oikomi =
        vec![mob("Nige", 0, (0., 0.)), uma("Senkou", 1), mob("Oikomi", 2, (0., 0.))];
    let n3_oikomi_pair = vec![uma("Senkou", 0), mob("Oikomi", 1, (0., 0.)), mob("Oikomi", 2, (0., 0.))];
    let n3_all_mixed = vec![mob("Sasi", 0, (0., 0.)), mob("Oikomi", 1, (0., 0.)), mob("Nige", 2, (0., 0.))];

    for (name, horses) in [
        ("uniform-senkou", n2_uniform),
        ("oikomi-vs-senkou", n2_mixed),
        ("senkou-spread", n3_senkou),
        ("nige-senkou-oikomi", n3_nige_senkou_oikomi),
        ("oikomi-pair", n3_oikomi_pair),
        ("sasi-oikomi-nige", n3_all_mixed),
    ] {
        all.extend(check_shape(name, &base, 10504, horses));
    }

    // The app's mob-layout field and the extra-lane 18-runner field.
    all.extend(check_shape("mob9", &base, 10504, field9(3)));
    all.extend(check_shape("mob18", &base, 10504, field18()));

    // The nine-style mixed field, uniform mob stats. It reproduced the
    // tick-579 passing-search divergence: the crowd walk needs the cached
    // rows of runners beyond the collect reach.
    let all9_mixed: Vec<Value> = (0..9)
        .map(|g| mob(["Nige", "Nige", "Senkou", "Senkou", "Senkou", "Sasi", "Sasi",
            "Oikomi", "Oikomi"][g], g, (0., 0.)))
        .collect();
    all.extend(check_shape("all9-mixed", &base, 10504, all9_mixed.clone()));

    // A second course: different corner layout and distance for the same
    // fields.
    all.extend(check_shape("mob9", &base, 10604, field9(3)));
    all.extend(check_shape("all9-mixed", &base, 10604, all9_mixed));

    if !all.is_empty() {
        panic!("{} problems:\n{}", all.len(), all.join("\n"));
    }
}

/// A field containing an Oonige refuses until the start-dash subsystem is
/// ported. Before this refusal it diverged at tick 1 without any flag.
#[test]
fn oonige_fields_are_refused_not_silently_wrong() {
    let base = v77_line();
    let request = request_with(
        &base,
        vec![uma("Senkou", 0), mob("Sasi", 1, (0., 0.)), mob("Nige", 2, (0., 0.)),
            horse("Oonige", [1600., 1200., 1200., 600., 1200.], "S", 0, 3)],
        10504,
        1,
    );
    let out = race_exec::run(&[(request.clone(), 1)]).expect("kernel run");
    assert_eq!(out.races[0].fallback.as_deref(), Some("Oonige start dash"));
    let facts = globalsim_gpu::facade::DeviceFacts {
        extensions: "cl_khr_fp64".into(),
        single_fp_config: 0xbf,
        double_fp_config: 0x3f,
    };
    assert_eq!(
        globalsim_gpu::facade::gate_reasons(&request, Some(&facts)).as_deref(),
        Some("Oonige start dash")
    );
}
