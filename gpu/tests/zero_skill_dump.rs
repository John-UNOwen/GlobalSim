//! Diagnostic (ignored): dump every frame column for the first divergent
//! runner around the first divergent tick, CPU vs GPU, for one request built
//! from the mob-layout field. Reads scratch too. Not a gate.

#![cfg(feature = "race")]

use globalsim_gpu::json::{self, Value};
use globalsim_gpu::{dep_cpu, race_exec};

const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn v77_line() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request")
        .lines()
        .next()
        .unwrap()
        .to_string()
}

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

fn mob(strategy: &str, gate: usize, spread: (f64, f64)) -> Value {
    horse(strategy, [1600. + spread.0, 1200., 1200., 600., 1200. + spread.1], "S", 0, gate)
}

fn uma(strategy: &str, gate: usize) -> Value {
    horse(strategy, [1000., 1000., 900., 800., 900.], "A", 2, gate)
}

fn request_with(base: &str, horses: Vec<Value>, course: i32, seed: i64) -> String {
    let mut value = json::parse(base).expect("template");
    let Value::Object(map) = &mut value else { unreachable!() };
    map.insert("horses".into(), json::arr(horses));
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    out
}

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

#[test]
#[ignore]
fn report_style_composition_probes() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let styles9 = ["Nige", "Nige", "Senkou", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let first_divergence = |name: &str, styles: &[&str]| {
        let horses: Vec<Value> =
            styles.iter().enumerate().map(|(g, s)| mob(s, g, (0., 0.))).collect();
        let seed = 1i64;
        let request = request_with(&base, horses.clone(), 10504, seed);
        let out = race_exec::run(&[(request.clone(), seed)]).expect("kernel");
        let race = &out.races[0];
        if let Some(f) = &race.fallback {
            println!("{name}: fallback {f}");
            return;
        }
        let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu");
        let common = cpu.frames.len().min(race.frames.len());
        for tick in 0..common {
            for runner in 0..cpu.runners {
                for column in 0..6 {
                    if cpu.frames[tick][runner][column].to_bits()
                        != race.frames[tick][runner][column].to_bits()
                    {
                        println!("{name}: tick {tick} runner {runner} col {} bits {:#018x}/{:#018x}",
                            COLUMNS[column],
                            cpu.frames[tick][runner][column].to_bits(),
                            race.frames[tick][runner][column].to_bits());
                        return;
                    }
                }
            }
        }
        if cpu.frames.len() != race.frames.len() {
            println!("{name}: tick count {} vs {}", cpu.frames.len(), race.frames.len());
        } else {
            println!("{name}: clean");
        }
    };
    first_divergence("all9-mixed", &styles9);
    first_divergence("no-nige7", &styles9[2..]);
    first_divergence("n2", &["Nige", "Nige"]);
    first_divergence("nige2-senkou1", &["Nige", "Nige", "Senkou"]);
    first_divergence("nige2-sasi1", &["Nige", "Nige", "Sasi"]);
    first_divergence("nige2-oikomi1", &["Nige", "Nige", "Oikomi"]);
    first_divergence("nige2-senkou2", &["Nige", "Nige", "Senkou", "Senkou"]);
    first_divergence("nige1-senkou2", &["Nige", "Senkou", "Senkou"]);
    first_divergence("senkou2-nige2-senkou2", &["Senkou", "Senkou", "Nige", "Nige", "Senkou", "Senkou"]);
    first_divergence("nige-sasi4", &["Nige", "Nige", "Sasi", "Sasi", "Sasi", "Sasi"]);
    first_divergence("nige-oikomi4", &["Nige", "Nige", "Oikomi", "Oikomi", "Oikomi", "Oikomi"]);
    first_divergence("senkou-oikomi4", &["Senkou", "Senkou", "Oikomi", "Oikomi"]);
    first_divergence("senkou-sasi-oikomi3", &["Senkou", "Sasi", "Oikomi"]);
    first_divergence("oikomi2", &["Senkou", "Oikomi", "Oikomi"]);
}

#[test]
#[ignore]
fn dump_engine_competition_events() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let seed = 1i64;
    let request = request_with(&base, field9(3), 10504, seed);
    let cpu = dep_cpu::run_cpu_binary(&request, seed).expect("cpu binary");
    let reply = globalsim_gpu::json::parse(&cpu.reply).expect("reply parses");
    let show = |v: &globalsim_gpu::json::Value| {
        let mut s = String::new();
        globalsim_gpu::json::to_string_into(v, &mut s);
        s
    };
    println!("competitionEvents:");
    match reply.get_array("competitionEvents") {
        Ok(rows) if !rows.is_empty() => {
            for row in rows {
                println!("  {}", show(row));
            }
        }
        Ok(_) => println!("  none"),
        Err(e) => println!("  missing: {e}"),
    }
    let results = reply.get_array("results").expect("results");
    println!("runner 0 result: {}", show(&results[0]));
}

#[test]
#[ignore]
fn dump_pace_snapshot_at_divergence() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let styles = ["Nige", "Nige", "Senkou"];
    let horses: Vec<Value> = styles.iter().enumerate().map(|(g, s)| mob(s, g, (0., 0.))).collect();
    let seed = 1i64;
    let request = request_with(&base, horses, 10504, seed);
    let out = race_exec::run_with_snapshot(&[(request.clone(), seed)], 124).expect("kernel");
    let race = &out.races[0];
    assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
    dump_snap(race, 3);
}

fn dump_snap(race: &globalsim_gpu::race_exec::RaceGpu, n: usize) {
    assert_eq!(race.debug_pace.len(), 14 * n, "debug blocks");
    let parts = [
        ["mode", "coef", "pm", "gap", "rank", "style"],
        ["target", "exit", "cool", "intvl", "flags", "spotT"],
        ["kakari", "rmode", "check", "maxrem", "tempt", "rrem"],
        ["lanemax", "lcool", "lxo", "sideIn", "sideIns", "sideOuts"],
        ["nearest", "best", "straight", "blocked", "front", "tgtN"],
        ["s0", "s1", "s2", "s3", "s4", "s5"],
        ["c3", "l3", "c4", "l4", "cand", "edge"],
    ];
    let rows: Vec<usize> = if n >= 8 { vec![6, 7, 8] } else { (0..n).collect() };
    for b in 0..2 {
        println!("== {} ==", ["pre-move", "post-move"][b]);
        for part in 0..7 {
            for &r in &rows {
                let row = race.debug_pace[(b * 7 + part) * n + r];
                print!("   r{r}:");
                for (k, v) in row.iter().enumerate() {
                    print!(" {}={v:.5}", parts[part][k]);
                }
                println!();
            }
        }
    }
}

#[test]
#[ignore]
fn dump_snap_all9_mixed() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);
    let styles = ["Nige", "Nige", "Senkou", "Senkou", "Senkou", "Sasi", "Sasi", "Oikomi", "Oikomi"];
    let horses: Vec<Value> = styles.iter().enumerate().map(|(g, s)| mob(s, g, (0., 0.))).collect();
    let seed = 1i64;
    let request = request_with(&base, horses.clone(), 10504, seed);
    let out = race_exec::run_with_snapshot(&[(request.clone(), seed)], 579).expect("kernel");
    let race = &out.races[0];
    assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
    let n = horses.len();
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu");
    for tick in 575..590 {
        for runner in 0..n {
            if tick > 579 && runner != 8 {
                continue;
            }
            let c = cpu.frames[tick][runner];
            let g = race.frames[tick][runner];
            let mut line = format!("tick {tick} r{runner}: ");
            for (k, col) in COLUMNS.iter().enumerate() {
                let mark = if c[k].to_bits() == g[k].to_bits() { " " } else { "*" };
                line.push_str(&format!("{col}{mark}={:.6}/{:.6} ", c[k], g[k]));
            }
            println!("{line}");
            if c[3].to_bits() != g[3].to_bits() {
                println!("   lane bits cpu={:016x} gpu={:016x} pos bits cpu={:016x} gpu={:016x}",
                    c[3].to_bits(), g[3].to_bits(), c[0].to_bits(), g[0].to_bits());
            }
        }
    }
    dump_snap(race, n);
}

#[test]
#[ignore]
fn dump_columns_around_first_divergence() {
    let base = v77_line();
    let geometry =
        std::fs::read_to_string(repo_file("data/course_geometry/10504.json")).expect("geometry");
    globalsim::register_course_geometry(10504, geometry);

    let seed = 1i64;
    let horses = field9(3);
    let n = horses.len();
    let request = request_with(&base, horses, 10504, seed);
    let out = race_exec::run(&[(request.clone(), seed)]).expect("kernel");
    let race = &out.races[0];
    assert!(race.fallback.is_none(), "fallback {:?}", race.fallback);
    let cpu = dep_cpu::run_cpu_race(&request, seed).expect("cpu");

    // Earliest divergence over all runners and columns.
    let mut first = (usize::MAX, usize::MAX, usize::MAX);
    'find: for tick in 0..cpu.frames.len().min(race.frames.len()) {
        for runner in 0..n {
            for column in 0..6 {
                if cpu.frames[tick][runner][column].to_bits()
                    != race.frames[tick][runner][column].to_bits()
                {
                    first = (tick, runner, column);
                    break 'find;
                }
            }
        }
    }
    let (ftick, frunner, _) = first;
    if ftick == usize::MAX {
        println!("no divergence: the scene is bit-exact");
        return;
    }
    println!("first divergence tick {ftick} runner {frunner} column {}", COLUMNS[first.2]);
    println!("styles: Nige Nige Senkou uma-Senkou Senkou Sasi Sasi Oikomi Oikomi (uma at gate 3)");
    for tick in ftick.saturating_sub(4)..=(ftick + 2).min(cpu.frames.len() - 1) {
        for runner in 0..n {
            let c = cpu.frames[tick][runner];
            let g = race.frames[tick][runner];
            let mut line = format!("tick {tick} r{runner}: ");
            for (k, col) in COLUMNS.iter().enumerate() {
                let mark = if c[k].to_bits() == g[k].to_bits() { " " } else { "*" };
                line.push_str(&format!("{col}{mark}={:.6}/{:.6} ", c[k], g[k]));
            }
            println!("{line}");
        }
    }
}
