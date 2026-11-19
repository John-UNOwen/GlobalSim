//! Wave-8 identity gates: GutsUp fires outside the constructor pass. Self
//! fires behind a phase gate refire on a cooldown and stack through the f32
//! accumulator, cross fires deliver through the receiver refresh, and every
//! guts-derived word (spurt plan maximum, duel entry snapshot, spot entry
//! snapshot, step-head minimum) recomputes from the live field at the points
//! the engine reads it. All frame-for-frame bit-exact on seeds 1..3, plus the
//! static-control and real-catalog pins and an ignored m=96 perf probe.

#![cfg(feature = "race")]

use globalsim_gpu::json::Value;
use globalsim_gpu::{dep_cpu, json, race_exec};

const SEEDS: [i64; 3] = [1, 2, 3];
const COLUMNS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn merged_skill_data() -> json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn v77_request() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("request file")
        .lines()
        .next()
        .expect("first line")
        .to_string()
}

fn register_geometry(course: i32) {
    let geometry =
        std::fs::read_to_string(repo_file(&format!("data/course_geometry/{course}.json")))
            .expect("geometry");
    globalsim::register_course_geometry(course, geometry);
}

fn compare_race(
    tag: &str,
    request: &str,
    seed: i64,
    race: &race_exec::RaceGpu,
    problems: &mut Vec<String>,
) {
    if let Some(reason) = &race.fallback {
        problems.push(format!("{tag}: fallback {reason}"));
        return;
    }
    let cpu = match dep_cpu::run_cpu_race(request, seed) {
        Ok(cpu) => cpu,
        Err(reason) => {
            problems.push(format!("{tag}: cpu race failed: {reason}"));
            return;
        }
    };
    if cpu.frames.len() != race.frames.len() {
        problems.push(format!(
            "{tag}: tick count cpu {} gpu {}",
            cpu.frames.len(),
            race.frames.len()
        ));
        return;
    }
    'frames: for (tick, (cpu_frame, race_frame)) in
        cpu.frames.iter().zip(race.frames.iter()).enumerate()
    {
        for (runner, (cpu_row, gpu_row)) in cpu_frame.iter().zip(race_frame.iter()).enumerate() {
            for (column, (cv, gv)) in cpu_row.iter().zip(gpu_row.iter()).enumerate() {
                if cv.to_bits() != gv.to_bits() {
                    problems.push(format!(
                        "{tag} tick {tick} runner {runner} {}: cpu {cv:?} ({}), gpu {gv:?} ({})",
                        COLUMNS[column],
                        cv.to_bits(),
                        gv.to_bits()
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

type HorsePatch<'a> = (usize, Option<&'a [i64]>, Option<&'a str>, Option<i64>, Option<i64>);

fn patched_requests(
    skill_data: &Value,
    horses: &[HorsePatch],
    seeds: &[i64],
) -> Vec<(String, i64)> {
    let mut value = json::parse(&v77_request()).expect("v77 parses");
    {
        let json::Value::Object(map) = &mut value else { unreachable!() };
        map.insert("skillData".into(), skill_data.clone());
        let json::Value::Array(entries) = map.get_mut("horses").expect("horses") else {
            unreachable!()
        };
        for (index, ids, strategy, pop, team) in horses {
            let json::Value::Object(horse) = &mut entries[*index] else { unreachable!() };
            if let Some(ids) = ids {
                horse.insert(
                    "skills".into(),
                    json::arr(
                        ids.iter()
                            .map(|&id| {
                                json::obj(vec![
                                    ("skillId", json::Value::Number(id as f64)),
                                    ("level", json::Value::Number(1.)),
                                ])
                            })
                            .collect(),
                    ),
                );
            }
            if let Some(style) = strategy {
                horse.insert("strategy".into(), json::Value::String((*style).into()));
            }
            if let Some(popularity) = pop {
                horse.insert(
                    "popularity".into(),
                    json::Value::Number((*popularity) as f64),
                );
            }
            if let Some(team_id) = team {
                horse.insert("teamId".into(), json::Value::Number((*team_id) as f64));
            }
        }
    }
    let mut requests = Vec::new();
    for &seed in seeds {
        let mut request = value.clone();
        let json::Value::Object(map) = &mut request else { unreachable!() };
        map.insert("seed".into(), json::Value::Number(seed as f64));
        let mut text = String::new();
        json::to_string_into(&request, &mut text);
        requests.push((text, seed));
    }
    requests
}

fn fired_ids(results: &[(String, i64)], out: &race_exec::RaceRun) -> (Vec<i64>, Vec<String>) {
    let mut fired: Vec<i64> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    for (index, (request, seed)) in results.iter().enumerate() {
        if let Some(reason) = &out.races[index].fallback {
            problems.push(format!("seed {seed}: fallback {reason}"));
            continue;
        }
        match dep_cpu::run_cpu_race(request, *seed) {
            Ok(cpu) => {
                for (_, sid, _, _) in &cpu.detail_events {
                    if !fired.contains(sid) {
                        fired.push(*sid);
                    }
                }
            }
            Err(reason) => problems.push(format!("seed {seed}: cpu race failed: {reason}")),
        }
    }
    fired.sort_unstable();
    (fired, problems)
}

fn run_gate(
    label: &str,
    skill_data: &Value,
    horses: &[HorsePatch],
    expect_fired: &[i64],
) {
    let results = patched_requests(skill_data, horses, &SEEDS);
    let out = race_exec::run(&results).expect("kernel run");
    let (fired, mut problems) = fired_ids(&results, &out);
    for sid in expect_fired {
        if !fired.contains(sid) {
            problems.push(format!("{sid} never activated on seeds 1..3"));
        }
    }
    for (index, (request, seed)) in results.iter().enumerate() {
        compare_race(&format!("{label} seed {seed}"), request, *seed, &out.races[index], &mut problems);
    }
    if !problems.is_empty() {
        panic!(
            "{} {label} problems, fired {fired:?}, problems:\n{}",
            problems.len(),
            problems.join("\n")
        );
    }
}

/// Adds battery definitions to the merged table.
fn with_battery(skill_data: &mut Value, defs: &[(i64, &str)]) {
    let json::Value::Object(sd) = skill_data else { unreachable!() };
    for (sid, def) in defs {
        sd.insert(sid.to_string(), json::parse(def).expect("battery def parses"));
    }
}

fn def(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":{base_duration},\"effects\":[{effects}]}}]}}"
    )
}

/// 669001 lifts its own guts at phase 1 and every 15 seconds after, 669002
/// drops it at phase 2 with a positive instance duration. The accumulator
/// stacks through both, the field clamps, and every later duel or spot entry
/// plus the next spurt plan and step-head minimum recompute from the moved
/// field, which the refire cycle re-enters many times per race.
#[test]
fn wave8_battery_guts_self_mid_refire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669001,
                &def("phase>=1", 15000, -1, r#"{"type":4,"target":1,"modifier":40000}"#).as_str(),
            ),
            (
                669002,
                &def("phase>=2", 60000, 30000,
                    r#"{"type":4,"target":1,"modifier":-80000}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669001i64][..]), None, None, None),
        (5, Some(&[669002i64][..]), None, None, None),
        (8, Some(&[669001, 669002][..]), None, None, None),
    ];
    run_gate("guts self mid refire", &skill_data, &horses, &[669001, 669002]);
}

/// 669030 is a construction-time bump, 669031 fires at phase 1 on the same
/// runner. The loop fire must continue from the unclamped f32 accumulator
/// the bump left behind, not from the clamped field.
#[test]
fn wave8_battery_static_chain_into_loop_fire_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669030,
                &def("always>=0", 0, -1, r#"{"type":4,"target":1,"modifier":30000}"#).as_str(),
            ),
            (
                669031,
                &def("phase>=1", 40000, -1, r#"{"type":4,"target":1,"modifier":20000}"#).as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669030i64, 669031][..]), None, None, None),
        (6, Some(&[669031, 669030][..]), None, None, None),
    ];
    run_gate("static chain into guts loop fire", &skill_data, &horses, &[669030, 669031]);
}

/// Cross guts fires: 669010 hits the runner ahead, 669011 the field behind
/// through the self-inclusive fan, 669012 the visible fan. Each receiver
/// takes the accumulator and field write, then the refresh that repatches
/// its spurt maximum and minimum from the moved field.
#[test]
fn wave8_battery_cross_guts_fans_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669010,
                &def("phase>=1", 20000, 30000, r#"{"type":4,"target":9,"modifier":40000}"#)
                    .as_str(),
            ),
            (
                669011,
                &def("phase>=1", 30000, -1, r#"{"type":4,"target":10,"modifier":20000}"#)
                    .as_str(),
            ),
            (
                669012,
                &def("phase>=2", 60000, 30000, r#"{"type":4,"target":4,"modifier":30000}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669010i64, 669011, 669012][..]), None, None, None),
        (5, Some(&[669011][..]), None, None, None),
        (8, Some(&[669010][..]), None, None, None),
    ];
    run_gate("cross guts fans", &skill_data, &horses, &[669010, 669011, 669012]);
}

/// The debuff filters on a negative cross guts fire: 669042 aims the visible
/// fan, the immune runner 669041 drops the hit at selection and the same-team
/// runner behind team 3 drops with it, while the rest of the field takes the
/// write and the refresh.
#[test]
fn wave8_battery_negative_guts_debuff_filters_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[
            (
                669042,
                &def("phase>=1", 20000, 30000, r#"{"type":4,"target":4,"modifier":-30000}"#)
                    .as_str(),
            ),
            (
                669041,
                &def("always>=0", 900000, 120000, r#"{"type":38,"target":1,"modifier":1}"#)
                    .as_str(),
            ),
        ],
    );
    let horses: Vec<HorsePatch> = vec![
        (2, Some(&[669042i64][..]), None, None, Some(3)),
        (3, Some(&[669042][..]), None, None, Some(3)),
        (4, Some(&[669041, 669042][..]), None, None, Some(3)),
    ];
    run_gate("negative guts debuff filters", &skill_data, &horses, &[669041, 669042]);
}

/// The guts-static control: 669009 is a constructor-only GutsUp, so the
/// race compiles with live_guts off and every guts word stays the
/// prepare-baked value. Frames must stay bit-exact, which pins the baked
/// path unchanged, the same proof the untouched wave-7 batteries carry
/// through this kernel.
#[test]
fn wave8_battery_guts_static_control_bit_exact() {
    register_geometry(10504);
    let mut skill_data = merged_skill_data();
    with_battery(
        &mut skill_data,
        &[(
            669009,
            &def("always>=0", 0, -1, r#"{"type":4,"target":1,"modifier":40000}"#).as_str(),
        )],
    );
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[669009i64][..]), None, None, None),
        (7, Some(&[669009][..]), None, None, None),
    ];
    run_gate("guts static control", &skill_data, &horses, &[]);
}

/// The real shipped static GutsUp ids: 200211 (weather==1, true for this
/// request), 202181 (always==1, negative), and 1200051 (course shape). All
/// are constructor bumps with the baked path, and the catalog-wide compile
/// side is pinned by wave8_folds.
#[test]
fn wave8_real_static_kind4_bit_exact() {
    register_geometry(10504);
    let skill_data = merged_skill_data();
    let horses: Vec<HorsePatch> = vec![
        (0, Some(&[200211i64][..]), None, None, None),
        (3, Some(&[202181][..]), None, None, None),
        (7, Some(&[1200051][..]), None, None, None),
    ];
    run_gate("real static kind 4", &skill_data, &horses, &[]);
}

/// Ignored perf probe, the source of the README's wave-8 line: one 96-race
/// batch with a live GutsUp carrier per race against the same batch without
/// any GutsUp record, printing both walls.
#[test]
#[ignore]
fn wave8_perf_m96_live_vs_baked_walls() {
    register_geometry(10504);
    let mut live_data = merged_skill_data();
    with_battery(
        &mut live_data,
        &[(
            669090,
            &def("phase>=1", 10000, -1, r#"{"type":4,"target":1,"modifier":30000}"#).as_str(),
        )],
    );
    let baked_data = merged_skill_data();
    let live = patched_requests(
        &live_data,
        &[(0, Some(&[669090i64][..]), None, None, None)],
        &(1i64..=96).collect::<Vec<_>>(),
    );
    let baked = patched_requests(
        &baked_data,
        &[(0, Some(&[200211i64][..]), None, None, None)],
        &(1i64..=96).collect::<Vec<_>>(),
    );
    // Warm both shapes once, then time the second launch.
    let _ = race_exec::run(&live).expect("live warmup");
    let _ = race_exec::run(&baked).expect("baked warmup");
    let t0 = std::time::Instant::now();
    let out_live = race_exec::run(&live).expect("live batch");
    let live_ms = t0.elapsed().as_secs_f64() * 1000.;
    let t1 = std::time::Instant::now();
    let out_baked = race_exec::run(&baked).expect("baked batch");
    let baked_ms = t1.elapsed().as_secs_f64() * 1000.;
    let fallbacks_live = out_live.races.iter().filter(|r| r.fallback.is_some()).count();
    let fallbacks_baked = out_baked.races.iter().filter(|r| r.fallback.is_some()).count();
    println!("wave8 m=96 walls: live guts {live_ms:.0} ms, baked guts {baked_ms:.0} ms, fallbacks {fallbacks_live}/{fallbacks_baked}");
    assert_eq!(fallbacks_live, 0, "live batch fell back");
    assert_eq!(fallbacks_baked, 0, "baked batch fell back");
}
