//! Bit-exact gate for the initialization kernel (handoff step 6, initialization
//! slice). Each chain runs hundreds of deterministic input cases through both
//! the kernel and the host reference, and every value is compared via to_bits.
//! The v77 test drives the real engine and pins the reply-visible values.
#![cfg(feature = "gpu")]
#![cfg(feature = "initialization")]

use globalsim_gpu::initialization::{
    self as ini, Case, HpCase, HpPhys, InitCase, InitParams, InitRunner, R30Case, StatsCase,
};

fn splitmix(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn spread(next: &mut impl FnMut() -> u64, lo: f64, hi: f64) -> f64 {
    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64;
    lo + (hi - lo) * unit
}

fn nextafter(value: f64, up: bool) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    let mut bits = value.to_bits();
    let negative = (bits >> 63) != 0;
    if negative != up {
        bits = bits.wrapping_add(1);
    } else {
        bits = bits.wrapping_sub(1);
    }
    f64::from_bits(bits)
}

struct Tally {
    checks: usize,
    failures: usize,
    shown: Vec<String>,
}

impl Tally {
    fn new() -> Tally {
        Tally { checks: 0, failures: 0, shown: Vec::new() }
    }
    fn eq64(&mut self, tag: &str, got: u64, want: u64) {
        self.checks += 1;
        if got != want {
            self.failures += 1;
            if self.shown.len() < 8 {
                self.shown.push(format!("{tag}: kernel {got:#018x} reference {want:#018x}"));
            }
        }
    }
    fn eq32(&mut self, tag: &str, got: u32, want: u32) {
        self.eq64(tag, got as u64, want as u64);
    }
    fn finish(&mut self, label: &str) {
        assert!(
            self.failures == 0,
            "{label}: {} of {} checks failed, first eight:\n{}",
            self.failures, self.checks, self.shown.join("\n")
        );
        println!("{label}: {} checks bit-exact", self.checks);
    }
}

fn real_params() -> (InitParams, HpPhys) {
    let root = ini::engine_parameters_value()
        .expect("global_parameters.json")
        .get("parameters").expect("parameters group").clone();
    let group = |name: &str| root.get(name).cloned().unwrap_or(ini::EngineValue::Null);
    let baset = group("BasetTargetSpeed");
    let temptation = group("Temptation");
    let hp = group("Hp");
    let params = InitParams {
        force: [
            root.get_f64("forceInMoveTargetSpeedAddRunningOut").expect("force1"),
            root.get_f64("forceInMoveTargetSpeedAddProceding").expect("force2"),
            root.get_f64("forceInMoveTargetSpeedAddInsertion").expect("force3"),
            root.get_f64("forceInMoveTargetSpeedAddChasing").expect("force4"),
        ],
        force_span: root.get_f64("forceInMoveTargetSpeedAddRandomRange").expect("span"),
        delay_max: root.get_f64("startDelayMax").expect("startDelayMax"),
        minus_val1: baset.get_f64("BaseTargetSpeedRandomMinusVal1").expect("minus1"),
        minus_val2: baset.get_f64("BaseTargetSpeedRandomMinusVal2").expect("minus2"),
        plus_val1: baset.get_f64("BaseTargetSpeedRandomPlusVal1").expect("plus1"),
        tempt_val1: temptation.get_f64("TemptationStartPerVal1").expect("tempt1"),
        lot_min: temptation.get_i64("TemptationLotSectionMin").expect("lotmin"),
        lot_max: temptation.get_i64("TemptationLotSectionMax").expect("lotmax"),
    };
    let phys = HpPhys {
        hp_initial: hp.get_f64("HpInitialVal1").expect("hp_initial"),
        hp_style: [
            hp.get_f64("HpMaxCoefNige").expect("nige"),
            hp.get_f64("HpMaxCoefSenko").expect("senko"),
            hp.get_f64("HpMaxCoefSashi").expect("sashi"),
            hp.get_f64("HpMaxCoefOikomi").expect("oikomi"),
            hp.get_f64("HpMaxCoefOoNige").expect("oonige"),
        ],
        course: [
            root.get_f64("raceBaseSpeedBaseSpeed").expect("c0"),
            root.get_f64("raceBaseSpeedBaseDistance").expect("c1"),
            root.get_f64("raceBaseSpeedDistanceRate").expect("c2"),
        ],
        drain3: hp.get_f64("HpGutsCoefSqrt").expect("drain3"),
        drain4: hp.get_f64("HpGutsCoef").expect("drain4"),
    };
    (params, phys)
}

#[test]
fn adjusted_stats_chain_bit_exact() {
    let mut cases: Vec<StatsCase> = Vec::new();
    // Boundary raws around the overcap (1200), the course-speed terms
    // (300/600/900/2000) and the clamp, crossed by mood and rank thresholds.
    let boundaries = [0.0, 1.0, 119.0, 300.0, 301.0, 600.0, 601.0, 900.0, 901.0,
        1200.0, 1200.0000001, 1400.0, 1600.0, 2000.0, 2001.0, 2400.0, 3600.0, 9999.0];
    let mut next = splitmix(0x57_5354_4154_53);
    for &raw in boundaries.iter() {
        for &mood in [-2, 0, 2].iter() {
            for &rank in [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 26, 30].iter() {
                cases.push(StatsCase {
                    raw: [raw, raw * 0.75, raw * 1.2, raw * 0.5, raw],
                    mood,
                    rank,
                    single_mode: false,
                    surface: 1 + (next() % 2) as i32,
                    strategy_apt: (next() % 8) as i32,
                    ground: 1 + (next() % 4) as i32,
                    css: match next() % 5 {
                        0 => vec![],
                        1 => vec![1],
                        2 => vec![5],
                        3 => vec![2, 4],
                        _ => vec![0, 3],
                    },
                });
            }
        }
    }
    while cases.len() < 704 {
        cases.push(StatsCase {
            raw: [
                spread(&mut next, 0.0, 2500.0),
                spread(&mut next, 0.0, 2500.0),
                spread(&mut next, 0.0, 3000.0),
                spread(&mut next, -50.0, 2500.0),
                spread(&mut next, 0.0, 100000.0),
            ],
            mood: (next() % 9) as i32 - 4,
            rank: (next() % 14) as i32 - 1,
            single_mode: false,
            surface: 1 + (next() % 2) as i32,
            strategy_apt: (next() % 8) as i32,
            ground: 1 + (next() % 4) as i32,
            css: match next() % 6 {
                0 => vec![],
                1 => vec![1],
                2 => vec![3],
                3 => vec![4],
                4 => vec![1, 3],
                _ => vec![2, 4],
            },
        });
    }
    let run = ini::run(&cases.iter().map(|c| Case::Stats(c.clone())).collect::<Vec<_>>())
        .expect("kernel run");
    println!("device {} build {} ms", run.device.name, run.build_ms);
    let mut tally = Tally::new();
    for (i, case) in cases.iter().enumerate() {
        let want = ini::stats_reference(case);
        let got = match &run.outs[i] {
            ini::Out::Stats(out) => out,
            _ => panic!("stats case decoded as another chain"),
        };
        assert_eq!(got.fallback, 0, "case {i} reported fallback {}", got.fallback);
        for (s, name) in ["speed", "stamina", "power", "guts", "wisdom"].iter().enumerate() {
            tally.eq64(&format!("stats {i} {name} base"), got.base[s], want.base[s]);
            tally.eq64(&format!("stats {i} {name} adjusted"), got.adjusted[s], want.adjusted[s]);
        }
        tally.eq64(&format!("stats {i} raw_stamina"), got.raw_stamina, want.raw_stamina);
        tally.eq64(&format!("stats {i} raw_power"), got.raw_power, want.raw_power);
    }
    tally.finish("adjusted stats chain");
}

#[test]
fn rule30_seeding_chain_bit_exact() {
    let mut next = splitmix(0x52_3330_43_4841_494E);
    let slope_sets: Vec<Vec<(f64, f64)>> = vec![
        vec![],
        vec![(125.0, 20000.0), (325.0, 15000.0), (625.0, -15000.0), (1825.0, 20000.0)],
        vec![(0.0, -5.0)],
        vec![(0.0, 3.0)],
        vec![(0.0, 0.0), (300.0, -2.0)],
        (0..7).map(|i| (i as f64 * 400.0 + 50.0, (i as f64 - 3.0) * 2.0)).collect(),
        (0..7).map(|i| (if i == 0 { 0.0 } else { i as f64 * 300.0 }, if i % 2 == 0 { -1.5 } else { 2.5 })).collect(),
    ];
    let mut cases: Vec<R30Case> = Vec::new();
    while cases.len() < 512 {
        let slopes = slope_sets[(next() % slope_sets.len() as u64) as usize].clone();
        cases.push(R30Case {
            seed: next() as i64,
            runner: (next() % 18) as u32,
            slopes,
            wisdom: spread(&mut next, 1.0, 2000.0),
        });
    }
    // Hill-boundary items: solve the gate downhill comparison roll < wisdom *
    // 0.0004 for wisdom and cross it with the raw neighbours.
    for seed in 1..=64i64 {
        let probe = R30Case { seed, runner: 0, slopes: vec![(0.0, -5.0)], wisdom: 1000.0 };
        let roll = f64::from_bits(ini::r30_reference(&probe).hill0_roll);
        let solved = roll / 0.0004;
        for wisdom in [solved, nextafter(solved, true), nextafter(solved, false), 0.0] {
            cases.push(R30Case { seed, runner: 0, slopes: vec![(0.0, -5.0)], wisdom });
        }
    }

    let run = ini::run(&cases.iter().map(|c| Case::R30(c.clone())).collect::<Vec<_>>())
        .expect("kernel run");
    println!("device {} build {} ms", run.device.name, run.build_ms);
    let mut tally = Tally::new();
    let mut gated = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let want = ini::r30_reference(case);
        let got = match &run.outs[i] {
            ini::Out::R30(out) => out,
            _ => panic!("r30 case decoded as another chain"),
        };
        assert_eq!(got.fallback, 0, "case {i} reported fallback {}", got.fallback);
        assert_eq!(got.cap_flag, 0, "case {i} hit the uniform cap");
        assert_eq!(want.cap_flag, 0, "case {i} reference hit the uniform cap");
        if want.hill_gate == 1 {
            gated += 1;
        }
        tally.eq32(&format!("r30 {i} d1"), got.d1, want.d1);
        tally.eq32(&format!("r30 {i} d2"), got.d2, want.d2);
        tally.eq32(&format!("r30 {i} hp_seed"), got.hp_seed, want.hp_seed);
        tally.eq32(&format!("r30 {i} hp hi"), got.hp_state.0, want.hp_state.0);
        tally.eq32(&format!("r30 {i} hp lo"), got.hp_state.1, want.hp_state.1);
        tally.eq32(&format!("r30 {i} gate_roll"), got.gate_roll, want.gate_roll);
        tally.eq32(&format!("r30 {i} random_lot"), got.random_lot, want.random_lot);
        tally.eq32(&format!("r30 {i} kakari_u"), got.kakari_uniform, want.kakari_uniform);
        tally.eq32(&format!("r30 {i} kakari_idx"), got.kakari_idx, want.kakari_idx);
        tally.eq32(&format!("r30 {i} hill_gate"), got.hill_gate, want.hill_gate);
        tally.eq32(&format!("r30 {i} downhill"), got.downhill, want.downhill);
        assert_eq!(got.hills.len(), want.hills.len(), "case {i} hill count");
        for (h, (g, w)) in got.hills.iter().zip(want.hills.iter()).enumerate() {
            tally.eq32(&format!("r30 {i} hill {h} lo"), g.0, w.0);
            tally.eq32(&format!("r30 {i} hill {h} hi"), g.1, w.1);
        }
        tally.eq32(&format!("r30 {i} hill0 hi"), got.hill0_post.0, want.hill0_post.0);
        tally.eq32(&format!("r30 {i} hill0 lo"), got.hill0_post.1, want.hill0_post.1);
        tally.eq32(&format!("r30 {i} solver hi"), got.solver_post.0, want.solver_post.0);
        tally.eq32(&format!("r30 {i} solver lo"), got.solver_post.1, want.solver_post.1);
        tally.eq64(&format!("r30 {i} start random"), got.start_delay_random, want.start_delay_random);
        tally.eq64(&format!("r30 {i} check random"), got.check_random, want.check_random);
        for (k, (g, w)) in got.r.iter().zip(want.r.iter()).enumerate() {
            tally.eq64(&format!("r30 {i} r{k}"), *g, *w);
        }
        for (s, (g, w)) in got.sections.iter().zip(want.sections.iter()).enumerate() {
            tally.eq64(&format!("r30 {i} section {s}"), *g, *w);
        }
        tally.eq64(&format!("r30 {i} hill0 roll"), got.hill0_roll, want.hill0_roll);
    }
    assert!(gated > 128, "only {gated} gate-fire cases, hill branch coverage is thin");
    println!("hill-gate fired in {gated} cases");
    tally.finish("rule30 seeding chain");
}

#[test]
fn retail_initialization_stream_bit_exact() {
    let (real, _phys) = real_params();
    let crafted: Vec<InitParams> = vec![
        InitParams { force: [-0.02, 0.01, 0.01, -0.03], force_span: 0.1, delay_max: 0.1, ..real },
        InitParams { force: [0.0, 0.0, 0.0, 0.0], force_span: 0.0, delay_max: 0.0, ..real },
        InitParams { force: [0.02, 0.01, 0.01, 0.03], force_span: -0.1, delay_max: 0.1, ..real },
        InitParams { minus_val1: 0.65, ..real },
        InitParams { minus_val2: 1.0, plus_val1: 1.0, ..real },
        InitParams { delay_max: -0.1, ..real },
        InitParams { tempt_val1: 0.0, ..real },
        InitParams { tempt_val1: -13.0, ..real },
        InitParams { lot_min: 0, lot_max: 0, ..real },
        InitParams { lot_min: 0, lot_max: 1, ..real },
        real,
    ];
    let wisdom_edges = [1.0f32, 1.0000001, 9.999999, 10.0, 10.000001, 100.0, 1000.0, 2000.0,
        112.49999, 112.5, 112.50001];
    let mut next = splitmix(0x49_4E49_545F_434841);
    let mut cases: Vec<InitCase> = Vec::new();
    while cases.len() < 576 {
        let n = 1 + (next() % 24) as usize;
        let params = crafted[(next() % crafted.len() as u64) as usize];
        let runners = (0..n)
            .map(|r| InitRunner {
                style: match next() % 7 {
                    0 => 0,
                    1 => 5,
                    2 => 8,
                    _ => 1 + (r % 4) as i64,
                },
                base_wisdom: spread(&mut next, 1.0, 3000.0),
                adjusted_wisdom: if next() % 4 == 0 {
                    wisdom_edges[(next() % wisdom_edges.len() as u64) as usize]
                } else {
                    spread(&mut next, 1.0, 2000.0) as f32
                },
                bonus: if next() % 6 == 0 {
                    [-100.0f32, 100.0, 0.5, -0.5, 12.999999, 13.000001][(next() % 6) as usize]
                } else {
                    spread(&mut next, -20.0, 20.0) as f32
                },
            })
            .collect();
        cases.push(InitCase {
            seed: next() as i32,
            runners,
            params,
            distance: [1200.0, 1400.0, 1600.0, 2000.0, 2000.5, 2400.0, 3200.0, 3600.0][(next() % 8) as usize],
            paired_seed: match next() % 3 {
                0 => None,
                1 => Some(next() as i32),
                _ => Some(-1),
            },
        });
    }
    let run = ini::run(&cases.iter().map(|c| Case::Init(c.clone())).collect::<Vec<_>>())
        .expect("kernel run");
    println!("device {} build {} ms", run.device.name, run.build_ms);
    let mut tally = Tally::new();
    let mut rushed = 0usize;
    let mut paired = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let want = ini::init_reference(case).expect("reference chain");
        let got = match &run.outs[i] {
            ini::Out::Init(out) => out,
            _ => panic!("init case decoded as another chain"),
        };
        assert_eq!(got.fallback, 0, "case {i} reported fallback {}", got.fallback);
        for (s, stage) in ["triggers", "ai", "wisdom", "delays", "sections", "rushing"].iter().enumerate() {
            tally.eq32(&format!("init {i} stage after_{stage}"), got.stages[s], want.stages[s]);
        }
        if want.stages[ini::STAGE_AFTER_RUSHING] > want.stages[ini::STAGE_AFTER_SECTIONS] {
            rushed += 1;
        }
        if case.paired_seed.is_some() {
            paired += 1;
        }
        assert_eq!(got.rows.len(), want.rows.len(), "case {i} row count");
        for (r, (g, w)) in got.rows.iter().zip(want.rows.iter()).enumerate() {
            let tag = format!("init {i} runner {r}");
            tally.eq32(&format!("{tag} force"), g.force_in_move, w.force_in_move);
            tally.eq32(&format!("{tag} chance"), g.chance, w.chance);
            tally.eq32(&format!("{tag} delay"), g.start_delay, w.start_delay);
            tally.eq32(&format!("{tag} rush chance"), g.rushing_chance, w.rushing_chance);
            tally.eq32(&format!("{tag} rush present"), g.rushing_present as u32, w.rushing_present as u32);
            tally.eq32(&format!("{tag} rush section"), g.rushing_section as u32, w.rushing_section as u32);
            for (s, (gs, ws)) in g.sections.iter().zip(w.sections.iter()).enumerate() {
                tally.eq32(&format!("{tag} section {s}"), *gs, *ws);
            }
            tally.eq64(&format!("{tag} kakari start"), g.kakari_start, w.kakari_start);
        }
        for (c, (g, w)) in got.cont_ints.iter().zip(want.cont_ints.iter()).enumerate() {
            tally.eq32(&format!("init {i} cont int {c}"), *g, *w);
        }
        for (c, (g, w)) in got.cont_doubles.iter().zip(want.cont_doubles.iter()).enumerate() {
            tally.eq64(&format!("init {i} cont double {c}"), *g, *w);
        }
    }
    assert!(rushed > 0, "no case drew a rushing lottery, gate is vacuous");
    println!("paired {paired} of {} cases, {rushed} cases drew rushing", cases.len());
    tally.finish("retail initialization stream");
}

#[test]
fn initial_hp_bit_exact() {
    let (_params, real) = real_params();
    let crafted_phys: Vec<HpPhys> = vec![
        HpPhys { hp_initial: 0.0, ..real },
        HpPhys { drain3: 1.0, drain4: 0.0, ..real },
        HpPhys { hp_style: [2.0, 0.5, 1.0, 0.0, 1.02], ..real },
        HpPhys { course: [15.0, 2000.0, 400.0], ..real },
        real,
    ];
    let stat_edges = [0.0f64, 1.0, 1.0000001, 1999.9999, 2000.0, 2000.0001, 5000.0, -4.0];
    let mut next = splitmix(0x48_505F43_4841_494E);
    let mut cases: Vec<HpCase> = Vec::new();
    while cases.len() < 576 {
        let edge = next() % 3 == 0;
        cases.push(HpCase {
            stamina: if edge { stat_edges[(next() % stat_edges.len() as u64) as usize] } else { spread(&mut next, 1.0, 2000.0) },
            guts: if edge { stat_edges[(next() % stat_edges.len() as u64) as usize] } else { spread(&mut next, 1.0, 2000.0) },
            wisdom: spread(&mut next, 1.0, 2000.0),
            distance: [0.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0, 2500.0, 3200.0, 3600.0][(next() % 12) as usize],
            strategy: 1 + (next() % 5) as i32,
            surface: 1 + (next() % 2) as i32,
            ground: 1 + (next() % 4) as i32,
            phys: crafted_phys[(next() % crafted_phys.len() as u64) as usize],
        });
    }
    let run = ini::run(&cases.iter().map(|c| Case::Hp(*c)).collect::<Vec<_>>()).expect("kernel run");
    println!("device {} build {} ms", run.device.name, run.build_ms);
    let mut tally = Tally::new();
    for (i, case) in cases.iter().enumerate() {
        let want = ini::hp_reference(case).expect("reference chain");
        let got = match &run.outs[i] {
            ini::Out::Hp(out) => out,
            _ => panic!("hp case decoded as another chain"),
        };
        assert_eq!(got.fallback, 0, "case {i} reported fallback {}", got.fallback);
        let tag = format!("hp {i}");
        tally.eq32(&format!("{tag} guts factor"), got.guts_factor, want.guts_factor);
        tally.eq32(&format!("{tag} wisdom"), got.wisdom, want.wisdom);
        tally.eq64(&format!("{tag} ground modifier"), got.ground_modifier, want.ground_modifier);
        tally.eq64(&format!("{tag} base speed"), got.base_speed, want.base_speed);
        tally.eq64(&format!("{tag} max hp"), got.max_hp, want.max_hp);
    }
    tally.finish("initial hp");
}

#[test]
fn v77_initialization_pinned_in_reply() {
    let geometry = ini::engine_file("data/course_geometry/10504.json").expect("geometry file");
    globalsim::register_course_geometry(10504, geometry);
    let request = ini::engine_file("examples/v77-race.jsonl").expect("request file")
        .lines().next().expect("first line").to_string();
    let seeds = [1i64, 2, 3, 4, 5, 6, 7, 8];
    let pins = ini::v77_reference(&request, &seeds).expect("engine races");

    let root = ini::engine_parse(&request).expect("request json");
    let horses = root.get_array("horses").expect("horses");
    let runner_count = horses.len();
    let courses = ini::engine_json("data/course_data.json").expect("course_data.json");
    let course = courses.get("10504").expect("course 10504");
    let distance = course.get_f64("distance").expect("distance");
    let surface = course.get_i64("surface").expect("surface") as i32;
    let css: Vec<i32> = course.get_array("courseSetStatus").unwrap_or(&Vec::new())
        .iter().filter_map(|v| v.as_f64()).map(|v| v as i32).collect();
    let slopes: Vec<(f64, f64)> = course.get_array("slopes").expect("slopes").iter()
        .map(|s| Ok((s.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0),
                      s.get("slope").and_then(|v| v.as_f64()).unwrap_or(0.0))))
        .collect::<Result<_, String>>().expect("slope fields");
    let (params, phys) = real_params();
    // v77 field: no skills, mood 2, no teamRank, strategy Senkou, aptitude A.
    let strategy = 2i32;
    let stats_cases: Vec<StatsCase> = horses.iter().map(|h| StatsCase {
        raw: [h.get_f64("speed").expect("speed"), h.get_f64("stamina").expect("stamina"),
              h.get_f64("power").expect("power"), h.get_f64("guts").expect("guts"),
              h.get_f64("wisdom").expect("wisdom")],
        mood: h.get_i64("mood").expect("mood") as i32,
        rank: 0,
        surface,
        strategy_apt: 1,
        ground: 1,
        css: css.clone(),
        single_mode: false,
    }).collect();

    // First run: the adjusted stats, whose wisdom feeds every later chain.
    let stats_run = ini::run(&stats_cases.iter().map(|c| Case::Stats(c.clone())).collect::<Vec<_>>())
        .expect("stats kernel run");
    let mut tally = Tally::new();
    let mut adjusted: Vec<[f64; 5]> = Vec::new();
    for (i, case) in stats_cases.iter().enumerate() {
        let want = ini::stats_reference(case);
        let got = match &stats_run.outs[i] {
            ini::Out::Stats(out) => out,
            _ => panic!("stats decode"),
        };
        for s in 0..5 {
            tally.eq64(&format!("v77 stats {i} base {s}"), got.base[s], want.base[s]);
            tally.eq64(&format!("v77 stats {i} adjusted {s}"), got.adjusted[s], want.adjusted[s]);
        }
        adjusted.push(adjusted_f64(got));
    }

    // Second run: rule30 seeding (all seeds and runners), one initialization
    // stream per seed, and the initial HP.
    let mut cases: Vec<Case> = Vec::new();
    for seed in seeds {
        for runner in 0..runner_count {
            cases.push(Case::R30(R30Case {
                seed,
                runner: runner as u32,
                slopes: slopes.clone(),
                wisdom: adjusted[runner][4],
            }));
        }
    }
    let init_cases: Vec<InitCase> = seeds.iter().map(|&seed| InitCase {
        seed: seed as i32,
        runners: (0..runner_count)
            .map(|r| InitRunner {
                style: strategy as i64,
                base_wisdom: f64::from_bits(ini::stats_reference(&stats_cases[r]).base[4]),
                adjusted_wisdom: adjusted[r][4] as f32,
                bonus: 0.0,
            })
            .collect(),
        params,
        distance,
        paired_seed: None,
    }).collect();
    for c in init_cases.iter() {
        cases.push(Case::Init(c.clone()));
    }
    let hp_cases: Vec<HpCase> = (0..runner_count)
        .map(|r| HpCase {
            stamina: adjusted[r][1],
            guts: adjusted[r][3],
            wisdom: adjusted[r][4],
            distance,
            strategy,
            surface,
            ground: 1,
            phys,
        })
        .collect();
    for c in hp_cases.iter() {
        cases.push(Case::Hp(*c));
    }
    let run = ini::run(&cases).expect("kernel run");
    println!("device {} build {} ms", run.device.name, run.build_ms);

    let (guts_order, wiz_order) = ini::order_reference(&adjusted);
    let mut cursor = 0usize;
    for seed in seeds {
        for runner in 0..runner_count {
            let case = R30Case { seed, runner: runner as u32, slopes: slopes.clone(), wisdom: adjusted[runner][4] };
            let want = ini::r30_reference(&case);
            let got = match &run.outs[cursor] {
                ini::Out::R30(out) => out.clone(),
                _ => panic!("r30 decode"),
            };
            cursor += 1;
            let tag = format!("v77 r30 seed {seed} runner {runner}");
            tally.eq32(&format!("{tag} d1"), got.d1, want.d1);
            tally.eq32(&format!("{tag} hp_seed"), got.hp_seed, want.hp_seed);
            tally.eq32(&format!("{tag} gate_roll"), got.gate_roll, want.gate_roll);
            tally.eq32(&format!("{tag} solver hi"), got.solver_post.0, want.solver_post.0);
            tally.eq32(&format!("{tag} solver lo"), got.solver_post.1, want.solver_post.1);
            for (s, (g, w)) in got.sections.iter().zip(want.sections.iter()).enumerate() {
                tally.eq64(&format!("{tag} section {s}"), *g, *w);
            }
        }
    }
    for (si, case) in init_cases.iter().enumerate() {
        let want = ini::init_reference(case).expect("v77 reference stream");
        let got = match &run.outs[cursor + si] {
            ini::Out::Init(out) => out,
            _ => panic!("init decode"),
        };
        let pin = &pins[si];
        assert_eq!(pin.seed, seeds[si], "pin seed order");
        assert_eq!(got.stages, want.stages, "seed {} stages", seeds[si]);
        for (r, (g, w)) in got.rows.iter().zip(want.rows.iter()).enumerate() {
            tally.eq32(&format!("v77 seed {} runner {r} delay", seeds[si]), g.start_delay, w.start_delay);
            // The reply carries the f32 start delay widened to f64.
            let widened = f64::from(f32::from_bits(g.start_delay)).to_bits();
            tally.eq64(&format!("v77 seed {} runner {r} startDelayTime reply", seeds[si]),
                widened, pin.start_delay[r]);
            for (s, (gs, ws)) in g.sections.iter().zip(w.sections.iter()).enumerate() {
                tally.eq32(&format!("v77 seed {} runner {r} section {s}", seeds[si]), *gs, *ws);
            }
        }
        assert_eq!(pin.guts_order, guts_order, "seed {} gutsOrder", seeds[si]);
        assert_eq!(pin.wiz_order, wiz_order, "seed {} wizOrder", seeds[si]);
    }
    cursor += init_cases.len();
    for (r, case) in hp_cases.iter().enumerate() {
        let want = ini::hp_reference(case).expect("v77 hp reference");
        let got = match &run.outs[cursor + r] {
            ini::Out::Hp(out) => out,
            _ => panic!("hp decode"),
        };
        tally.eq64(&format!("v77 hp runner {r} max"), got.max_hp, want.max_hp);
        for pin in pins.iter() {
            tally.eq64(&format!("v77 hp runner {r} reply seed {}", pin.seed),
                pin.hp0[r], want.max_hp);
        }
    }
    println!("v77 reply fields pinned: startDelayTime, gutsOrder, wizOrder, frame-0 hp ({} runners x {} seeds)",
        runner_count, seeds.len());
    tally.finish("v77 initialization reply");
}

fn adjusted_f64(out: &ini::StatsOut) -> [f64; 5] {
    out.adjusted_f64()
}

#[test]
fn engine_source_pinned_mirrors_untouched() {
    let engine = |rel: &str| ini::engine_file(rel).expect("engine source");
    let builder = engine("src/race_solver_builder.rs");
    for needle in [
        "1200.0 + (stat - 1200.0) / 2.0",
        "if rank <= 0 { return 0.0; }",
        "[(6, 50.), (11, 30.), (16, 20.), (21, 10.), (26, 0.), (99999, 0.)]",
        "let base = adjust_overcap(raw) as f32 + team_rank_bonus(rank);",
        "let motiv_coef = 1.0 + 0.02 * mood as f64;",
        "raw_stamina: desc.stamina * motiv_coef,",
        "+ (desc.power - 1200.0).max(0.0) * 0.5,",
        "GROUND_SPEED_MODIFIER[course.surface as usize][ground]",
        "GROUND_POWER_MODIFIER[course.surface as usize][ground]",
        "[11000.,10000.,8500.,7500.,6000.,4000.,2000.,1000.][base.strategy_aptitude as usize] * 0.0001_f32",
        "seed ^ skill_id ^ 0x57495344",
        "0x85ebca6b",
        "0xc2b2ae35",
        "if hi == 0 { 1 } else { hi }",
        "let solver_rng = Rule30CARng::new(self.rng.int32(), 0);",
        "Rule30CARng::new(solver_rng.int32(), 0)",
    ] {
        assert!(builder.contains(needle), "race_solver_builder.rs no longer contains {needle:?}");
    }
    let solver = engine("src/race_solver.rs");
    for needle in [
        "solver.gate_roll = solver.rng_uniform(12252240.0) as i64;",
        "solver.random_lot = solver.rng_uniform(100.0) as i64;",
        "solver.start_delay = 0.1 * solver.rng.random();",
        "Rule30CARng::new(self.rng.int32(), self.rng.int32())",
        "self.hill_start.last() == Some(&0.0)",
        "let roll = self.hill_rng[0].random();",
        "if self.slope_per < 0.0 && roll < self.horse.wisdom * 0.0004",
        "for _ in 0..24 {",
    ] {
        assert!(solver.contains(needle), "race_solver.rs no longer contains {needle:?}");
    }
    let physics = engine("src/retail_physics.rs");
    for needle in [
        "(bounded_stat(stamina) as f64 * self.hp_initial as f64) * self.hp_style[strategy - 1] as f64",
        "self.course[0] - (distance - self.course[1]) / self.course[2]",
        "self.hp_drain[4] as f64",
        "root(bounded_stat(guts) * self.hp_drain[3])",
        "(value as f64).sqrt() as f32",
    ] {
        assert!(physics.contains(needle), "retail_physics.rs no longer contains {needle:?}");
    }
    let hp = engine("src/hp_policy.rs");
    for needle in [
        "[0.0, 1.0, 1.0, 1.02, 1.02]",
        "[0.0, 1.0, 1.0, 1.01, 1.02]",
        "state.base_speed=parameters.course_speed(course.distance as f32) as f64;",
        "self.max_hp=retail.parameters.initial_hp(horse.stamina as f32,horse.strategy as usize,self.distance as f32) as f64;",
    ] {
        assert!(hp.contains(needle), "hp_policy.rs no longer contains {needle:?}");
    }
    let course = engine("src/course_data.rs");
    for needle in [
        "let count=course.course_set_status.iter().filter(|&&stat|stat!=0).count();",
        "coefficient+=term/count as f32;",
        "let mut coefficient=1.0_f32;",
    ] {
        assert!(course.contains(needle), "course_data.rs no longer contains {needle:?}");
    }
    let retail = engine("src/retail_initialization.rs");
    for needle in [
        "let coefficient = (10000 + 200 * mood.clamp(-2, 2)) as f32 * 0.0001_f32;",
        "((raw as f32) * coefficient).clamp(1.0, 2000.0)",
        "rng.integer(0,100) as f32",
    ] {
        assert!(retail.contains(needle), "retail_initialization.rs no longer contains {needle:?}");
    }
    let lib = engine("src/lib.rs");
    for needle in [
        "let seed_lo = (seed + i as i64 * 7919) as u32;",
        "(1e-6_f32 * left.abs().max(right.abs()))",
        "startDelayTime",
    ] {
        assert!(lib.contains(needle), "lib.rs no longer contains {needle:?}");
    }
    println!("engine source pins present");
}
