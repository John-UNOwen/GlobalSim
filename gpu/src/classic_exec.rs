//! Host lane for the classic (non-retail) physics kernel. Construction runs
//! here with the engine's own Rust float math (stats, seeding chain, sections,
//! duel and spot constants), so the kernel carries no powf or log10 and every
//! constructor value reaches the device exactly as RaceSolver::new computes it.
//! The kernel runs the tick loop and the per-tick rng consumers.

use crate::course_data;
use crate::dep_cpu;
use crate::initialization::{self, StatsCase};
use crate::json::{self, Value};
use crate::race::{FallbackReason, RaceOutcome, RaceRunner};
use crate::random_gen::Rule30CARng;
use ocl::{Buffer, Queue, flags};

pub const CLS_MAXR: usize = 120;
pub const CLS_MAX_TICKS: usize = 6144;
pub const CLS_FRAME_COLS: usize = 14;
pub const CLS_MAX_SLOPES: usize = 16;

pub const CLS_RI_SEED: usize = 0;
pub const CLS_RI_N: usize = 1;
pub const CLS_RI_NSLOPES: usize = 2;
pub const CLS_RI_TRACK: usize = 3;
pub const CLS_RI_SURFACE: usize = 4;
pub const CLS_RI_GROUND: usize = 5;
pub const CLS_RI_FLAGS: usize = 6;
pub const CLS_RI_DUELON: usize = 7;
pub const CLS_RI_NMAX: usize = 8;
pub const CLS_RI_STRIDE: usize = 16 + 48 * CLS_MAXR;
pub const CLS_RI_RUN0: usize = 16;
pub const CLS_RI_RUN_STRIDE: usize = 48;
pub const R_STRATEGY: usize = 0;
pub const R_DISTAPT: usize = 1;
pub const R_SURFAPT: usize = 2;
pub const R_STAPT: usize = 3;
pub const R_SOLVER_HI: usize = 6;
pub const R_SOLVER_LO: usize = 7;
pub const R_HP_HI: usize = 8;
pub const R_HP_LO: usize = 9;
pub const R_PACE_HI: usize = 10;
pub const R_PACE_LO: usize = 11;
pub const R_HILL0: usize = 12;
pub const R_SLOPE_IDX: usize = 44;
pub const R_INIT_DOWNHILL: usize = 46;
pub const R_CAPFLAG: usize = 47;

pub const CLS_RD_RUN0: usize = 16;
pub const CLS_RD_RUN_STRIDE: usize = 64;
pub const D_DISTANCE: usize = 0;
pub const D_MAXLANE: usize = 1;
pub const D_FIRSTMOVE: usize = 2;
pub const D_SECTION: usize = 3;
pub const D_LASTSTR_START: usize = 4;
pub const D_LASTSTR_END: usize = 5;
pub const D_LASTCORNER_START: usize = 6;
pub const D_BASESPEED: usize = 7;
pub const D_GROUNDMOD: usize = 9;
pub const CLS_RD_SLOPES: usize = 16 + 64 * CLS_MAXR;
pub const CLS_RD_STRIDE: usize = CLS_RD_SLOPES + 48;
pub const F_LANE0: usize = 0;
pub const F_SPEED_STAT: usize = 1;
pub const F_STAMINA: usize = 2;
pub const F_POWER: usize = 3;
pub const F_GUTS: usize = 4;
pub const F_WISDOM: usize = 5;
pub const F_DELAY: usize = 6;
pub const F_SECT0: usize = 7;
pub const F_KSTART: usize = 32;
pub const F_KDUR: usize = 33;
pub const F_BTS0: usize = 34;
pub const F_LSS: usize = 37;
pub const F_MINSPEED: usize = 38;
pub const F_BACC0: usize = 39;
pub const F_MAXHP: usize = 45;
pub const F_GUTSMOD: usize = 46;
pub const F_PKMIN: usize = 47;
pub const F_PKMAX: usize = 48;
pub const F_PKEND: usize = 49;
pub const F_PKCONT: usize = 50;
pub const F_PCT15: usize = 51;
pub const F_PCT20: usize = 52;
pub const F_DTS: usize = 53;
pub const F_DA: usize = 54;
pub const F_SPOTM: usize = 55;
pub const F_SPOTD: usize = 56;
pub const F_SUBPAR: usize = 57;
pub const F_SLOPE0: usize = 58;
pub const F_STYLECNT: usize = 59;
pub const F_NIGECNT: usize = 60;

pub const CLS_ST_STRIDE: usize = 16 + 17 * CLS_MAXR;
pub const S_FALLBACK: usize = 0;
pub const S_TICKS: usize = 1;
pub const S_END: usize = 2;
pub const CLS_ST_RUN0: usize = 16;
pub const CLS_ST_RUN_STRIDE: usize = 17;

pub const CLS_SU_STRIDE: usize = 24400;
pub const CLS_SD_STRIDE: usize = 54784;

pub const S_LSTTRANS: usize = 6;
pub const S_SPURTT: usize = 8;
pub const S_SPURTD: usize = 10;
pub const S_TEMPT: usize = 12;
pub const S_RSCNT: usize = 13;
pub const S_SPURTTICK: usize = 14;

pub const CLS_RP_COMPLEN: usize = 0;
pub const CLS_RP_PHRCOUNT: usize = 1;
pub const CLS_RP_SKLEN: usize = 2;
pub const CLS_RP_COMP0: usize = 8;
pub const CLS_RP_COMP_CAP: usize = 2400;
pub const CLS_RP_PHR0: usize = CLS_RP_COMP0 + CLS_RP_COMP_CAP;
pub const CLS_RP_PHR_CAP: usize = 800;
pub const CLS_RP_SK0: usize = CLS_RP_PHR0 + CLS_RP_PHR_CAP;
pub const CLS_RP_SK_CAP: usize = 640;
pub const CLS_RP_STRIDE: usize = CLS_RP_SK0 + CLS_RP_SK_CAP;

const HP_STRATEGY_COEFFICIENT: [f64; 6] = [0.0, 0.95, 0.89, 1.0, 0.995, 0.86];
const HP_CONSUMPTION_GROUND_MODIFIER: [[f64; 5]; 3] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.02, 1.02],
    [0.0, 1.0, 1.0, 1.01, 1.02],
];
const SPEED_STRATEGY_PHASE_COEFFICIENT: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.98, 0.962],
    [0.978, 0.991, 0.975],
    [0.938, 0.998, 0.994],
    [0.931, 1.0, 1.0],
    [1.063, 0.962, 0.95],
];
const SPEED_DISTANCE_PROFICIENCY_MODIFIER: [f64; 8] = [1.05, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1];
const ACCEL_STRATEGY_PHASE_COEFFICIENT: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 0.996],
    [0.985, 1.0, 0.996],
    [0.975, 1.0, 1.0],
    [0.945, 1.0, 0.997],
    [1.17, 0.94, 0.956],
];
const ACCEL_GROUND_TYPE_PROFICIENCY_MODIFIER: [f64; 8] = [1.05, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1];
const ACCEL_DISTANCE_PROFICIENCY_MODIFIER: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4];
const BASE_ACCEL: f64 = 0.0006;
const UPHILL_BASE_ACCEL: f64 = 0.0004;
const STRATEGY_PROFICIENCY_MODIFIER_APPROX: [f64; 8] = [1.1, 1.0, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1];
const POSITION_KEEP_BASE_MIN_THRESHOLD: [f64; 6] = [0.0, 0.0, 3.0, 6.5, 7.5, f64::NAN];
const POSITION_KEEP_BASE_MAX_THRESHOLD: [f64; 6] = [0.0, 0.0, 5.0, 7.0, 8.0, f64::NAN];

fn base_speed(distance: f64) -> f64 {
    20.0 - (distance - 2000.0) / 1000.0
}

fn base_target_speed(strategy: usize, distance_apt: usize, speed: f64, distance: f64, phase: usize) -> f64 {
    base_speed(distance) * SPEED_STRATEGY_PHASE_COEFFICIENT[strategy][phase]
        + (if phase == 2 { 1.0 } else { 0.0 }) * (500.0 * speed).sqrt()
            * SPEED_DISTANCE_PROFICIENCY_MODIFIER[distance_apt] * 0.002
}

fn last_spurt_speed(strategy: usize, distance_apt: usize, speed: f64, guts: f64, distance: f64) -> f64 {
    (base_target_speed(strategy, distance_apt, speed, distance, 2) + 0.01 * base_speed(distance)) * 1.05
        + (500.0 * speed).sqrt() * SPEED_DISTANCE_PROFICIENCY_MODIFIER[distance_apt] * 0.002
        + (450.0 * guts).powf(0.597) * 0.0001
}

fn calc_base_accel(base_accel_const: f64, strategy: usize, phase: usize, power: f64,
    surface_apt: usize, distance_apt: usize) -> f64 {
    base_accel_const
        * (500.0 * power).sqrt()
        * ACCEL_STRATEGY_PHASE_COEFFICIENT[strategy][phase]
        * ACCEL_GROUND_TYPE_PROFICIENCY_MODIFIER[surface_apt]
        * ACCEL_DISTANCE_PROFICIENCY_MODIFIER[distance_apt]
}

fn position_keep_course_factor(distance: f64) -> f64 {
    (0.0008 * (distance - 1000.0) + 1.0).max(1.0)
}

fn position_keep_min_threshold(strategy: usize, distance: f64) -> f64 {
    let factor = if strategy == 2 { 1.0 } else { position_keep_course_factor(distance) };
    POSITION_KEEP_BASE_MIN_THRESHOLD[strategy] * factor
}

fn position_keep_max_threshold(strategy: usize, distance: f64) -> f64 {
    POSITION_KEEP_BASE_MAX_THRESHOLD[strategy] * position_keep_course_factor(distance)
}

/// Mirror of race_solver_builder.rs race_seed_hi, pinned to the engine source.
fn ref_race_seed_hi(horse_seed: u32) -> u32 {
    let mut seed = horse_seed ^ 0x52414345 ^ 0x57495344;
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x85ebca6b);
    seed = (seed ^ (seed >> 13)).wrapping_mul(0xc2b2ae35);
    seed ^= seed >> 16;
    if seed == 0 { 1 } else { seed }
}

fn strategy_id(name: &str) -> Result<i32, String> {
    match name.to_ascii_uppercase().as_str() {
        "NIGE" => Ok(1),
        "SENKOU" => Ok(2),
        "SASI" | "SASHI" => Ok(3),
        "OIKOMI" => Ok(4),
        "OONIGE" => Ok(5),
        other => Err(format!("unknown strategy {other:?}")),
    }
}

fn aptitude_id(name: &str) -> Result<i32, String> {
    match name.to_ascii_uppercase().as_str() {
        "S" => Ok(0),
        "A" => Ok(1),
        "B" => Ok(2),
        "C" => Ok(3),
        "D" => Ok(4),
        "E" => Ok(5),
        "F" => Ok(6),
        "G" => Ok(7),
        other => Err(format!("unknown aptitude {other:?}")),
    }
}

fn ground_id(name: &str) -> Result<i32, String> {
    match name.to_ascii_uppercase().as_str() {
        "GOOD" => Ok(1),
        "YIELDING" => Ok(2),
        "SOFT" => Ok(3),
        "HEAVY" => Ok(4),
        other => Err(format!("unknown ground condition {other:?}")),
    }
}

/// The non-retail branch of starting_grid.rs lane(): the extra divider added
/// to the dense gate rank before the /18 division, in full f64.
fn starting_lane(gate: usize, track_id: i32, count: usize, grids: &Value) -> f64 {
    let (kind, half) = grids
        .get(&track_id.to_string())
        .map(|g| {
            (
                g.get_i64("initialLaneType").unwrap_or(1),
                g.get_bool("enableHalfGate").unwrap_or(false) && count <= 9,
            )
        })
        .unwrap_or((1, false));
    let extra = if (kind == 1 && gate >= 9) || (kind == 4 && !half && gate >= 8) {
        0.6
    } else if kind == 3 && gate >= 14 {
        1.86
    } else {
        0.0
    };
    (gate as f64 + extra) / 18.0
}

/// One race's kernel inputs. A domain the classic kernel does not answer
/// bit-exactly comes back as `host_fallback` and skips the launch.
pub(crate) struct Prepared {
    ri: Vec<u32>,
    rd: Vec<f64>,
    n: usize,
    host_fallback: Option<String>,
}

/// Course fields the classic lane reads, extracted like the engine's
/// load_course_table (src/lib.rs:112-157): slopes sorted ascending by start.
struct ClassicCourse {
    distance: f64,
    surface: i32,
    track: i32,
    max_lane: f64,
    first_move: f64,
    empty_in_move: bool,
    last_straight: Option<(f64, f64)>,
    last_corner_start: Option<f64>,
    duel_on: bool,
    slopes: Vec<(f64, f64, f64)>,
    css: Vec<i32>,
}

fn classic_course(course: &Value) -> Result<ClassicCourse, String> {
    let mut slopes: Vec<(f64, f64, f64)> = course
        .get("slopes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|s| Ok((s.get_f64("start")?, s.get_f64("length")?, s.get_f64("slope")?)))
        .collect::<Result<_, String>>()?;
    if !course_data::is_sorted_by_start_f(slopes.iter().map(|s| s.0)) {
        slopes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let mut corners: Vec<f64> = Vec::new();
    if let Some(list) = course.get("corners").and_then(Value::as_array) {
        for c in list {
            corners.push(c.get_f64("start")?);
        }
    }
    let mut straights: Vec<(f64, f64)> = Vec::new();
    if let Some(list) = course.get("straights").and_then(Value::as_array) {
        for s in list {
            straights.push((s.get_f64("start")?, s.get_f64("end")?));
        }
    }
    let css = course
        .get("courseSetStatus")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_i64)
        .map(|x| x as i32)
        .collect();
    let first_move = course
        .get("firstMoveLanePoint")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| corners.first().map_or(0.0, |c| c.max(0.0)));
    Ok(ClassicCourse {
        distance: course.get_f64("distance")?,
        surface: course.get_i64("surface")? as i32,
        track: course.get_i64("raceTrackId")? as i32,
        max_lane: course.get("laneMax").and_then(Value::as_f64).unwrap_or(13000.0) / 10000.0,
        first_move,
        empty_in_move: first_move > 0.
            && course.get("emptyInMove").and_then(Value::as_bool).unwrap_or(true),
        last_straight: straights.last().copied(),
        last_corner_start: corners.last().copied(),
        duel_on: !straights.is_empty() && !corners.is_empty(),
        slopes,
        css,
    })
}

fn flagged(n: usize, reason: String) -> Prepared {
    Prepared { ri: vec![0u32; CLS_RI_STRIDE], rd: vec![0f64; CLS_RD_STRIDE], n, host_fallback: Some(reason) }
}

/// A flagged entry for a request that did not parse, so the facade's batch
/// keeps its shape and the race comes back as a fallback.
pub(crate) fn stub(reason: String) -> Prepared {
    flagged(0, reason)
}

fn prepare(request: &str, seed: i64, grids: &Value, courses: &Value) -> Result<Prepared, String> {
    let root = json::parse(request)?;
    let course_id = root.get_i64("courseId")? as i32;
    let horses = root.get_array("horses")?;
    let n = horses.len();
    let course = classic_course(
        courses.get(&course_id.to_string()).ok_or("course_data.json has no entry for the course")?,
    )?;

    // Domain gates. Each maps to a host fallback, never to an approximation.
    // Any of the four retail flags builds the retail config (lib.rs:1514-1523),
    // and the retail config switches the solver onto the retail physics.
    if ["retailRuntime", "experimentalRetailRuntime", "experimentalRetailInitialization",
        "experimentalRetailSections", "experimentalRetailGeometry", "experimentalRetailStartup"]
        .iter()
        .any(|key| root.get(*key).and_then(Value::as_bool).unwrap_or(false))
    {
        return Ok(flagged(n, "retail runtime belongs to the retail lane".into()));
    }
    if root.get("experimentalGrid").is_some() {
        return Ok(flagged(n, "experimentalGrid belongs to the retail lane".into()));
    }
    if root.get("singleMode").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(flagged(n, "singleMode's +400 stat bonus is outside the GPU lane".into()));
    }
    if root.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(flagged(n, "later stamina mechanics are not in the classic kernel".into()));
    }
    if root.get("traceConservePower").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(flagged(n, "conserve-power tracing is not in the classic kernel".into()));
    }
    if horses.iter().any(|h| h.get("skills").and_then(Value::as_array).is_some_and(|s| !s.is_empty())) {
        return Ok(flagged(n, "skills belong to the retail kernel lane".into()));
    }
    if n == 0 || n > CLS_MAXR {
        return Ok(flagged(n, "roster size outside the kernel's range".into()));
    }
    if course.slopes.len() > CLS_MAX_SLOPES {
        return Ok(flagged(n, "more slopes than the classic kernel carries".into()));
    }
    let ground = ground_id(root.get_str("groundCondition")?)?;
    let raws: Vec<[f64; 5]> = horses
        .iter()
        .map(|h| {
            Ok([
                h.get_f64("speed")?,
                h.get_f64("stamina")?,
                h.get_f64("power")?,
                h.get_f64("guts")?,
                h.get_f64("wisdom")?,
            ])
        })
        .collect::<Result<_, String>>()?;
    if raws.iter().any(|r| !r.iter().all(|v| v.is_finite())) {
        return Ok(flagged(n, "raw stats beyond the host stats authority".into()));
    }

    let mut ri = vec![0u32; CLS_RI_STRIDE];
    let mut rd = vec![0f64; CLS_RD_STRIDE];
    ri[CLS_RI_SEED] = seed as u32;
    ri[CLS_RI_N] = n as u32;
    ri[CLS_RI_NSLOPES] = course.slopes.len() as u32;
    ri[CLS_RI_TRACK] = course.track as u32;
    ri[CLS_RI_SURFACE] = course.surface as u32;
    ri[CLS_RI_GROUND] = ground as u32;
    ri[CLS_RI_FLAGS] = course.empty_in_move as u32;
    ri[CLS_RI_DUELON] = course.duel_on as u32;
    ri[CLS_RI_NMAX] = n as u32;
    rd[D_DISTANCE] = course.distance;
    rd[D_MAXLANE] = course.max_lane;
    rd[D_FIRSTMOVE] = course.first_move;
    rd[D_SECTION] = course.distance / 24.0;
    rd[D_LASTSTR_START] = course.last_straight.map_or(-1.0, |s| s.0);
    rd[D_LASTSTR_END] = course.last_straight.map_or(-1.0, |s| s.1);
    rd[D_LASTCORNER_START] = course.last_corner_start.unwrap_or(f64::INFINITY);
    rd[D_BASESPEED] = base_speed(course.distance);
    rd[D_GROUNDMOD] = HP_CONSUMPTION_GROUND_MODIFIER[course.surface as usize][ground as usize];
    for (k, &(start, length, slope)) in course.slopes.iter().enumerate() {
        rd[CLS_RD_SLOPES + 3 * k] = start;
        rd[CLS_RD_SLOPES + 3 * k + 1] = start + length;
        rd[CLS_RD_SLOPES + 3 * k + 2] = slope;
    }

    // Dense gate rank drives the lane, exactly like src/lib.rs:603-609.
    let mut gate_order: Vec<usize> = (0..n).collect();
    gate_order.sort_by_key(|&i| horses[i].get_i64("gateIndex").unwrap_or(0));
    let strategies: Vec<usize> = horses
        .iter()
        .map(|h| strategy_id(h.get_str("strategy").unwrap_or("")).unwrap_or(0) as usize)
        .collect();
    let style_count = |target: usize| -> usize { strategies.iter().filter(|&&s| s == target).count() };
    let nige_count = style_count(1);

    for (r, horse) in horses.iter().enumerate() {
        let strategy = strategy_id(horse.get_str("strategy")?)? as usize;
        let distapt = aptitude_id(horse.get_str("distanceAptitude")?)? as usize;
        let surfapt = aptitude_id(horse.get_str("surfaceAptitude")?)? as usize;
        let stapt = aptitude_id(horse.get_str("strategyAptitude")?)? as usize;
        let rank = horse.get("teamRank").and_then(Value::as_i64).unwrap_or(0);
        let stats_case = StatsCase {
            raw: raws[r],
            mood: horse.get_i64("mood")? as i32,
            rank: rank as i32,
            surface: course.surface,
            strategy_apt: stapt as i32,
            ground,
            css: course.css.clone(),
        };
        let stats = initialization::stats_reference(&stats_case);
        if stats.fallback != 0 {
            return Ok(flagged(n, "the initialization reference flagged".into()));
        }
        // The injected asitame arms on the mood and rank adjusted power, not on
        // the request's power field (race_solver_builder.rs:802 and :295).
        if f64::from_bits(stats.raw_power) > 1200.0 {
            return Ok(flagged(n, "asitame release (adjusted raw power above 1200) lands in wave 12b".into()));
        }
        let [speed, stamina, power, guts, wisdom] = stats.adjusted_f64();

        // The per-runner seeding chain, the engine draw order kept exact:
        // builder pair, hp seed, goroshi and pace, gate roll, random lot,
        // start delay, hill pairs, gate hill roll, kakari, sections.
        let seed_lo = (seed + r as i64 * 7919) as u32;
        let mut builder = Rule30CARng::new(seed_lo, ref_race_seed_hi(seed_lo));
        let d1 = builder.int32();
        let _d2 = builder.int32();
        let mut solver = Rule30CARng::new(d1, 0);
        let hp_seed = solver.int32();
        let _gorosi = solver.int32();
        let pace_lo = solver.int32();
        let _gate_roll = solver.uniform(12252240.0);
        let _random_lot = solver.uniform(100.0);
        let start_delay = 0.1 * solver.random();
        let mut hills: Vec<(u32, u32)> = course
            .slopes
            .iter()
            .map(|_| (solver.int32(), solver.int32()))
            .collect();
        let mut init_downhill = 0u32;
        let mut slope_idx = 0u32;
        let mut slope_per0 = 0.0;
        if let Some(&first) = course.slopes.first() {
            if first.0 == 0.0 {
                let (lo, hi) = hills[0];
                let mut hill0 = Rule30CARng::new(lo, hi);
                let roll = hill0.random();
                if first.2 < 0.0 && roll < wisdom * 0.0004 {
                    init_downhill = 1;
                }
                hills[0] = (hill0.lo, hill0.hi);
                slope_idx = 1;
                slope_per0 = first.2;
            }
        }
        let section_length = course.distance / 24.0;
        let mut kakari_start = (2.0 + solver.uniform(7.0) as f64) * section_length;
        if solver.random() > (0.65 / (0.1 * wisdom + 1.0).log10()).powi(2) {
            kakari_start = course.distance + 9999.0;
        }
        let r1 = solver.random();
        let r2 = solver.random();
        let r3 = solver.random();
        let kakari_duration = 3.0
            * [0.0, r1, r2, r3, 1.0]
                .iter()
                .position(|&x| x > 0.45)
                .unwrap_or(0) as f64;
        let max_v = wisdom / 5500.0 * (wisdom * 0.1).log10();
        let mut sections = [0.0; 25];
        for slot in &mut sections[..24] {
            *slot = base_speed(course.distance) * (max_v - 0.65 + solver.random() * 0.65) / 100.0;
        }
        let (solver_hi, solver_lo) = (solver.hi, solver.lo);

        let db = CLS_RD_RUN0 + CLS_RD_RUN_STRIDE * r;
        let rb = CLS_RI_RUN0 + CLS_RI_RUN_STRIDE * r;
        let gate = gate_order.iter().position(|&i| i == r).unwrap_or(0);
        rd[db + F_LANE0] = starting_lane(gate, course.track, n, grids);
        rd[db + F_SPEED_STAT] = speed;
        rd[db + F_STAMINA] = stamina;
        rd[db + F_POWER] = power;
        rd[db + F_GUTS] = guts;
        rd[db + F_WISDOM] = wisdom;
        rd[db + F_DELAY] = start_delay;
        rd[db + F_SECT0..db + F_SECT0 + 25].copy_from_slice(&sections);
        rd[db + F_KSTART] = kakari_start;
        rd[db + F_KDUR] = kakari_duration;
        for phase in 0..3 {
            rd[db + F_BTS0 + phase] = base_target_speed(strategy, distapt, speed, course.distance, phase);
        }
        rd[db + F_LSS] = last_spurt_speed(strategy, distapt, speed, guts, course.distance);
        rd[db + F_MINSPEED] = 0.85 * base_speed(course.distance) + (200.0 * guts).sqrt() * 0.001;
        for slot in 0..6 {
            rd[db + F_BACC0 + slot] = calc_base_accel(
                if slot > 2 { UPHILL_BASE_ACCEL } else { BASE_ACCEL },
                strategy,
                slot % 3,
                power,
                surfapt,
                distapt,
            );
        }
        rd[db + F_MAXHP] = 0.8 * HP_STRATEGY_COEFFICIENT[strategy] * stamina + course.distance;
        rd[db + F_GUTSMOD] = 1.0 + 200.0 / (600.0 * guts).sqrt();
        rd[db + F_PKMIN] = position_keep_min_threshold(strategy, course.distance);
        rd[db + F_PKMAX] = position_keep_max_threshold(strategy, course.distance);
        rd[db + F_PKEND] = (course.distance / 24.0).floor() * 10.0;
        rd[db + F_PKCONT] = (course.distance / 24.0).floor() * if strategy == 5 { 3.0 } else { 1.0 };
        rd[db + F_PCT15] = (15.0 * (wisdom.max(1.0) * 0.1).log10()).max(0.0);
        rd[db + F_PCT20] = (20.0 * (wisdom.max(1.0) * 0.1).log10()).max(0.0);
        rd[db + F_DTS] = (200.0 * guts.max(1.0)).powf(0.708) * 0.0001;
        rd[db + F_DA] = (160.0 * guts.max(1.0)).powf(0.59) * 0.0001;
        rd[db + F_SPOTM] = (500.0 * guts.max(1.0)).powf(0.6) * 0.0001;
        rd[db + F_SPOTD] = (700.0 * guts.max(1.0)).powf(0.5) * 0.012
            * STRATEGY_PROFICIENCY_MODIFIER_APPROX[stapt];
        rd[db + F_SUBPAR] = ((15.0 + 0.05 * wisdom) * 1000.0).round();
        rd[db + F_SLOPE0] = slope_per0;
        rd[db + F_STYLECNT] = style_count(strategy) as f64;
        rd[db + F_NIGECNT] = nige_count as f64;

        ri[rb + R_STRATEGY] = strategy as u32;
        ri[rb + R_DISTAPT] = distapt as u32;
        ri[rb + R_SURFAPT] = surfapt as u32;
        ri[rb + R_STAPT] = stapt as u32;
        ri[rb + R_SOLVER_HI] = solver_hi;
        ri[rb + R_SOLVER_LO] = solver_lo;
        ri[rb + R_HP_HI] = 0;
        ri[rb + R_HP_LO] = hp_seed;
        ri[rb + R_PACE_HI] = 0;
        ri[rb + R_PACE_LO] = pace_lo;
        for (k, &(lo, hi)) in hills.iter().enumerate() {
            ri[rb + R_HILL0 + 2 * k] = hi;
            ri[rb + R_HILL0 + 2 * k + 1] = lo;
        }
        ri[rb + R_SLOPE_IDX] = slope_idx;
        ri[rb + R_INIT_DOWNHILL] = init_downhill;
    }
    Ok(Prepared { ri, rd, n, host_fallback: None })
}

/// One race as the classic kernel ran it. Rows carry the six replay columns
/// plus eight debug columns (target speed, lane target, pos-keep coef and
/// mode, spot, duel, active slope, downhill mode). The reply exports mirror
/// the engine's per-runner result fields. `comp` holds competition starts as
/// (tick, kind, runner) triples, `phr` positioning phases as
/// (runner, phase) pairs, and `kakari` skillEvents records as
/// (runner, start bits, duration bits) quintuples, all in engine push order.
#[derive(Clone)]
pub struct ClassicGpu {
    pub frames: Vec<Vec<[f64; CLS_FRAME_COLS]>>,
    pub ticks: usize,
    pub fallback: Option<String>,
    pub kernel_flag: u32,
    pub finish_step_all: Vec<Option<f64>>,
    pub finish_raw_all: Vec<Option<f64>>,
    pub loop_end: f64,
    pub start_delay_all: Vec<f64>,
    pub strategies: Vec<u32>,
    pub lsttrans_all: Vec<f64>,
    pub spurt_time_all: Vec<Option<f64>>,
    pub spurt_dist_all: Vec<Option<f64>>,
    pub tempt_all: Vec<u32>,
    pub rscore_all: Vec<u32>,
    pub spurt_tick_all: Vec<u32>,
    pub comp: Vec<u32>,
    pub phr: Vec<u32>,
    pub kakari: Vec<u32>,
}

pub struct ClassicRun {
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
    pub races: Vec<ClassicGpu>,
}

/// The scheduler-facing view of one classic lane race.
pub fn classic_outcome(race: &ClassicGpu) -> RaceOutcome {
    match &race.fallback {
        Some(reason) => RaceOutcome {
            frames: Vec::new(),
            fallback: Some(FallbackReason::UnsupportedDomain(reason.clone())),
        },
        None => RaceOutcome {
            frames: race
                .frames
                .iter()
                .map(|tick| tick.iter().map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]]).collect())
                .collect(),
            fallback: None,
        },
    }
}

pub struct ClassicContext {
    grids: Value,
    courses: Value,
    spike: crate::gpu::Spike,
    program: ocl::Program,
}

impl ClassicContext {
    pub fn open() -> Result<ClassicContext, String> {
        let spike = crate::gpu::Spike::open()?;
        let program = spike.build_program(crate::gpu::CLASSIC_KERNEL_SOURCE)?;
        let grids = json::parse(&dep_cpu::engine_data("starting_grids.json")?)?;
        let courses = initialization::engine_json("data/course_data.json")?;
        Ok(ClassicContext { grids, courses, spike, program })
    }

    pub(crate) fn prepare(&self, request: &str, seed: i64) -> Result<Prepared, String> {
        prepare(request, seed, &self.grids, &self.courses)
    }

    pub(crate) fn run_prepared(&self, prepared: &mut [Prepared]) -> Result<ClassicRun, String> {
        let launched: Vec<usize> = (0..prepared.len())
            .filter(|&i| prepared[i].host_fallback.is_none())
            .collect();
        let m = launched.len();
        let nmax = launched.iter().map(|&i| prepared[i].n).max().unwrap_or(1);
        for &i in &launched {
            prepared[i].ri[CLS_RI_NMAX] = nmax as u32;
        }
        let queue = self.spike.queue();
        let mut status = vec![0u32; m.max(1) * CLS_ST_STRIDE];
        let mut rp_records = vec![0u32; m.max(1) * CLS_RP_STRIDE];
        let mut frames_all = vec![0f64; m.max(1) * CLS_MAX_TICKS * nmax * CLS_FRAME_COLS];
        let frames_bytes = CLS_MAX_TICKS * nmax * CLS_FRAME_COLS * std::mem::size_of::<f64>();
        const FRAME_BYTES_BUDGET: usize = 512 << 20;
        let chunk_len = (FRAME_BYTES_BUDGET / frames_bytes.max(1)).max(1);
        for chunk_start in (0..m).step_by(chunk_len) {
            let mc = chunk_len.min(m - chunk_start);
            let mut ri_all = vec![0u32; mc * CLS_RI_STRIDE];
            let mut rd_all = vec![0f64; mc * CLS_RD_STRIDE];
            for (slot, &i) in launched[chunk_start..chunk_start + mc].iter().enumerate() {
                ri_all[slot * CLS_RI_STRIDE..][..CLS_RI_STRIDE].copy_from_slice(&prepared[i].ri);
                rd_all[slot * CLS_RD_STRIDE..][..CLS_RD_STRIDE].copy_from_slice(&prepared[i].rd);
            }
            let b_ri = read_only(&queue, &ri_all)?;
            let b_rd = read_only(&queue, &rd_all)?;
            let b_su = read_write(&queue, &vec![0u32; mc * CLS_SU_STRIDE])?;
            let b_sd = read_write(&queue, &vec![0f64; mc * CLS_SD_STRIDE])?;
            let b_frames = read_write(&queue, &vec![0f64; mc * CLS_MAX_TICKS * nmax * CLS_FRAME_COLS])?;
            let b_status = read_write(&queue, &vec![0u32; mc * CLS_ST_STRIDE])?;
            let b_rp = read_write(&queue, &vec![0u32; mc * CLS_RP_STRIDE])?;
            let kernel = ocl::Kernel::builder()
                .program(&self.program)
                .name("race_classic")
                .queue(queue.clone())
                .global_work_size(mc)
                .arg(&b_ri)
                .arg(&b_rd)
                .arg(&b_su)
                .arg(&b_sd)
                .arg(&b_frames)
                .arg(&b_status)
                .arg(&b_rp)
                .build()
                .map_err(|e| e.to_string())?;
            unsafe { kernel.enq().map_err(|e| e.to_string())? };
            queue.finish().map_err(|e| e.to_string())?;
            b_status
                .cmd()
                .read(&mut status[chunk_start * CLS_ST_STRIDE..][..mc * CLS_ST_STRIDE])
                .enq()
                .map_err(|e| e.to_string())?;
            b_rp
                .cmd()
                .read(&mut rp_records[chunk_start * CLS_RP_STRIDE..][..mc * CLS_RP_STRIDE])
                .enq()
                .map_err(|e| e.to_string())?;
            b_frames
                .cmd()
                .read(&mut frames_all[chunk_start * CLS_MAX_TICKS * nmax * CLS_FRAME_COLS..][..mc * CLS_MAX_TICKS * nmax * CLS_FRAME_COLS])
                .enq()
                .map_err(|e| e.to_string())?;
            queue.finish().map_err(|e| e.to_string())?;
        }
        let mut races: Vec<ClassicGpu> = Vec::with_capacity(prepared.len());
        let mut slot = 0usize;
        for race in prepared.iter() {
            let mut out = ClassicGpu {
                frames: Vec::new(),
                ticks: 0,
                fallback: race.host_fallback.clone(),
                kernel_flag: 0,
                finish_step_all: Vec::new(),
                finish_raw_all: Vec::new(),
                loop_end: 0.,
                start_delay_all: Vec::new(),
                strategies: Vec::new(),
                lsttrans_all: Vec::new(),
                spurt_time_all: Vec::new(),
                spurt_dist_all: Vec::new(),
                tempt_all: Vec::new(),
                rscore_all: Vec::new(),
                spurt_tick_all: Vec::new(),
                comp: Vec::new(),
                phr: Vec::new(),
                kakari: Vec::new(),
            };
            if race.host_fallback.is_none() {
                let st = &status[slot * CLS_ST_STRIDE..][..CLS_ST_STRIDE];
                slot += 1;
                out.kernel_flag = st[S_FALLBACK];
                out.ticks = st[S_TICKS] as usize;
                out.loop_end = read_f64(st, S_END).unwrap_or(0.0);
                out.fallback = classic_kernel_fallback(out.kernel_flag);
                if out.fallback.is_none() {
                    let base = (slot - 1) * CLS_MAX_TICKS * nmax * CLS_FRAME_COLS;
                    // Classic captures one frame per loop pass, no pre-loop
                    // frame, so the replay holds exactly `ticks` frames.
                    out.frames = (0..out.ticks)
                        .map(|t| {
                            (0..race.n)
                                .map(|r| {
                                    let at = base + (t * nmax + r) * CLS_FRAME_COLS;
                                    let mut row = [0f64; CLS_FRAME_COLS];
                                    row.copy_from_slice(&frames_all[at..at + CLS_FRAME_COLS]);
                                    row
                                })
                                .collect()
                        })
                        .collect();
                    out.finish_raw_all = (0..race.n)
                        .map(|r| read_f64(st, S_FIN0 + r * CLS_ST_RUN_STRIDE))
                        .collect();
                    out.finish_step_all = (0..race.n)
                        .map(|r| read_f64(st, S_FIN0 + r * CLS_ST_RUN_STRIDE + 2))
                        .collect();
                    out.start_delay_all = (0..race.n)
                        .map(|r| race.rd[CLS_RD_RUN0 + CLS_RD_RUN_STRIDE * r + F_DELAY])
                        .collect();
                    out.strategies = (0..race.n)
                        .map(|r| race.ri[CLS_RI_RUN0 + CLS_RI_RUN_STRIDE * r + R_STRATEGY])
                        .collect();
                    out.lsttrans_all = (0..race.n)
                        .map(|r| read_f64(st, S_FIN0 + r * CLS_ST_RUN_STRIDE + S_LSTTRANS).unwrap_or(-1.0))
                        .collect();
                    out.spurt_time_all = (0..race.n)
                        .map(|r| read_f64(st, S_FIN0 + r * CLS_ST_RUN_STRIDE + S_SPURTT))
                        .collect();
                    out.spurt_dist_all = (0..race.n)
                        .map(|r| read_f64(st, S_FIN0 + r * CLS_ST_RUN_STRIDE + S_SPURTD))
                        .collect();
                    out.tempt_all = (0..race.n)
                        .map(|r| st[S_FIN0 + r * CLS_ST_RUN_STRIDE + S_TEMPT])
                        .collect();
                    out.rscore_all = (0..race.n)
                        .map(|r| st[S_FIN0 + r * CLS_ST_RUN_STRIDE + S_RSCNT])
                        .collect();
                    out.spurt_tick_all = (0..race.n)
                        .map(|r| st[S_FIN0 + r * CLS_ST_RUN_STRIDE + S_SPURTTICK])
                        .collect();
                    let rp = &rp_records[(slot - 1) * CLS_RP_STRIDE..][..CLS_RP_STRIDE];
                    let clen = (rp[CLS_RP_COMPLEN] as usize).min(CLS_RP_COMP_CAP);
                    let plen = (rp[CLS_RP_PHRCOUNT] as usize).min(CLS_RP_PHR_CAP);
                    let klen = (rp[CLS_RP_SKLEN] as usize).min(CLS_RP_SK_CAP);
                    out.comp = rp[CLS_RP_COMP0..CLS_RP_COMP0 + clen].to_vec();
                    out.phr = rp[CLS_RP_PHR0..CLS_RP_PHR0 + plen].to_vec();
                    out.kakari = rp[CLS_RP_SK0..CLS_RP_SK0 + klen].to_vec();
                }
            }
            races.push(out);
        }
        Ok(ClassicRun { device: self.spike.device.clone(), build_ms: self.spike.build_ms, races })
    }
}

const S_FIN0: usize = CLS_ST_RUN0;

fn read_f64(st: &[u32], word: usize) -> Option<f64> {
    let bits = u64::from(st[word]) | (u64::from(st[word + 1]) << 32);
    (bits != 0).then(|| f64::from_bits(bits))
}

fn classic_kernel_fallback(flag: u32) -> Option<String> {
    (flag != 0).then(|| match flag {
        1 => "uniform lottery cap hit".into(),
        2 => "tick count beyond the kernel's frame buffer".into(),
        3 => "field or slope count beyond the kernel's range".into(),
        4 => "reply record channel overflow".into(),
        other => format!("classic kernel fallback {other}"),
    })
}

fn read_only<T: ocl::OclPrm>(queue: &Queue, data: &[T]) -> Result<Buffer<T>, String> {
    let buffer = Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .len(data.len())
        .build()
        .map_err(|e| e.to_string())?;
    buffer.cmd().write(data).enq().map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn read_write<T: ocl::OclPrm>(queue: &Queue, data: &[T]) -> Result<Buffer<T>, String> {
    let buffer = Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(data.len())
        .build()
        .map_err(|e| e.to_string())?;
    buffer.cmd().write(data).enq().map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// Open a context, prepare every request and launch once.
pub fn run(requests: &[(String, i64)]) -> Result<ClassicRun, String> {
    let ctx = ClassicContext::open()?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        prepared.push(ctx.prepare(request, *seed)?);
    }
    ctx.run_prepared(&mut prepared)
}

/// The same run against a caller-supplied course table, the shape of
/// `data/course_data.json`. The CPU oracle embeds its table at compile time
/// (src/lib.rs:66 `include_str!`, src/api.rs:4-10, src/course_data.rs:102),
/// so scenes that vary the table itself run on the GPU lane only.
pub fn run_with_courses_text(requests: &[(String, i64)], courses_text: &str) -> Result<ClassicRun, String> {
    let mut ctx = ClassicContext::open()?;
    ctx.courses = json::parse(courses_text)?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        prepared.push(ctx.prepare(request, *seed)?);
    }
    ctx.run_prepared(&mut prepared)
}

/// The classic lane behind the RaceRunner seam.
pub struct ClassicRace;

impl RaceRunner for ClassicRace {
    fn run_race(&mut self, request: &str, seed: i64) -> RaceOutcome {
        match run(&[(request.to_string(), seed)]) {
            Ok(out) if out.races.len() == 1 => classic_outcome(&out.races[0]),
            Ok(_) => RaceOutcome {
                frames: Vec::new(),
                fallback: Some(FallbackReason::DeviceUnavailable("no race ran".into())),
            },
            Err(reason) => RaceOutcome {
                frames: Vec::new(),
                fallback: Some(FallbackReason::DeviceUnavailable(reason)),
            },
        }
    }
}
