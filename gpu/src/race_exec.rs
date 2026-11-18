//! Host for the race-loop kernel (handoff step 6 items 3-5 minus skills).
//! Parses a race request with its course and global parameters, packs one
//! fixed-stride input block per race, runs each whole no-skill race in one
//! kernel work item and reads the replay frames back.
//!
//! The stride and slot numbers mirror gpu/src/kernels/retail_core.cl, and
//! tests/layout.rs pins them.

use crate::dep_cpu;
use crate::initialization::{self, Case, Out, StatsCase};
use crate::json::{self, Value};
use crate::race::{FallbackReason, RaceOutcome, RaceRunner};
use crate::retail_geometry;
use ocl::{Buffer, Queue, flags};

pub const RACE_MAXR: usize = 24;
pub const RACE_MAX_TICKS: usize = 6144;
pub const RI_STRIDE: usize = 320;
pub const RIL_STRIDE: usize = 448;
pub const ST_STRIDE: usize = 112;
pub const SCR_U_WORDS: usize = SCR_RACE + 128;
pub const SRL_WORDS: usize = SRL_HP_OUT_L + 24 * 64;
pub const DT: f32 = 0.0666;

// Slot bases, mirrors of retail_core.cl.
pub const RI_SEED: usize = 0;
pub const RI_N: usize = 1;
pub const RI_COURSE: usize = 2;
pub const RI_GROUND: usize = 3;
pub const RI_SURFACE: usize = 4;
pub const RI_PAIRED: usize = 5;
pub const RI_STARTUP: usize = 6;
pub const RI_GEOM_ON: usize = 7;
pub const RI_HOST_FLAG: usize = 8;
pub const RI_NSLOPES: usize = 10;
pub const RI_LOT_MIN: usize = 13;
pub const RI_LOT_MAX: usize = 14;
pub const RI_EMPTY_MOVE: usize = 15;
pub const RI_RUN0: usize = 16;
pub const RI_RUN_STRIDE: usize = 12;
pub const RR_STYLE: usize = 0;
pub const RR_STRATEGY: usize = 1;
pub const RR_DISTAPT: usize = 2;
pub const RR_MOOD: usize = 3;
pub const RR_RANK: usize = 4;
pub const RR_CSS0: usize = 5;
pub const RR_CSSN: usize = 9;
pub const RR_STAPT: usize = 10;
pub const RR_SUAPT: usize = 11;
pub const PT_POWF: usize = 84;
pub const PT_SPURT: usize = 51;
pub const PI_DISTANCE: usize = 106;
pub const PC_LAST_STRAIGHT_START: usize = 184;
pub const PC_LAST_CORNER_START: usize = 186;
pub const PC_FIRST_MOVE_POINT: usize = 187;
pub const PC_MAX_LANE: usize = 188;
pub const PC_SLOPES: usize = 192;
pub const PI_RUNNER0: usize = 240;
// Finish bits per runner live at ST_FINISH0 = 8 (step, then raw, lo/hi u32).
pub const ST_FALLBACK: usize = 0;
pub const ST_ARG: usize = 1;
pub const ST_TICKS: usize = 2;
pub const ST_FINISH0: usize = 8;
// Scratch bases, mirrors of retail_core.cl.
const IIU: usize = 96;
const SF_STRIDE: usize = 44;
const LIST_STRIDE: usize = 48;
const CAND_MAX: usize = 101;
const SCR_STATS_IN: usize = 0;
const SCR_STATS_OUT: usize = SCR_STATS_IN + 24 * IIU;
const SCR_INIT_IN: usize = SCR_STATS_OUT + 24 * 160;
const SCR_INIT_OUT: usize = SCR_INIT_IN + IIU;
const SCR_HP_IN: usize = SCR_INIT_OUT + 992;
const SCR_HP_OUT: usize = SCR_HP_IN + 24 * IIU;
const SCR_STATE: usize = SCR_HP_OUT + 24 * 180;
const SCR_SF: usize = SCR_STATE + 48 * 24;
const SCR_LISTS: usize = SCR_SF + SF_STRIDE * 24;
const SCR_CAND: usize = SCR_LISTS + LIST_STRIDE * 24;
const SCR_CACHED: usize = (SCR_CAND + 3 * CAND_MAX + 63) & !63;
const SCR_RACE: usize = SCR_CACHED + 4 * LIST_STRIDE * 24;
const SRL_HP: usize = 40 * 24;
const RD_WORDS: usize = 8 * 24 + SRL_HP + 32;
const SRL_STATS_IN_L: usize = RD_WORDS;
const SRL_INIT_IN_L: usize = SRL_STATS_IN_L + 24 * 48;
const SRL_INIT_OUT_L: usize = SRL_INIT_IN_L + 48;
const SRL_HP_IN_L: usize = SRL_INIT_OUT_L + 64;
const SRL_HP_OUT_L: usize = SRL_HP_IN_L + 24 * 48;

// Fallback codes the kernel reports, mirrors of race.cl FB_*.
pub const FB_N: u32 = 1;
pub const FB_PAIRED: u32 = 2;
pub const FB_STARTUP: u32 = 3;
pub const FB_HOST: u32 = 4;
pub const FB_INIT: u32 = 5;
pub const FB_GEOMETRY: u32 = 6;

/// One race as the kernel ran it. `frames` holds the ticks plus the pre-loop
/// frame 0, each row the six replay values per runner. `finish_*` stays None
/// exactly when the runner never crossed the line or the race is flagged.
pub struct RaceGpu {
    pub frames: Vec<Vec<[f64; 6]>>,
    pub ticks: usize,
    pub fallback: Option<String>,
    pub kernel_flag: u32,
    pub kernel_arg: u32,
    pub finish_step: Option<f64>,
    pub finish_raw: Option<f64>,
    /// The loop's own end time, the engine's finish value for runners that
    /// never crossed the line.
    pub loop_end: f64,
}

pub struct RaceRun {
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
    pub races: Vec<RaceGpu>,
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

/// The engine's retail_starting_lane from data/starting_grids.json.
fn retail_starting_lane(gate: usize, track_id: i32, count: usize, grids: &Value) -> f64 {
    let (kind, half) = match grids.get(&track_id.to_string()) {
        Some(g) => (
            g.get_i64("initialLaneType").unwrap_or(1),
            g.get_bool("enableHalfGate").unwrap_or(false) && count <= 9,
        ),
        None => (1, false),
    };
    let extra = if (kind == 1 && gate >= 9) || (kind == 4 && !half && gate >= 8) {
        0.6
    } else if kind == 3 && gate >= 14 {
        1.86
    } else {
        0.0
    };
    let lane = gate as f32 * (1f32 / 18f32);
    (lane
        + if extra == 0.6 {
            0.033333335f32
        } else if extra == 1.86 {
            0.10333333f32
        } else {
            0.0
        }) as f64
}

fn group<'a>(value: &'a Value, name: &str) -> Result<&'a Value, String> {
    value.get(name).ok_or_else(|| format!("missing group {name}"))
}

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value.get_f64(key)
}

fn phase_rows(value: &Value, regular: &str, runaway: &str) -> Result<[f64; 20], String> {
    let mut rows = group(value, regular)?
        .as_array()
        .ok_or("phase array missing")?
        .iter()
        .chain(group(value, runaway)?.as_array().ok_or("ex phase array missing")?)
        .map(|row| [
            number(row, "Start").unwrap_or(0.),
            number(row, "Middle").unwrap_or(0.),
            number(row, "End").unwrap_or(0.),
            number(row, "Last").unwrap_or(0.),
        ]);
    let mut out = [0.; 20];
    for (slot, row) in out.chunks_mut(4).zip(&mut rows) {
        slot.copy_from_slice(&row[..]);
    }
    Ok(out)
}

/// The scalar parameters every race shares, raw JSON f64s. The kernel
/// narrows each with one f32 convert, matching `Parameters::from_value`.
fn pack_shared_parameters(globals: &Value, pd: &mut [f64]) -> Result<(), String> {
    let p = group(globals, "parameters")?;

    let base = group(p, "BasetTargetSpeed")?;
    let speed = group(p, "Speed")?;
    let block = group(p, "Block")?;
    let slope = group(p, "Slope")?;
    let hp = group(p, "Hp")?;
    let temptation = group(p, "Temptation")?;
    let keep = group(p, "PositionKeep")?;

    copy_keys(pd, 0, p, &["raceBaseSpeedBaseSpeed", "raceBaseSpeedBaseDistance", "raceBaseSpeedDistanceRate"])?;
    pd[3..23].copy_from_slice(&phase_rows(base, "PhaseBaseTargetSpeedPerArray", "PhaseBaseTargetSpeedPerExArray")?);
    pd[23..43].copy_from_slice(&phase_rows(speed, "PhaseAccelCoefArray", "PhaseAccelCoefExArray")?);
    copy_keys(pd, 43, base, &["PhaseEndBaseTargetSpeedCoef"])?;
    copy_keys(pd, 44, p, &["addSpeedParamCoef"])?;
    copy_keys(pd, 45, speed, &["MinSpeedRate", "MinSpeedGutsCoefSqrt", "MinSpeedGutsCoef"])?;
    copy_keys(pd, 48, block, &["FrontBlockDistanceGap", "FrontBlockMaxSpeedRateMin", "FrontBlockMaxSpeedRateMax"])?;
    copy_keys(pd, 51, p, &[
        "lastSpurtBaseTargetSpeedAddCoef",
        "LastSpurtTargetSpeedCoefSqrt",
        "LastSpurtTargetSpeedGutsCoef1",
        "LastSpurtTargetSpeedGutsPow",
        "LastSpurtTargetSpeedGutsCoef2",
        "baseTargetSpeedCoef",
    ])?;
    copy_keys(pd, 57, slope, &["UpSlopeAddSpeedVal1", "DownSlopeAddSpeedVal1", "DownSlopeAddSpeedVal2"])?;
    copy_keys(pd, 60, speed, &["TargetSpeedMin"])?;
    copy_keys(pd, 61, p, &["AccelPowCoefSqrt", "accelPowCoef", "accelPowCoefUpSlope", "StartAccelAdd"])?;
    copy_keys(pd, 65, hp, &["HpInitialVal1"])?;
    copy_keys(pd, 66, hp, &["HpMaxCoefNige", "HpMaxCoefSenko", "HpMaxCoefSashi", "HpMaxCoefOikomi", "HpMaxCoefOoNige"])?;
    copy_keys(pd, 71, hp, &[
        "HpDecRateBaseNormal",
        "HpDecRateBasePositionKeepPaseDown",
        "HpDecRateBaseTemptation",
        "HpDecRateBaseCompeteTopNige",
        "HpDecRateBaseCompeteTopOonige",
        "HpDecRateBaseTemptationAndCompeteTopNige",
        "HpDecRateBaseTemptationAndCompeteTopOonige",
        "HpDecRateMultiplyDownSlopeAccelMode",
    ])?;
    copy_keys(pd, 79, hp, &["SpeedGapParam1", "SpeedGapParam1Pow", "HpDecBase", "HpGutsCoefSqrt", "HpGutsCoef"])?;

    copy_keys(pd, 96, p, &[
        "forceInMoveTargetSpeedAddRunningOut",
        "forceInMoveTargetSpeedAddProceding",
        "forceInMoveTargetSpeedAddInsertion",
        "forceInMoveTargetSpeedAddChasing",
    ])?;
    copy_keys(pd, 100, p, &["forceInMoveTargetSpeedAddRandomRange"])?;
    copy_keys(pd, 101, p, &["startDelayMax"])?;
    copy_keys(pd, 102, base, &["BaseTargetSpeedRandomMinusVal1", "BaseTargetSpeedRandomMinusVal2", "BaseTargetSpeedRandomPlusVal1"])?;
    copy_keys(pd, 105, temptation, &["TemptationStartPerVal1"])?;

    let keep_number = |suffix: &str| -> Result<f64, String> { number(keep, &format!("PositionKeep{suffix}")) };
    pd[112] = keep_number("StartSection")?;
    pd[113] = keep_number("EndSection")?;
    pd[114] = keep_number("CoolDownSec")?;
    pd[115] = keep_number("CheckIntervalSec")?;
    pd[116] = keep_number("ContinueSection")?;
    pd[117] = keep
        .get("PositionKeepContinueSectionExArray")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .ok_or("no Runaway pace duration")?
        .as_f64()
        .ok_or("Runaway pace duration is not a number")?;
    pd[118] = keep_number("PaseBaseDistance")?;
    pd[119] = keep_number("PaseBaseDistanceCoef")?;
    for (i, name) in ["Senko", "Sashi", "Oikomi"].iter().enumerate() {
        pd[120 + 2 * i] = keep_number(&format!("PaseUpDownDistanceDiffMin{name}"))?;
        pd[121 + 2 * i] = keep_number(&format!("PaseUpDownDistanceDiffMax{name}"))?;
    }
    pd[126] = keep_number("PaseUpDownDistanceDiffMaxCoef")?;
    pd[127] = keep_number("SpeedUpEndDistanceDiff")?;
    pd[128] = keep_number("SpeedUpEndDistanceDiffOne")?;
    pd[129] = keep_number("SpeedUpEndDistanceDiffOonige")?;
    pd[130] = keep_number("OvertakeEndDistanceDiff")?;
    pd[131] = keep_number("OvertakeEndDistanceDiffOonige")?;
    pd[132] = keep_number("SpeedUpStartPerVal1")?;
    pd[133] = keep_number("SpeedUpStartPerVal1Oonige")?;
    pd[134] = keep_number("OvertakeStartPerVal1")?;
    pd[135] = keep_number("OvertakeStartPerVal1Oonige")?;
    pd[136] = keep_number("PaseUpStartPerVal1")?;
    pd[137] = pd[136];
    pd[138] = keep_number("SpeedUpBaseTargetSpeedMultiply")?;
    pd[139] = keep_number("SpeedUpBaseTargetSpeedMultiplyOonige")?;
    pd[140] = keep_number("OvertakeBaseTargetSpeedMultiply")?;
    pd[141] = keep_number("OvertakeBaseTargetSpeedMultiplyOonige")?;
    pd[142] = keep_number("PaseUpBaseTargetSpeedMultiply")?;
    pd[143] = pd[142];
    pd[144] = keep_number("PaseDownBaseTargetSpeedMultiply")?;
    pd[145] = keep_number("PaseDownBaseTargetSpeedMultiplyPhaseMiddle")?;
    pd[146] = keep_number("PaseUpExBaseTargetSpeedMultiply")?;
    pd[147] = pd[146];

    copy_keys(pd, 160, temptation, &[
        "TemptationEndCheckTime",
        "TemptationForceEndTime",
        "TemptationEndPer",
        "SenkoToNigePer",
        "SashiToSenkoPer",
        "SashiToNigePer",
        "OikomiToSashiPer",
        "OikomiToSenkoPer",
        "OikomiToNigePer",
    ])?;
    copy_keys(pd, 176, slope, &["DownSlopeAddSpeedEndPer", "DownSlopeAddSpeedStartWizRate"])?;

    // The Block gaps and the root-level Target/Move parameters.
    copy_keys(pd, 384, block, &["FrontBlockLaneGap", "SideBlockDistanceGap", "SideBlockLaneGap"])?;
    copy_keys(pd, 387, p, &[
        "forceInMoveLaneThreshold",
        "overTakeCoolDownTime",
        "lastMoveOutLaneNormalizeVal",
        "lastMoveOutLaneCoef",
        "lastMoveOutLaneAddRandom",
        "laneMoveSpeedPowCoef",
        "laneMoveSpeedAdd",
        "laneMoveSpeedBase",
        "laneMoveSpeedOutCoef",
        "laneMoveAccelMoveSpeedBaseCoef",
    ])?;
    Ok(())
}

fn copy_keys(pd: &mut [f64], slot: usize, v: &Value, keys: &[&str]) -> Result<(), String> {
    for (k, key) in keys.iter().enumerate() {
        pd[slot + k] = number(v, key)?;
    }
    Ok(())
}
/// One course's race scalars and slope events into one pd row. Slopes sort
/// by start exactly like the engine's course table builder.
fn pack_course_scalars(course: &Value, pd: &mut [f64]) -> Result<usize, String> {    pd[PI_DISTANCE] = course.get_f64("distance")?;
    let empty: Vec<Value> = Vec::new();
    let corners = course.get("corners").and_then(Value::as_array).unwrap_or(&empty);
    let straights = course.get("straights").and_then(Value::as_array).unwrap_or(&empty);
    if let Some(s) = straights.last() {
        pd[PC_LAST_STRAIGHT_START] = s.get_f64("start")?;
        pd[PC_LAST_STRAIGHT_START + 1] = s.get_f64("end")?;
    }
    pd[PC_LAST_CORNER_START] =
        corners.last().map_or(f64::INFINITY, |c| c.get_f64("start").unwrap_or(f64::INFINITY));
    pd[PC_MAX_LANE] = course.get("laneMax").and_then(Value::as_f64).unwrap_or(13000.0) / 10000.0;
    let mut slopes: Vec<&Value> = course.get("slopes").and_then(Value::as_array).unwrap_or(&empty).iter().collect();
    slopes.sort_by(|a, b| {
        a.get_f64("start").unwrap_or(0.).partial_cmp(&b.get_f64("start").unwrap_or(0.)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let count = slopes.len().min((RIL_STRIDE - PC_SLOPES) / 3);
    for (i, slope) in slopes.iter().enumerate().take(count) {
        pd[PC_SLOPES + 3 * i] = slope.get_f64("start")?;
        pd[PC_SLOPES + 3 * i + 1] = slope.get("length").and_then(Value::as_f64).unwrap_or(0.);
        pd[PC_SLOPES + 3 * i + 2] = slope.get_f64("slope")?;
    }
    Ok(count)
}

/// Everything the launch needs about one race before the powf slot (it
/// waits for the adjusted-guts pre-pass) and before the geometry tables.
struct Prepared {
    ri: Vec<u32>,
    pd: Vec<f64>,
    n: usize,
    stats: Vec<StatsCase>,
    host_fallback: Option<String>,
    geometry: Option<retail_geometry::FlatCourse>,
    seed: i32,
}

/// Parses one request into kernel inputs. A domain the kernel cannot answer
/// bit-exactly comes back as `host_fallback`, and the race skips the launch.
fn prepare(request: &str, seed: i64, globals: &Value, grids: &Value, courses: &Value) -> Result<Prepared, String> {
    let root = json::parse(request)?;
    let course_id = root.get_i64("courseId")? as i32;
    let seed_i32 = i32::try_from(seed).map_err(|_| "seed does not fit an i32".to_string())?;
    let horses = root.get_array("horses")?;
    let course = courses.get(&course_id.to_string()).ok_or("course_data.json has no entry for the course")?;

    // Domain gates first; each maps to a kernel fallback code.
    let flagged = |reason: &str| Ok(Prepared {
        ri: vec![0u32; RI_STRIDE],
        pd: vec![0f64; RIL_STRIDE],
        n: horses.len(),
        stats: Vec::new(),
        host_fallback: Some(reason.into()),
        geometry: None,
        seed: seed_i32,
    });
    if horses.is_empty() || horses.len() > RACE_MAXR {
        return flagged("roster size outside the kernel's range");
    }
    if root.get("pairedRandomStreams").and_then(Value::as_bool) == Some(true) {
        return flagged("paired random streams");
    }
    if root.get("experimentalRetailStartup").and_then(Value::as_bool) == Some(true) {
        return flagged("startup mode");
    }
    if root.get("experimentalRetailRuntime").and_then(Value::as_bool) != Some(true) {
        return flagged("non-retail runtime");
    }

    let mut ri = vec![0u32; RI_STRIDE];
    let mut pd = vec![0f64; RIL_STRIDE];
    let mut stats_cases: Vec<StatsCase> = Vec::new();
    let mut geometry = None;
    pack_shared_parameters(globals, &mut pd)?;
    let nslopes = pack_course_scalars(course, &mut pd)?;

    let surface = course.get_i64("surface")? as i32;
    let ground = ground_id(root.get_str("groundCondition")?)?;
    let distance = course.get_f64("distance")?;
    let empty_vec: Vec<Value> = Vec::new();
    let css: Vec<i32> = course
        .get("courseSetStatus")
        .and_then(Value::as_array)
        .unwrap_or(&empty_vec)
        .iter()
        .filter_map(Value::as_i64)
        .map(|v| v as i32)
        .collect();
    let first_move = course
        .get("firstMoveLanePoint")
        .and_then(Value::as_f64)
        .or_else(|| {
            course
                .get("corners")
                .and_then(Value::as_array)
                .and_then(|c| c.first())
                .and_then(|c| c.get_f64("start").ok())
                .map(|s| s.max(0.0))
        })
        .unwrap_or(0.);
    let empty_move = first_move > 0.0 && course.get("emptyInMove").and_then(Value::as_bool).unwrap_or(true);

    ri[RI_SEED] = seed_i32 as u32;
    ri[RI_N] = horses.len() as u32;
    ri[RI_GROUND] = ground as u32;
    ri[RI_SURFACE] = surface as u32;
    ri[RI_NSLOPES] = nslopes as u32;
    let temptation = group(group(globals, "parameters")?, "Temptation")?;
    ri[RI_LOT_MIN] = temptation.get_i64("TemptationLotSectionMin")? as u32;
    ri[RI_LOT_MAX] = temptation.get_i64("TemptationLotSectionMax")? as u32;
    ri[RI_EMPTY_MOVE] = empty_move as u32;
    pd[PI_DISTANCE] = distance;
    pd[PC_FIRST_MOVE_POINT] = first_move;

    if root.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/course_geometry");
        let loaded = retail_geometry::CourseGeometry::load(&dir, course_id, distance as f32)
            .map_err(|e| format!("world geometry for {course_id}: {e}"))?;
        // With geometry on, the lane machine reads first_move_lane_point
        // from the geometry, not the course file.
        pd[PC_FIRST_MOVE_POINT] = f64::from(loaded.first_move_lane_point);
        geometry = Some(retail_geometry::flatten_course(&loaded));
    }

    // Dense gate rank drives the retail starting lane.
    let mut gate_order: Vec<usize> = (0..horses.len()).collect();
    gate_order.sort_by_key(|&i| horses[i].get_i64("gateIndex").unwrap_or(0));
    let track = course.get_i64("raceTrackId")? as i32;

    for (r, horse) in horses.iter().enumerate() {
        let strategy = strategy_id(horse.get_str("strategy")?)?;
        if horse.get("skills").and_then(Value::as_array).is_some_and(|s| !s.is_empty()) {
            return flagged("skills are not ported");
        }
        let rank = gate_order.iter().position(|&i| i == r).unwrap_or(0);
        let lane = retail_starting_lane(rank, track, horses.len(), grids);
        let b = RI_RUN0 + RI_RUN_STRIDE * r;
        ri[b + RR_STYLE] = strategy as u32;
        ri[b + RR_STRATEGY] = strategy as u32;
        ri[b + RR_DISTAPT] = aptitude_id(horse.get_str("distanceAptitude")?)? as u32;
        ri[b + RR_MOOD] = horse.get_i64("mood")? as u32;
        ri[b + RR_RANK] = horse.get("teamRank").and_then(Value::as_i64).unwrap_or(0) as u32;
        ri[b + RR_STAPT] = aptitude_id(horse.get_str("strategyAptitude")?)? as u32;
        ri[b + RR_SUAPT] = aptitude_id(horse.get_str("surfaceAptitude")?)? as u32;
        ri[b + RR_CSSN] = css.len().min(4) as u32;
        for (k, value) in css.iter().take(4).enumerate() {
            ri[b + RR_CSS0 + k] = *value as u32;
        }
        let stats = [
            horse.get_f64("speed")?,
            horse.get_f64("stamina")?,
            horse.get_f64("power")?,
            horse.get_f64("guts")?,
            horse.get_f64("wisdom")?,
        ];
        for (k, value) in stats.iter().enumerate() {
            pd[PI_RUNNER0 + 6 * r + k] = *value;
        }
        // The lane round-trips f32 in the engine's start table.
        pd[PI_RUNNER0 + 6 * r + 5] = f64::from(lane as f32);
        stats_cases.push(StatsCase {
            raw: stats,
            mood: horse.get_i64("mood")? as i32,
            rank: ri[b + RR_RANK] as i32,
            surface,
            strategy_apt: ri[b + RR_STAPT] as i32,
            ground,
            css: css.clone(),
        });
    }
    Ok(Prepared { ri, pd, n: horses.len(), stats: stats_cases, host_fallback: None, geometry, seed: seed_i32 })
}

/// Runs every request end to end on the gated device. Races whose domain the
/// kernel cannot answer stay flagged without a launch.
pub fn run(requests: &[(String, i64)]) -> Result<RaceRun, String> {
    let globals = initialization::engine_parameters_value()?;
    let grids = json::parse(&dep_cpu::engine_data("starting_grids.json")?)?;
    let courses = initialization::engine_json("data/course_data.json")?;

    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        prepared.push(prepare(request, *seed, &globals, &grids, &courses)?);
    }
    let launched: Vec<usize> = (0..prepared.len()).filter(|&i| prepared[i].host_fallback.is_none()).collect();
    if launched.windows(2).any(|w| prepared[w[0]].n != prepared[w[1]].n) {
        return Err("one launch needs one uniform runner count".into());
    }
    let n = launched.first().map_or(1, |&i| prepared[i].n);

    // The stats chain runs once per race as an initialization-kernel
    // pre-pass so the host can supply the exact f32 powf (the kernel
    // re-runs the chain itself, from the same inputs, for the rest).
    let mut cases: Vec<Case> = Vec::new();
    for &i in &launched {
        for stats in &prepared[i].stats {
            cases.push(Case::Stats(stats.clone()));
        }
    }
    let prepass = if cases.is_empty() { None } else { Some(initialization::run(&cases)?) };
    let mut offset = 0usize;
    for &i in &launched {
        let count = prepared[i].stats.len();
        // PT_POWF carries one value, so it is exact only for one runner.
        // Multi-runner launches reach a kernel fallback before the slot.
        let guts = match &prepass {
            Some(out) => match &out.outs[offset] {
                Out::Stats(s) => s.adjusted_f64()[3],
                _ => return Err("stats pre-pass decode".into()),
            },
            None => 0.,
        };
        offset += count;
        let bounded = (guts as f32).clamp(1., 2000.);
        let spurt_coef = prepared[i].pd[PT_SPURT + 2] as f32;
        let spurt_pow = prepared[i].pd[PT_SPURT + 3] as f32;
        prepared[i].pd[PT_POWF] = f64::from((bounded * spurt_coef).powf(spurt_pow).to_bits());
    }

    // One geometry table set shared by every launched race.
    let mut table_courses: Vec<i32> = Vec::new();
    let mut table_sets: Vec<retail_geometry::FlatCourse> = Vec::new();
    for &i in &launched {
        if let Some(course) = prepared[i].geometry.take() {
            let id = course.course_id;
            if !table_courses.contains(&id) {
                table_courses.push(id);
                table_sets.push(course);
            }
            let row = table_courses.iter().position(|&k| k == id).unwrap();
            prepared[i].ri[RI_COURSE] = row as u32;
            prepared[i].ri[RI_GEOM_ON] = 1u32;
        }
    }
    if table_sets.is_empty() {
        // OpenCL rejects zero-length buffers; a geometry-off launch never
        // reads these.
        let zeros = |k: usize| vec![0f64; k];
        table_sets.push(retail_geometry::FlatCourse {
            course_id: 0,
            distance: 0.,
            distance_to_world: 0.,
            initial_lane_max: 0.,
            rotation_fix: [0.; 4],
            overrun_present: false,
            overrun_distance: 0.,
            seg_x: zeros(2 * crate::geometry::NSEG),
            seg_xd: zeros(2 * crate::geometry::NSEG),
            seg_z: zeros(2 * crate::geometry::NSEG),
            seg_zd: zeros(2 * crate::geometry::NSEG),
            seg_q: vec![0f32; 8 * crate::geometry::NSEG],
            seg_qd: vec![0f32; 8 * crate::geometry::NSEG],
            lm_boundary: Vec::new(),
            lm_limit: Vec::new(),
            ce_start: Vec::new(),
            ce_end: Vec::new(),
            ce_id: Vec::new(),
            ce_flags: Vec::new(),
        });
    }
    let tables = crate::geometry::course_tables(&table_sets);

    let spike = crate::gpu::Spike::open()?;
    let program = spike.build_program(crate::gpu::RACE_KERNEL_SOURCE)?;
    let queue = spike.queue();

    let m = launched.len();
    let mut ri_all = vec![0u32; m.max(1) * RI_STRIDE];
    let mut pd_all = vec![0f64; m.max(1) * RIL_STRIDE];
    for (slot, &i) in launched.iter().enumerate() {
        ri_all[slot * RI_STRIDE..][..RI_STRIDE].copy_from_slice(&prepared[i].ri);
        pd_all[slot * RIL_STRIDE..][..RIL_STRIDE].copy_from_slice(&prepared[i].pd);
    }
    let b_ri = read_write(&queue, &ri_all)?;
    let b_pd = read_write(&queue, &pd_all)?;
    let b_scr_u = read_write(&queue, &vec![0u32; m.max(1) * SCR_U_WORDS])?;
    let b_scr_d = read_write(&queue, &vec![0f64; m.max(1) * SRL_WORDS])?;
    let b_frames = read_write(&queue, &vec![0f64; m.max(1) * RACE_MAX_TICKS * n * 6])?;
    let b_status = read_write(&queue, &vec![0u32; m.max(1) * ST_STRIDE])?;
    let b_cs = read_only(&queue, &tables.cs)?;
    let b_sg_x = read_only(&queue, &tables.sg_x)?;
    let b_sg_xd = read_only(&queue, &tables.sg_xd)?;
    let b_sg_z = read_only(&queue, &tables.sg_z)?;
    let b_sg_zd = read_only(&queue, &tables.sg_zd)?;
    let b_sg_q = read_only(&queue, &tables.sg_q)?;
    let b_sg_qd = read_only(&queue, &tables.sg_qd)?;
    let b_lm_b = read_only(&queue, &tables.lm_b)?;
    let b_lm_l = read_only(&queue, &tables.lm_l)?;
    let b_ce_b = read_only(&queue, &tables.ce_b)?;
    let b_ce_e = read_only(&queue, &tables.ce_e)?;
    let b_ce_i = read_only(&queue, &tables.ce_i)?;
    let b_ce_f = read_only(&queue, &tables.ce_f)?;

    if m > 0 {
        let kernel = ocl::Kernel::builder()
            .program(&program)
            .name("race")
            .queue(queue.clone())
            .global_work_size(m)
            .arg(&b_ri)
            .arg(&b_pd)
            .arg(&b_scr_u)
            .arg(&b_scr_d)
            .arg(&b_frames)
            .arg(&b_status)
            .arg(&b_cs)
            .arg(&b_sg_x)
            .arg(&b_sg_xd)
            .arg(&b_sg_z)
            .arg(&b_sg_zd)
            .arg(&b_sg_q)
            .arg(&b_sg_qd)
            .arg(&b_lm_b)
            .arg(&b_lm_l)
            .arg(&b_ce_b)
            .arg(&b_ce_e)
            .arg(&b_ce_i)
            .arg(&b_ce_f)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
    }
    queue.finish().map_err(|e| e.to_string())?;

    let mut status = vec![0u32; m.max(1) * ST_STRIDE];
    let mut frames_all = vec![0f64; m.max(1) * RACE_MAX_TICKS * n * 6];
    if m > 0 {
        b_status.cmd().read(&mut status).enq().map_err(|e| e.to_string())?;
        b_frames.cmd().read(&mut frames_all).enq().map_err(|e| e.to_string())?;
    }

    let mut races: Vec<RaceGpu> = Vec::with_capacity(requests.len());
    let mut slot = 0usize;
    for race in prepared.iter() {
        let mut out = RaceGpu {
            frames: Vec::new(),
            ticks: 0,
            fallback: race.host_fallback.clone(),
            kernel_flag: 0,
            kernel_arg: 0,
            finish_step: None,
            finish_raw: None,
            loop_end: 0.,
        };
        if race.host_fallback.is_none() {
            let st = &status[slot * ST_STRIDE..][..ST_STRIDE];
            slot += 1;
            out.kernel_flag = st[ST_FALLBACK];
            out.kernel_arg = st[ST_ARG];
            out.ticks = st[ST_TICKS] as usize;
            out.fallback = kernel_fallback(out.kernel_flag, out.kernel_arg);
            out.loop_end = loop_end(out.ticks);
            if out.fallback.is_none() {
                let base = (slot - 1) * RACE_MAX_TICKS * n * 6;
                out.frames = (0..=out.ticks)
                    .map(|t| {
                        (0..n)
                            .map(|r| {
                                let mut row = [0f64; 6];
                                let at = base + t * n * 6 + r * 6;
                                row.copy_from_slice(&frames_all[at..at + 6]);
                                row
                            })
                            .collect()
                    })
                    .collect();
                out.finish_step = read_f64(st, ST_FINISH0);
                out.finish_raw = read_f64(st, ST_FINISH0 + 2);
            }
        }
        races.push(out);
    }

    Ok(RaceRun { device: spike.device, build_ms: spike.build_ms, races })
}

/// The finish bits the kernel stored, or None for a runner that never
/// finished (the engine then uses the loop end time).
fn read_f64(st: &[u32], word: usize) -> Option<f64> {
    let bits = u64::from(st[word]) | (u64::from(st[word + 1]) << 32);
    (bits != 0).then(|| f64::from_bits(bits))
}

fn kernel_fallback(flag: u32, arg: u32) -> Option<String> {
    (flag != 0).then(|| match (flag, arg) {
        (FB_N, _) => "roster size the kernel rejects".into(),
        (FB_PAIRED, _) => "paired streams reached the kernel".into(),
        (FB_STARTUP, _) => "startup reached the kernel".into(),
        (FB_HOST, code) => format!("kernel host flag {code}"),
        (FB_INIT, _) => "the initialization chain flagged".into(),
        (FB_GEOMETRY, _) => "geometry rejected a query".into(),
        (other, a) => format!("kernel fallback {other} arg {a}"),
    })
}

/// The v77 request as a solo field: roster line for the course, horse 0 in
/// gate 0, the given seed. The gates drive one-runner races from it.
pub fn solo_request(request: &str, course: i32, seed: i64) -> Result<String, String> {
    let mut value = json::parse(request)?;
    let Value::Object(map) = &mut value else { return Err("request is not a JSON object".into()) };
    let mut horse = match map.get_mut("horses") {
        Some(Value::Array(horses)) if !horses.is_empty() => horses.swap_remove(0),
        _ => return Err("request has no horses array".into()),
    };
    if let Value::Object(horse_map) = &mut horse {
        horse_map.insert("gateIndex".into(), Value::Number(0.));
    }
    if let Some(Value::Array(horses)) = map.get_mut("horses") {
        horses.clear();
        horses.push(horse);
    }
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    Ok(out)
}

/// The engine's loop time after `ticks` iterations of the f32 sum.
pub fn loop_end(ticks: usize) -> f64 {
    let mut t = 0f64;
    for _ in 0..ticks {
        t = ((t as f32) + DT) as f64;
    }
    t
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

/// The whole-race GPU runner behind the handoff's RaceRunner seam. A race
/// the kernel flagged re-runs on the CPU through the scheduler's queue.
pub struct GpuRace;

impl RaceRunner for GpuRace {
    fn run_race(&mut self, request: &str, seed: i64) -> RaceOutcome {
        match run(&[(request.to_string(), seed)]) {
            Ok(out) if out.races.len() == 1 => {
                let race = &out.races[0];
                match &race.fallback {
                    Some(reason) => RaceOutcome {
                        frames: Vec::new(),
                        fallback: Some(FallbackReason::UnsupportedDomain(reason.clone())),
                    },
                    None => RaceOutcome {
                        frames: race
                            .frames
                            .iter()
                            .map(|frame| {
                                frame.iter().map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]]).collect()
                            })
                            .collect(),
                        fallback: None,
                    },
                }
            }
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











