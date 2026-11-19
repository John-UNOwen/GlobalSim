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

pub const RACE_MAXR: usize = 120;
pub const RACE_MAX_TICKS: usize = 6144;
pub const RACE_MAX_SLOPES: usize = 16;
pub const RI_STRIDE: usize = RI_DEBUG_TICK + 16;
pub const RIL_STRIDE: usize = PD_RUN0 + PD_RUN_STRIDE * RACE_MAXR;
pub const ST_STRIDE: usize = ST_LSS0 + RACE_MAXR;
// Skill scratch mirrors of retail_core.cl and skills.cl.
pub const SK_STRIDE: usize = crate::skills_compile::SK_STRIDE;
const SK_RMAX: usize = crate::skills_compile::SK_RMAX;
const SKR_STRIDE: usize = 8;
const SK_SEQN: usize = 40;
const SKD_STRIDE: usize = 148;
const SCR_SK: usize = SCR_BANKS + RACE_MAXR * BANK_WORDS;
const SK_TSEQ: usize = SCR_SK + SK_RMAX * SKR_STRIDE;
const SK_ASEQ: usize = SK_TSEQ + SK_RMAX * SK_SEQN;
const SK_CSEQ: usize = SK_ASEQ + SK_RMAX * SK_SEQN;
const SK_LSEQ: usize = SK_CSEQ + SK_RMAX * SK_SEQN;
const SK_VSEQ: usize = SK_LSEQ + SK_RMAX * SK_SEQN;
const SK_SUB: usize = SK_VSEQ + SK_RMAX * SK_SEQN;
const SK_STAGED: usize = SK_SUB + BANK_WORDS;
pub const SCR_U_WORDS: usize = SK_STAGED + 2 * SK_RMAX;
pub const SRL_SKD: usize = SRL_HP_OUT_L + RACE_MAXR * OOL;
pub const SRL_WORDS: usize = SRL_SKD + SK_RMAX * SKD_STRIDE;
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
pub const RI_NSTR: usize = 11;
pub const RI_NCOR: usize = 12;
pub const RI_LOT_MIN: usize = 13;
pub const RI_LOT_MAX: usize = 14;
pub const RI_EMPTY_MOVE: usize = 15;
pub const RI_DEBUG_TICK: usize = RI_RUN0 + RI_RUN_STRIDE * RACE_MAXR;
pub const RI_RUN0: usize = 16;
pub const RI_RUN_STRIDE: usize = 13;
pub const RR_STYLE: usize = 0;
pub const RR_STRATEGY: usize = 1;
pub const RR_DISTAPT: usize = 2;
pub const RR_MOOD: usize = 3;
pub const RR_RANK: usize = 4;
pub const RR_CSS0: usize = 5;
pub const RR_CSSN: usize = 9;
pub const RR_STAPT: usize = 10;
pub const RR_SUAPT: usize = 11;
pub const RR_TEAM: usize = 12;
pub const PT_SPURT: usize = 51;
pub const PI_DISTANCE: usize = 106;
pub const PD_CF_DIST: usize = PL_FRONT_LANE + 16;
pub const PD_CF_LANE: usize = PD_CF_DIST + 1;
pub const PD_CF_SPEED: usize = PD_CF_DIST + 2;
pub const PD_CF_ACCEL: usize = PD_CF_DIST + 5;
pub const PD_PM_RANGE: usize = PD_CF_DIST + 8;
pub const PD_PM_NOTFWD: usize = PD_CF_DIST + 9;
pub const PD_DTYPE: usize = PD_CF_DIST + 10;
pub const PD_FC_HALF0: usize = PD_CF_DIST + 11;
pub const PD_FC_HALF1: usize = PD_CF_DIST + 12;
pub const PD_RUN0: usize = PL_FRONT_LANE + 32;
pub const PD_RUN_STRIDE: usize = 8;
pub const PRD_POWF: usize = 0;
pub const PRD_DUEL_T: usize = 1;
pub const PRD_DUEL_A: usize = 2;
pub const PRD_SPOT_M: usize = 3;
pub const PRD_SPOT_D: usize = 4;
// Construction-time stat bumps from the skill table, 0 when the runner has
// none. The boosted field values replace the adjusted stats in the kernel's
// stats-out block, and the rushing bonus carries the f32 bits as an integer.
pub const PRD_BOOST_GUTS: usize = 5;
pub const PRD_BOOST_WIS: usize = 6;
pub const PRD_RUSH_BONUS: usize = 7;
pub const PC_LAST_STRAIGHT_START: usize = 184;
pub const PC_LAST_CORNER_START: usize = 186;
pub const PC_FIRST_MOVE_POINT: usize = 187;
pub const PC_MAX_LANE: usize = 188;
pub const PC_SLOPES: usize = 192;
pub const PI_RUNNER0: usize = 240;
pub const PI_RUN_STRIDE: usize = 7;
pub const PI_RUN_LANE: usize = 5;
pub const PI_RUN_POS: usize = 6;
// The TargetParameters lane block follows the per-runner pd block.
pub const PL_FRONT_LANE: usize = PI_RUNNER0 + PI_RUN_STRIDE * RACE_MAXR;
pub const PL_FORCE_IN_THRESHOLD: usize = PL_FRONT_LANE + 3;
// Finish bits per runner live at ST_FINISH0 = 8 (step, then raw, lo/hi u32).
pub const ST_FALLBACK: usize = 0;
pub const ST_ARG: usize = 1;
pub const ST_TICKS: usize = 2;
pub const ST_FINISH0: usize = 8;
// Compact-reply exports per runner, mirrors of retail_core.cl. The f32 ones
// carry raw bits, which the facade widens exactly like the engine's casts.
pub const ST_TEMPT0: usize = ST_FINISH0 + 4 * RACE_MAXR;
pub const ST_RSCORE0: usize = ST_TEMPT0 + RACE_MAXR;
pub const ST_HPEMPTY0: usize = ST_RSCORE0 + RACE_MAXR;
pub const ST_DELAY0: usize = ST_HPEMPTY0 + RACE_MAXR;
pub const ST_LSS0: usize = ST_DELAY0 + RACE_MAXR;
// skillEvents log, mirror of retail_core.cl: word 0 is the entry count,
// then (runner, id) pairs, id EV_KAKARI for a completed rush episode.
pub const EV_MAX: usize = 512;
pub const EV_STRIDE: usize = 1 + 2 * EV_MAX;
pub const EV_KAKARI: u32 = 0xFFFF_FFFF;
// Scratch bases, mirrors of retail_core.cl.
const IIU: usize = crate::initialization::IN_U_STRIDE;
const IIL: usize = crate::initialization::IN_L_STRIDE;
const OOU: usize = crate::initialization::OUT_U_STRIDE;
const OOL: usize = crate::initialization::OUT_L_STRIDE;
const SF_STRIDE: usize = 56;
const LIST_STRIDE: usize = 2 * RACE_MAXR;
const CACHE_STRIDE: usize = RACE_MAXR;
const CAND_MAX: usize = 101;
const SCR_STATS_IN: usize = 0;
const SCR_STATS_OUT: usize = SCR_STATS_IN + RACE_MAXR * IIU;
const SCR_INIT_IN: usize = SCR_STATS_OUT + RACE_MAXR * 160;
const SCR_INIT_OUT: usize = SCR_INIT_IN + IIU;
const SCR_HP_IN: usize = SCR_INIT_OUT + OOU;
const SCR_HP_OUT: usize = SCR_HP_IN + RACE_MAXR * IIU;
const SCR_STATE: usize = SCR_HP_OUT + RACE_MAXR * 180;
const SCR_SF: usize = SCR_STATE + 83 * RACE_MAXR;
const SCR_LISTS: usize = SCR_SF + SF_STRIDE * RACE_MAXR;
const SCR_CAND: usize = SCR_LISTS + LIST_STRIDE * RACE_MAXR;
const SCR_CACHED: usize = (SCR_CAND + 3 * CAND_MAX + 63) & !63;
const SCR_RACE: usize = SCR_CACHED + 4 * CACHE_STRIDE * RACE_MAXR;
// Race block tail: order and prev-rank arrays, the pacemaker fields, the
// aligned strategy counts, then the four scan arrays and the stepped flags.
const RS_PM: usize = 8 + 2 * RACE_MAXR;
const RS_COUNTS: usize = (RS_PM + 8 + 7) & !7;
const RS_ADV: usize = RS_COUNTS + 8 + 4 * RACE_MAXR;
const SCR_BANKS: usize = SCR_RACE + RS_ADV + RACE_MAXR; // paired mode: one RUN bank per runner
const BANK_WORDS: usize = 60;
const SRL_HP: usize = (2 * RACE_MAXR + 70) * RACE_MAXR;
const RD_WORDS: usize = 8 * RACE_MAXR + SRL_HP + 32;
const SRL_STATS_IN_L: usize = RD_WORDS;
const SRL_INIT_IN_L: usize = SRL_STATS_IN_L + RACE_MAXR * IIL;
const SRL_INIT_OUT_L: usize = SRL_INIT_IN_L + IIL;
const SRL_HP_IN_L: usize = SRL_INIT_OUT_L + OOL;
const SRL_HP_OUT_L: usize = SRL_HP_IN_L + RACE_MAXR * IIL;

// Fallback codes the kernel reports, mirrors of race.cl FB_*.
pub const FB_N: u32 = 1;
pub const FB_HOST: u32 = 4;
pub const FB_INIT: u32 = 5;
pub const FB_GEOMETRY: u32 = 6;
pub const FB_EVENT: u32 = 7;
pub const FB_SKILL_SEQ: u32 = 8;

/// One race as the kernel ran it. `frames` holds the ticks plus the pre-loop
/// frame 0, each row the six replay values per runner. `finish_*` stays None
/// exactly when the runner never crossed the line or the race is flagged.
pub struct RaceGpu {
    pub frames: Vec<Vec<[f64; 6]>>,
    pub ticks: usize,
    pub fallback: Option<String>,
    /// The engine's grid-mode starts were in force for this race, so the
    /// compact reply carries the experimentalGridVersion tag like the engine's.
    pub grid: bool,
    pub kernel_flag: u32,
    pub kernel_arg: u32,
    pub finish_step: Option<f64>,
    pub finish_raw: Option<f64>,
    /// Per runner in roster order, the same bits the solo fields carry for
    /// runner 0. Runners that never crossed read as None.
    pub finish_step_all: Vec<Option<f64>>,
    pub finish_raw_all: Vec<Option<f64>>,
    /// The loop's own end time, the engine's finish value for runners that
    /// never crossed the line.
    pub loop_end: f64,
    /// Compact-reply exports per runner in roster order: the two count words,
    /// the hp-empty flag and the two f32 quantities the engine widens to f64.
    pub tempt_all: Vec<u32>,
    pub rscore_all: Vec<u32>,
    pub hp_empty_all: Vec<bool>,
    pub delay_all: Vec<f64>,
    pub lss_all: Vec<f64>,
    /// The race's skillEvents (horseIndex, id) pairs in kernel order, id
    /// EV_KAKARI for a completed rush episode.
    pub events: Vec<(usize, u32)>,
    /// Test diagnostics only, empty unless the request set RI_DEBUG_TICK.
    /// The fourteen reserved tail frame slots read back as fourteen blocks
    /// of `runners` rows of six: pre-move pace (mode, coefficient,
    /// pacemaker, gap, rank, RS_STYLE), pre-move clock (target speed, exit
    /// gap, cooldown, interval, competition flags, spot timer), pre-move
    /// rush, pre-move lane room (lane max, lane cooldown, x-overlap, side
    /// inner, side inside, side outside), then the passing-search trace
    /// (nearest, best target, straight, clear flag, front and target count,
    /// retained-target flags, the second target's crowd-walk inputs), then
    /// the same seven blocks after the move.
    /// Present to bisect a divergence, never in a replay reply.
    pub debug_pace: Vec<[f64; 6]>,
    /// Test diagnostics only, None unless the caller asked for the skill
    /// scratch: the merged trigger windows and the record state as the scan
    /// gate sees them after initialization.
    pub debug_sk: Option<SkillScratch>,
}

/// One race's kernel skill scratch read back for a divergence hunt: the
/// per-record double block at `SKD_STRIDE` per record slot and the per-record
/// state words at `SKR_STRIDE` per record slot.
pub struct SkillScratch {
    pub skd: Vec<f64>,
    pub sks: Vec<u32>,
}

pub struct RaceRun {
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
    /// Where this call's milliseconds went inside `run_prepared`.
    pub stage_ms: StageTimings,
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
    pd[PD_PM_RANGE] = number(keep, "PaseMakerChangeForwardRunningStyleRange")?;
    pd[PD_PM_NOTFWD] = number(keep, "PaseMakerChangeTopHorseNotMostForwardStyleCount")?;
    let fight = group(p, "CompeteFight")?;
    copy_keys(pd, PD_CF_DIST, fight, &[
        "DistanceGap",
        "LaneGap",
        "AddParam1Coef1",
        "AddParam1Coef2",
        "AddParam1Coef3",
        "AddParam2Coef1",
        "AddParam2Coef2",
        "AddParam2Coef3",
    ])?;

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
    copy_keys(pd, PL_FRONT_LANE, block, &["FrontBlockLaneGap", "SideBlockDistanceGap", "SideBlockLaneGap"])?;
    copy_keys(pd, PL_FORCE_IN_THRESHOLD, p, &[
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
    let count = slopes.len().min(RACE_MAX_SLOPES);
    for (i, slope) in slopes.iter().enumerate().take(count) {
        pd[PC_SLOPES + 3 * i] = slope.get_f64("start")?;
        pd[PC_SLOPES + 3 * i + 1] = slope.get("length").and_then(Value::as_f64).unwrap_or(0.);
        pd[PC_SLOPES + 3 * i + 2] = slope.get_f64("slope")?;
    }
    Ok(count)
}

/// Everything the launch needs about one race before the powf slot (it
/// waits for the adjusted-guts pre-pass) and before the geometry tables.
/// The seed enters this data only through the `ri[RI_SEED]` word: the stats
/// cases, the skill table, the parameter rows and the geometry are all
/// seed-independent, and the kernel reseeds every stream from that word.
/// A clone with `with_seed` is therefore an exact prepare for another seed.
#[derive(Clone)]
pub struct Prepared {
    ri: Vec<u32>,
    pd: Vec<f64>,
    n: usize,
    stats: Vec<StatsCase>,
    host_fallback: Option<String>,
    geometry: Option<retail_geometry::FlatCourse>,
    seed: i32,
    sk: Vec<u32>,
    bumps: Vec<(usize, i32, f64)>,
    debug_scratch: bool,
    grid: bool,
}

impl Prepared {
    /// A flagged placeholder keeping the batch aligned when one request fails
    /// to parse at all. It never reaches the launch.
    pub(crate) fn stub(reason: String) -> Prepared {
        Prepared {
            ri: Vec::new(),
            pd: Vec::new(),
            n: 0,
            stats: Vec::new(),
            host_fallback: Some(reason),
            geometry: None,
            seed: 0,
            sk: Vec::new(),
            bumps: Vec::new(),
            debug_scratch: false,
            grid: false,
        }
    }

    /// The clone of this prepared race that `prepare(request, seed)` would
    /// return for another seed: one `ri` copy with `RI_SEED` rewritten.
    /// Only valid for a parsed prepare (a stub has no `ri`).
    pub fn with_seed(&self, seed: i32) -> Prepared {
        let mut clone = self.clone();
        clone.ri[RI_SEED] = seed as u32;
        clone.seed = seed;
        clone
    }
}

/// Parses one request into kernel inputs. A domain the kernel cannot answer
/// bit-exactly comes back as `host_fallback`, and the race skips the launch.
/// Geometry loads and flattens once per course through the registry, an
/// injected copy ahead of the file fallback.
fn prepare(
    request: &str,
    seed: i64,
    globals: &Value,
    grids: &Value,
    courses: &Value,
    registry: &mut GeometryRegistry,
) -> Result<Prepared, String> {
    let root = json::parse(request)?;
    let course_id = root.get_i64("courseId")? as i32;
    let seed_i32 = i32::try_from(seed).map_err(|_| "seed does not fit an i32".to_string())?;
    let horses = root.get_array("horses")?;
    let course = courses.get(&course_id.to_string()).ok_or("course_data.json has no entry for the course")?;

    // Domain gates first, each mapping to a kernel fallback code.
    let flagged = |reason: &str| Ok(Prepared {
        ri: vec![0u32; RI_STRIDE],
        pd: vec![0f64; RIL_STRIDE],
        n: horses.len(),
        bumps: Vec::new(),
        stats: Vec::new(),
        host_fallback: Some(reason.into()),
        geometry: None,
        seed: seed_i32,
        sk: Vec::new(),
        debug_scratch: false,
        grid: false,
    });
    // One validity test mirrors experimental_grid_enabled (src/lib.rs:262-272),
    // including its message, so invalid shapes flag with the engine's reason.
    // It runs ahead of the roster cap: the engine validates the grid key first.
    let grid = match root.get("experimentalGrid") {
        None => false,
        Some(g) => {
            let valid = g.as_object().is_some_and(|o| o.len() == 2)
                && g.get("columns").and_then(Value::as_f64) == Some(20.0)
                && g.get("rowSpacing").and_then(Value::as_f64) == Some(3.0)
                && (21..=120).contains(&horses.len());
            if !valid {
                return flagged(
                    "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0.",
                );
            }
            true
        }
    };
    if horses.is_empty() || horses.len() > RACE_MAXR {
        return flagged("roster size outside the kernel's range");
    }
    // The engine enables the startup pass when the flag is true and the
    // runtime is retail (lib.rs:1507-1513). The kernel replays it as one dt=0
    // tick ahead of the frame-0 capture. A non-boolean flag errors the CPU.
    let startup = match root.get("experimentalRetailStartup") {
        None => false,
        Some(value) => match value.as_bool() {
            Some(b) => b,
            None => return flagged("experimentalRetailStartup must be a boolean"),
        },
    };
    if root.get("experimentalRetailRuntime").and_then(Value::as_bool) != Some(true) {
        return flagged("non-retail runtime");
    }
    // The engine refuses geometry with grid starts (api.rs:68-71).
    if grid && root.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true) {
        return flagged("World geometry requires the experimental runtime and standard starting grid");
    }
    let mut ri = vec![0u32; RI_STRIDE];
    let mut pd = vec![0f64; RIL_STRIDE];
    let mut stats_cases: Vec<StatsCase> = Vec::new();
    let mut geometry = None;
    pack_shared_parameters(globals, &mut pd)?;
    let nslopes = pack_course_scalars(course, &mut pd)?;
    // The duel path only runs when the course has both straights and corners,
    // matching update_dueling's guard.
    ri[RI_NSTR] = course.get("straights").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) as u32;
    ri[RI_NCOR] = course.get("corners").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) as u32;

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
    ri[RI_PAIRED] =
        (root.get("pairedRandomStreams").and_then(Value::as_bool) == Some(true)) as u32;
    ri[RI_STARTUP] = startup as u32;
    ri[RI_NSLOPES] = nslopes as u32;
    let temptation = group(group(globals, "parameters")?, "Temptation")?;
    ri[RI_LOT_MIN] = temptation.get_i64("TemptationLotSectionMin")? as u32;
    ri[RI_LOT_MAX] = temptation.get_i64("TemptationLotSectionMax")? as u32;
    ri[RI_EMPTY_MOVE] = empty_move as u32;
    pd[PI_DISTANCE] = distance;
    pd[PC_FIRST_MOVE_POINT] = first_move;
    pd[PD_DTYPE] = course.get_i64("distanceType")? as f64;
    // The final-corner half window, NaN meaning "no window" exactly like the
    // engine's geometry-less Option making the condition false.
    pd[PD_FC_HALF0] = f64::NAN;
    pd[PD_FC_HALF1] = f64::NAN;

    if root.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true) {
        // first_move_lane_point drives the lane machine when geometry is on,
        // so the registry carries it beside the flattened tables.
        let (flat, first_move) = registry.course(course_id, distance as f32)?;
        pd[PC_FIRST_MOVE_POINT] = f64::from(first_move);
        if let Some((h0, h1)) = flat.final_corner_half {
            pd[PD_FC_HALF0] = f64::from(h0);
            pd[PD_FC_HALF1] = f64::from(h1);
        }
        geometry = Some(flat);
    }

    // Dense gate rank drives the retail starting lane.
    let mut gate_order: Vec<usize> = (0..horses.len()).collect();
    gate_order.sort_by_key(|&i| horses[i].get_i64("gateIndex").unwrap_or(0));
    let track = course.get_i64("raceTrackId")? as i32;

    // Hand the skill set to the kernel's numeric table. One unsupported
    // construct flags the whole race rather than approximating the engine.
    let sk_plan = crate::skills_compile::plan(&root, course);
    if let Some(reason) = &sk_plan.reason {
        return flagged(reason);
    }

    for (r, horse) in horses.iter().enumerate() {
        let mut strategy = strategy_id(horse.get_str("strategy")?)?;
        // HorseData.InitRunningStyleEx (race_solver_builder.rs:858-865): a
        // Nige that learned a type 6 effect races as Oonige, decided before
        // triggers and wisdom lotteries and invisible to the request.
        if strategy == 1 && learned_runaway(horse, root.get("skillData")) {
            strategy = 5;
        }
        let rank = gate_order.iter().position(|&i| i == r).unwrap_or(0);
        // Grid mode takes the engine's row-lane pair (src/lib.rs:274-279, the
        // f32 round-trip at lib.rs:610), retail takes the track's gate lane.
        let (start_pos, lane) = if grid {
            let column = rank % 20;
            let lane = (column as f64 + if column >= 9 { 0.6 } else { 0.0 }) / 18.0;
            (f64::from((-3.0 * (rank / 20) as f64) as f32), f64::from(lane as f32))
        } else {
            (0.0, retail_starting_lane(rank, track, horses.len(), grids))
        };
        let b = RI_RUN0 + RI_RUN_STRIDE * r;
        // Retail initialization sees the base style: an Oonige takes the
        // RunningOut force coefficient and its shared-stream draw (lib.rs:1533).
        ri[b + RR_STYLE] = (if strategy == 5 { 1 } else { strategy }) as u32;
        ri[b + RR_STRATEGY] = strategy as u32;
        ri[b + RR_DISTAPT] = aptitude_id(horse.get_str("distanceAptitude")?)? as u32;
        ri[b + RR_MOOD] = horse.get_i64("mood")? as u32;
        ri[b + RR_RANK] = horse.get("teamRank").and_then(Value::as_i64).unwrap_or(0) as u32;
        ri[b + RR_STAPT] = aptitude_id(horse.get_str("strategyAptitude")?)? as u32;
        ri[b + RR_SUAPT] = aptitude_id(horse.get_str("surfaceAptitude")?)? as u32;
        // The fixed team id, the same value the engine reads for the debuff
        // same-team drop (lib.rs:335).
        ri[b + RR_TEAM] = horse.get("teamId").and_then(Value::as_f64).unwrap_or(0.0) as i64 as u32;
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
            pd[PI_RUNNER0 + PI_RUN_STRIDE * r + k] = *value;
        }
        // The lane and start position round-trip f32 in the engine's start table.
        pd[PI_RUNNER0 + PI_RUN_STRIDE * r + PI_RUN_LANE] = f64::from(lane as f32);
        pd[PI_RUNNER0 + PI_RUN_STRIDE * r + PI_RUN_POS] = start_pos;
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
    Ok(Prepared { ri, pd, n: horses.len(), stats: stats_cases, host_fallback: None, geometry, seed: seed_i32, sk: sk_plan.words, bumps: sk_plan.bumps, debug_scratch: false, grid })
}

/// Engine rule: an installed skill whose definition carries a type 6 effect
/// upgrades a Nige to Oonige at build time.
fn learned_runaway(horse: &Value, skill_data: Option<&Value>) -> bool {
    let Some(Value::Array(list)) = horse.get("skills") else { return false };
    let Some(table) = skill_data else { return false };
    list.iter().any(|entry| {
        let Some((id, _)) = crate::skills_compile::normalize_skill_entry(entry) else {
            return false;
        };
        table
            .get(&id)
            .and_then(|def| def.get("alternatives"))
            .and_then(Value::as_array)
            .is_some_and(|alts| {
                alts.iter().any(|alt| {
                    alt.get("effects")
                        .and_then(Value::as_array)
                        .is_some_and(|effects| {
                            effects
                                .iter()
                                .any(|ef| ef.get("type").and_then(Value::as_i64) == Some(6))
                        })
                })
            })
    })
}

/// Runs every request end to end on the gated device. Races whose domain the
/// kernel cannot answer stay flagged without a launch.
pub fn run(requests: &[(String, i64)]) -> Result<RaceRun, String> {
    Ok(run_inner(requests, None)?.0)
}

/// Kernel-only cost probe: prepare and upload once, then warm the launch and
/// time `timed` further launches of the same buffers. No frames read-back,
/// so launch_ms is device execution plus per-launch sync only.
pub struct KernelBench {
    pub device: String,
    pub build_ms: f64,
    pub races: usize,
    pub runners: usize,
    /// Race ticks summed over the batch for one launch.
    pub total_ticks: usize,
    pub launch_ms: f64,
}

pub fn run_kernel_only(requests: &[(String, i64)], warm: usize, timed: usize) -> Result<KernelBench, String> {
    let (_, bench) = run_inner(requests, Some((warm.max(1), timed.max(1))))?;
    bench.ok_or_else(|| "kernel-only run found no launchable race".into())
}

/// Flattened course tables, cached per course by `RaceContext` and by the
/// module-level runs. The f32 is the geometry's first_move_lane_point.
/// `injected` holds the app's registered JSON per course: the authority over
/// the `data/course_geometry` files, which stay the fallback for courses
/// nothing registered (tests and standalone tools).
#[derive(Default)]
pub struct GeometryRegistry {
    injected: std::collections::HashMap<i32, String>,
    courses: std::collections::HashMap<i32, (retail_geometry::FlatCourse, f32)>,
}

impl GeometryRegistry {
    /// Make `json_text` (the app's `data/course_geometry/<course_id>.json`
    /// contents) the kernel's geometry for the course, replacing any earlier
    /// copy, injected or file, from the next prepare. The text is fully
    /// verified now against its own captured identity, so a rejected course
    /// names itself at registration instead of mid-batch.
    pub fn register(&mut self, course_id: i32, json_text: &str) -> Result<(), String> {
        let value = json::parse(json_text).map_err(|e| format!("injected geometry for {course_id}: {e}"))?;
        let identity = value
            .get("course_identity")
            .ok_or_else(|| format!("injected geometry for {course_id}: Missing verified course identity"))?;
        let distance = identity.get_f64("distance")? as f32;
        retail_geometry::CourseGeometry::from_value(&value, course_id, distance)
            .map_err(|e| format!("injected geometry for {course_id}: {e}"))?;
        self.injected.insert(course_id, json_text.to_string());
        // Drop the flattened cache so the next prepare flattens the new copy.
        self.courses.remove(&course_id);
        Ok(())
    }

    fn course(&mut self, course_id: i32, distance: f32) -> Result<(retail_geometry::FlatCourse, f32), String> {
        if let Some((flat, first_move)) = self.courses.get(&course_id) {
            return Ok((flat.clone(), *first_move));
        }
        // An injected copy is the authority: a bad one fails this prepare
        // (the facade answers `fallback`) instead of substituting the file.
        let loaded = match self.injected.get(&course_id) {
            Some(text) => {
                let value = json::parse(text).map_err(|e| format!("injected geometry for {course_id}: {e}"))?;
                retail_geometry::CourseGeometry::from_value(&value, course_id, distance)
                    .map_err(|e| format!("injected geometry for {course_id}: {e}"))?
            }
            None => {
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/course_geometry");
                retail_geometry::CourseGeometry::load(&dir, course_id, distance)
                    .map_err(|e| format!("world geometry for {course_id}: {e}"))?
            }
        };
        let first_move = loaded.first_move_lane_point;
        let flat = retail_geometry::flatten_course(&loaded);
        self.courses.insert(course_id, (flat.clone(), first_move));
        Ok((flat, first_move))
    }
}

/// Where `RaceContext::open` spent its milliseconds: the device gate (with
/// its restricted_tick program), the race program build, the initialization
/// program build, and the engine data load.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenTimings {
    pub spike_ms: f64,
    pub race_program_ms: f64,
    pub init_program_ms: f64,
    pub data_ms: f64,
    pub total_ms: f64,
}

/// Where `run_prepared` spent its milliseconds. Every batch pays these once
/// per call. `prepass_ms` is the adjusted-stats launch (its program is built
/// at `RaceContext::open`, not per call). `tables_ms` is the geometry dedup
/// and table assembly, `upload_ms` the buffer allocation and writes,
/// `launch_ms` the kernel enqueue plus the device sync, `readback_ms` the
/// result reads, and `decode_ms` the host-side decode of every race row.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimings {
    pub total_ms: f64,
    pub prepass_ms: f64,
    pub tables_ms: f64,
    pub upload_ms: f64,
    pub launch_ms: f64,
    pub readback_ms: f64,
    pub decode_ms: f64,
}

/// What `run_prepared` does at the launch: run once, or time launches.
/// `Burst` enqueues back to back with one sync at the end, which separates
/// work time from the device's per-launch wake-up cost.
pub enum LaunchMode {
    Once,
    /// warm launches, a sync, then `timed` launches each with its own sync.
    Timed { warm: usize, timed: usize },
    /// warm launches, a sync, then `count` back-to-back launches and one sync.
    Burst { warm: usize, count: usize },
}

impl LaunchMode {
    /// True when the launch is a measurement and frames stay undecoded.
    fn timed(&self) -> bool {
        !matches!(self, LaunchMode::Once)
    }
}

/// Launch state kept between batches: one gated device, one built race
/// program, one built initialization program, engine data loaded once and
/// the geometry registry (injected app copies ahead of the file fallback).
/// Not Sync, the OpenCL queue and program belong to the thread that opened
/// them.
pub struct RaceContext {
    globals: Value,
    grids: Value,
    courses: Value,
    spike: crate::gpu::Spike,
    program: ocl::Program,
    init_program: ocl::Program,
    registry: GeometryRegistry,
    pub open_ms: OpenTimings,
}

impl RaceContext {
    /// Runs the device gate, loads the engine data and builds the race and
    /// initialization programs once. A failing gate never loads the data.
    pub fn open() -> Result<RaceContext, String> {
        let started = std::time::Instant::now();
        let spike = crate::gpu::Spike::open()?;
        let spike_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = std::time::Instant::now();
        let program = spike.build_program(crate::gpu::RACE_KERNEL_SOURCE)?;
        let race_program_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = std::time::Instant::now();
        let init_program = spike.build_program(crate::gpu::INITIALIZATION_KERNEL_SOURCE)?;
        let init_program_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = std::time::Instant::now();
        let globals = initialization::engine_parameters_value()?;
        let grids = json::parse(&dep_cpu::engine_data("starting_grids.json")?)?;
        let courses = initialization::engine_json("data/course_data.json")?;
        let data_ms = started.elapsed().as_secs_f64() * 1e3;
        let total_ms = started.elapsed().as_secs_f64() * 1e3 + spike_ms + race_program_ms + init_program_ms;
        Ok(RaceContext {
            globals,
            grids,
            courses,
            spike,
            program,
            init_program,
            registry: GeometryRegistry::default(),
            open_ms: OpenTimings {
                spike_ms,
                race_program_ms,
                init_program_ms,
                data_ms,
                total_ms,
            },
        })
    }

    /// The gated device the batches run on.
    pub fn device(&self) -> &crate::gpu::DeviceReport {
        &self.spike.device
    }

    /// Register the app's course JSON as the kernel's geometry authority for
    /// the course, ahead of the file fallback (see `GeometryRegistry`).
    pub fn register_geometry(&mut self, course_id: i32, json_text: &str) -> Result<(), String> {
        self.registry.register(course_id, json_text)
    }

    /// One request into kernel inputs. The Err arm is a parse failure the
    /// batch caller reports per race instead of failing the batch.
    pub fn prepare(&mut self, request: &str, seed: i64) -> Result<Prepared, String> {
        prepare(request, seed, &self.globals, &self.grids, &self.courses, &mut self.registry)
    }

    /// Launches the unflagged entries in one kernel run and decodes every
    /// entry, flagged ones included, in input order. The facade and the
    /// benches drive batches through here. `run` and the module-level entries
    /// open a context and prepare on top of it.
    pub fn run_prepared(&mut self, prepared: &mut [Prepared], timing: Option<(usize, usize)>) -> Result<(RaceRun, Option<KernelBench>), String> {
        self.run_prepared_mode(
            prepared,
            match timing {
                None => LaunchMode::Once,
                Some((warm, timed)) => LaunchMode::Timed { warm, timed },
            },
        )
    }

    /// `run_prepared` in `LaunchMode::Burst`: the timed launches go back to
    /// back with a single sync, so the measurement shows what the launches
    /// cost when the device never goes idle between them.
    pub fn run_prepared_burst(&mut self, prepared: &mut [Prepared], warm: usize, count: usize) -> Result<(RaceRun, Option<KernelBench>), String> {
        self.run_prepared_mode(prepared, LaunchMode::Burst { warm: warm.max(1), count: count.max(1) })
    }

    fn run_prepared_mode(&mut self, prepared: &mut [Prepared], mode: LaunchMode) -> Result<(RaceRun, Option<KernelBench>), String> {
    let t_total = std::time::Instant::now();
    let launched: Vec<usize> = (0..prepared.len()).filter(|&i| prepared[i].host_fallback.is_none()).collect();
    if launched.windows(2).any(|w| prepared[w[0]].n != prepared[w[1]].n) {
        return Err("one launch needs one uniform runner count".into());
    }
    let n = launched.first().map_or(1, |&i| prepared[i].n);

    // The stats chain runs once per race as an initialization-kernel
    // pre-pass so the host can supply the exact f32 powf (the kernel
    // re-runs the chain itself, from the same inputs, for the rest).
    let t_stage = std::time::Instant::now();
    let mut cases: Vec<Case> = Vec::new();
    for &i in &launched {
        for stats in &prepared[i].stats {
            cases.push(Case::Stats(stats.clone()));
        }
    }
    let prepass = if cases.is_empty() { None } else { Some(initialization::run_with_program(&cases, &self.spike, &self.init_program)?) };
    let mut offset = 0usize;
    for &i in &launched {
        let count = prepared[i].stats.len();
        let spurt_coef = prepared[i].pd[PT_SPURT + 2] as f32;
        let spurt_pow = prepared[i].pd[PT_SPURT + 3] as f32;
        let duel_speed = [
            prepared[i].pd[PD_CF_SPEED] as f32,
            prepared[i].pd[PD_CF_SPEED + 1] as f32,
            prepared[i].pd[PD_CF_SPEED + 2] as f32,
        ];
        let duel_accel = [
            prepared[i].pd[PD_CF_ACCEL] as f32,
            prepared[i].pd[PD_CF_ACCEL + 1] as f32,
            prepared[i].pd[PD_CF_ACCEL + 2] as f32,
        ];
        let proficiency = [1.1f64, 1.0, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1];
        for r in 0..count {
            let adjusted = match &prepass {
                Some(out) => match &out.outs[offset + r] {
                    Out::Stats(s) => s.adjusted_f64(),
                    _ => return Err("stats pre-pass decode".into()),
                },
                None => [0.; 5],
            };
            let raw_guts = adjusted[3];
            // The engine's constructor pass chains the static GutsUp and
            // WisdomUp bumps as an f32 accumulator and clamps the field on
            // every read, and the kakari bonus runs in f64.
            let mut acc_guts = raw_guts as f32;
            let mut acc_wis = adjusted[4] as f32;
            let mut bonus = 0.0f64;
            let mut bump_guts = false;
            let mut bump_wis = false;
            for (_runner, kind, modifier) in prepared[i].bumps.iter().filter(|(b, _, _)| *b == r) {
                match *kind {
                    4 => {
                        acc_guts += *modifier as f32;
                        bump_guts = true;
                    }
                    5 => {
                        acc_wis += *modifier as f32;
                        bump_wis = true;
                    }
                    _ => bonus += *modifier / 100.0,
                }
            }
            let guts_eff = if bump_guts { acc_guts.clamp(1.0, 2000.0) as f64 } else { raw_guts };
            let base = PD_RUN0 + PD_RUN_STRIDE * r;
            let bounded = (guts_eff as f32).clamp(1., 2000.);
            prepared[i].pd[base + PRD_POWF] =
                f64::from((bounded * spurt_coef).powf(spurt_pow).to_bits());
            let guts = guts_eff.max(1.0);
            let duel = |c: [f32; 3]| {
                let argument = (guts as f32) * c[0];
                let power = (argument as f64).powf(c[1] as f64) as f32;
                (power * c[2]) as f64
            };
            prepared[i].pd[base + PRD_DUEL_T] = duel(duel_speed);
            prepared[i].pd[base + PRD_DUEL_A] = duel(duel_accel);
            let argument = (guts as f32) * 500.0_f32;
            let power = (argument as f64).powf(0.6_f32 as f64) as f32;
            prepared[i].pd[base + PRD_SPOT_M] = (power * 0.0001_f32) as f64;
            let stapt = prepared[i].ri[RI_RUN0 + RI_RUN_STRIDE * r + RR_STAPT] as usize;
            prepared[i].pd[base + PRD_SPOT_D] =
                (700.0 * guts).powf(0.5) * 0.012 * proficiency[stapt];
            if bump_guts {
                prepared[i].pd[base + PRD_BOOST_GUTS] = acc_guts.clamp(1.0, 2000.0) as f64;
            }
            if bump_wis {
                prepared[i].pd[base + PRD_BOOST_WIS] = acc_wis.clamp(1.0, 2000.0) as f64;
            }
            prepared[i].pd[base + PRD_RUSH_BONUS] =
                f64::from(((bonus * 100.0) as f32).to_bits());
        }
        offset += count;
    }
    let prepass_ms = t_stage.elapsed().as_secs_f64() * 1e3;

    // One geometry table set shared by every launched race.
    let t_stage = std::time::Instant::now();
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
        // OpenCL rejects zero-length buffers, so a geometry-off launch never
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
            lm_boundary: vec![0f32],
            lm_limit: vec![0f32],
            // OpenCL rejects empty buffers, and a count of zero keeps the
            // kernel from ever reading these rows.
            ce_start: vec![0f32],
            ce_end: vec![0f32],
            ce_id: vec![0i32],
            ce_flags: vec![0u32],
            final_corner_half: None,
        });
    }
    let tables = crate::geometry::course_tables(&table_sets);
    let tables_ms = t_stage.elapsed().as_secs_f64() * 1e3;

    let queue = self.spike.queue();

    let m = launched.len();
    // Frame-byte budget per launch. One 120-horse race writes
    // RACE_MAX_TICKS * n * 6 f64 (about 35.4 MB), so a big batch splits into
    // several launches; every current suite fits a single one.
    const FRAME_BYTES_BUDGET: usize = 512 << 20;
    let race_frames_bytes = RACE_MAX_TICKS * n * 6 * std::mem::size_of::<f64>();
    let chunk_len = (FRAME_BYTES_BUDGET / race_frames_bytes.max(1)).max(1);

    let mut status = vec![0u32; m.max(1) * ST_STRIDE];
    let mut events_all = vec![0u32; m.max(1) * EV_STRIDE];
    let mut frames_all = if mode.timed() { Vec::new() } else { vec![0f64; m.max(1) * RACE_MAX_TICKS * n * 6] };
    // The skill scratch readback stays off the normal path: the kernel leaves
    // the merged windows and the record state there, and a divergence hunt
    // asks for them per race through `run_with_skill_scratch`.
    let want_sk = m > 0 && prepared.iter().any(|p| p.debug_scratch);
    let mut skd_all: Vec<f64> = Vec::new();
    let mut sks_all: Vec<u32> = Vec::new();
    if want_sk {
        skd_all = vec![0f64; m * SRL_WORDS];
        sks_all = vec![0u32; m * SCR_U_WORDS];
    }
    let mut upload_ms = 0.;
    let mut launch_ms = 0.;
    let mut readback_ms = 0.;
    let mut bench_ticks = 0usize;
    let mut bench_launch_ms = 0.;
    for chunk_start in (0..m).step_by(chunk_len) {
        let mc = chunk_len.min(m - chunk_start);
        let chunk = &launched[chunk_start..chunk_start + mc];
        let slot0 = chunk_start;
        let t_stage = std::time::Instant::now();
        let mut ri_all = vec![0u32; mc * RI_STRIDE];
        let mut pd_all = vec![0f64; mc * RIL_STRIDE];
        let mut sk_all = vec![0u32; mc * SK_STRIDE];
        for (slot, &i) in chunk.iter().enumerate() {
            ri_all[slot * RI_STRIDE..][..RI_STRIDE].copy_from_slice(&prepared[i].ri);
            pd_all[slot * RIL_STRIDE..][..RIL_STRIDE].copy_from_slice(&prepared[i].pd);
            if !prepared[i].sk.is_empty() {
                sk_all[slot * SK_STRIDE..][..SK_STRIDE].copy_from_slice(&prepared[i].sk);
            }
        }
        let b_ri = read_write(&queue, &ri_all)?;
        let b_pd = read_write(&queue, &pd_all)?;
        let b_sk = read_only(&queue, &sk_all)?;
        let b_scr_u = read_write(&queue, &vec![0u32; mc * SCR_U_WORDS])?;
        let b_scr_d = read_write(&queue, &vec![0f64; mc * SRL_WORDS])?;
        let b_frames = read_write(&queue, &vec![0f64; mc * RACE_MAX_TICKS * n * 6])?;
        let b_status = read_write(&queue, &vec![0u32; mc * ST_STRIDE])?;
        let b_events = read_write(&queue, &vec![0u32; mc * EV_STRIDE])?;
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
        upload_ms += t_stage.elapsed().as_secs_f64() * 1e3;

        let t_stage = std::time::Instant::now();
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name("race")
            .queue(queue.clone())
            .global_work_size(mc)
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
            .arg(&b_sk)
            .arg(&b_events)
            .build()
            .map_err(|e| e.to_string())?;
        match mode {
            LaunchMode::Timed { warm, timed } => {
                for _ in 0..warm {
                    unsafe { kernel.enq().map_err(|e| e.to_string())? };
                }
                queue.finish().map_err(|e| e.to_string())?;
                let mut probe = vec![0u32; mc * ST_STRIDE];
                b_status.cmd().read(&mut probe).enq().map_err(|e| e.to_string())?;
                queue.finish().map_err(|e| e.to_string())?;
                bench_ticks += probe.chunks(ST_STRIDE).map(|s| s[ST_TICKS] as usize).sum::<usize>();
                let start = std::time::Instant::now();
                for _ in 0..timed {
                    unsafe { kernel.enq().map_err(|e| e.to_string())? };
                    queue.finish().map_err(|e| e.to_string())?;
                }
                bench_launch_ms += start.elapsed().as_secs_f64() * 1e3 / timed as f64;
            }
            LaunchMode::Burst { warm, count } => {
                for _ in 0..warm {
                    unsafe { kernel.enq().map_err(|e| e.to_string())? };
                }
                queue.finish().map_err(|e| e.to_string())?;
                let mut probe = vec![0u32; mc * ST_STRIDE];
                b_status.cmd().read(&mut probe).enq().map_err(|e| e.to_string())?;
                queue.finish().map_err(|e| e.to_string())?;
                bench_ticks += probe.chunks(ST_STRIDE).map(|s| s[ST_TICKS] as usize).sum::<usize>();
                let start = std::time::Instant::now();
                for _ in 0..count {
                    unsafe { kernel.enq().map_err(|e| e.to_string())? };
                }
                queue.finish().map_err(|e| e.to_string())?;
                bench_launch_ms += start.elapsed().as_secs_f64() * 1e3 / count as f64;
            }
            LaunchMode::Once => unsafe { kernel.enq().map_err(|e| e.to_string())? },
        }
        queue.finish().map_err(|e| e.to_string())?;
        launch_ms += t_stage.elapsed().as_secs_f64() * 1e3;

        let t_stage = std::time::Instant::now();
        if !mode.timed() {
            b_status.cmd().read(&mut status[slot0 * ST_STRIDE..][..mc * ST_STRIDE]).enq().map_err(|e| e.to_string())?;
            b_events.cmd().read(&mut events_all[slot0 * EV_STRIDE..][..mc * EV_STRIDE]).enq().map_err(|e| e.to_string())?;
            b_frames.cmd().read(&mut frames_all[slot0 * RACE_MAX_TICKS * n * 6..][..mc * RACE_MAX_TICKS * n * 6]).enq().map_err(|e| e.to_string())?;
            if want_sk {
                b_scr_d.cmd().read(&mut skd_all[slot0 * SRL_WORDS..][..mc * SRL_WORDS]).enq().map_err(|e| e.to_string())?;
                b_scr_u.cmd().read(&mut sks_all[slot0 * SCR_U_WORDS..][..mc * SCR_U_WORDS]).enq().map_err(|e| e.to_string())?;
            }
            queue.finish().map_err(|e| e.to_string())?;
        }
        readback_ms += t_stage.elapsed().as_secs_f64() * 1e3;
    }
    let bench = if matches!(mode, LaunchMode::Timed { .. } | LaunchMode::Burst { .. }) && m > 0 {
        Some(KernelBench {
            device: self.spike.device.name.clone(),
            build_ms: self.spike.build_ms,
            races: m,
            runners: m * n,
            total_ticks: bench_ticks,
            launch_ms: bench_launch_ms,
        })
    } else {
        None
    };

    let t_stage = std::time::Instant::now();
    let mut races: Vec<RaceGpu> = Vec::with_capacity(prepared.len());
    let mut slot = 0usize;
    for race in prepared.iter() {
        let mut out = RaceGpu {
            frames: Vec::new(),
            ticks: 0,
            fallback: race.host_fallback.clone(),
            grid: race.grid,
            kernel_flag: 0,
            kernel_arg: 0,
            finish_step: None,
            finish_raw: None,
            finish_step_all: Vec::new(),
            finish_raw_all: Vec::new(),
            loop_end: 0.,
            tempt_all: Vec::new(),
            rscore_all: Vec::new(),
            hp_empty_all: Vec::new(),
            delay_all: Vec::new(),
            lss_all: Vec::new(),
            events: Vec::new(),
            debug_pace: Vec::new(),
            debug_sk: None,
        };
        if race.host_fallback.is_none() && !mode.timed() {
            let st = &status[slot * ST_STRIDE..][..ST_STRIDE];
            let ev = &events_all[slot * EV_STRIDE..][..EV_STRIDE];
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
                if race.ri[RI_DEBUG_TICK] != 0 {
                    let tail = base + (RACE_MAX_TICKS - 14) * n * 6;
                    out.debug_pace = (0..14 * n)
                        .map(|row| {
                            let mut v = [0f64; 6];
                            v.copy_from_slice(&frames_all[tail + row * 6..tail + row * 6 + 6]);
                            v
                        })
                        .collect();
                }
                if race.debug_scratch {
                    let d = (slot - 1) * SRL_WORDS + SRL_SKD;
                    let u = (slot - 1) * SCR_U_WORDS + SCR_SK;
                    out.debug_sk = Some(SkillScratch {
                        skd: skd_all[d..d + SK_RMAX * SKD_STRIDE].to_vec(),
                        sks: sks_all[u..u + SK_RMAX * SKR_STRIDE].to_vec(),
                    });
                }
                out.finish_step = read_f64(st, ST_FINISH0);
                out.finish_raw = read_f64(st, ST_FINISH0 + 2);
                out.finish_step_all = (0..race.n)
                    .map(|r| read_f64(st, ST_FINISH0 + 4 * r))
                    .collect();
                out.finish_raw_all = (0..race.n)
                    .map(|r| read_f64(st, ST_FINISH0 + 4 * r + 2))
                    .collect();
                out.tempt_all = (0..race.n).map(|r| st[ST_TEMPT0 + r]).collect();
                out.rscore_all = (0..race.n).map(|r| st[ST_RSCORE0 + r]).collect();
                out.hp_empty_all = (0..race.n).map(|r| st[ST_HPEMPTY0 + r] != 0).collect();
                out.delay_all = (0..race.n)
                    .map(|r| f64::from(f32::from_bits(st[ST_DELAY0 + r])))
                    .collect();
                out.lss_all = (0..race.n)
                    .map(|r| f64::from(f32::from_bits(st[ST_LSS0 + r])))
                    .collect();
                let count = (ev[0] as usize).min(EV_MAX);
                out.events = (0..count)
                    .map(|k| (ev[1 + 2 * k] as usize, ev[2 + 2 * k]))
                    .collect();
            }
        }
        races.push(out);
    }
    let decode_ms = t_stage.elapsed().as_secs_f64() * 1e3;

    let stage_ms = StageTimings {
        total_ms: t_total.elapsed().as_secs_f64() * 1e3,
        prepass_ms,
        tables_ms,
        upload_ms,
        launch_ms,
        readback_ms,
        decode_ms,
    };
    Ok((RaceRun { device: self.spike.device.clone(), build_ms: self.spike.build_ms, stage_ms, races }, bench))
    }
}

/// Diagnostic version of `run`: snapshot the pace decision state entering
/// and leaving the movement pass of frame `debug_tick`, carried in
/// `RaceGpu::debug_pace`. Zero leaves the kernel snapshot off.
pub fn run_with_snapshot(
    requests: &[(String, i64)],
    debug_tick: usize,
) -> Result<RaceRun, String> {
    let mut ctx = RaceContext::open()?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        let mut p = ctx.prepare(request, *seed)?;
        if debug_tick > 0 && p.ri.len() > RI_DEBUG_TICK {
            p.ri[RI_DEBUG_TICK] = debug_tick as u32;
        }
        prepared.push(p);
    }
    Ok(ctx.run_prepared(&mut prepared, None)?.0)
}

/// Diagnostic version of `run`: read every race's kernel skill scratch back
/// into `RaceGpu::debug_sk` after the launch, so a divergence hunt can read
/// the merged trigger windows and the record state the scan gate used.
pub fn run_with_skill_scratch(requests: &[(String, i64)]) -> Result<RaceRun, String> {
    let mut ctx = RaceContext::open()?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        let mut p = ctx.prepare(request, *seed)?;
        p.debug_scratch = true;
        prepared.push(p);
    }
    Ok(ctx.run_prepared(&mut prepared, None)?.0)
}

/// The module-level entry: open a context, prepare every request (one parse
/// failure fails the whole call, as it always has) and launch once.
fn run_inner(
    requests: &[(String, i64)],
    timing: Option<(usize, usize)>,
) -> Result<(RaceRun, Option<KernelBench>), String> {
    let mut ctx = RaceContext::open()?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        prepared.push(ctx.prepare(request, *seed)?);
    }
    ctx.run_prepared(&mut prepared, timing)
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
        (FB_HOST, code) => format!("kernel host flag {code}"),
        (FB_INIT, _) => "the initialization chain flagged".into(),
        (FB_GEOMETRY, _) => "geometry rejected a query".into(),
        (FB_EVENT, _) => "event log overflow".into(),
        (FB_SKILL_SEQ, _) => "skill duration order overflow".into(),
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

/// The v77 roster line for a course and a seed, every horse kept in place.
pub fn multi_request(request: &str, course: i32, seed: i64) -> Result<String, String> {
    let mut value = json::parse(request)?;
    let Value::Object(map) = &mut value else { return Err("request is not a JSON object".into()) };
    if !map.get("horses").is_some_and(|h| h.as_array().is_some()) {
        return Err("request has no horses array".into());
    }
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    Ok(out)
}

/// `multi_request` with the `pairedRandomStreams` key set.
pub fn paired_request(request: &str, course: i32, seed: i64) -> Result<String, String> {
    let mut value = json::parse(request)?;
    let Value::Object(map) = &mut value else { return Err("request is not a JSON object".into()) };
    if !map.get("horses").is_some_and(|h| h.as_array().is_some()) {
        return Err("request has no horses array".into());
    }
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    map.insert("pairedRandomStreams".into(), Value::Bool(true));
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    Ok(out)
}

/// `multi_request` with these skill ids attached to one horse at level 1.
/// `usage_patch` rewrites one skillData entry's alternatives[0].durationUsage
/// so the gate can also build a request the compiler must reject.
pub fn skills_request(
    request: &str,
    course: i32,
    seed: i64,
    horse: usize,
    skill_ids: &[i64],
    usage_patch: Option<(i64, i64)>,
) -> Result<String, String> {
    let mut value = json::parse(request)?;
    let Value::Object(map) = &mut value else { return Err("request is not a JSON object".into()) };
    map.insert("courseId".into(), Value::Number(f64::from(course)));
    map.insert("seed".into(), Value::Number(seed as f64));
    {
        let Value::Array(horses) = map.get_mut("horses").ok_or("request has no horses array")? else {
            return Err("request has no horses array".into());
        };
        let Some(Value::Object(hm)) = horses.get_mut(horse) else {
            return Err("requested horse is out of range".into());
        };
        hm.insert(
            "skills".into(),
            json::arr(skill_ids.iter().map(|&id| {
                json::obj(vec![
                    ("skillId", Value::Number(id as f64)),
                    ("level", Value::Number(1.)),
                ])
            }).collect()),
        );
    }
    if let Some((skill_id, usage)) = usage_patch {
        let Value::Object(sd) = map.get_mut("skillData").ok_or("request has no skillData")? else {
            return Err("request has no skillData".into());
        };
        let Value::Object(entry) = sd
            .get_mut(&skill_id.to_string())
            .ok_or("skillData lacks the patched skill")?
        else {
            return Err("skillData lacks the patched skill".into());
        };
        let Value::Array(alts) = entry.get_mut("alternatives").ok_or("entry has no alternatives")? else {
            return Err("entry has no alternatives".into());
        };
        let Some(Value::Object(alt)) = alts.get_mut(0) else {
            return Err("entry has no alternatives".into());
        };
        alt.insert("durationUsage".into(), Value::Number(usage as f64));
    }
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











