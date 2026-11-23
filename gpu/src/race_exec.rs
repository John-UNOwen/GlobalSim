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
use crate::random_gen::Rule30CARng;
use crate::retail_geometry;
use crate::gpu::RaceShape;
use ocl::{Buffer, Queue, flags};
use std::sync::Arc;

// The layout bound every runner-indexed region sizes to: the game's 18-horse
// field limit (`gpu::ROSTER_CAP`, the single source). Rosters above it refuse
// by size, before the kernel.
pub const RACE_MAXR: usize = crate::gpu::ROSTER_CAP;

/// Non-grid fields above this runner count are outside the validated lane
/// band on both lanes. Grid mode keeps the engine's 21 to 120 (README
/// "Wave 15: the non-grid band refusal and the cap back at 120").
pub const NON_GRID_MAX_RUNNERS: usize = 24;
/// The band refusal's text, the single source for prepare, the classic
/// prepare and facade::gate_reasons.
pub const NON_GRID_BAND_REFUSAL: &str =
    "non-grid fields above 24 runners are outside the validated lane band";

pub const RACE_MAX_TICKS: usize = 6144;
// The frame window sizes to a per-course tick bound, not the 400 s engine
// ceiling. A race still running at the bound flags FB_TICKS and reruns on the
// CPU, so a tight bound costs reruns, never wrong frames.
pub const TICK_BOUND_SPEED: f64 = 9.0;
pub const TICK_BOUND_MARGIN: usize = 640;
pub const TICK_BOUND_MIN: usize = 640;
pub fn course_tick_bound(distance: f64) -> usize {
    let dt = DT as f64;
    let raw = (distance / TICK_BOUND_SPEED / dt).ceil() + TICK_BOUND_MARGIN as f64;
    (raw.clamp(TICK_BOUND_MIN as f64, RACE_MAX_TICKS as f64)) as usize
}
pub const RACE_MAX_SLOPES: usize = 16;
pub const RI_STRIDE: usize = RI_DEBUG_TICK + 24;
pub const RIL_STRIDE: usize = PD_RUN0 + PD_RUN_STRIDE * RACE_MAXR;
pub const ST_STRIDE: usize = ST_HEALC + 1;
// Skill scratch mirrors of retail_core.cl and skills.cl.
pub const SK_STRIDE: usize = crate::skills_compile::SK_STRIDE;
// The compiled table always keeps the SK_RMAX layout. Only the scratch
// block is sized down, to the smallest bucket that holds the batch. The
// 192 rung cuts a 162-record request (the CM field) from 256 down: 94
// scratch records per lane fewer, and the slot still fits the batch.
pub const SK_RMAX: usize = crate::skills_compile::SK_RMAX;
pub const SK_BUCKETS: [usize; 5] = [64, 128, 192, 256, SK_RMAX];
pub const fn sk_rcap(nrec: usize) -> usize {
    let mut i = 0;
    while i < SK_BUCKETS.len() {
        if nrec <= SK_BUCKETS[i] {
            return SK_BUCKETS[i];
        }
        i += 1;
    }
    SK_RMAX
}
// Mirrors of the record blocks in skills.cl, derived from the same arm cap.
// State: 4 header words, one window count per arm slot, 19 tail words.
pub const SKR_STRIDE: usize = 4 + crate::skills_compile::SK_ARMMAX + 19;
const SK_SEQN: usize = 40;
// Dynamic: 140 head words, one 16-pair chain per arm slot, 8 effect timers.
pub const SKD_STRIDE: usize = 140 + 2 * 16 * crate::skills_compile::SK_ARMMAX + 8;
const SCR_SK: usize = SCR_BANKS + RACE_MAXR * BANK_WORDS;
const fn sk_tseq(rcap: usize) -> usize { SCR_SK + rcap * SKR_STRIDE }
const fn sk_aseq(rcap: usize) -> usize { sk_tseq(rcap) + rcap * SK_SEQN }
const fn sk_cseq(rcap: usize) -> usize { sk_aseq(rcap) + rcap * SK_SEQN }
const fn sk_lseq(rcap: usize) -> usize { sk_cseq(rcap) + rcap * SK_SEQN }
const fn sk_vseq(rcap: usize) -> usize { sk_lseq(rcap) + rcap * SK_SEQN }
const fn sk_sub(rcap: usize) -> usize { sk_vseq(rcap) + rcap * SK_SEQN }
const fn sk_staged(rcap: usize) -> usize { sk_sub(rcap) + BANK_WORDS }
const SK_AAPN: usize = 128;
const SK_AAPW: usize = 6;
const SK_RUNN: usize = 8;
const fn sk_aap(rcap: usize) -> usize { sk_staged(rcap) + 2 * rcap }
const fn sk_run(rcap: usize) -> usize { sk_aap(rcap) + SK_AAPN * SK_AAPW }
const fn sk_vval(rcap: usize) -> usize { sk_run(rcap) + RACE_MAXR * (1 + SK_RUNN) }
// The u32 scratch ends at the journal tick mirror. The window before it
// exists only in the -D SK_DIAG program variant.
pub const fn sk_jrn0(rcap: usize) -> usize { sk_vval(rcap) + rcap * 16 }
pub const SK_JRN_N: usize = 16384;
pub const fn scr_u_words(rcap: usize, diag: bool) -> usize {
    sk_jrn0(rcap) + 1 + if diag { SK_JRN_N } else { 0 }
}
pub const SCR_U_WORDS: usize = scr_u_words(SK_RMAX, false);
pub const SCR_U_WORDS_DIAG: usize = scr_u_words(SK_RMAX, true);
// Full-reply channels, rc-relative, each region counting its own records.
// Only full-request slots pay the block, other races share the host's
// zero-filled sink block 0.
pub const SCR_RC_ON: usize = 0;
pub const SCR_RC_SEQ: usize = SCR_RC_ON + 1;
pub const SCR_RC_LEAD: usize = SCR_RC_SEQ + 1;
pub const SCR_RC_RPT: usize = SCR_RC_LEAD + RACE_MAXR;
pub const SCR_RC_ACT: usize = SCR_RC_RPT + 1;
pub const RC_ACT_CAP: usize = 24576;
pub const SCR_RC_TGT: usize = SCR_RC_ACT + 1 + RC_ACT_CAP;
pub const RC_TGT_CAP: usize = 49152;
pub const SCR_RC_FAN: usize = SCR_RC_TGT + 1 + RC_TGT_CAP;
pub const RC_FAN_CAP: usize = 16384;
pub const SCR_RC_CONS: usize = SCR_RC_FAN + 1 + RC_FAN_CAP;
pub const RC_CONS_CAP: usize = 768;
pub const SCR_RC_COMP: usize = SCR_RC_CONS + 1 + RC_CONS_CAP;
pub const RC_COMP_CAP: usize = 768;
pub const SCR_RC_PHR: usize = SCR_RC_COMP + 1 + RC_COMP_CAP;
pub const RC_PHR_CAP: usize = 8192;
pub const SCR_RC_KAK: usize = SCR_RC_PHR + 1 + RC_PHR_CAP;
pub const RC_KAK_CAP: usize = 1024;
pub const RC_WORDS: usize = SCR_RC_KAK + 1 + RC_KAK_CAP;
pub const SRL_SKD: usize = SRL_HP_OUT_L + RACE_MAXR * OOL;
pub const fn srl_words(rcap: usize) -> usize { SRL_SKD + rcap * SKD_STRIDE }
pub const SRL_WORDS: usize = srl_words(SK_RMAX);
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
// Slot 9: the singleMode flag, read by the kernel's im_stats pre-pass.
pub const RI_SINGLE: usize = 9;
pub const RI_NSLOPES: usize = 10;
pub const RI_NSTR: usize = 11;
pub const RI_NCOR: usize = 12;
pub const RI_LOT_MIN: usize = 13;
pub const RI_LOT_MAX: usize = 14;
pub const RI_EMPTY_MOVE: usize = 15;
pub const RI_DEBUG_TICK: usize = RI_RUN0 + RI_RUN_STRIDE * RACE_MAXR;
pub const RI_FULL: usize = RI_DEBUG_TICK + 14;
// Compact launch row mask, mirrors of retail_core.cl: kept frame row count
// (0 writes every runner's row) and the runner index behind frame slot 0/1.
pub const RI_FRAMES: usize = RI_DEBUG_TICK + 15;
pub const RI_ROW0: usize = RI_DEBUG_TICK + 16;
pub const RI_ROW1: usize = RI_DEBUG_TICK + 17;
// Reply-channel slot in the rc buffer, mirror of retail_core.cl.
pub const RI_RC: usize = RI_DEBUG_TICK + 18;
// Request-table row this slot reads, the pd row plus the skill table shared by
// every seed of one request. Mirror of retail_core.cl.
pub const RI_REQ: usize = RI_DEBUG_TICK + 19;
pub const RI_RUN0: usize = 16;
pub const RI_RUN_STRIDE: usize = 14;
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
pub const RR_CHARA: usize = 13;
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
// 1 when some GutsUp arm can fire outside the constructor pass: the kernel
// then recomputes the guts-derived powf words from the live field.
pub const PD_LIVE_GUTS: usize = PD_CF_DIST + 13;
pub const PD_RUN0: usize = PL_FRONT_LANE + 32;
pub const PD_RUN_STRIDE: usize = 22;
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
// The unclamped wisdom accumulator, NaN when no WisdomUp bump chained. The
// field word above is clamped, so this keeps a later mid-race fire exact.
pub const PRD_WIS_ACC: usize = 8;
// The guts accumulator mirror of the wisdom one, and the strategy proficiency
// the live spot duration scales by.
pub const PRD_GUTS_ACC: usize = 9;
pub const PRD_SPOT_PROF: usize = 10;
// The later-stamina request flag and the shared solver stream pair left by
// the engine's construction draws, one row per runner (Job B, wave 18).
pub const PRD_LATER: usize = 11;
pub const PRD_R30_HI: usize = 12;
pub const PRD_R30_LO: usize = 13;
// The construction SpeedUp bump pair: the clamped speed field the plan words
// freeze from, and the unclamped accumulator, NaN when no bump chained.
pub const PRD_BOOST_SPEED: usize = 14;
pub const PRD_SPEED_ACC: usize = 15;
// The construction stamina, power and raw-field bump words, same pair shape:
// clamped field, NaN-marked accumulator, and the f64 raw max(1.0) chain the
// StaminaUp, PowerUp and AllStatusUp constructor fires chain beside the field.
pub const PRD_BOOST_STAM: usize = 16;
pub const PRD_STAM_ACC: usize = 17;
pub const PRD_BOOST_POWER: usize = 18;
pub const PRD_POWER_ACC: usize = 19;
pub const PRD_RAW_STAM: usize = 20;
pub const PRD_RAW_POWER: usize = 21;
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
/// F7 slice accounting, the kernel's `ST_DONE`. A launched lane that is
/// neither done nor flagged has work for the host's next slice launch.
pub const ST_DONE: usize = 3;
pub const ST_FINISH0: usize = 8;
// Compact-reply exports per runner, mirrors of retail_core.cl. The f32 ones
// carry raw bits, which the facade widens exactly like the engine's casts.
pub const ST_TEMPT0: usize = ST_FINISH0 + 4 * RACE_MAXR;
pub const ST_RSCORE0: usize = ST_TEMPT0 + RACE_MAXR;
pub const ST_HPEMPTY0: usize = ST_RSCORE0 + RACE_MAXR;
pub const ST_DELAY0: usize = ST_HPEMPTY0 + RACE_MAXR;
pub const ST_LSS0: usize = ST_DELAY0 + RACE_MAXR;
// Full-reply exports, mirrors of retail_core.cl. The snapshot columns carry
// horse stat bits, the spurt columns the actual start pair, and ST_SPFIRST0
// the first frame whose step left the runner spurting (0 when never).
pub const ST_SNAPG0: usize = ST_LSS0 + RACE_MAXR;
pub const ST_SNAPW0: usize = ST_SNAPG0 + 2 * RACE_MAXR;
pub const ST_SPURTT0: usize = ST_SNAPW0 + 2 * RACE_MAXR;
pub const ST_SPURTD0: usize = ST_SPURTT0 + 2 * RACE_MAXR;
pub const ST_SPFIRST0: usize = ST_SPURTD0 + 2 * RACE_MAXR;
// Per-race later-stamina activation counters: keep, compete, secure, the
// keep-gated skips of both entry checks, and the staminasyoubu fires.
pub const ST_LATER_KEEP: usize = ST_SPFIRST0 + RACE_MAXR;
pub const ST_LATER_CMP: usize = ST_LATER_KEEP + 1;
pub const ST_LATER_SEC: usize = ST_LATER_CMP + 1;
pub const ST_LATER_GATE: usize = ST_LATER_SEC + 1;
pub const ST_LATER_SYB: usize = ST_LATER_GATE + 1;
// The field heal ordinal and its tick-end snapshot, mirror of retail_core.cl.
pub const ST_HEALN: usize = ST_LATER_SYB + 1;
pub const ST_HEALC: usize = ST_HEALN + 1;
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
// SI_STRIDE mirror: retail_core.cl sizes the scalar block by SI_SLOTS.
const SCR_SF: usize = SCR_STATE + 96 * RACE_MAXR;
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
const SRL_HP: usize = (2 * RACE_MAXR + 94) * RACE_MAXR;
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
pub const FB_RC: u32 = 9;
pub const FB_TICKS: u32 = 10;

/// Mirror of race_solver_builder.rs race_seed_hi, pinned to the engine source.
fn ref_race_seed_hi(horse_seed: u32) -> u32 {
    let mut seed = horse_seed ^ 0x52414345 ^ 0x57495344;
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x85ebca6b);
    seed = (seed ^ (seed >> 13)).wrapping_mul(0xc2b2ae35);
    seed ^= seed >> 16;
    if seed == 0 { 1 } else { seed }
}

/// The kernel's raw full-reply channel streams, in write order. The
/// full_reply module decodes these into the engine's reply shape.
#[derive(Clone)]
pub struct ReplyRaw {
    /// Ten-word activations: runner, sid, detail, clock lo/hi, duration
    /// lo/hi, ordinal plus one, flags, tick. Flag bit 2 marks a finalize.
    pub act: Vec<u32>,
    /// Targets: runner, sid, detail, clock lo/hi, effect count, then one
    /// length-prefixed list per effect, 0 encoding null.
    pub tgt: Vec<u32>,
    /// Fan deliveries: runner, sid, detail, clock lo/hi, length-prefixed
    /// list, 0 encoding null.
    pub fan: Vec<u32>,
    /// Conserve releases: runner, clock lo/hi.
    pub cons: Vec<u32>,
    /// Competition starts: tick, kind, runner.
    pub comp: Vec<u32>,
    /// Positioning phases: runner, phase.
    pub phr: Vec<u32>,
    /// Rush edges: kind 0 start 1 end, runner, clock lo/hi.
    pub kak: Vec<u32>,
    /// Order-key stat snapshots per runner, raw f64 bits widened from f32.
    pub snap_guts: Vec<f64>,
    pub snap_wis: Vec<f64>,
    /// The actual last-spurt start pair, None when the edge never fired.
    pub spurt_time: Vec<Option<f64>>,
    pub spurt_dist: Vec<Option<f64>>,
    /// First frame index with the runner in the last spurt, 0 when never.
    pub spfirst: Vec<usize>,
}

/// One race as the kernel ran it. `frames` holds the ticks plus the pre-loop
/// frame 0, each row the six replay values per runner. `finish_*` stays None
/// exactly when the runner never crossed the line or the race is flagged.
#[derive(Clone)]
pub struct RaceGpu {
    pub frames: Vec<Vec<[f64; 6]>>,
    pub ticks: usize,
    pub fallback: Option<String>,
    /// The engine's grid-mode starts were in force for this race, so the
    /// compact reply carries the experimentalGridVersion tag like the engine's.
    pub grid: bool,
    /// Grid start positions in roster order, the f32-narrowed values the
    /// field ranks its recording set by (src/race_field.rs:118-125).
    pub grid_pos: Vec<f64>,
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
    /// Later-stamina activation counters: keep, compete, secure, gated,
    /// staminasyoubu. Zero while the request flag is off.
    pub later_stats: [u32; 5],
    pub delay_all: Vec<f64>,
    pub lss_all: Vec<f64>,
    /// The race's skillEvents (horseIndex, id) pairs in kernel order, id
    /// EV_KAKARI for a completed rush episode.
    pub events: Vec<(usize, u32)>,
    /// Full-reply channel streams and snapshot exports, read back only when
    /// the race was prepared with `set_full`. Word 0 semantics: each channel
    /// Vec is the flat record stream in kernel write order.
    pub reply: Option<ReplyRaw>,
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
#[derive(Clone)]
pub struct SkillScratch {
    pub skd: Vec<f64>,
    pub sks: Vec<u32>,
    /// Raw ten-word journal records, the kernel's SK_JRN window contents.
    pub journal: Vec<u32>,
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

/// Request ids. One prepare takes one, and its `with_seed` clones share it, so
/// a launch can tell the races that read the same request data (F1).
static NEXT_REQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn next_req() -> u32 {
    NEXT_REQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// The pd words the stats pre-pass derivation leaves on a race, per runner
/// in `PRD_DERIVED` order, plus the race-level live-guts word. They read the
/// pre-pass adjusted stats, the request's construction bumps and request or
/// engine-fixed scalars, so they are seed-independent and one row serves
/// every seed and later launch of the request (F2). Words the derivation
/// leaves untouched on a fresh zeroed pd are stored 0.0, so applying the row
/// writes exactly the bytes the fresh derivation would.
struct DerivedRow {
    live_guts: f64,
    /// n runners x `PRD_DERIVED.len()` words, row-major.
    words: Vec<f64>,
}

/// The runner-row pd slots the pre-pass derivation writes, in the order
/// `DerivedRow::words` stores them.
const PRD_DERIVED: [usize; 19] = [
    PRD_POWF, PRD_DUEL_T, PRD_DUEL_A, PRD_SPOT_M, PRD_SPOT_D,
    PRD_BOOST_GUTS, PRD_BOOST_WIS, PRD_RUSH_BONUS, PRD_WIS_ACC,
    PRD_GUTS_ACC, PRD_SPOT_PROF, PRD_BOOST_SPEED, PRD_SPEED_ACC,
    PRD_BOOST_STAM, PRD_STAM_ACC, PRD_BOOST_POWER, PRD_POWER_ACC,
    PRD_RAW_STAM, PRD_RAW_POWER,
];

/// One request's `DerivedRow` in the context cache, keyed by request text
/// (the FNV hash only indexes it). Nothing in a row varies after the context
/// opens: the globals, courses and grids are fixed at `RaceContext::open`,
/// and a request carries its own skill table, so re-registered geometry does
/// not stale them.
struct CachedDerived {
    key: u64,
    request: Arc<str>,
    row: Arc<DerivedRow>,
}

/// One prepared request in the context cache (F3): the compiled plan, parsed
/// runner rows and attached geometry, stored pre-launch with geometry intact.
/// A hit clones the template and reseeds the solver pair per seed, byte-exact.
struct CachedPrepared {
    key: u64,
    request: String,
    plan: bool,
    epoch: u64,
    template: Prepared,
}

/// Entries kept per request cache, newest last. The app's batch lane replays
/// one request, so a small bound holds the hot entry without keeping every
/// request a long session ever sent.
const REQUEST_CACHE_MAX: usize = 8;

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
    /// The request this race's pd row and skill table come from. Races with
    /// one id share one table row in the launch.
    req: u32,
    n: usize,
    stats: Vec<StatsCase>,
    host_fallback: Option<String>,
    geometry: Option<Arc<retail_geometry::FlatCourse>>,
    seed: i32,
    sk: Vec<u32>,
    bumps: Vec<(usize, i32, f64)>,
    live_guts: bool,
    debug_scratch: bool,
    grid: bool,
    full: bool,
    /// The effect-cap bucket `sk` was packed at (0 for a flagged race that
    /// never reaches a launch). The launch picks the program built with the
    /// matching `-D SK_EMAX`.
    sk_emax: usize,
    /// The tag-cap bucket `sk` was packed at, the `-D SK_TMAX` twin.
    sk_tmax: usize,
    // Compact launch row mask: the kept runner indices, empty for the full
    // launch that writes every runner's frame row.
    frame_rows: Vec<usize>,
    /// FNV-1a of the request bytes this race was prepared from. It indexes
    /// the context's request caches, and the entries keep the text so equal
    /// bytes answer and a colliding hash only costs a miss (F2).
    req_hash: u64,
    req_text: Arc<str>,
    /// The request's derived stats words, set when the context has launched
    /// or prepared the request before (F2). A launch applies them instead of
    /// running the stats pre-pass for this race again.
    derived: Option<Arc<DerivedRow>>,
}

impl Prepared {
    /// Why the host refused this race for the kernel, if it did.
    pub fn host_fallback_reason(&self) -> Option<&str> {
        self.host_fallback.as_deref()
    }

    /// The effect-cap bucket of the compiled skill table.
    pub fn sk_emax(&self) -> usize {
        self.sk_emax
    }

    /// The tag-cap bucket of the compiled skill table.
    pub fn sk_tmax(&self) -> usize {
        self.sk_tmax
    }

    /// The race's runner count (one launch needs one uniform count).
    pub fn runners(&self) -> usize {
        self.n
    }

    /// The host-side kernel inputs as raw bits: the ri row, the pd row and
    /// the compiled skill table. The cache gate uses them to check that a
    /// prepared-cache clone carries exactly what a fresh parse produces.
    pub fn host_input_bits(&self) -> (Vec<u32>, Vec<u64>, Vec<u32>) {
        (self.ri.clone(), self.pd.iter().map(|v| v.to_bits()).collect(), self.sk.clone())
    }

    /// Compiled skill records, the number that picks the scratch bucket.
    pub fn records(&self) -> usize {
        self.sk[crate::skills_compile::SK_NREC] as usize
    }

    /// The wire id and flags word of every compiled record, for the gate
    /// that checks which records carry the plan-walk marking.
    pub fn sk_record_flags(&self) -> Vec<(u32, u32)> {
        (0..self.records())
            .map(|i| {
                let at = crate::skills_compile::SK_RECORDS
                    + crate::skills_compile::SK_RSTRIDE * i;
                (self.sk[at], self.sk[at + 2])
            })
            .collect()
    }

    /// Ask for the full-reply channels: the kernel writes the activation,
    /// target, fan, conserve, competition, phase and rush streams plus the
    /// snapshot columns the full reply needs. Only valid for a parsed
    /// prepare, and it must stay off a prepared race the kernel ran already.
    pub(crate) fn set_full(&mut self) {
        if self.ri.len() > RI_FULL {
            self.ri[RI_FULL] = 1;
            self.full = true;
        }
    }

    /// Write only these runners' frame rows, one and two kept rows for the
    /// app's compact lane. The frame block strides by the kept count, and the
    /// kept order is the frame row order. A debug race keeps every runner's
    /// row (its tail block needs them), so debug and compact refuse.
    /// Returns false when the mask is not accepted. The race then stays a
    /// full-frame launch.
    pub fn set_frame_rows(&mut self, rows: &[usize]) -> bool {
        if rows.is_empty() || rows.len() > 2 || self.ri[RI_DEBUG_TICK] != 0 {
            return false;
        }
        if rows.iter().any(|&r| r >= self.n) {
            return false;
        }
        self.ri[RI_FRAMES] = rows.len() as u32;
        self.ri[RI_ROW0] = rows[0] as u32;
        self.ri[RI_ROW1] = if rows.len() > 1 { rows[1] as u32 } else { u32::MAX };
        self.frame_rows = rows.to_vec();
        true
    }

    /// The frame row count of this race's launch: the kept rows, or every
    /// runner in a full launch.
    pub(crate) fn frame_row_count(&self) -> usize {
        if self.frame_rows.is_empty() {
            self.n
        } else {
            self.frame_rows.len()
        }
    }

    /// The runner index behind each kept frame row, in row order. Empty for
    /// a full launch.
    pub fn kept_frame_rows(&self) -> &[usize] {
        &self.frame_rows
    }

    /// The course length the kernel ran against, the compact fold's plan
    /// boundary base.
    pub fn course_distance(&self) -> f64 {
        self.pd[PI_DISTANCE]
    }

    /// A flagged placeholder keeping the batch aligned when one request fails
    /// to parse at all. It never reaches the launch.
    pub(crate) fn stub(reason: String) -> Prepared {
        Prepared {
            ri: Vec::new(),
            pd: Vec::new(),
            req: next_req(),
            n: 0,
            stats: Vec::new(),
            host_fallback: Some(reason),
            geometry: None,
            seed: 0,
            sk: Vec::new(),
            bumps: Vec::new(),
            live_guts: false,
            debug_scratch: false,
            grid: false,
            full: false,
            sk_emax: 0,
            sk_tmax: 0,
            frame_rows: Vec::new(),
            req_hash: 0,
            req_text: Arc::from(""),
            derived: None,
        }
    }

    /// The clone of this prepared race for another seed: one `ri` copy with
    /// `RI_SEED` rewritten. The per-runner solver pair keeps the template's,
    /// `reseed_solver_rows` rewrites it where a seed-exact clone needs that.
    /// Only valid for a parsed prepare (a stub has no `ri`).
    pub fn with_seed(&self, seed: i32) -> Prepared {
        let mut clone = self.clone();
        clone.ri[RI_SEED] = seed as u32;
        clone.seed = seed;
        clone
    }

    /// Rewrite the pd solver pair per runner for this race's own seed, the
    /// same draws a fresh parse writes. The pair is the only seed-dependent
    /// pd word, and the kernel spends it only when `PRD_LATER` is set.
    fn reseed_solver_rows(&mut self) {
        let nslopes = self.ri[RI_NSLOPES] as usize;
        for r in 0..self.n {
            let (hi, lo) = construction_solver_pair(self.seed, r, nslopes);
            let base = PD_RUN0 + PD_RUN_STRIDE * r;
            self.pd[base + PRD_R30_HI] = hi as f64;
            self.pd[base + PRD_R30_LO] = lo as f64;
        }
    }
}

/// Column of one runner-row slot inside a `DerivedRow` word row.
const fn prd_col(slot: usize) -> usize {
    let mut i = 0;
    while i < PRD_DERIVED.len() {
        if PRD_DERIVED[i] == slot {
            return i;
        }
        i += 1;
    }
    0
}

/// Derive one request's `DerivedRow` from its first race's inputs and the
/// pre-pass output at `offset`, arithmetic unchanged from the in-place
/// derivation. Slots a fresh zeroed pd would leave untouched are stored 0.0.
fn derive_row(
    race: &Prepared,
    prepass: Option<&initialization::RunOutput>,
    offset: usize,
) -> Result<DerivedRow, String> {
    let count = race.stats.len();
    let mut words = vec![0.0f64; count * PRD_DERIVED.len()];
    let spurt_coef = race.pd[PT_SPURT + 2] as f32;
    let spurt_pow = race.pd[PT_SPURT + 3] as f32;
    let duel_speed = [
        race.pd[PD_CF_SPEED] as f32,
        race.pd[PD_CF_SPEED + 1] as f32,
        race.pd[PD_CF_SPEED + 2] as f32,
    ];
    let duel_accel = [
        race.pd[PD_CF_ACCEL] as f32,
        race.pd[PD_CF_ACCEL + 1] as f32,
        race.pd[PD_CF_ACCEL + 2] as f32,
    ];
    let proficiency = [1.1f64, 1.0, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1];
    for r in 0..count {
        let (adjusted, mut raw_stam, mut raw_pow) = match prepass {
            Some(out) => match &out.outs[offset + r] {
                Out::Stats(s) => (
                    s.adjusted_f64(),
                    f64::from_bits(s.raw_stamina),
                    f64::from_bits(s.raw_power),
                ),
                _ => return Err("stats pre-pass decode".into()),
            },
            None => ([0.; 5], 0., 0.),
        };
        let raw_guts = adjusted[3];
        // The engine's constructor pass chains the static stat bumps as f32
        // accumulators with the field clamped on every read, moves the raw
        // fields in f64 with a max(1.0), and runs the bonus in f64.
        let mut acc_speed = adjusted[0] as f32;
        let mut acc_stam = adjusted[1] as f32;
        let mut acc_pow = adjusted[2] as f32;
        let mut acc_guts = raw_guts as f32;
        let mut acc_wis = adjusted[4] as f32;
        let mut bonus = 0.0f64;
        let mut bump_speed = false;
        let mut bump_stam = false;
        let mut bump_pow = false;
        let mut bump_guts = false;
        let mut bump_wis = false;
        for (_runner, kind, modifier) in race.bumps.iter().filter(|(b, _, _)| *b == r) {
            match *kind {
                1 => {
                    acc_speed += *modifier as f32;
                    bump_speed = true;
                }
                2 => {
                    acc_stam += *modifier as f32;
                    raw_stam = (raw_stam + *modifier).max(1.0);
                    bump_stam = true;
                }
                3 => {
                    acc_pow += *modifier as f32;
                    raw_pow = (raw_pow + *modifier).max(1.0);
                    bump_pow = true;
                }
                4 => {
                    acc_guts += *modifier as f32;
                    bump_guts = true;
                }
                5 => {
                    acc_wis += *modifier as f32;
                    bump_wis = true;
                }
                // One AllStatusUp chains five stats and both raw fields,
                // the engine's :2569-2577 block read straight.
                32 => {
                    acc_speed += *modifier as f32;
                    bump_speed = true;
                    acc_stam += *modifier as f32;
                    raw_stam = (raw_stam + *modifier).max(1.0);
                    bump_stam = true;
                    acc_pow += *modifier as f32;
                    raw_pow = (raw_pow + *modifier).max(1.0);
                    bump_pow = true;
                    acc_guts += *modifier as f32;
                    bump_guts = true;
                    acc_wis += *modifier as f32;
                    bump_wis = true;
                }
                _ => bonus += *modifier / 100.0,
            }
        }
        let guts_eff = if bump_guts { acc_guts.clamp(1.0, 2000.0) as f64 } else { raw_guts };
        let row = &mut words[r * PRD_DERIVED.len()..][..PRD_DERIVED.len()];
        let bounded = (guts_eff as f32).clamp(1., 2000.);
        row[prd_col(PRD_POWF)] = f64::from((bounded * spurt_coef).powf(spurt_pow).to_bits());
        let guts = guts_eff.max(1.0);
        let duel = |c: [f32; 3]| {
            let argument = (guts as f32) * c[0];
            let power = (argument as f64).powf(c[1] as f64) as f32;
            (power * c[2]) as f64
        };
        row[prd_col(PRD_DUEL_T)] = duel(duel_speed);
        row[prd_col(PRD_DUEL_A)] = duel(duel_accel);
        let argument = (guts as f32) * 500.0_f32;
        let power = (argument as f64).powf(0.6_f32 as f64) as f32;
        row[prd_col(PRD_SPOT_M)] = (power * 0.0001_f32) as f64;
        let stapt = race.ri[RI_RUN0 + RI_RUN_STRIDE * r + RR_STAPT] as usize;
        row[prd_col(PRD_SPOT_D)] = (700.0 * guts).powf(0.5) * 0.012 * proficiency[stapt];
        row[prd_col(PRD_SPOT_PROF)] = proficiency[stapt];
        row[prd_col(PRD_BOOST_GUTS)] =
            if bump_guts { acc_guts.clamp(1.0, 2000.0) as f64 } else { 0.0 };
        row[prd_col(PRD_BOOST_WIS)] =
            if bump_wis { acc_wis.clamp(1.0, 2000.0) as f64 } else { 0.0 };
        // The kernel seeds the wisdom accumulator from the unclamped f32
        // chain; NaN marks the field seed for runners with no bump.
        row[prd_col(PRD_WIS_ACC)] = if bump_wis { f64::from(acc_wis) } else { f64::NAN };
        row[prd_col(PRD_GUTS_ACC)] = if bump_guts { f64::from(acc_guts) } else { f64::NAN };
        row[prd_col(PRD_RUSH_BONUS)] = f64::from(((bonus * 100.0) as f32).to_bits());
        row[prd_col(PRD_BOOST_SPEED)] =
            if bump_speed { acc_speed.clamp(1.0, 2000.0) as f64 } else { 0.0 };
        row[prd_col(PRD_SPEED_ACC)] = if bump_speed { f64::from(acc_speed) } else { f64::NAN };
        row[prd_col(PRD_BOOST_STAM)] =
            if bump_stam { acc_stam.clamp(1.0, 2000.0) as f64 } else { 0.0 };
        row[prd_col(PRD_STAM_ACC)] = if bump_stam { f64::from(acc_stam) } else { f64::NAN };
        row[prd_col(PRD_BOOST_POWER)] =
            if bump_pow { acc_pow.clamp(1.0, 2000.0) as f64 } else { 0.0 };
        row[prd_col(PRD_POWER_ACC)] = if bump_pow { f64::from(acc_pow) } else { f64::NAN };
        row[prd_col(PRD_RAW_STAM)] = if bump_stam { raw_stam } else { f64::NAN };
        row[prd_col(PRD_RAW_POWER)] = if bump_pow { raw_pow } else { f64::NAN };
    }
    Ok(DerivedRow { live_guts: if race.live_guts { 1.0 } else { 0.0 }, words })
}

/// Write one request's derived words into a race's pd row, the same slots
/// the removed per-race derivation wrote.
fn apply_derived(race: &mut Prepared, row: &DerivedRow) {
    race.pd[PD_LIVE_GUTS] = row.live_guts;
    let mut at = 0usize;
    for r in 0..race.stats.len() {
        let base = PD_RUN0 + PD_RUN_STRIDE * r;
        for slot in PRD_DERIVED {
            race.pd[base + slot] = row.words[at];
            at += 1;
        }
    }
}

/// The shared solver stream pair left by the engine's construction draws for
/// one runner (the `RaceSolver::new` chain, race_solver.rs:975-1067). The
/// `(hi, lo)` state after the chain. Nothing else in retail runtime draws on it.
fn construction_solver_pair(seed_i32: i32, r: usize, nslopes: usize) -> (u32, u32) {
    let seed_lo = (seed_i32 as i64 + r as i64 * 7919) as u32;
    let mut builder = Rule30CARng::new(seed_lo, ref_race_seed_hi(seed_lo));
    let d1 = builder.int32();
    let _d2 = builder.int32();
    let mut solver = Rule30CARng::new(d1, 0);
    let _hp_seed = solver.int32();
    let _gorosi = solver.int32();
    let _pace = solver.int32();
    let _gate_roll = solver.uniform(12252240.0);
    let _random_lot = solver.uniform(100.0);
    let _start_delay = solver.random();
    for _ in 0..nslopes {
        let _hill_lo = solver.int32();
        let _hill_hi = solver.int32();
    }
    let _kakari = solver.uniform(7.0);
    let _kakari_skip = solver.random();
    let _r1 = solver.random();
    let _r2 = solver.random();
    let _r3 = solver.random();
    for _ in 0..24 {
        let _section = solver.random();
    }
    (solver.hi, solver.lo)
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
    plan_enabled: bool,
) -> Result<Prepared, String> {
    let req_hash = crate::gpu::fnv1a64(request.as_bytes());
    let req_text: Arc<str> = Arc::from(request);
    let root = json::parse(request)?;
    let course_id = root.get_i64("courseId")? as i32;
    let seed_i32 = i32::try_from(seed).map_err(|_| "seed does not fit an i32".to_string())?;
    let horses = root.get_array("horses")?;
    let course = courses.get(&course_id.to_string()).ok_or("course_data.json has no entry for the course")?;

    // Domain gates first, each mapping to a kernel fallback code.
    let flagged = |reason: &str| Ok(Prepared {
        ri: vec![0u32; RI_STRIDE],
        pd: vec![0f64; RIL_STRIDE],
        req: next_req(),
        n: horses.len(),
        bumps: Vec::new(),
        stats: Vec::new(),
        host_fallback: Some(reason.into()),
        geometry: None,
        seed: seed_i32,
        sk: Vec::new(),
        live_guts: false,
        debug_scratch: false,
        grid: false,
        full: false,
        sk_emax: 0,
        sk_tmax: 0,
        frame_rows: Vec::new(),
        req_hash,
        req_text: req_text.clone(),
        derived: None,
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
    // Dense non-grid fields above 24 stack at the lane rail, where the
    // kernel's resolve_overlaps parks runners at rail+0.4/18 while the
    // engine settles at the rail. The band refusal keeps them on the host.
    if !grid && horses.len() > NON_GRID_MAX_RUNNERS {
        return flagged(NON_GRID_BAND_REFUSAL);
    }
    // The engine rejects non-boolean values for the two experimental keys at
    // parse time, ahead of every runtime check (lib.rs:343-349). Quoting its
    // messages keeps a shape it rejects flagged here with byte-equal text.
    if let Some(value) = root.get("experimentalLaterStaminaMechanics") {
        if value.as_bool().is_none() {
            return flagged("experimentalLaterStaminaMechanics must be a boolean.");
        }
    }
    if let Some(value) = root.get("traceConservePower") {
        if value.as_bool().is_none() {
            return flagged("traceConservePower must be a boolean");
        }
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
    // The three later-stamina systems draw from the shared solver stream,
    // whose only host-side use is construction (race_solver.rs:2767-3043).
    // The kernel replays the construction pair and keeps the stream live.
    let later_stamina = match root.get("experimentalLaterStaminaMechanics") {
        None => false,
        Some(value) => match value.as_bool() {
            Some(b) => b,
            None => return flagged("experimentalLaterStaminaMechanics must be a boolean"),
        },
    };
    // singleMode is ported: the +400 lands inside the im_stats clamps
    // (race_solver_builder.rs:304-321). A non-boolean flag is ignored (lib.rs:342).
    let single_mode = root.get("singleMode").and_then(Value::as_bool).unwrap_or(false);
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
    ri[RI_SINGLE] = single_mode as u32;
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

    // The solver stream state after the engine's construction draws, one
    // pair per runner (RaceSolver::new, race_solver.rs:975-1067). Runtime
    // retail spends no other draw on this stream.
    for r in 0..horses.len() {
        let (hi, lo) = construction_solver_pair(seed_i32, r, nslopes);
        let base = PD_RUN0 + PD_RUN_STRIDE * r;
        pd[base + PRD_LATER] = later_stamina as u8 as f64;
        pd[base + PRD_R30_HI] = hi as f64;
        pd[base + PRD_R30_LO] = lo as f64;
    }

    // Dense gate rank drives the retail starting lane.
    let mut gate_order: Vec<usize> = (0..horses.len()).collect();
    gate_order.sort_by_key(|&i| horses[i].get_i64("gateIndex").unwrap_or(0));
    let track = course.get_i64("raceTrackId")? as i32;

    // Hand the skill set to the kernel's numeric table. One unsupported
    // construct flags the whole race rather than approximating the engine.
    let extras = crate::skills_compile::ClassicExtras {
        raw_power: Vec::new(),
        strategy: Vec::new(),
        later_stamina,
    };
    let sk_plan =
        crate::skills_compile::plan_retail_with_plan(&root, course, &extras, plan_enabled);
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
        // The fixed chara id the engine's chara fan compares (lib.rs:336).
        ri[b + RR_CHARA] = horse.get("charaId").and_then(Value::as_f64).unwrap_or(0.0) as i64 as u32;
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
            single_mode,
        });
    }
    Ok(Prepared { ri, pd, req: next_req(), n: horses.len(), stats: stats_cases, host_fallback: None, geometry, seed: seed_i32, sk: sk_plan.words, bumps: sk_plan.bumps, live_guts: sk_plan.live_guts, debug_scratch: false, grid, full: false, sk_emax: sk_plan.emax, sk_tmax: sk_plan.tmax, frame_rows: Vec::new(), req_hash, req_text, derived: None })
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
/// Calls made at the same time from several threads share one launch (see
/// `crate::coalesce`).
pub fn run(requests: &[(String, i64)]) -> Result<RaceRun, String> {
    if crate::coalesce::enabled() {
        return shared_run(requests, false);
    }
    Ok(run_inner(requests, None)?.0)
}

/// One caller's share of a shared launch: its requests and whether they
/// carry the full-reply channels.
struct SharedJob {
    requests: Vec<(String, i64)>,
    full: bool,
}

static SHARED: crate::coalesce::Coalescer<RaceContext, SharedJob, RaceRun> = crate::coalesce::Coalescer::new();

fn shared_run(requests: &[(String, i64)], full: bool) -> Result<RaceRun, String> {
    SHARED.submit(SharedJob { requests: requests.to_vec(), full }, RaceContext::open, shared_batch)
}

/// Prepare every job, launch the jobs that agree on a runner count together
/// (one launch needs one uniform count), and hand each job its own races.
/// A job that cannot share (a prepare error, or mixed counts of its own) gets
/// exactly the answer a call of its own would have.
fn shared_batch(ctx: &mut RaceContext, jobs: Vec<SharedJob>) -> Vec<Result<RaceRun, String>> {
    let mut answers: Vec<Option<Result<RaceRun, String>>> = (0..jobs.len()).map(|_| None).collect();
    let mut groups: Vec<(usize, Vec<(usize, Vec<Prepared>)>)> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        let mut prepared = Vec::with_capacity(job.requests.len());
        let mut failed = None;
        for (request, seed) in &job.requests {
            match ctx.prepare(request, *seed) {
                Ok(mut p) => {
                    if job.full {
                        p.set_full();
                    }
                    prepared.push(p);
                }
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = failed {
            answers[j] = Some(Err(e));
            continue;
        }
        let mut counts: Vec<usize> = prepared.iter().filter(|p| p.host_fallback.is_none()).map(|p| p.n).collect();
        counts.sort_unstable();
        counts.dedup();
        if counts.len() != 1 {
            answers[j] = Some(ctx.run_prepared(&mut prepared, None).map(|r| r.0));
            continue;
        }
        match groups.iter_mut().find(|(n, _)| *n == counts[0]) {
            Some((_, members)) => members.push((j, prepared)),
            None => groups.push((counts[0], vec![(j, prepared)])),
        }
    }
    for (_, members) in groups {
        let lens: Vec<usize> = members.iter().map(|(_, p)| p.len()).collect();
        let mut all: Vec<Prepared> = Vec::with_capacity(lens.iter().sum());
        let mut owners = Vec::with_capacity(members.len());
        for (j, prepared) in members {
            owners.push(j);
            all.extend(prepared);
        }
        match ctx.run_prepared(&mut all, None) {
            Ok((run, _)) => {
                let mut races = run.races.into_iter();
                for (&j, &len) in owners.iter().zip(&lens) {
                    answers[j] = Some(Ok(RaceRun {
                        device: run.device.clone(),
                        build_ms: run.build_ms,
                        stage_ms: run.stage_ms,
                        races: races.by_ref().take(len).collect(),
                    }));
                }
            }
            Err(_) => {
                // Each job alone, so a launch error lands only on the job
                // that causes it.
                let mut rest = all;
                for (&j, &len) in owners.iter().zip(&lens) {
                    let tail = rest.split_off(len);
                    answers[j] = Some(ctx.run_prepared(&mut rest, None).map(|r| r.0));
                    rest = tail;
                }
            }
        }
    }
    answers.into_iter().map(|a| a.expect("every job answered")).collect()
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
    courses: std::collections::HashMap<i32, (Arc<retail_geometry::FlatCourse>, f32)>,
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

    fn course(&mut self, course_id: i32, distance: f32) -> Result<(Arc<retail_geometry::FlatCourse>, f32), String> {
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
        let flat = Arc::new(retail_geometry::flatten_course(&loaded));
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
    /// F7 slice accounting over the batch's chunks: launch count and the
    /// slowest single launch. Zero launches stay counted for `Off` runs.
    pub slices: usize,
    pub slice_max_ms: f64,
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

/// How `LaunchMode::Once` splits a launch into tick ranges (F7, handoff
/// "Keep launches under the Windows timeout"). Windows kills device work
/// past `TdrDelay` (2 s on a stock PC), and the widest CM launches ran 7.5
/// to 8.8 s in one kernel. A slice covers ticks `[start, start + len)` for
/// every lane. Per-race state lives in scratch, which the grow-only lane
/// keeps between launches, so the host relaunches the same buffers at the
/// next range until every lane is done (`ST_DONE`) or flagged. The bench
/// modes never slice, so their launch numbers stay raw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlicePolicy {
    /// One launch per chunk, the pre-F7 behavior.
    Off,
    /// Every slice runs exactly this many ticks (the gate tests' knob).
    Fixed(u32),
    /// Adaptive sizing: the first slice runs `SLICE_FIRST_TICKS`, each next
    /// length scales the previous by `target_ms / elapsed` and clamps to
    /// `SLICE_MIN_TICKS..=RACE_MAX_TICKS - 1`, so launches converge near
    /// `target_ms` and stay a margin under the stock timeout.
    Adaptive { target_ms: f64 },
}

/// The adaptive first slice, in ticks. At the widest CM launch a tick costs
/// about 10 ms of wall time, so 32 ticks is about 0.3 s, a quarter of the
/// stock 2 s `TdrDelay`: the first slice cannot time out on any PC.
pub(crate) const SLICE_FIRST_TICKS: u32 = 32;
/// The adaptive floor: below this many ticks a slice is launch overhead.
pub(crate) const SLICE_MIN_TICKS: u32 = 16;

/// The race kernel's pinned local (work-group) size, the C2 handoff route. One
/// work item is one whole race and lanes share no `__local` memory or barriers,
/// so the value schedules and never changes arithmetic: any value is bit-exact.
/// `0` leaves the choice to the driver (the pre-C2 launch). A positive value
/// pads the 1-D launch to a work-group multiple and the kernel returns the idle
/// tail lanes through `n_lanes`. `GSIM_GPU_LOCAL` overrides it for experiments.
///
/// The B580 sweep (handoff "C2 sweep") pins 8: at the memory-bound full launch
/// (~4,300 races, the app's batch width) a work group of 6 to 8 reaches 670
/// races/s against the driver's 625 (a 7% lift), while 32 and above sit on the
/// driver's number. `bin lws_sweep` re-runs that sweep.
pub const RACE_LOCAL_WORK_SIZE: usize = 8;

/// The race launch local size: the pinned `RACE_LOCAL_WORK_SIZE` unless
/// `GSIM_GPU_LOCAL` names another. Unparsable or negative falls back to pinned.
pub(crate) fn race_local_work_size() -> usize {
    std::env::var("GSIM_GPU_LOCAL")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|&v| v >= 0)
        .map_or(RACE_LOCAL_WORK_SIZE, |v| v as usize)
}
/// The adaptive per-launch target, in milliseconds.
const SLICE_TARGET_MS: f64 = 500.0;

/// The slice policy from the environment. `GSIM_GPU_SLICE=off` runs one
/// launch, `GSIM_GPU_SLICE_TICKS=N` fixes every slice at N ticks,
/// `GSIM_GPU_SLICE_TARGET_MS` moves the adaptive target, and
/// `GSIM_GPU_SLICE_LOG=1` prints each slice's tick range and time.
pub(crate) fn slice_policy_from_env() -> SlicePolicy {
    if std::env::var("GSIM_GPU_SLICE").is_ok_and(|v| v.eq_ignore_ascii_case("off")) {
        return SlicePolicy::Off;
    }
    if let Some(t) = std::env::var("GSIM_GPU_SLICE_TICKS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&t| t > 0)
    {
        return SlicePolicy::Fixed(t);
    }
    let target_ms = std::env::var("GSIM_GPU_SLICE_TARGET_MS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&t| t > 0.)
        .unwrap_or(SLICE_TARGET_MS);
    SlicePolicy::Adaptive { target_ms }
}

impl LaunchMode {
    /// True when the launch is a measurement and frames stay undecoded.
    fn timed(&self) -> bool {
        !matches!(self, LaunchMode::Once)
    }
}

/// Launch buffers for one race lane, bound to one queue. Grow-only: a set
/// is rebuilt only when a launch needs more races, more distinct requests, a
/// wider runner count, wider frame rows, a wider skill table, a wider scratch
/// stride or more reply-channel blocks than its capacity holds.
struct LaneSet {
    cap: usize,
    // Request-table rows: the pd rows and skill tables one launch uploads,
    // one per distinct request rather than one per race (F1).
    reqs: usize,
    n: usize,
    // Skill table stride, the launch's effect-cap bucket.
    skw: usize,
    rows: usize,
    // Scratch strides of the lane's program variant, journal and bucket.
    words: usize,
    srl: usize,
    // Frame window rows, the launch's per-course tick bound.
    ticks: usize,
    queue: Queue,
    b_ri: Buffer<u32>,
    b_pd: Buffer<f64>,
    b_sk: Buffer<u32>,
    b_scr_u: Buffer<u32>,
    // Reply-channel blocks, one per full request plus the zero-filled sink.
    rc_blocks: usize,
    b_rc: Buffer<u32>,
    b_scr_d: Buffer<f64>,
    b_frames: Buffer<f64>,
    b_status: Buffer<u32>,
    b_events: Buffer<u32>,
}

impl LaneSet {
    fn new(
        queue: &Queue, cap: usize, reqs: usize, n: usize, skw: usize, rows: usize,
        words: usize, srl: usize, ticks: usize, rc_blocks: usize,
    ) -> Result<LaneSet, String> {
        Ok(LaneSet {
            cap,
            reqs,
            n,
            skw,
            rows,
            words,
            srl,
            ticks,
            queue: queue.clone(),
            b_ri: empty_buf(queue, cap * RI_STRIDE, flags::MEM_READ_WRITE)?,
            b_pd: empty_buf(queue, reqs * RIL_STRIDE, flags::MEM_READ_WRITE)?,
            b_sk: empty_buf(queue, reqs * skw, flags::MEM_READ_ONLY)?,
            b_scr_u: empty_buf(queue, cap * words, flags::MEM_READ_WRITE)?,
            rc_blocks,
            b_rc: empty_buf(queue, rc_blocks.max(1) * RC_WORDS, flags::MEM_READ_WRITE)?,
            b_scr_d: empty_buf(queue, cap * srl, flags::MEM_READ_WRITE)?,
            b_frames: empty_buf(queue, cap * ticks * rows.max(1) * 6, flags::MEM_READ_WRITE)?,
            b_status: empty_buf(queue, cap * ST_STRIDE, flags::MEM_READ_WRITE)?,
            b_events: empty_buf(queue, cap * EV_STRIDE, flags::MEM_READ_WRITE)?,
        })
    }
}

/// Uploaded geometry tables for one course id list. The list order is the
/// row order `RI_COURSE` indexes, so it is part of the key. Re-registered
/// course geometry empties the cache.
struct TableSet {
    ids: Vec<i32>,
    cs: Buffer<u32>,
    sg_x: Buffer<f64>,
    sg_xd: Buffer<f64>,
    sg_z: Buffer<f64>,
    sg_zd: Buffer<f64>,
    sg_q: Buffer<f32>,
    sg_qd: Buffer<f32>,
    lm_b: Buffer<f32>,
    lm_l: Buffer<f32>,
    ce_b: Buffer<f32>,
    ce_e: Buffer<f32>,
    ce_i: Buffer<u32>,
    ce_f: Buffer<u32>,
}

/// Launch state kept between batches: one gated device, one built race
/// program per effect-cap bucket, one built initialization program, engine
/// data loaded once and the geometry registry (injected app copies ahead of
/// the file fallback). Not Sync, the OpenCL queue and program belong to the
/// thread that opened them.
pub struct RaceContext {
    globals: Value,
    grids: Value,
    courses: Value,
    spike: crate::gpu::Spike,
    init_program: ocl::Program,
    // Race programs by (effect bucket, tag bucket, record bucket, launch
    // shape), built on first use. The top record bucket in the sampling
    // shape is built in open().
    programs: Vec<(usize, usize, usize, RaceShape, ocl::Program)>,
    registry: GeometryRegistry,
    lanes: Vec<LaneSet>,
    table_cache: Vec<TableSet>,
    slice_override: Option<SlicePolicy>,
    // A calibration ceiling on the frame window, from GSIM_GPU_TICK_CAP_LIMIT.
    // Unset keeps the per-course bound. It only tightens the bound, so the
    // overflow fallback still covers any race the ceiling cuts short.
    tick_cap_limit: Option<usize>,
    // SK_FLAG_PLAN marking on the retail scan gate, off only for the gate
    // that proves the plan walk against the general stack walk.
    plan_enabled: bool,
    // The C2 local (work-group) size override. None takes the environment
    // (`GSIM_GPU_LOCAL`, else the pinned default), Some forces a size on this
    // context, the gate test's hook that keeps off the shared process env.
    local_size_override: Option<usize>,
    // Derived stats words per request text this context has launched, newest
    // last (F2). A launch of a known request skips the stats pre-pass.
    stats_cache: Vec<CachedDerived>,
    // Prepared request hosts this context has parsed, newest last (F3). The
    // geometry epoch invalidates them: a re-registered course changes what a
    // prepare of the same text must produce.
    prepared_cache: Vec<CachedPrepared>,
    geom_epoch: u64,
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
        let default_emax = crate::skills_compile::SK_EMAX_BUCKETS[0];
        let default_tmax = crate::skills_compile::SK_TMAX;
        let program = spike.build_race_program_cached(default_emax, default_tmax, SK_RMAX, RaceShape::Sampling)?;
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
        let tick_cap_limit = std::env::var("GSIM_GPU_TICK_CAP_LIMIT").ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|c| c.clamp(TICK_BOUND_MIN, RACE_MAX_TICKS));
        Ok(RaceContext {
            globals,
            grids,
            courses,
            spike,
            init_program,
            programs: vec![(default_emax, default_tmax, SK_RMAX, RaceShape::Sampling, program)],
            registry: GeometryRegistry::default(),
            lanes: Vec::new(),
            table_cache: Vec::new(),
            slice_override: None,
            tick_cap_limit,
            plan_enabled: true,
            local_size_override: None,
            stats_cache: Vec::new(),
            prepared_cache: Vec::new(),
            geom_epoch: 0,
            open_ms: OpenTimings {
                spike_ms,
                race_program_ms,
                init_program_ms,
                data_ms,
                total_ms,
            },
        })
    }

    /// Index of the race program for one define set, built once per variant
    /// on first use.
    fn race_program(
        &mut self, emax: usize, tmax: usize, rcap: usize, shape: RaceShape,
    ) -> Result<usize, String> {
        if let Some(k) = self
            .programs
            .iter()
            .position(|(e, t, r, s, _)| *e == emax && *t == tmax && *r == rcap && *s == shape)
        {
            return Ok(k);
        }
        let started = std::time::Instant::now();
        let built = self.spike.build_race_program_cached(emax, tmax, rcap, shape)?;
        if std::env::var_os("GSIM_GPU_PROGRAM_LOG").is_some() {
            eprintln!(
                "program ready: emax {emax} tmax {tmax} rcap {rcap} {shape:?} {:.1} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        self.programs.push((emax, tmax, rcap, shape, built));
        Ok(self.programs.len() - 1)
    }

    /// The (effect bucket, tag bucket, record bucket, shape) keys of the
    /// programs this context has built, in build order (the gate tests' hook).
    pub fn program_keys(&self) -> Vec<(usize, usize, usize, RaceShape)> {
        self.programs
            .iter()
            .map(|(e, t, r, s, _)| (*e, *t, *r, *s))
            .collect()
    }

    /// The gated device the batches run on.
    pub fn device(&self) -> &crate::gpu::DeviceReport {
        &self.spike.device
    }

    /// Override the F7 slice policy for this context's `Once` launches (the
    /// gate tests' hook). None keeps the environment-resolved default.
    pub fn set_slice_policy(&mut self, policy: Option<SlicePolicy>) {
        self.slice_override = policy;
    }

    /// Switch the SK_FLAG_PLAN marking of prepared retail tables (the gate
    /// tests' hook). Off prepares every record for the general stack walk.
    pub fn set_plan_enabled(&mut self, on: bool) {
        self.plan_enabled = on;
    }

    /// Force the race kernel's local (work-group) size on this context (the
    /// C2 gate test's hook). `None` takes the environment and pinned default.
    /// `0` leaves the work-group size to the driver.
    pub fn set_local_work_size(&mut self, size: Option<usize>) {
        self.local_size_override = size;
    }

    /// Prepare every request and launch once under `policy` (None keeps the
    /// environment default). The gate tests' hook for repeated batches on
    /// one opened context.
    pub fn run_requests(
        &mut self,
        requests: &[(String, i64)],
        policy: Option<SlicePolicy>,
    ) -> Result<RaceRun, String> {
        self.slice_override = policy;
        let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
        for (request, seed) in requests {
            prepared.push(self.prepare(request, *seed)?);
        }
        Ok(self.run_prepared(&mut prepared, None)?.0)
    }

    /// The device's global memory and its largest single allocation, in
    /// bytes, for sizing launches (bin saturate).
    pub fn device_memory(&self) -> Result<(u64, u64), String> {        use ocl::enums::{DeviceInfo, DeviceInfoResult};
        let device = self.spike.queue().device();
        let read = |info| match device.info(info) {
            Ok(DeviceInfoResult::GlobalMemSize(v)) | Ok(DeviceInfoResult::MaxMemAllocSize(v)) => Ok(v),
            Ok(other) => Err(format!("unexpected device info {other:?}")),
            Err(e) => Err(e.to_string()),
        };
        Ok((read(DeviceInfo::GlobalMemSize)?, read(DeviceInfo::MaxMemAllocSize)?))
    }

    /// Register the app's course JSON as the kernel's geometry authority for
    /// the course, ahead of the file fallback (see `GeometryRegistry`).
    pub fn register_geometry(&mut self, course_id: i32, json_text: &str) -> Result<(), String> {
        self.registry.register(course_id, json_text)?;
        // Cached uploads and prepared hosts stale out with the registry
        // content they came from.
        self.table_cache.clear();
        self.prepared_cache.clear();
        self.geom_epoch += 1;
        Ok(())
    }

    /// The derived stats row for one request text, if this context derived it
    /// before (F2). The hash indexes the scan and the text comparison decides.
    fn cached_row(&self, key: u64, text: &str) -> Option<Arc<DerivedRow>> {
        self.stats_cache
            .iter()
            .find(|c| c.key == key && &*c.request == text)
            .map(|c| c.row.clone())
    }

    /// One request into kernel inputs. The Err arm is a parse failure the
    /// batch caller reports per race instead of failing the batch. A text the
    /// context prepared before answers from the prepared cache (F3): the
    /// stored template cloned and reseeded, byte-identical to a fresh parse.
    pub fn prepare(&mut self, request: &str, seed: i64) -> Result<Prepared, String> {
        let seed_i32 = i32::try_from(seed).map_err(|_| "seed does not fit an i32".to_string())?;
        let key = crate::gpu::fnv1a64(request.as_bytes());
        let hit = self.prepared_cache.iter().find(|e| {
            e.key == key
                && e.plan == self.plan_enabled
                && e.epoch == self.geom_epoch
                && e.request == request
        });
        if let Some(entry) = hit {
            let mut out = entry.template.with_seed(seed_i32);
            out.reseed_solver_rows();
            out.derived = self.cached_row(key, request);
            return Ok(out);
        }
        let mut prepared = prepare(request, seed, &self.globals, &self.grids, &self.courses,
            &mut self.registry, self.plan_enabled)?;
        if prepared.host_fallback.is_none() {
            prepared.derived = self.cached_row(prepared.req_hash, &prepared.req_text);
            if self.prepared_cache.len() >= REQUEST_CACHE_MAX {
                self.prepared_cache.remove(0);
            }
            self.prepared_cache.push(CachedPrepared {
                key,
                request: request.to_string(),
                plan: self.plan_enabled,
                epoch: self.geom_epoch,
                template: prepared.clone(),
            });
        }
        Ok(prepared)
    }

    /// Cache occupancy for the gate tests: (prepared entries, derived rows).
    pub fn cache_probe(&self) -> (usize, usize) {
        (self.prepared_cache.len(), self.stats_cache.len())
    }

    /// Grows lane `k`'s buffer set to serve a launch of `mc` races, `n`
    /// runners, a skill table stride of `skw` words, `rows` frame rows, a u32
    /// scratch stride of `words`, a d64 scratch stride of `srl`, a `ticks`
    /// frame window and `rc_blocks` reply-channel blocks. Lanes stay
    /// grow-only, and a set rebuilt for a wider launch keeps its queue, so a
    /// lane's buffers and commands never cross.
    fn lane(
        &mut self, k: usize, mc: usize, reqs: usize, n: usize, skw: usize, rows: usize,
        words: usize, srl: usize, ticks: usize, rc_blocks: usize,
    ) -> Result<(), String> {
        // Fill upward, so a compact first launch still leaves lane 0 holding
        // a full-frame set. The grow-only maxima keep every earlier launch
        // answerable without a rebuild.
        for _ in self.lanes.len()..=k {
            let queue = self.spike.queue();
            self.lanes.push(LaneSet::new(&queue, mc, reqs, n, skw, rows, words, srl, ticks, rc_blocks)?);
        }
        if self.lanes[k].cap >= mc && self.lanes[k].reqs >= reqs && self.lanes[k].n >= n
            && self.lanes[k].skw >= skw && self.lanes[k].rows >= rows
            && self.lanes[k].words >= words && self.lanes[k].srl >= srl && self.lanes[k].ticks >= ticks
            && self.lanes[k].rc_blocks >= rc_blocks
        {
            return Ok(());
        }
        let set = &self.lanes[k];
        let queue = set.queue.clone();
        self.lanes[k] = LaneSet::new(
            &queue, mc, reqs.max(set.reqs), n, skw.max(set.skw),
            rows, words, srl, ticks, rc_blocks,
        )?;
        Ok(())
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
    // A compact launch keeps the same frame row count in every race, so one
    // stride sizes the frame block. The kept runner indices may differ.
    let frows = launched.first().map_or(1, |&i| prepared[i].frame_row_count());
    if launched.iter().any(|&i| prepared[i].frame_row_count() != frows) {
        return Err("one launch needs one uniform frame row count".into());
    }

    // The stats chain runs once per distinct request as an initialization-kernel
    // pre-pass so the host can supply the exact f32 powf. Clones share a request
    // id, and a text the context derived before applies the stored row (F2).
    let t_stage = std::time::Instant::now();
    // Group the launched races by request id: a group is one request row of
    // the launch, so its races share one derived row.
    let mut group_of = vec![usize::MAX; prepared.len()];
    let mut groups: Vec<usize> = Vec::new();
    for &i in &launched {
        let g = match groups.iter().position(|&f| prepared[f].req == prepared[i].req) {
            Some(g) => g,
            None => {
                groups.push(i);
                groups.len() - 1
            }
        };
        group_of[i] = g;
    }
    let mut rows: Vec<Option<Arc<DerivedRow>>> =
        groups.iter().map(|&f| prepared[f].derived.clone()).collect();
    for g in 0..groups.len() {
        if rows[g].is_none() {
            let first = groups[g];
            rows[g] = self.cached_row(prepared[first].req_hash, &prepared[first].req_text);
        }
    }
    let stale: Vec<usize> = (0..groups.len()).filter(|&g| rows[g].is_none()).collect();
    let mut cases: Vec<Case> = Vec::new();
    for &g in &stale {
        for stats in &prepared[groups[g]].stats {
            cases.push(Case::Stats(stats.clone()));
        }
    }
    let prepass = if cases.is_empty() { None } else { Some(initialization::run_with_program(&cases, &self.spike, &self.init_program)?) };
    let mut offset = 0usize;
    for &g in &stale {
        let first = groups[g];
        let count = prepared[first].stats.len();
        let row = Arc::new(derive_row(&prepared[first], prepass.as_ref(), offset)?);
        offset += count;
        // Two stale groups of one text (a batch of same-text requests
        // prepared separately) share the row the first of them stores.
        rows[g] = Some(match self.stats_cache.iter().find(|c| {
            c.key == prepared[first].req_hash && &*c.request == &*prepared[first].req_text
        }) {
            Some(hit) => hit.row.clone(),
            None => {
                if self.stats_cache.len() >= REQUEST_CACHE_MAX {
                    self.stats_cache.remove(0);
                }
                self.stats_cache.push(CachedDerived {
                    key: prepared[first].req_hash,
                    request: prepared[first].req_text.clone(),
                    row: row.clone(),
                });
                row
            }
        });
    }
    for &i in &launched {
        let row = rows[group_of[i]].as_ref().ok_or("missing stats pre-pass row")?;
        apply_derived(&mut prepared[i], row);
    }
    let prepass_ms = t_stage.elapsed().as_secs_f64() * 1e3;

    // One geometry table set shared by every launched race. The Arc clone
    // keeps the Prepared's geometry attached, so a prepared race can be
    // launched again (the prepared cache relaunches a template's clones).
    let t_stage = std::time::Instant::now();
    let mut table_courses: Vec<i32> = Vec::new();
    let mut table_sets: Vec<Arc<retail_geometry::FlatCourse>> = Vec::new();
    for &i in &launched {
        if let Some(course) = prepared[i].geometry.clone() {
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
        table_sets.push(Arc::new(retail_geometry::FlatCourse {
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
        }));
    }
    let table_ids: Vec<i32> = table_sets.iter().map(|c| c.course_id).collect();
    let ti = match self.table_cache.iter().position(|t| t.ids == table_ids) {
        Some(pos) => pos,
        None => {
            let tables = crate::geometry::course_tables(table_sets.iter().map(Arc::as_ref));
            let q = self.spike.queue();
            self.table_cache.push(TableSet {
                ids: table_ids,
                cs: read_only(&q, &tables.cs)?,
                sg_x: read_only(&q, &tables.sg_x)?,
                sg_xd: read_only(&q, &tables.sg_xd)?,
                sg_z: read_only(&q, &tables.sg_z)?,
                sg_zd: read_only(&q, &tables.sg_zd)?,
                sg_q: read_only(&q, &tables.sg_q)?,
                sg_qd: read_only(&q, &tables.sg_qd)?,
                lm_b: read_only(&q, &tables.lm_b)?,
                lm_l: read_only(&q, &tables.lm_l)?,
                ce_b: read_only(&q, &tables.ce_b)?,
                ce_e: read_only(&q, &tables.ce_e)?,
                ce_i: read_only(&q, &tables.ce_i)?,
                ce_f: read_only(&q, &tables.ce_f)?,
            });
            self.table_cache.len() - 1
        }
    };
    let tables_ms = t_stage.elapsed().as_secs_f64() * 1e3;

    let m = launched.len();
    // Frame-byte budget per launch. A full race writes tick_cap * n * 6 f64
    // and a compact launch tick_cap * frows * 6, where tick_cap is the course
    // tick bound (RACE_MAX_TICKS for a debug launch). A big batch splits into
    // several launches; every current suite fits a single one. Debug keeps the
    // full window, so a debug batch budgets on RACE_MAX_TICKS.
    // GSIM_GPU_FRAME_BUDGET_MB overrides the 512 MiB default, so a benchmark
    // can launch as many races at once as device memory holds (bin saturate).
    const FRAME_BYTES_BUDGET: usize = 512 << 20;
    let frame_bytes_budget = std::env::var("GSIM_GPU_FRAME_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .map_or(FRAME_BYTES_BUDGET, |mb| mb << 20);
    let batch_ticks = if launched.iter().any(|&i| prepared[i].debug_scratch) {
        RACE_MAX_TICKS
    } else {
        let bound = launched
            .iter()
            .map(|&i| course_tick_bound(prepared[i].course_distance()))
            .max()
            .unwrap_or(RACE_MAX_TICKS);
        self.tick_cap_limit.map_or(bound, |limit| bound.min(limit))
    };
    let race_frames_bytes = batch_ticks * frows * 6 * std::mem::size_of::<f64>();
    let chunk_len = (frame_bytes_budget / race_frames_bytes.max(1)).max(1);

    // Per-chunk host staging is filled by the readback and drained by the
    // chunk's decode, so no batch pins more frames than its used ticks.
    // Only debug races stage the u32/d64 scratch; full requests read their
    // channels from the rc buffer.
    let want_sk = m > 0 && prepared.iter().any(|p| p.debug_scratch);
    // Grouped reads with one event each and one decoder thread per group
    // pipeline decode against the queue: group k decodes while the device
    // reads group k+1. Two race kernels at once share the B580 poorly.
    let groups = if mode.timed() { 1 } else { (m / 96).clamp(1, 4) };
    let mut decoded: Vec<Option<RaceGpu>> = (0..m).map(|_| None).collect();
    let mut upload_ms = 0.;
    let mut launch_ms = 0.;
    let mut readback_ms = 0.;
    let mut decode_ms = 0.;
    let mut bench_ticks = 0usize;
    let mut bench_launch_ms = 0.;
    let policy = self.slice_override.unwrap_or_else(slice_policy_from_env);
    let mut slice_launches = 0usize;
    let mut slice_max_ms = 0f64;
    let prepared: &[Prepared] = prepared;
    // Races at one (effect, tag) bucket pair share their program's table
    // layout, so every same-pair race launches together. Runs of equal pairs
    // left a heterogeneous batch launching near one race per chunk.
    let mut runs: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (pos, &i) in launched.iter().enumerate() {
        runs.entry((prepared[i].sk_emax, prepared[i].sk_tmax))
            .or_default()
            .push(pos);
    }
    for (bucket, group) in &runs {
        // The bucket pair picks the program's table layout, so the lane's
        // skill buffer and the kernel's sk_all stride both follow it.
        let skw = crate::skills_compile::sk_stride_at(bucket.0, bucket.1);
        for chunk_start in (0..group.len()).step_by(chunk_len) {
            let mc = chunk_len.min(group.len() - chunk_start);
            let chunk: Vec<usize> = group[chunk_start..chunk_start + mc].iter().map(|&p| launched[p]).collect();
            let slot0 = chunk_start;
            let t_stage = std::time::Instant::now();
            let mut ri_all = vec![0u32; mc * RI_STRIDE];
            // F1: the pd row and the skill table belong to the request, so a
            // chunk uploads one row per distinct request and each slot names
            // its row in RI_REQ. The seed clones of one prepare share a row.
            let mut req_row: Vec<usize> = Vec::new();
            for (slot, &i) in chunk.iter().enumerate() {
                ri_all[slot * RI_STRIDE..][..RI_STRIDE].copy_from_slice(&prepared[i].ri);
                let row = match req_row.iter().position(|&r| prepared[r].req == prepared[i].req) {
                    Some(row) => row,
                    None => {
                        req_row.push(i);
                        req_row.len() - 1
                    }
                };
                ri_all[slot * RI_STRIDE + RI_REQ] = row as u32;
            }
            let reqs = req_row.len();
            let mut pd_all = vec![0f64; reqs * RIL_STRIDE];
            for (row, &i) in req_row.iter().enumerate() {
                pd_all[row * RIL_STRIDE..][..RIL_STRIDE].copy_from_slice(&prepared[i].pd);
            }
            // A debug-scratch race reads the journal, so its chunk runs the
            // diag program: journal writes plus the wider scratch stride. Debug
            // stays at the top bucket so the scratch dump keeps the full width.
            let diag = chunk.iter().any(|&i| prepared[i].debug_scratch);
            let full = chunk.iter().any(|&i| prepared[i].full);
            let shape = RaceShape::from_flags(full, diag);
            // Full requests own one rc block each, blocks 1..n. The rest point
            // at block 0, the zero-filled sink they only ever read off. A
            // no-reply launch never touches the block at all.
            let mut rc_blocks = 1usize;
            let mut rc_id = vec![0u32; mc];
            if full {
                for (slot, &i) in chunk.iter().enumerate() {
                    if prepared[i].full {
                        rc_id[slot] = rc_blocks as u32;
                        ri_all[slot * RI_STRIDE + RI_RC] = rc_blocks as u32;
                        rc_blocks += 1;
                    }
                }
            }
            let rcap = if diag {
                SK_RMAX
            } else {
                let nrec = chunk.iter().map(|&i| prepared[i].sk[crate::skills_compile::SK_NREC] as usize).max().unwrap_or(0);
                sk_rcap(nrec)
            };
            let words = scr_u_words(rcap, diag);
            let srl = srl_words(rcap);
            // Debug keeps the full frame window (its tail sits at the last rows).
            // Other launches size it to the longest course in the chunk.
            let bound = chunk.iter().map(|&i| course_tick_bound(prepared[i].course_distance())).max().unwrap_or(RACE_MAX_TICKS);
            let tick_cap = if diag {
                RACE_MAX_TICKS
            } else {
                match self.tick_cap_limit {
                    Some(limit) => bound.min(limit),
                    None => bound,
                }
            };
            // F7 slice ends stay inside the launch's frame window, the point
            // where the kernel stops a still-open race with FB_TICKS.
            let slice_cap = (tick_cap - 1).max(1) as u32;
            // Lane 1 holds compact launches, so switching between the modes does
            // not rebuild the full-frame lane's buffers.
            let lane_k = if frows == n { 0 } else { 1 };
            self.lane(lane_k, mc, reqs, n, skw, frows, words, srl, tick_cap, rc_blocks)?;
            let pk = self.race_program(bucket.0, bucket.1, rcap, shape)?;
            let program = &self.programs[pk].4;
            let lane = &self.lanes[lane_k];
            let tabs = &self.table_cache[ti];
            let queue = &lane.queue;
            let b_ri = &lane.b_ri;
            let b_pd = &lane.b_pd;
            let b_sk = &lane.b_sk;
            let b_scr_u = &lane.b_scr_u;
            let b_rc = &lane.b_rc;
            let b_scr_d = &lane.b_scr_d;
            let b_frames = &lane.b_frames;
            let b_status = &lane.b_status;
            let b_events = &lane.b_events;
            let b_cs = &tabs.cs;
            let b_sg_x = &tabs.sg_x;
            let b_sg_xd = &tabs.sg_xd;
            let b_sg_z = &tabs.sg_z;
            let b_sg_zd = &tabs.sg_zd;
            let b_sg_q = &tabs.sg_q;
            let b_sg_qd = &tabs.sg_qd;
            let b_lm_b = &tabs.lm_b;
            let b_lm_l = &tabs.lm_l;
            let b_ce_b = &tabs.ce_b;
            let b_ce_e = &tabs.ce_e;
            let b_ce_i = &tabs.ce_i;
            let b_ce_f = &tabs.ce_f;
            b_ri.cmd().write(&ri_all).enq().map_err(|e| e.to_string())?;
            b_pd.cmd().write(&pd_all).enq().map_err(|e| e.to_string())?;
            // No host zeros go over PCIe. The device clears each reused buffer
            // the kernel may read before writing, so a lane serves batches of
            // any earlier width. Frames need no clear: the kernel writes every
            // frame word decode reads, debug tail blocks aside.
            b_sk.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
            for (row, &i) in req_row.iter().enumerate() {
                let hw = sk_high_water(&prepared[i].sk);
                if hw > 0 {
                    b_sk.cmd()
                        .write(&prepared[i].sk[..hw])
                        .offset(row * skw)
                        .enq()
                        .map_err(|e| e.to_string())?;
                }
            }
            b_scr_u.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
            if full {
                // The sink block only serves non-full slots of a full launch.
                // A no-reply program never reads rc, so its launch skips the fill.
                b_rc.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
            }
            b_scr_d.cmd().fill(0.0f64, None).enq().map_err(|e| e.to_string())?;
            b_status.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
            b_events.cmd().fill(0u32, None).enq().map_err(|e| e.to_string())?;
            if chunk.iter().any(|&i| prepared[i].ri[RI_DEBUG_TICK] != 0) {
                b_frames.cmd().fill(0.0f64, None).enq().map_err(|e| e.to_string())?;
            }
            upload_ms += t_stage.elapsed().as_secs_f64() * 1e3;

            let t_stage = std::time::Instant::now();
            // The C2 local work size: a forced value pads the 1-D launch to a
            // work-group multiple and passes the real race count as `n_lanes`,
            // so the kernel returns the idle tail lanes unchanged. `0` keeps the
            // pre-C2 launch, global size to race count, driver picks the group.
            let lws = self.local_size_override.unwrap_or_else(race_local_work_size);
            let n_lanes = mc as u32;
            let gws = if lws > 0 { mc.div_ceil(lws) * lws } else { mc };
            let mk_kernel = |resume: u32, slice_end: u32| -> Result<ocl::Kernel, String> {
                let mut kb = ocl::Kernel::builder();
                kb.program(program).name("race").queue(queue.clone()).global_work_size(gws);
                if lws > 0 {
                    kb.local_work_size(lws);
                }
                kb.arg(b_ri)
                    .arg(b_pd)
                    .arg(b_scr_u)
                    .arg(b_rc)
                    .arg(b_scr_d)
                    .arg(b_frames)
                    .arg(b_status)
                    .arg(b_cs)
                    .arg(b_sg_x)
                    .arg(b_sg_xd)
                    .arg(b_sg_z)
                    .arg(b_sg_zd)
                    .arg(b_sg_q)
                    .arg(b_sg_qd)
                    .arg(b_lm_b)
                    .arg(b_lm_l)
                    .arg(b_ce_b)
                    .arg(b_ce_e)
                    .arg(b_ce_i)
                    .arg(b_ce_f)
                    .arg(b_sk)
                    .arg(b_events)
                    .arg(tick_cap as u32)
                    .arg(slice_end)
                    .arg(resume)
                    .arg(n_lanes)
                    .build()
                    .map_err(|e| e.to_string())
            };
            match mode {
                LaunchMode::Timed { warm, timed } => {
                    let kernel = mk_kernel(0, u32::MAX)?;
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
                    let kernel = mk_kernel(0, u32::MAX)?;
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
                LaunchMode::Once => {
                    // F7 slicing: one launch per tick range and a status
                    // readback between ranges. The tick end grows each slice,
                    // and past the tick cap the kernel reports every lane done.
                    let log = std::env::var_os("GSIM_GPU_SLICE_LOG").is_some();
                    let mut probe = vec![0u32; mc * ST_STRIDE];
                    let mut resume = 0u32;
                    let mut slice_start = 0u32;
                    let mut slice_len = match policy {
                        SlicePolicy::Off => u32::MAX,
                        SlicePolicy::Fixed(t) => t,
                        SlicePolicy::Adaptive { .. } => SLICE_FIRST_TICKS.min(slice_cap),
                    };
                    loop {
                        let slice_end = slice_start.saturating_add(slice_len);
                        let kernel = mk_kernel(resume, slice_end)?;
                        let t_slice = std::time::Instant::now();
                        unsafe { kernel.enq().map_err(|e| e.to_string())? };
                        queue.finish().map_err(|e| e.to_string())?;
                        let el_ms = t_slice.elapsed().as_secs_f64() * 1e3;
                        slice_launches += 1;
                        slice_max_ms = slice_max_ms.max(el_ms);
                        if policy == SlicePolicy::Off {
                            break;
                        }
                        if log {
                            eprintln!(
                                "gpu slice {slice_start}..{slice_end} in {el_ms:.1} ms"
                            );
                        }
                        b_status.cmd().read(&mut probe).enq().map_err(|e| e.to_string())?;
                        queue.finish().map_err(|e| e.to_string())?;
                        let unfinished = probe
                            .chunks(ST_STRIDE)
                            .any(|s| s[ST_FALLBACK] == 0 && s[ST_DONE] == 0);
                        if !unfinished {
                            break;
                        }
                        // The slice end has saturated and lanes still report
                        // unfinished: advancing is impossible, so fail loudly
                        // instead of relaunching empty slices forever. A common
                        // cause is a host/kernel ST_STRIDE disagreement.
                        if slice_end == u32::MAX {
                            return Err(
                                "gpu slice loop saturated at u32::MAX with unfinished races".into(),
                            );
                        }
                        let next = match policy {
                            SlicePolicy::Fixed(t) => t,
                            SlicePolicy::Adaptive { target_ms } => {
                                let scaled =
                                    (slice_len as f64) * (target_ms / el_ms.max(1.0));
                                (scaled.round() as i64)
                                    .clamp(SLICE_MIN_TICKS as i64, slice_cap as i64)
                                    as u32
                            }
                            SlicePolicy::Off => break,
                        };
                        slice_start = slice_end;
                        slice_len = next;
                        resume = 1;
                    }
                }
            }
            queue.finish().map_err(|e| e.to_string())?;
            launch_ms += t_stage.elapsed().as_secs_f64() * 1e3;

            let t_stage = std::time::Instant::now();
            if !mode.timed() {
                // Read back only the ticks that ran. Status first, then one frame
                // prefix read per race at its own height, packed side by side, so
                // readback shrinks with the used ticks, not the frame budget.
                let mut c_status = vec![0u32; mc * ST_STRIDE];
                let mut c_events = vec![0u32; mc * EV_STRIDE];
                b_status.cmd().read(&mut c_status).enq().map_err(|e| e.to_string())?;
                b_events.cmd().read(&mut c_events).enq().map_err(|e| e.to_string())?;
                queue.finish().map_err(|e| e.to_string())?;
                let mut frame_off = vec![0usize; mc + 1];
                let mut tail_off = vec![0usize; mc + 1];
                for (slot, &i) in chunk.iter().enumerate() {
                    let used = c_status[slot * ST_STRIDE + ST_TICKS] as usize + 1;
                    frame_off[slot + 1] = frame_off[slot] + used * frows * 6;
                    let dbg = if prepared[i].ri[RI_DEBUG_TICK] != 0 { 14 * n * 6 } else { 0 };
                    tail_off[slot + 1] = tail_off[slot] + dbg;
                }
                let mut c_frames = vec![0f64; frame_off[mc]];
                let mut c_tail = vec![0f64; tail_off[mc]];
                let mut c_skd: Vec<f64> = Vec::new();
                let mut c_sks: Vec<u32> = Vec::new();
                if want_sk && chunk.iter().any(|&i| prepared[i].debug_scratch) {
                    c_skd = vec![0f64; mc * srl];
                    c_sks = vec![0u32; mc * words];
                    b_scr_d.cmd().read(&mut c_skd).enq().map_err(|e| e.to_string())?;
                    b_scr_u.cmd().read(&mut c_sks).enq().map_err(|e| e.to_string())?;
                }
                let mut c_rc: Vec<u32> = Vec::new();
                if rc_blocks > 1 {
                    c_rc = vec![0u32; rc_blocks * RC_WORDS];
                    b_rc.cmd().read(&mut c_rc).enq().map_err(|e| e.to_string())?;
                }
                let gspan = mc.div_ceil(groups.max(1)).max(1);
                let mut sentinel = vec![0u32; 1];
                let mut gevs: Vec<(usize, ocl::Event)> = Vec::new();
                for lo in (0..mc).step_by(gspan) {
                    let hi = (lo + gspan).min(mc);
                    for slot in lo..hi {
                        b_frames.cmd()
                            .read(&mut c_frames[frame_off[slot]..frame_off[slot + 1]])
                            .offset(slot * tick_cap * frows * 6)
                            .enq()
                            .map_err(|e| e.to_string())?;
                        if prepared[chunk[slot]].ri[RI_DEBUG_TICK] != 0 {
                            // The debug blocks sit in the last 14 rows of the race's
                            // frame window, past the used-tick prefix.
                            b_frames.cmd()
                                .read(&mut c_tail[tail_off[slot]..tail_off[slot + 1]])
                                .offset((slot * tick_cap + tick_cap - 14) * n * 6)
                                .enq()
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    // This in-order queue runs the group's reads first, so the
                    // sentinel read closes the group and its event says the data
                    // landed.
                    let mut gev = ocl::Event::empty();
                    b_status.cmd()
                        .read(&mut sentinel)
                        .enew(&mut gev)
                        .enq()
                        .map_err(|e| e.to_string())?;
                    gevs.push((lo, gev));
                }
                let (st, evs) = (&c_status, &c_events);
                let (foff, toff) = (&frame_off, &tail_off);
                let (fr, tl) = (&c_frames, &c_tail);
                let (skd, sks) = (&c_skd, &c_sks);
                let rc_all = &c_rc;
                let rcid = &rc_id;
                let races = prepared;
                let idx = chunk.as_slice();
                let t_start = t_stage;
                let mut slots: Vec<Option<RaceGpu>> = (0..mc).map(|_| None).collect();
                let mut rb_max: f64 = 0.;
                let mut dk_max: f64 = 0.;
                std::thread::scope(|s| -> Result<(), String> {
                    let mut hs = Vec::new();
                    for (gi, part) in slots.chunks_mut(gspan).enumerate() {
                        let (lo, gev) = (&gevs[gi].0, &gevs[gi].1);
                        let lo = *lo;
                        hs.push(s.spawn(move || -> Result<(f64, f64), String> {
                            gev.wait_for().map_err(|e| e.to_string())?;
                            let fire = t_start.elapsed().as_secs_f64() * 1e3;
                            for (j, dest) in part.iter_mut().enumerate() {
                                let slot = lo + j;
                                let rc = if rc_all.is_empty() {
                                    &[][..]
                                } else {
                                    let b = rcid[slot] as usize;
                                    &rc_all[b * RC_WORDS..][..RC_WORDS]
                                };
                                *dest = Some(decode_slot(
                                    &races[idx[slot]], slot, n, frows, words, rcap, srl, rc,
                                    st, evs, foff, toff, fr, tl, skd, sks,
                                ));
                            }
                            Ok((fire, t_start.elapsed().as_secs_f64() * 1e3))
                        }));
                    }
                    for h in hs {
                        let (fire, done) = h
                            .join()
                            .unwrap_or_else(|_| Err("a decode thread panicked".to_string()))?;
                        rb_max = rb_max.max(fire);
                        dk_max = dk_max.max(done - fire);
                    }
                    Ok(())
                })?;
                for (j, d) in slots.into_iter().enumerate() {
                    decoded[group[slot0 + j]] = d;
                }
                readback_ms += rb_max;
                decode_ms += dk_max;
            } else {
                readback_ms += t_stage.elapsed().as_secs_f64() * 1e3;
            }
        }
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

    let mut races: Vec<RaceGpu> = Vec::with_capacity(prepared.len());
    let mut slot = 0usize;
    for race in prepared.iter() {
        let mut out = if race.host_fallback.is_none() && !mode.timed() {
            let d = decoded[slot].take().expect("every launched race decoded in its chunk");
            slot += 1;
            d
        } else {
            RaceGpu {
                frames: Vec::new(),
                ticks: 0,
                fallback: race.host_fallback.clone(),
                grid: race.grid,
                grid_pos: Vec::new(),
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
                later_stats: [0; 5],
                delay_all: Vec::new(),
                lss_all: Vec::new(),
                events: Vec::new(),
                reply: None,
                debug_pace: Vec::new(),
                debug_sk: None,
            }
        };
        if race.grid {
            // The engine ranks its recording set by these construction-time
            // positions (src/race_field.rs:118-125), the key its competition
            // reorder sorts by.
            out.grid_pos = (0..race.n)
                .map(|r| race.pd[PI_RUNNER0 + PI_RUN_STRIDE * r + PI_RUN_POS])
                .collect();
        }
        races.push(out);
    }

    let stage_ms = StageTimings {
        total_ms: t_total.elapsed().as_secs_f64() * 1e3,
        prepass_ms,
        tables_ms,
        upload_ms,
        launch_ms,
        readback_ms,
        decode_ms,
        slices: slice_launches,
        slice_max_ms,
    };
    Ok((RaceRun { device: self.spike.device.clone(), build_ms: self.spike.build_ms, stage_ms, races }, bench))
    }
}

/// Decodes one launched race from its chunk's staging. Frames arrive packed
/// to the used-tick prefix, the offsets in frame_off and tail_off, u32/d64
/// scratch staging only for chunks holding debug races, and `rc` is the
/// race's reply-channel block.
fn decode_slot(
    race: &Prepared,
    slot: usize,
    n: usize,
    rows: usize,
    words: usize,
    rcap: usize,
    srl: usize,
    rc: &[u32],
    status: &[u32],
    events_all: &[u32],
    frame_off: &[usize],
    tail_off: &[usize],
    frames_all: &[f64],
    tail_all: &[f64],
    skd_all: &[f64],
    sks_all: &[u32],
) -> RaceGpu {
    let st = &status[slot * ST_STRIDE..][..ST_STRIDE];
    let ev = &events_all[slot * EV_STRIDE..][..EV_STRIDE];
    let mut out = RaceGpu {
        frames: Vec::new(),
        ticks: 0,
        fallback: None,
        grid: race.grid,
        grid_pos: Vec::new(),
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
        later_stats: [0; 5],
        delay_all: Vec::new(),
        lss_all: Vec::new(),
        events: Vec::new(),
        reply: None,
        debug_pace: Vec::new(),
        debug_sk: None,
    };
    out.kernel_flag = st[ST_FALLBACK];
    out.kernel_arg = st[ST_ARG];
    out.ticks = st[ST_TICKS] as usize;
    out.fallback = kernel_fallback(out.kernel_flag, out.kernel_arg);
    out.loop_end = loop_end(out.ticks);
    if out.fallback.is_none() {
        let base = frame_off[slot];
        let span = (out.ticks + 1) * rows * 6;
        out.frames = frames_all[base..base + span]
            .chunks_exact(rows * 6)
            .map(|tick| {
                tick.chunks_exact(6)
                    .map(|w| {
                        let mut row = [0f64; 6];
                        row.copy_from_slice(w);
                        row
                    })
                    .collect()
            })
            .collect();
        if race.ri[RI_DEBUG_TICK] != 0 {
            let tail = tail_off[slot];
            out.debug_pace = (0..14 * n)
                .map(|row| {
                    let mut v = [0f64; 6];
                    v.copy_from_slice(&tail_all[tail + row * 6..tail + row * 6 + 6]);
                    v
                })
                .collect();
        }
        if race.debug_scratch {
            let d = slot * srl + SRL_SKD;
            let u = slot * words + SCR_SK;
            let j = slot * words + sk_jrn0(rcap);
            let jc = (sks_all[j] as usize).min(SK_JRN_N - 9);
            out.debug_sk = Some(SkillScratch {
                skd: skd_all[d..d + rcap * SKD_STRIDE].to_vec(),
                sks: sks_all[u..u + rcap * SKR_STRIDE].to_vec(),
                journal: sks_all[j + 1..j + 1 + jc].to_vec(),
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
        out.later_stats = [
            st[ST_LATER_KEEP],
            st[ST_LATER_CMP],
            st[ST_LATER_SEC],
            st[ST_LATER_GATE],
            st[ST_LATER_SYB],
        ];
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
        if race.full {
            let channel = |base: usize, cap: usize| -> Vec<u32> {
                let count = (rc[base] as usize).min(cap);
                rc[base + 1..base + 1 + count].to_vec()
            };
            let bits = |word: usize| -> f64 {
                f64::from_bits(
                    u64::from(st[word]) | (u64::from(st[word + 1]) << 32),
                )
            };
            out.reply = Some(ReplyRaw {
                act: channel(SCR_RC_ACT, RC_ACT_CAP),
                tgt: channel(SCR_RC_TGT, RC_TGT_CAP),
                fan: channel(SCR_RC_FAN, RC_FAN_CAP),
                cons: channel(SCR_RC_CONS, RC_CONS_CAP),
                comp: channel(SCR_RC_COMP, RC_COMP_CAP),
                phr: channel(SCR_RC_PHR, RC_PHR_CAP),
                kak: channel(SCR_RC_KAK, RC_KAK_CAP),
                snap_guts: (0..race.n)
                    .map(|r| bits(ST_SNAPG0 + 2 * r))
                    .collect(),
                snap_wis: (0..race.n)
                    .map(|r| bits(ST_SNAPW0 + 2 * r))
                    .collect(),
                spurt_time: (0..race.n)
                    .map(|r| read_f64(st, ST_SPURTT0 + 2 * r))
                    .collect(),
                spurt_dist: (0..race.n)
                    .map(|r| read_f64(st, ST_SPURTD0 + 2 * r))
                    .collect(),
                spfirst: (0..race.n)
                    .map(|r| st[ST_SPFIRST0 + r] as usize)
                    .collect(),
            });
        }
    }
    out
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

/// Prepare every request with the full-reply channels on and launch once.
/// Each race's streams arrive in `RaceGpu::reply` for `full_reply::build`.
pub fn run_full(requests: &[(String, i64)]) -> Result<RaceRun, String> {
    if crate::coalesce::enabled() {
        return shared_run(requests, true);
    }
    let mut ctx = RaceContext::open()?;
    let mut prepared: Vec<Prepared> = Vec::with_capacity(requests.len());
    for (request, seed) in requests {
        let mut p = ctx.prepare(request, *seed)?;
        p.set_full();
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
        (FB_RC, _) => "full-reply channel overflow".into(),
        (FB_TICKS, _) => "the race outgrew the course tick bound".into(),
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

/// A device-side buffer with no host content. The caller clears it with a
/// fill when the kernel could read a word before writing it.
fn empty_buf<T: ocl::OclPrm>(
    queue: &Queue,
    len: usize,
    mem_flags: flags::MemFlags,
) -> Result<Buffer<T>, String> {
    Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(mem_flags)
        .len(len)
        .build()
        .map_err(|e| e.to_string())
}

/// Words through the last non-zero one of a compiled skill row. Everything
/// past it is zero, and the device fill covers it.
fn sk_high_water(row: &[u32]) -> usize {
    let mut end = row.len();
    while end > 0 {
        let lo = end.saturating_sub(4096);
        if row[lo..end].iter().any(|&w| w != 0) {
            return lo + row[lo..end].iter().rposition(|&w| w != 0).unwrap() + 1;
        }
        end = lo;
    }
    0
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











