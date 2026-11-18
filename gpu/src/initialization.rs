//! Host side of the initialization kernel (handoff step 6, initialization
//! slice). Four chains run per work item: adjusted stats, the Rule30 solver
//! seeding chain, the field-wide retail initialization stream (force-in-move,
//! start delays, section modifiers, rushing lotteries), and the retail initial
//! HP. Every chain is pinned bit-exactly by tests/initialization_gate.rs.
//!
//! The reference side here calls the vendored engine modules directly where
//! they exist (retail_initialization, retail_random, retail_physics,
//! course_data, random_gen, retail_sort). The two chains that live inside
//! race_solver_builder.rs and race_solver.rs, which pull in Rc/RefCell graphs
//! the kernel port cannot reuse, are mirrored below and pinned by
//! source_text_untouched_pinned_mirrors in the gate.

use crate::json;
use crate::json::Value;

pub use crate::json::Value as EngineValue;

pub fn engine_file(rel: &str) -> Result<String, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(rel);
    std::fs::read_to_string(path).map_err(|e| e.to_string())
}

pub fn engine_json(rel: &str) -> Result<EngineValue, String> {
    json::parse(&engine_file(rel)?)
}

pub fn engine_parse(text: &str) -> Result<EngineValue, String> {
    json::parse(text)
}

// Kernel layout, mirrors gpu/src/kernels/initialization.cl.
pub const IN_U_STRIDE: usize = 96;
pub const IN_L_STRIDE: usize = 48;
pub const OUT_U_STRIDE: usize = 992;
pub const OUT_L_STRIDE: usize = 64;

pub const MODE_STATS: u32 = 0;
pub const MODE_R30: u32 = 1;
pub const MODE_INIT: u32 = 2;
pub const MODE_HP: u32 = 3;

pub const MAX_RUNNERS: usize = 24;
pub const MAX_SLOPES: usize = 8;

// Input u32 slots.
const US_MODE: usize = 0;
const US_A: usize = 1;
const US_B: usize = 2;
const US_C: usize = 3;
const US_D: usize = 4;
const US_E: usize = 5;
const US_F: usize = 6;
const US_CSS0: usize = 7;
const US_CSS1: usize = 8;
const US_CSS2: usize = 9;
const US_CSS3: usize = 10;
const US_R0: usize = 16;
const IIS_STRIDE: usize = 3;

// Output u32 slots.
pub const OS_BANK: usize = 0;
pub const OS_FLAG: usize = 120;
pub const OS_STAGES: usize = 121;
pub const OS_RUN0: usize = 128;
pub const IRS_STRIDE: usize = 30;
pub const OR_FORCE: usize = 0;
pub const OR_CHANCE: usize = 1;
pub const OR_DELAY: usize = 2;
pub const OR_RCHANCE: usize = 3;
pub const OR_RSECTION: usize = 4;
pub const OR_RPRESENT: usize = 5;
pub const OR_SECTIONS: usize = 6;
pub const O3_D1: usize = 128;
pub const O3_D2: usize = 129;
pub const O3_HP_SEED: usize = 130;
pub const O3_HP_HI: usize = 131;
pub const O3_HP_LO: usize = 132;
pub const O3_GATE_ROLL: usize = 133;
pub const O3_RANDOM_LOT: usize = 134;
pub const O3_KAKARI_U: usize = 135;
pub const O3_KAKARI_IDX: usize = 136;
pub const O3_HILL0_ROLL: usize = 137;
pub const O3_DOWNHILL: usize = 138;
pub const O3_HILLS: usize = 139;
pub const O3_HILL0_HI: usize = 155;
pub const O3_HILL0_LO: usize = 156;
pub const O3_CAPFLAG: usize = 157;
pub const O3_SOLVER_HI: usize = 158;
pub const O3_SOLVER_LO: usize = 159;
pub const OS_CONT_I: usize = 848;
pub const OH_GUTS: usize = 170;
pub const OH_WISDOM: usize = 171;

pub const STAGE_AFTER_TRIGGERS: usize = 0;
pub const STAGE_AFTER_AI: usize = 1;
pub const STAGE_AFTER_WISDOM: usize = 2;
pub const STAGE_AFTER_DELAYS: usize = 3;
pub const STAGE_AFTER_SECTIONS: usize = 4;
pub const STAGE_AFTER_RUSHING: usize = 5;
pub const CONT_DOUBLES_L: usize = 32;

#[derive(Clone, Debug)]
pub struct StatsCase {
    /// Raw request stats in f64, speed, stamina, power, guts, wisdom.
    pub raw: [f64; 5],
    pub mood: i32,
    pub rank: i32,
    pub surface: i32,
    pub strategy_apt: i32,
    pub ground: i32,
    /// course_set_status entries, at most 4.
    pub css: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct R30Case {
    /// The request seed i64 and the runner index. The chain starts at
    /// seed_lo = (seed + runner * 7919) as u32.
    pub seed: i64,
    pub runner: u32,
    /// (start, slope) of every course slope, first slope read for the gate
    /// downhill roll, at most MAX_SLOPES.
    pub slopes: Vec<(f64, f64)>,
    /// The adjusted wisdom in f64, as RaceSolver::new receives it.
    pub wisdom: f64,
}

#[derive(Clone, Debug)]
pub struct InitRunner {
    /// running_style as base_running_style, 1..=4 take a force coefficient.
    pub style: i64,
    /// HorseData base_wisdom as read by number(horse, "base_wisdom").
    pub base_wisdom: f64,
    /// The adjusted solver wisdom as f32, the section-range input.
    pub adjusted_wisdom: f32,
    /// The rushing bonus in percentage points (0.0 without skills).
    pub bonus: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct InitParams {
    /// forceInMoveTargetSpeedAdd{RunningOut,Proceding,Insertion,Chasing}.
    pub force: [f64; 4],
    pub force_span: f64,
    pub delay_max: f64,
    /// BasetTargetSpeed group.
    pub minus_val1: f64,
    pub minus_val2: f64,
    pub plus_val1: f64,
    /// Temptation group.
    pub tempt_val1: f64,
    pub lot_min: i64,
    pub lot_max: i64,
}

#[derive(Clone, Debug)]
pub struct InitCase {
    pub seed: i32,
    pub runners: Vec<InitRunner>,
    pub params: InitParams,
    /// course.distance as f64.
    pub distance: f64,
    /// Some(seed) gives each runner's rushing lottery the (seed, runner,
    /// "RUSH") substream, mirroring initialize_sections_and_rushing.
    pub paired_seed: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct HpPhys {
    pub hp_initial: f64,
    pub hp_style: [f64; 5],
    /// raceBaseSpeedBaseSpeed, raceBaseSpeedBaseDistance, raceBaseSpeedDistanceRate.
    pub course: [f64; 3],
    /// HpGutsCoefSqrt and HpGutsCoef (hp_drain[3], hp_drain[4]).
    pub drain3: f64,
    pub drain4: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct HpCase {
    pub stamina: f64,
    pub guts: f64,
    pub wisdom: f64,
    pub distance: f64,
    pub strategy: i32,
    pub surface: i32,
    pub ground: i32,
    pub phys: HpPhys,
}

#[derive(Clone, Debug)]
pub enum Case {
    Stats(StatsCase),
    R30(R30Case),
    Init(InitCase),
    Hp(HpCase),
}

impl Case {
    pub fn mode(&self) -> u32 {
        match self {
            Case::Stats(_) => MODE_STATS,
            Case::R30(_) => MODE_R30,
            Case::Init(_) => MODE_INIT,
            Case::Hp(_) => MODE_HP,
        }
    }

    pub fn words_u(&self) -> Vec<u32> {
        let mut w = vec![0u32; IN_U_STRIDE];
        w[US_MODE] = self.mode();
        match self {
            Case::Stats(c) => {
                w[US_A] = c.mood as u32;
                w[US_B] = c.rank as u32;
                w[US_C] = c.surface as u32;
                w[US_D] = c.strategy_apt as u32;
                w[US_E] = c.ground as u32;
                w[US_F] = c.css.len() as u32;
                for (i, v) in c.css.iter().take(4).enumerate() {
                    w[US_CSS0 + i] = *v as u32;
                }
            }
            Case::R30(c) => {
                w[US_A] = c.seed as u32;
                w[US_B] = (c.seed >> 32) as u32;
                w[US_C] = c.runner;
                w[US_D] = c.slopes.len() as u32;
            }
            Case::Init(c) => {
                w[US_A] = c.seed as u32;
                w[US_B] = c.runners.len() as u32;
                w[US_C] = c.paired_seed.is_some() as u32;
                w[US_D] = c.paired_seed.unwrap_or(0) as u32;
                w[US_E] = c.params.lot_min as u32;
                w[US_F] = c.params.lot_max as u32;
                for (r, runner) in c.runners.iter().enumerate() {
                    let b = US_R0 + IIS_STRIDE * r;
                    w[b] = runner.style as u32;
                    w[b + 1] = runner.adjusted_wisdom.to_bits();
                    w[b + 2] = runner.bonus.to_bits();
                }
            }
            Case::Hp(c) => {
                w[US_A] = c.strategy as u32;
                w[US_B] = c.surface as u32;
                w[US_C] = c.ground as u32;
            }
        }
        w
    }

    pub fn words_l(&self) -> Vec<u64> {
        let mut w = vec![0u64; IN_L_STRIDE];
        match self {
            Case::Stats(c) => {
                for (i, v) in c.raw.iter().enumerate() {
                    w[i] = v.to_bits();
                }
            }
            Case::R30(c) => {
                w[0] = c.slopes.first().map_or(0.0, |s| s.0).to_bits();
                w[1] = c.slopes.first().map_or(0.0, |s| s.1).to_bits();
                w[2] = c.wisdom.to_bits();
            }
            Case::Init(c) => {
                w[0] = c.distance.to_bits();
                for (r, runner) in c.runners.iter().enumerate() {
                    w[8 + r] = runner.base_wisdom.to_bits();
                }
                w[32] = c.params.force[0].to_bits();
                w[33] = c.params.force[1].to_bits();
                w[34] = c.params.force[2].to_bits();
                w[35] = c.params.force[3].to_bits();
                w[36] = c.params.force_span.to_bits();
                w[37] = c.params.delay_max.to_bits();
                w[38] = c.params.minus_val1.to_bits();
                w[39] = c.params.minus_val2.to_bits();
                w[40] = c.params.plus_val1.to_bits();
                w[41] = c.params.tempt_val1.to_bits();
            }
            Case::Hp(c) => {
                w[0] = c.stamina.to_bits();
                w[1] = c.guts.to_bits();
                w[2] = c.wisdom.to_bits();
                w[3] = c.distance.to_bits();
                w[4] = c.phys.hp_initial.to_bits();
                for (i, v) in c.phys.hp_style.iter().enumerate() {
                    w[5 + i] = v.to_bits();
                }
                for (i, v) in c.phys.course.iter().enumerate() {
                    w[10 + i] = v.to_bits();
                }
                w[13] = c.phys.drain3.to_bits();
                w[14] = c.phys.drain4.to_bits();
            }
        }
        w
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatsOut {
    pub base: [u64; 5],
    pub raw_stamina: u64,
    pub raw_power: u64,
    pub adjusted: [u64; 5],
    pub fallback: u32,
}

impl StatsOut {
    pub fn adjusted_f64(&self) -> [f64; 5] {
        self.adjusted.map(f64::from_bits)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R30Out {
    pub d1: u32,
    pub d2: u32,
    pub hp_seed: u32,
    pub hp_state: (u32, u32),
    pub gate_roll: u32,
    pub random_lot: u32,
    pub kakari_uniform: u32,
    pub kakari_idx: u32,
    pub hill_gate: u32,
    pub downhill: u32,
    /// (lo, hi) seed words of every per-hill Rule30CARng.
    pub hills: Vec<(u32, u32)>,
    pub hill0_post: (u32, u32),
    pub solver_post: (u32, u32),
    pub cap_flag: u32,
    pub start_delay_random: u64,
    pub check_random: u64,
    pub r: [u64; 3],
    pub sections: [u64; 24],
    pub hill0_roll: u64,
    pub fallback: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitRow {
    pub force_in_move: u32,
    pub chance: u32,
    pub start_delay: u32,
    pub rushing_chance: u32,
    pub rushing_section: i32,
    pub rushing_present: bool,
    pub sections: [u32; 24],
    pub kakari_start: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitOut {
    pub stages: [u32; 6],
    pub rows: Vec<InitRow>,
    pub cont_ints: [u32; 8],
    pub cont_doubles: [u64; 8],
    pub fallback: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HpOut {
    pub guts_factor: u32,
    pub wisdom: u32,
    pub ground_modifier: u64,
    pub base_speed: u64,
    pub max_hp: u64,
    pub fallback: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Out {
    Stats(StatsOut),
    R30(R30Out),
    Init(InitOut),
    Hp(HpOut),
}

fn f64_at(l: &[u64], slot: usize) -> f64 {
    f64::from_bits(l[slot])
}

fn decode(case: &Case, u: &[u32], l: &[u64]) -> Out {
    let fallback = u[OS_FLAG];
    match case {
        Case::Stats(_) => Out::Stats(StatsOut {
            base: (0..5).map(|i| l[i]).collect::<Vec<_>>().try_into().unwrap(),
            raw_stamina: l[5],
            raw_power: l[6],
            adjusted: (0..5).map(|i| l[7 + i]).collect::<Vec<_>>().try_into().unwrap(),
            fallback,
        }),
        Case::R30(c) => Out::R30(R30Out {
            d1: u[O3_D1],
            d2: u[O3_D2],
            hp_seed: u[O3_HP_SEED],
            hp_state: (u[O3_HP_HI], u[O3_HP_LO]),
            gate_roll: u[O3_GATE_ROLL],
            random_lot: u[O3_RANDOM_LOT],
            kakari_uniform: u[O3_KAKARI_U],
            kakari_idx: u[O3_KAKARI_IDX],
            hill_gate: u[O3_HILL0_ROLL],
            downhill: u[O3_DOWNHILL],
            hills: (0..c.slopes.len())
                .map(|i| (u[O3_HILLS + 2 * i], u[O3_HILLS + 2 * i + 1]))
                .collect(),
            hill0_post: (u[O3_HILL0_HI], u[O3_HILL0_LO]),
            solver_post: (u[O3_SOLVER_HI], u[O3_SOLVER_LO]),
            cap_flag: u[O3_CAPFLAG],
            start_delay_random: l[0],
            check_random: l[1],
            r: (2..5).map(|i| l[i]).collect::<Vec<_>>().try_into().unwrap(),
            sections: (5..29).map(|i| l[i]).collect::<Vec<_>>().try_into().unwrap(),
            hill0_roll: l[29],
            fallback,
        }),
        Case::Init(c) => Out::Init(InitOut {
            stages: (0..6).map(|i| u[OS_STAGES + i]).collect::<Vec<_>>().try_into().unwrap(),
            rows: (0..c.runners.len())
                .map(|r| InitRow {
                    force_in_move: u[OS_RUN0 + IRS_STRIDE * r + OR_FORCE],
                    chance: u[OS_RUN0 + IRS_STRIDE * r + OR_CHANCE],
                    start_delay: u[OS_RUN0 + IRS_STRIDE * r + OR_DELAY],
                    rushing_chance: u[OS_RUN0 + IRS_STRIDE * r + OR_RCHANCE],
                    rushing_section: u[OS_RUN0 + IRS_STRIDE * r + OR_RSECTION] as i32,
                    rushing_present: u[OS_RUN0 + IRS_STRIDE * r + OR_RPRESENT] != 0,
                    sections: (0..24)
                        .map(|s| u[OS_RUN0 + IRS_STRIDE * r + OR_SECTIONS + s])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    kakari_start: l[r],
                })
                .collect(),
            cont_ints: (0..8).map(|i| u[OS_CONT_I + i]).collect::<Vec<_>>().try_into().unwrap(),
            cont_doubles: (0..8)
                .map(|i| l[CONT_DOUBLES_L + i])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            fallback,
        }),
        Case::Hp(_) => Out::Hp(HpOut {
            guts_factor: u[OH_GUTS],
            wisdom: u[OH_WISDOM],
            ground_modifier: l[0],
            base_speed: l[1],
            max_hp: l[2],
            fallback,
        }),
    }
}


pub struct RunOutput {
    pub outs: Vec<Out>,
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
}

#[cfg(feature = "gpu")]
fn buffer<T: ocl::core::OclPrm>(queue: &ocl::Queue, data: &[T]) -> Result<ocl::Buffer<T>, String> {
    let b = ocl::Buffer::<T>::builder()
        .queue(queue.clone())
        .len(data.len())
        .build()
        .map_err(|e| e.to_string())?;
    b.cmd().write(data).enq().map_err(|e| e.to_string())?;
    Ok(b)
}

/// Runs every case through the initialization kernel on the gated device.
#[cfg(feature = "gpu")]
pub fn run(cases: &[Case]) -> Result<RunOutput, String> {
    let spike = crate::gpu::Spike::open()?;
    let program = spike.build_program(crate::gpu::INITIALIZATION_KERNEL_SOURCE)?;
    let queue = spike.queue();
    let in_u: Vec<u32> = cases.iter().flat_map(|c| c.words_u()).collect();
    let in_l: Vec<u64> = cases.iter().flat_map(|c| c.words_l()).collect();
    let b_in_u = buffer(&queue, &in_u)?;
    let b_in_l = buffer(&queue, &in_l)?;
    let b_out_u = ocl::Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(cases.len() * OUT_U_STRIDE)
        .build()
        .map_err(|e| e.to_string())?;
    let b_out_l = ocl::Buffer::<u64>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(cases.len() * OUT_L_STRIDE)
        .build()
        .map_err(|e| e.to_string())?;
    let kernel = ocl::Kernel::builder()
        .program(&program)
        .name("initialization")
        .queue(queue.clone())
        .global_work_size(cases.len())
        .arg(&b_in_u)
        .arg(&b_in_l)
        .arg(&b_out_u)
        .arg(&b_out_l)
        .build()
        .map_err(|e| e.to_string())?;
    unsafe { kernel.enq().map_err(|e| e.to_string())? };
    queue.finish().map_err(|e| e.to_string())?;
    let mut out_u = vec![0u32; cases.len() * OUT_U_STRIDE];
    let mut out_l = vec![0u64; cases.len() * OUT_L_STRIDE];
    b_out_u.cmd().read(&mut out_u).enq().map_err(|e| e.to_string())?;
    b_out_l.cmd().read(&mut out_l).enq().map_err(|e| e.to_string())?;
    let outs = cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            decode(
                c,
                &out_u[i * OUT_U_STRIDE..(i + 1) * OUT_U_STRIDE],
                &out_l[i * OUT_L_STRIDE..(i + 1) * OUT_L_STRIDE],
            )
        })
        .collect();
    Ok(RunOutput { outs, device: spike.device, build_ms: spike.build_ms })
}


// Mirrors of the pure race_solver_builder.rs helpers (source-pinned by the
// gate): adjust_overcap, team_rank_bonus and ranked_base_stat.
fn ref_adjust_overcap(stat: f64) -> f64 {
    if stat > 1200.0 {
        1200.0 + (stat - 1200.0) / 2.0
    } else {
        stat
    }
}

fn ref_team_rank_bonus(rank: i32) -> f32 {
    if rank <= 0 {
        return 0.0;
    }
    for (threshold, bonus) in [(6, 50.0f32), (11, 30.0), (16, 20.0), (21, 10.0), (26, 0.0), (99999, 0.0)]
    {
        if rank < threshold {
            return bonus;
        }
    }
    0.0
}

fn ref_ranked_base_stat(raw: f64, mood: i32, rank: i32) -> f32 {
    let base = ref_adjust_overcap(raw) as f32 + ref_team_rank_bonus(rank);
    crate::retail_initialization::initial_base_wisdom(base as f64, mood)
}

/// The adjusted-stats chain over the vendored initial_base_wisdom and
/// course_speed_modifier, mirroring build_base_stats_with_team_rank and
/// build_adjusted_stats_for_mode with single_mode=false.
pub fn stats_reference(c: &StatsCase) -> StatsOut {
    let motiv = 1.0 + 0.02 * c.mood as f64;
    let base = [
        ref_ranked_base_stat(c.raw[0], c.mood, c.rank) as f64,
        ref_ranked_base_stat(c.raw[1], c.mood, c.rank) as f64,
        ref_ranked_base_stat(c.raw[2], c.mood, c.rank) as f64,
        ref_ranked_base_stat(c.raw[3], c.mood, c.rank) as f64,
        ref_ranked_base_stat(c.raw[4], c.mood, c.rank) as f64,
    ];
    let course = crate::course_data::CourseData {
        race_track_id: 0,
        distance: 0.0,
        distance_type: 0,
        surface: c.surface,
        turn: 0,
        course_set_status: c.css.clone(),
        corners: Vec::new(),
        straights: Vec::new(),
        slopes: Vec::new(),
        max_lane: 0.0,
        first_move_lane_point: 0.0,
        empty_in_move: false,
    };
    let race_course_modifier = crate::course_data::course_speed_modifier(
        &course, base[0], base[1], base[2], base[3], base[4],
    );
    let gspd = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -50.0],
        [0.0, 0.0, 0.0, 0.0, -50.0],
    ];
    let gpow = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -50.0, -50.0, -50.0],
        [0.0, -100.0, -50.0, -100.0, -100.0],
    ];
    let clamp1 = |v: f64| v.clamp(1.0, 2000.0);
    let adjusted = [
        ((base[0] as f32 as f64 * race_course_modifier as f32 as f64
            + gspd[c.surface as usize][c.ground as usize]) as f32)
            .clamp(1.0, 2000.0) as f64,
        clamp1(base[1]),
        clamp1(base[2] + gpow[c.surface as usize][c.ground as usize]),
        clamp1(base[3]),
        ((base[4] as f32 as f64
            * ([11000.0f32, 10000.0, 8500.0, 7500.0, 6000.0, 4000.0, 2000.0, 1000.0]
                [c.strategy_apt as usize]
                * 0.0001_f32) as f64) as f32)
            .clamp(1.0, 2000.0) as f64,
    ];
    StatsOut {
        base: base.map(|v| v.to_bits()),
        raw_stamina: (c.raw[1] * motiv).to_bits(),
        raw_power: (base[2] + (c.raw[2] - 1200.0).max(0.0) * 0.5).to_bits(),
        adjusted: adjusted.map(|v| v.to_bits()),
        fallback: 0,
    }
}

/// Mirrors of wisdom_seed and race_seed_hi from race_solver_builder.rs (the
/// module pulls in the whole skill graph and cannot be vendored), pinned to
/// the engine source by the gate.
fn ref_race_seed_hi(horse_seed: u32) -> u32 {
    let mut seed = horse_seed ^ 0x52414345 ^ 0x57495344;
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x85ebca6b);
    seed = (seed ^ (seed >> 13)).wrapping_mul(0xc2b2ae35);
    seed ^= seed >> 16;
    if seed == 0 { 1 } else { seed }
}

/// The seeding chain of one solver: the builder draw pair, the hp seed draw,
/// then the rng-consuming part of RaceSolver::new (gate roll, random lot,
/// start-delay random, hill seeds, gate hill roll, kakari and section draws).
pub fn r30_reference(c: &R30Case) -> R30Out {
    let seed_lo = (c.seed + c.runner as i64 * 7919) as u32;
    let mut builder = crate::random_gen::Rule30CARng::new(seed_lo, ref_race_seed_hi(seed_lo));
    let d1 = builder.int32();
    let d2 = builder.int32();

    let mut solver = crate::random_gen::Rule30CARng::new(d1, 0);
    let hp_seed = solver.int32();
    let mut cap_flag = 0u32;
    let (gate_roll, flag) = crate::rng::uniform_guarded(&mut solver, 12252240.0, crate::rng::UNIFORM_CAP);
    cap_flag |= flag;
    let (random_lot, flag) = crate::rng::uniform_guarded(&mut solver, 100.0, crate::rng::UNIFORM_CAP);
    cap_flag |= flag;
    let start_delay_random = 0.1 * solver.random();

    let hills: Vec<(u32, u32)> = (0..c.slopes.len())
        .map(|_| {
            let lo = solver.int32();
            let hi = solver.int32();
            (lo, hi)
        })
        .collect();
    let mut hill_gate = 0u32;
    let mut downhill = 0u32;
    let mut hill0_post = (0u32, 0u32);
    let mut hill0_roll = 0.0f64;
    if let (Some(&(start, per)), Some(&(lo, hi))) = (c.slopes.first(), hills.first()) {
        if start == 0.0 {
            let mut h0 = crate::random_gen::Rule30CARng::new(lo, hi);
            let roll = h0.random();
            hill0_roll = roll;
            hill0_post = (h0.hi, h0.lo);
            if per < 0.0 && roll < c.wisdom * 0.0004 {
                downhill = 1;
            }
            hill_gate = 1;
        }
    }

    let (kakari_uniform, flag) = crate::rng::uniform_guarded(&mut solver, 7.0, crate::rng::UNIFORM_CAP);
    cap_flag |= flag;
    let check_random = solver.random();
    let r1 = solver.random();
    let r2 = solver.random();
    let r3 = solver.random();
    let seq = [0.0, r1, r2, r3, 1.0];
    let kakari_idx = seq.iter().position(|&x| x > 0.45).unwrap() as u32;
    let sections: [u64; 24] = (0..24)
        .map(|_| solver.random().to_bits())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let (solver_hi, solver_lo) = (solver.hi, solver.lo);

    R30Out {
        d1,
        d2,
        hp_seed,
        hp_state: (0, hp_seed),
        gate_roll: gate_roll as u32,
        random_lot: random_lot as u32,
        kakari_uniform: kakari_uniform as u32,
        kakari_idx,
        hill_gate,
        downhill,
        hills,
        hill0_post,
        solver_post: (solver_hi, solver_lo),
        cap_flag,
        start_delay_random: start_delay_random.to_bits(),
        check_random: check_random.to_bits(),
        r: [r1.to_bits(), r2.to_bits(), r3.to_bits()],
        sections,
        hill0_roll: hill0_roll.to_bits(),
        fallback: 0,
    }
}


pub fn engine_parameters_value() -> Result<Value, String> {
    engine_json("data/global_parameters.json")
}

fn init_horse_value(r: &InitRunner) -> Value {
    json::obj(vec![
        ("skills", Value::Array(Vec::new())),
        ("running_style", json::num(r.style as f64)),
        ("base_wisdom", json::num(r.base_wisdom)),
    ])
}

fn init_params_value(p: &InitParams) -> Value {
    json::obj(vec![
        ("forceInMoveTargetSpeedAddRunningOut", json::num(p.force[0])),
        ("forceInMoveTargetSpeedAddProceding", json::num(p.force[1])),
        ("forceInMoveTargetSpeedAddInsertion", json::num(p.force[2])),
        ("forceInMoveTargetSpeedAddChasing", json::num(p.force[3])),
        ("forceInMoveTargetSpeedAddRandomRange", json::num(p.force_span)),
        ("startDelayMax", json::num(p.delay_max)),
        (
            "BasetTargetSpeed",
            json::obj(vec![
                ("BaseTargetSpeedRandomMinusVal1", json::num(p.minus_val1)),
                ("BaseTargetSpeedRandomMinusVal2", json::num(p.minus_val2)),
                ("BaseTargetSpeedRandomPlusVal1", json::num(p.plus_val1)),
            ]),
        ),
        (
            "Temptation",
            json::obj(vec![
                ("TemptationStartPerVal1", json::num(p.tempt_val1)),
                ("TemptationLotSectionMin", json::num(p.lot_min as f64)),
                ("TemptationLotSectionMax", json::num(p.lot_max as f64)),
            ]),
        ),
    ])
}

/// The field-wide retail initialization for one no-skill field, run through
/// the vendored engine functions themselves: initialize then
/// initialize_sections_and_rushing, the same pair run_race calls.
pub fn init_reference(c: &InitCase) -> Result<InitOut, String> {
    let horses: Vec<Value> = c.runners.iter().map(init_horse_value).collect();
    let course = json::obj(vec![("distance", json::num(c.distance))]);
    let skills = json::obj(Vec::new());
    let parameters = init_params_value(&c.params);
    let mut init = crate::retail_initialization::initialize(
        c.seed,
        &horses,
        &course,
        &skills,
        &parameters,
        false,
    )?;
    let pairs: Vec<(f32, f32)> = c
        .runners
        .iter()
        .map(|r| (r.adjusted_wisdom, r.bonus))
        .collect();
    crate::retail_initialization::initialize_sections_and_rushing(
        &mut init,
        &pairs,
        &course,
        &parameters,
        c.paired_seed,
    )?;
    let stages = ["after_triggers", "after_ai", "after_wisdom", "after_delays", "after_sections", "after_rushing"];
    let stage = |i: usize| -> u32 {
        *init.stages.get(stages[i]).expect("initialize inserts every stage") as u32
    };
    let distance_f32 = c.distance as f32;
    let rows: Vec<InitRow> = init
        .horses
        .iter()
        .map(|row| {
            let sections: [u32; 24] = row
                .section_modifiers
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                .try_into()
                .expect("initialize_sections_and_rushing fills 24");
            let kakari_start = row
                .rushing_section
                .map(|s| (s - 1) as f32 * (distance_f32 / 24.0))
                .unwrap_or(distance_f32 + 9999.0) as f64;
            InitRow {
                force_in_move: row.force_in_move_coefficient.to_bits(),
                chance: 0,
                start_delay: row.start_delay.to_bits(),
                rushing_chance: row.rushing_chance.to_bits(),
                rushing_section: row.rushing_section.unwrap_or(0),
                rushing_present: row.rushing_section.is_some(),
                sections,
                kakari_start: kakari_start.to_bits(),
            }
        })
        .collect();
    // chance is recomputed the way initialize does, the engine row keeps only
    // the skill_enabled flags which a skill-free field never fills.
    let rows = rows
        .into_iter()
        .zip(c.runners.iter())
        .map(|(mut row, r)| {
            let chance = (100.0_f32 - 9000.0_f32 / (r.base_wisdom as f32)).max(20.0);
            row.chance = chance.to_bits();
            row
        })
        .collect();
    let mut cont_ints = [0u32; 8];
    for v in cont_ints.iter_mut() {
        *v = init.rng.next_int() as u32;
    }
    let mut cont_doubles = [0u64; 8];
    for v in cont_doubles.iter_mut() {
        *v = init.rng.next_double().to_bits();
    }
    Ok(InitOut {
        stages: [stage(0), stage(1), stage(2), stage(3), stage(4), stage(5)],
        rows,
        cont_ints,
        cont_doubles,
        fallback: 0,
    })
}


/// HP_CONSUMPTION_GROUND_MODIFIER from hp_policy.rs (a private const), pinned
/// to the engine source by the gate.
const HP_GROUND_MODIFIER: [[f64; 5]; 3] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.02, 1.02],
    [0.0, 1.0, 1.0, 1.01, 1.02],
];

/// The retail initial HP chain: Parameters::from_value over the case physics,
/// then initial_hp, course_speed and hp_guts_factor, mirroring
/// GameHpState::with_retail + init.
pub fn hp_reference(c: &HpCase) -> Result<HpOut, String> {
    let mut parameters = engine_parameters_value()?
        .get("parameters").ok_or("Missing retail parameters")?.clone();
    {
        let Value::Object(map) = &mut parameters else {
            return Err("Retail parameters root is not an object".into());
        };
        map.insert("raceBaseSpeedBaseSpeed".into(), json::num(c.phys.course[0]));
        map.insert("raceBaseSpeedBaseDistance".into(), json::num(c.phys.course[1]));
        map.insert("raceBaseSpeedDistanceRate".into(), json::num(c.phys.course[2]));
        let Value::Object(hp) = map
            .get_mut("Hp").ok_or("Missing Hp group")?
        else {
            return Err("Hp group is not an object".into());
        };
        hp.insert("HpInitialVal1".into(), json::num(c.phys.hp_initial));
        let names = ["HpMaxCoefNige", "HpMaxCoefSenko", "HpMaxCoefSashi", "HpMaxCoefOikomi", "HpMaxCoefOoNige"];
        for (i, name) in names.iter().enumerate() {
            hp.insert((*name).into(), json::num(c.phys.hp_style[i]));
        }
        hp.insert("HpGutsCoefSqrt".into(), json::num(c.phys.drain3));
        hp.insert("HpGutsCoef".into(), json::num(c.phys.drain4));
    }
    let physics = crate::retail_physics::Parameters::from_value(&parameters)?;
    let stamina = c.stamina as f32;
    let distance = c.distance as f32;
    let max_hp = physics.initial_hp(stamina, c.strategy as usize, distance) as f64;
    let base_speed = physics.course_speed(distance) as f64;
    let guts_factor = physics.hp_guts_factor(c.guts as f32);
    Ok(HpOut {
        guts_factor: guts_factor.to_bits(),
        wisdom: (c.wisdom as f32).to_bits(),
        ground_modifier: HP_GROUND_MODIFIER[c.surface as usize][c.ground as usize].to_bits(),
        base_speed: base_speed.to_bits(),
        max_hp: max_hp.to_bits(),
        fallback: 0,
    })
}


/// parameter_orders from lib.rs: bounded adjusted guts and wisdom ranked by
/// the loaded-framework introsort, tolerance-ranked with gate-index ties.
pub fn order_reference(adjusted: &[[f64; 5]]) -> (Vec<usize>, Vec<usize>) {
    let rank = |stat: usize| -> Vec<usize> {
        let mut order: Vec<usize> = (0..adjusted.len()).collect();
        let value = |i: usize| crate::retail_physics::bounded_stat(adjusted[i][stat] as f32);
        crate::retail_sort::comparison_sort(&mut order, |&a, &b| {
            let (left, right) = (value(a), value(b));
            let tolerance = (1e-6_f32 * left.abs().max(right.abs())).max(8.0 * f32::from_bits(1));
            if (left - right).abs() < tolerance {
                a.cmp(&b)
            } else {
                right.total_cmp(&left)
            }
        });
        let mut ranks = vec![0usize; order.len()];
        for (rank, i) in order.into_iter().enumerate() {
            ranks[i] = rank;
        }
        ranks
    };
    // adjusted layout: speed, stamina, power, guts, wisdom.
    (rank(3), rank(4))
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V77Pin {
    pub seed: i64,
    /// startDelayTime as the reply f64 (f32 start_delay widened), per horse.
    pub start_delay: Vec<u64>,
    pub guts_order: Vec<usize>,
    pub wiz_order: Vec<usize>,
    /// The tick-0 hp value per horse, the initial HP widened to the f64 frame.
    pub hp0: Vec<u64>,
}

/// Drives the real engine over the v77 request line with the given seeds and
/// returns the reply-visible initialization values.
pub fn v77_reference(request: &str, seeds: &[i64]) -> Result<Vec<V77Pin>, String> {
    let mut pins = Vec::new();
    for &seed in seeds {
        let request = {
            let mut value = json::parse(request)?;
            let Value::Object(map) = &mut value else {
                return Err("request is not an object".into());
            };
            map.insert("seed".into(), json::num(seed as f64));
            map.insert("retainAllFrames".into(), Value::Bool(true));
            json::to_string(&value)
        };
        let reply = {
            let mut prepared = globalsim::PreparedRace::new(&request)?;
            let simulation = prepared.run(seed)?;
            let hp0 = (0..simulation.frames().horse_count())
                .map(|r| simulation.frames().frame(0).1[r][2].to_bits())
                .collect::<Vec<_>>();
            let mut text = String::new();
            simulation.write_json(&mut text);
            (json::parse(&text)?, hp0)
        };
        let (reply, hp0) = reply;
        let results = reply.get_array("results").map_err(|_| "reply has no results")?;
        let start_delay = results
            .iter()
            .map(|r| r.get_f64("startDelayTime").map(f64::to_bits))
            .collect::<Result<Vec<_>, _>>()?;
        let guts_order = results
            .iter()
            .map(|r| r.get_i64("gutsOrder").map(|v| v as usize))
            .collect::<Result<Vec<_>, String>>()?;
        let wiz_order = results
            .iter()
            .map(|r| r.get_i64("wizOrder").map(|v| v as usize))
            .collect::<Result<Vec<_>, String>>()?;
        pins.push(V77Pin { seed, start_delay, guts_order, wiz_order, hp0 });
    }
    Ok(pins)
}

