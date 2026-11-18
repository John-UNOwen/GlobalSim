// Retail runtime core shared by the race kernel (handoff step 6, race loop).
// Ports, line for line, the no-skill call surface the race tick loop reaches:
// src/retail_physics.rs (bounded_stat, root, course_speed, base_target_speed,
// minimum_speed, blocking_speed_cap, maximum_spurt_speed, the target and accel
// chains, the HP chain, advance_start_delay, finish_time, phase helpers),
// src/retail_sort.rs (comparison_sort), src/retail_spurt.rs (choose and
// SpurtState), src/retail_states.rs (SlopeState, RushingState, DownhillState),
// src/retail_position_keep.rs (State::update) and the retail branch of
// src/hp_policy.rs (GameHpState ticks). rng_core.cl and initialization.cl are
// prepended by the host; initialization.cl already carries the bounded-stat
// clamps and engine_log10.
//
// Exactness per docs/GPU_HANDOFF.md: both FP_CONTRACT OFF pragmas and the
// build flag live in the prepended files; every width follows the Rust
// source, and every as-f32 store is one convert_float_rte.
//
// maximum_spurt_speed is the only powf in the no-skill loop. Its inputs are
// race-constant without skills, so the host precomputes the exact f32 powf
// result with the vendored engine and passes it in (slot PT_POWF). The kernel
// never evaluates a transcendental power.

// Course and runner caps. The catalog's largest field is 18; the
// initialization kernel's cap is 24, so 24 is the binding layout bound.
#define RACE_MAXR 24
// 400 s of race time at dt=0.0666 is under 6007 ticks.
#define RACE_MAX_TICKS 6144
// Slopes per course, from course_data.json entries.
#define RACE_MAX_SLOPES 16

// ---------------------------------------------------------------------------
// Race input, u32 words per race, stride RI_STRIDE.

#define RI_STRIDE 320
#define RI_SEED 0            // i32 request seed (RetailRandom seed)
#define RI_N 1               // runners
#define RI_COURSE 2          // geometry table index
#define RI_GROUND 3          // 1..4
#define RI_SURFACE 4         // 1..2
#define RI_PAIRED 5          // 1 = pairedRandomStreams (fallback, unsupported)
#define RI_STARTUP 6         // 1 = experimentalRetailStartup (fallback)
#define RI_GEOM_ON 7         // 1 = experimentalRetailGeometry
#define RI_HOST_FLAG 8       // nonzero: the host already knows this race needs
                             // a fallback (skills, paired streams, no geometry)
#define RI_HOST_ARG 9        // which domain the host flag stands for
#define RI_NSLOPES 10
#define RI_NSTR 11           // straights present
#define RI_NCOR 12           // corners present
#define RI_LOT_MIN 13
#define RI_LOT_MAX 14
#define RI_EMPTY_MOVE 15       // course empty_in_move (first_move_lane_point > 0)
// Per-runner block, slot RI_RUN0 + RI_RUN_STRIDE * r.
#define RI_RUN0 16
#define RI_RUN_STRIDE 12
#define RR_STYLE 0           // base_running_style (1..5)
#define RR_STRATEGY 1        // resolved strategy (1..5)
#define RR_DISTAPT 2         // distance aptitude 0..7
#define RR_MOOD 3
#define RR_RANK 4            // team_rank
#define RR_CSS0 5            // up to four course_set_status entries
#define RR_CSS1 6
#define RR_CSS2 7
#define RR_CSS3 8
#define RR_CSSN 9
#define RR_STAPT 10          // strategy aptitude 0..7
#define RR_SUAPT 11          // surface aptitude 0..7

// ---------------------------------------------------------------------------
// Race input, f64 words (raw bits) per race, stride RIL_STRIDE. Float
// parameters are stored as the raw JSON f64 and narrowed once in the kernel,
// mirroring Parameters::from_value's `get_f64 as f32`.

#define RIL_STRIDE 448
// 0..83: retail physics Parameters, the layout the retail_physics port reads.
#define PT_COURSE0 0
#define PT_COURSE1 1
#define PT_COURSE2 2
#define PT_PHASE_SPEED 3     // [5][4], strategy-major then Start,Middle,End,Last
#define PT_PHASE_ACCEL 23    // [5][4]
#define PT_END_SQRT 43
#define PT_ADD_SPEED 44
#define PT_MINIMUM 45        // [3]
#define PT_BLOCKING 48       // [3]
#define PT_SPURT 51          // [6]
#define PT_SLOPE 57          // [3]
#define PT_TARGET_MIN 60
#define PT_ACCEL 61          // [4]
#define PT_HP_INITIAL 65
#define PT_HP_STYLE 66       // [5]
#define PT_HP_RATES 71       // [8]
#define PT_HP_DRAIN 79       // [5]
#define PT_POWF 84           // f32 powf result bits, stored as the integer value
// 96..: initialization parameters.
#define PI_FORCE1 96
#define PI_FORCE2 97
#define PI_FORCE3 98
#define PI_FORCE4 99
#define PI_FORCE_SPAN 100
#define PI_DELAY_MAX 101
#define PI_MINUS1 102
#define PI_MINUS2 103
#define PI_PLUS1 104
#define PI_TEMPT1 105
#define PI_DISTANCE 106
// 112..: position-keep parameters (36).
#define PK_START_SECTION 112
#define PK_END_SECTION 113
#define PK_COOLDOWN 114
#define PK_INTERVAL 115
#define PK_CONTINUE 116      // [2]
#define PK_BASE_DISTANCE 118
#define PK_DISTANCE_COEF 119
#define PK_GAPS 120          // [3][2]
#define PK_MIDDLE_GAP 126
#define PK_SPEED_END 127     // [3]
#define PK_OVERTAKE_END 130  // [2]
#define PK_WISDOM 132        // [3][2]
#define PK_COEFFICIENTS 138  // [5][2]
// 160..: rushing parameters (9).
#define PR_CHECK_TIME 160
#define PR_MAXIMUM_TIME 161
#define PR_END_PERCENT 162
#define PR_SENKO_NIGE 163
#define PR_SASHI_SENKO 164
#define PR_SASHI_NIGE 165
#define PR_OIKOMI_SASHI 166
#define PR_OIKOMI_SENKO 167
#define PR_OIKOMI_NIGE 168
// 176..: downhill parameters (2).
#define PD_END_PERCENT 176
#define PD_WISDOM_RATE 177
// 184..: course and geometry scalars.
#define PC_LAST_STRAIGHT_START 184
#define PC_LAST_STRAIGHT_END 185
#define PC_LAST_CORNER_START 186
#define PC_FIRST_MOVE_POINT 187
#define PC_MAX_LANE 188
// Slope events, slot 192 + 3*k: start, length, slope.
#define PC_SLOPES 192
// 240..: per-runner raw stats and initial lanes. Per runner, six slots:
// speed, stamina, power, guts, wisdom (raw request f64), lane.
#define PI_RUNNER0 240
// 384..: TargetParameters from src/retail_lane.rs, the keys of
// TargetParameters::from_value and MoveParameters::from_value, plus the
// Block-group lane gaps.
#define PL_FRONT_LANE 384      // FrontBlockLaneGap
#define PL_SIDE_DISTANCE 385   // SideBlockDistanceGap
#define PL_SIDE_LANE 386       // SideBlockLaneGap
#define PL_FORCE_IN_THRESHOLD 387 // forceInMoveLaneThreshold
#define PL_COOLDOWN 388        // overTakeCoolDownTime
#define PL_OUTWARD_NORMALIZE 389 // lastMoveOutLaneNormalizeVal
#define PL_OUTWARD_COEFFICIENT 390 // lastMoveOutLaneCoef
#define PL_OUTWARD_RANDOM 391  // lastMoveOutLaneAddRandom
#define PL_POWER_COEFFICIENT 392 // laneMoveSpeedPowCoef
#define PL_SPEED_ADD 393       // laneMoveSpeedAdd
#define PL_BASE_SPEED 394      // laneMoveSpeedBase
#define PL_OUTER_COEFFICIENT 395 // laneMoveSpeedOutCoef
#define PL_ACCELERATION 396    // laneMoveAccelMoveSpeedBaseCoef

// ---------------------------------------------------------------------------
// Per-race scratch bases, u32 words at scratch + SCR_* (the im_* blocks) and
// the runner/race state that follows. Runner-state columns: field base + r.

// initialization sub-blocks, sized to the im_* strides so the race kernel
// can call the initialization functions in place. The ulong views need the
// block bases even, and every stride here is even.
#define SCR_STATS_IN 0                 // 24 * IIU
#define SCR_STATS_OUT (SCR_STATS_IN + 24 * IIU)  // 24 * 160, ob 128 then olb
#define SCR_INIT_IN (SCR_STATS_OUT + 24 * 160)   // IIU
#define SCR_INIT_OUT (SCR_INIT_IN + IIU)         // OOU
#define SCR_HP_IN (SCR_INIT_OUT + OOU)           // 24 * IIU
#define SCR_HP_OUT (SCR_HP_IN + 24 * IIU)        // 24 * 180, ob 172 then olb
#define SCR_STATE (SCR_HP_OUT + 24 * 180)
// Integer state, SI_STRIDE columns per runner.
#define SI_STRIDE 48
#define SI_STRATEGY 0
#define SI_DISTAPT 1
#define SI_SUAPT 2            // surface aptitude (drives the accel ground factor)
#define SI_RANK 3
#define SI_PHASE 4            // solver phase 0..2
#define SI_RPHASE 5           // retail phase count, unclamped 0..4
#define SI_RMODE 6            // rushing_mode 0..4
#define SI_KAKARI 7
#define SI_PACEDOWN 8
#define SI_DOWNHILL 9
#define SI_DASH 10
#define SI_PREPARED 11
#define SI_EVENTS_READY 12
#define SI_CORNER 13
#define SI_FINALE 14          // final_corner
#define SI_LS_ENTERED 15      // last_straight_entered
#define SI_FINISHED 16
#define SI_BLOCKER 17         // front_blocker solver index, -1 none
#define SI_HPEMPTY 18
#define SI_DIRTY 19           // live_roster_dirty
#define SI_SECT 20            // rushing start_section, -1 none
#define SI_RS_ACTIVE 21
#define SI_RS_USED 22
#define SI_RS_MODE 23
#define SI_DS_ACTIVE 24
#define SI_SL_NEXT 25         // slope next_event
#define SI_SL_REGION 26       // slope region, -1 none
#define SI_PS_RUSHING 27
#define SI_PS_STYLE 28
#define SI_PS_MODE 29         // pace mode 0..4, 7 none
#define SI_L_FORCE 30         // lane force_check
#define SI_L_XOP 31           // extra_outward present
#define SI_L_PREV_CANIN 32    // -1 None, 0 Some(false), 1 Some(true)
#define SI_SP_PENDING 33
#define SI_SP_COUNT 34
#define SI_SP_RESULT 35
#define SI_MV_DIR 36          // move continue_direction, -1 initial
#define SI_SEC_CUR 37         // SectionIndex current
#define SI_BMIN_BITS 38       // base_minimum guts bits
#define SI_L_INNER_IDX 39     // side inner_blocker index, -1 none
#define SI_L_TGTN 40          // lane targets list length
#define SI_L_VISN 41          // lane visible list length
#define SI_L_FRONT_IDX 42     // blocking sample front index, -1 none
#define SI_L_CANIN 43         // side room can_in
#define SI_L_CANOUT 44        // side room can_out
#define SI_L_FORCEIN 45       // lane force_in
#define SI_RANKCOL 46         // field rank_of, 1-based (0 before the first update)
#define SI_SP_ACTIVE 47       // is_last_spurt
// Integer columns end.
#define SCR_SF (SCR_STATE + 48 * 24)
// f32 state (raw-bit floats), SF_STRIDE columns per runner.
#define SF_STRIDE 44
#define SF_LANEMAX 0
#define SF_OBSERVED 1
#define SF_MIN 2              // retail minimum
#define SF_BMIN_VAL 3         // base_minimum value
#define SF_ACCEL_BASE 4       // adjusted power (the accel cache is inert)
#define SF_FORCE_IN 5         // force_in_coefficient
#define SF_PENDING_HP 6
#define SF_AFTER_GOAL 7
#define SF_DELAY 8            // delay_remaining
#define SF_RS_CHECK 9
#define SF_RS_MAXREM 10
#define SF_DS_REM 11
#define SF_SL_END 12
#define SF_SL_VALUE 13
#define SF_PS_COOL 14
#define SF_PS_INTVL 15
#define SF_PS_COEF 16
#define SF_PS_ESTART 17
#define SF_PS_EEND 18
#define SF_PS_EXIT 19
#define SF_L_COOLDOWN 20
#define SF_L_XO 21
#define SF_L_SIDE_INNER 22    // side inner_blocker lane value
#define SF_L_FRONT_POS 23     // front_cache owner position
#define SF_L_FRONT_LANE 24    // front_cache owner lane
#define SF_L_FRONT_DIST 25    // unused slot (kept spacing)
#define SF_L_FRONT_LIMIT 26   // aliased below as SF_SPURT_MAX
#define SF_L_FRONT_IDX 27     // front_cache front index (float, -1 none)
#define SF_SP_START 28
#define SF_SP_SPEED 29
#define SF_MV_SPEED 30        // move_speed
#define SF_MV_TARGET 31       // move target lane
#define SF_MV_TIME 32         // move continue_time
#define SF_HPG 33
#define SF_HPW 34
#define SF_MV_LANE 35         // MoveState lane
#define SF_L_SIDE_INS 36      // side room in_space
#define SF_L_SIDE_OUTS 37     // side room out_space
#define SF_WORLD_X 38         // geometry world cursor x
#define SF_WORLD_Y 39         // geometry world cursor y
#define SF_BLOCKER_GAP 40     // front_blocker gap, f32
#define SF_SPURT_MAX 26       // race-constant maximum_spurt_speed (front_cache slot)
#define SCR_LISTS (SCR_SF + SF_STRIDE * 24)
// Lane lists, SCR_LISTS + LIST_STRIDE * r. The targets list occupies the
// first RACE_MAXR slots, then the visible list.
#define LIST_STRIDE 48
#define SCR_CAND (SCR_LISTS + LIST_STRIDE * 24)
#define CAND_MAX 101
// Race-level integer block after the candidate floats.
#define SCR_CACHED ((SCR_CAND + 3 * CAND_MAX + 63) & ~63)
// Cached horses, SCR_CACHED + 4 * (LIST_STRIDE * r + j): position, lane,
// speed and target of owner r's view, roster slot j (the owner's own slot
// is never read).
#define SCR_RACE (SCR_CACHED + 4 * LIST_STRIDE * 24)
#define RS_TICKS 0
#define RS_FRAMES 1
#define RS_FALLBACK 2
#define RS_FB_ARG 3
#define RS_DONE 4
#define RS_ORDER 8            // 24: roster indices sorted by position
#define RS_PREVRANK 32        // 24: rank from the previous tick
#define SCR_U_WORDS (SCR_RACE + 128)

// Per-race scratch f64. Runner columns: field base + r.
#define SD_STRIDE 40
#define SD_POS 0
#define SD_LANE 1
#define SD_LANET 2             // lane_target
#define SD_LCS 3               // lane_change_speed
#define SD_LCT 4               // move_lane_continue_time
#define SD_CS 5                // current_speed
#define SD_TS 6                // target_speed
#define SD_MS 7                // min_speed
#define SD_LSS 8               // last_spurt_speed
#define SD_BTS 9               // base_target_speed[3]
#define SD_PKCOEF 12           // pos_keep_speed_coef
#define SD_SECMOD 15           // section_modifier[25]
#define SRL_RUNNER 0
#define SRL_HP (40 * 24)       // hp columns, SH_STRIDE per runner
#define SH_STRIDE 8
#define SH_MAX 0
#define SH_HP 1
#define SH_BASESPEED 2
#define SH_GROUNDMOD 3
// Race f64 scalars.
#define RD_T (SH_STRIDE * 24 + SRL_HP)   // loop time, the engine f64 f32-sum
#define RD_WORDS (RD_T + 32)
// im_* f64 block bases in the same scratch (double indices).
#define SRL_STATS_IN_L RD_WORDS                    // 24 * IIL
#define SRL_INIT_IN_L (SRL_STATS_IN_L + 24 * IIL)  // IIL
#define SRL_INIT_OUT_L (SRL_INIT_IN_L + IIL)       // OOL
#define SRL_HP_IN_L (SRL_INIT_OUT_L + OOL)         // 24 * IIL
#define SRL_HP_OUT_L (SRL_HP_IN_L + 24 * IIL)      // 24 * OOL
#define SRL_WORDS (SRL_HP_OUT_L + 24 * OOL)

// SectionModifier length.
#define SEC_MOD 25

// ---------------------------------------------------------------------------
// Per-race status output, u32 words at status + id * ST_STRIDE.
#define ST_STRIDE 112
#define ST_FALLBACK 0          // 0 none, 1 unsupported domain (tick in ST_ARG)
#define ST_ARG 1               // tick or argument of the fallback
#define ST_TICKS 2             // loop iterations (frames carry ST_TICKS+1)
#define ST_FINISH0 8           // per runner: finish_step bits then finish_raw

// Race frames output: frames + (id * RACE_MAX_TICKS + tick) * n_runners * 6
// + runner * 6, in the engine's row order pos, speed, hp, lane, rushing,
// blocker.
#define FR_COLS 6

// ---------------------------------------------------------------------------
// Physics ports (src/retail_physics.rs). pd indexes the race's f64 block.

static float phys_f(__global const double* pd, const int slot) {
    return convert_float_rte(pd[slot]);
}

static float bounded_stat(const float v) {
    return clamp_1_2000_f32(v);
}

static float phys_root(const float v) {
    return convert_float_rte(sqrt((double)v));
}

static float phys_course_speed(__global const double* pd, const float distance) {
    return phys_f(pd, PT_COURSE0)
        - (distance - phys_f(pd, PT_COURSE1)) / phys_f(pd, PT_COURSE2);
}

static float phys_phase_speed(__global const double* pd, const int strategy,
    const int phase) {
    return phys_f(pd, PT_PHASE_SPEED + 4 * (strategy - 1) + phase);
}

static float phys_phase_accel(__global const double* pd, const int strategy,
    const int phase) {
    return phys_f(pd, PT_PHASE_ACCEL + 4 * (strategy - 1) + phase);
}

static float phys_base_target_speed(__global const double* pd, const float speed,
    const int strategy, const int phase, const float aptitude, const float distance) {
    // The Rust locals are evaluated at double with one rounding on the store.
    double base = (double)phys_phase_speed(pd, strategy, phase) * (double)0.01f
        * (double)phys_course_speed(pd, distance);
    if (phase >= 2) {
        const float end_add = convert_float_rte(
            ((double)phys_root(bounded_stat(speed) * phys_f(pd, PT_END_SQRT))
                * (double)aptitude) * (double)phys_f(pd, PT_ADD_SPEED));
        base += (double)end_add;
    }
    return convert_float_rte(base);
}

static float phys_start_dash_speed(__global const double* pd, const float distance) {
    return phys_course_speed(pd, distance) * phys_f(pd, PT_MINIMUM);
}

static float phys_minimum_speed(__global const double* pd, const float guts,
    const float distance) {
    return convert_float_rte((double)phys_start_dash_speed(pd, distance)
        + (double)phys_root(bounded_stat(guts) * phys_f(pd, PT_MINIMUM + 1))
            * (double)phys_f(pd, PT_MINIMUM + 2));
}

static float phys_blocking_speed_cap(__global const double* pd, const float minimum,
    const float blocker_speed, const float gap) {
    const float ratio = clamp(gap / phys_f(pd, PT_BLOCKING), 0.f, 1.f);
    const float rate = convert_float_rte((double)phys_f(pd, PT_BLOCKING + 1)
        + (double)ratio * ((double)phys_f(pd, PT_BLOCKING + 2)
            - (double)phys_f(pd, PT_BLOCKING + 1)));
    return fmax(minimum, blocker_speed * rate);
}

// The host supplies powf(bounded_stat(guts) * spurt[2], spurt[3]) as f32 in
// PT_POWF; the inputs are race-constant without skills.
static float phys_maximum_spurt_speed(__global const double* pd, const float speed,
    const int strategy, const float aptitude, const float distance) {
    const float base = convert_float_rte(
        (double)phys_base_target_speed(pd, speed, strategy, 2, aptitude, distance)
            + (double)phys_course_speed(pd, distance) * (double)phys_f(pd, PT_SPURT));
    const double addition = ((double)phys_root(bounded_stat(speed) * phys_f(pd, PT_SPURT + 1))
        * (double)aptitude) * (double)phys_f(pd, PT_ADD_SPEED);
    const double guts_add = (double)as_float((uint)pd[PT_POWF]) * (double)phys_f(pd, PT_SPURT + 4);
    return convert_float_rte(
        ((double)base * (double)phys_f(pd, PT_SPURT + 5) + addition) + guts_add);
}

// target_speed_indices from retail_physics.rs.
static int phys_section(const float position, const float distance) {
    return clamp(convert_int_rtz((double)position / (double)(distance / 24.0f)), 0, 23);
}

static int phys_phase_of_section(const int section) {
    return (section < 4) ? 0 : ((section < 16) ? 1 : 2);
}

// advance_start_delay from retail_physics.rs. On 1 the caller takes
// `fraction` as the movement gate.
static int advance_start_delay(float* remaining, const float dt, double* fraction) {
    if (dt == 0.f) {
        return 0;
    }
    if (*remaining > 0.f) {
        *remaining -= dt;
        if (*remaining > 0.f) {
            return 0;
        }
    }
    const float leftover = *remaining;
    *remaining = 0.f;
    *fraction = (leftover < 0.f) ? -(double)leftover / (double)dt : 1.0;
    return 1;
}

// finish_time from retail_physics.rs (CalcFinishTime).
static float phys_finish_time(const float frame_time, const float dt,
    const float previous, const float distance, const float course) {
    const float span = distance - previous;
    const float fraction = clamp((distance - course) / span, 0.f, 1.f);
    return convert_float_rte((double)frame_time - (double)fraction * (double)dt);
}

// apply_additive_modifiers from retail_physics.rs: fold of f32 additions.
static float apply_additive_modifiers(float base, __global const float* mods, const int len) {
    int i = 0;
    while (i < len) {
        base += mods[i];
        i += 1;
    }
    return base;
}

// TargetSpeed fields the no-skill loop fills; the struct arrives as scalars.
// spot_struggle and bonus are inert without skills, passed as 0.
static float phys_target_speed(__global const double* pd, const float base,
    const float section, const float pace, const float force_in, const float power,
    const float slope, const int downhill, const float minimum, const int hp_remaining,
    __global const float* mods, const int nmods) {
    float speed = (base + section) * pace + force_in;
    speed = apply_additive_modifiers(speed, mods, nmods);
    // The slope correction evaluates at double and rounds once (doc 13.186).
    float correction;
    if (slope > 0.) {
        correction = -convert_float_rte((double)slope
            * ((double)phys_f(pd, PT_SLOPE) / (double)bounded_stat(power)));
    } else if (slope < 0. && downhill) {
        correction = convert_float_rte((double)phys_f(pd, PT_SLOPE + 1)
            + fabs(slope) / (double)phys_f(pd, PT_SLOPE + 2));
    } else {
        correction = 0.f;
    }
    speed += correction;
    if (!hp_remaining) {
        speed = minimum;
    }
    return fmax(fmin(speed, 30.f), phys_f(pd, PT_TARGET_MIN));
}

// base_acceleration from retail_physics.rs. The double chain rounds once.
static float phys_base_acceleration(__global const double* pd, const float power,
    const float ground, const float distance, const int strategy, const int phase,
    const int uphill) {
    return convert_float_rte(
        ((((double)phys_root(bounded_stat(power) * phys_f(pd, PT_ACCEL))
            * (double)phys_f(pd, PT_ACCEL + (uphill ? 2 : 1)))
            * (double)ground)
            * (double)distance)
            * (double)phys_phase_accel(pd, strategy, phase));
}

// acceleration_from_base_iter from retail_physics.rs.
static float phys_acceleration_from_base(__global const double* pd, float base,
    const int start_dash, __global const float* mods, const int nmods) {
    base = apply_additive_modifiers(base, mods, nmods);
    if (start_dash) {
        base += phys_f(pd, PT_ACCEL + 3);
    }
    return fmax(base, 0.f);
}

// ---------------------------------------------------------------------------
// comparison_sort from src/retail_sort.rs, iterative. The comparator is
// recomputed per pair exactly as the Rust closure does. The explicit
// (lo, hi, depth) stack replaces the single recursive right-partition call.

static int cs_time_cmp(const float a, const float b) {
    // The comparator truncates the difference, not the individual times.
    const int key = convert_int_rtz((a - b) * 10000.f);
    return (key > 0) ? 1 : ((key < 0) ? -1 : 0);
}

static void cs_swap(__global float* c_start, __global float* c_speed,
    __global float* c_time, const int i, const int j) {
    float tmp = c_time[i];
    c_time[i] = c_time[j];
    c_time[j] = tmp;
    tmp = c_speed[i];
    c_speed[i] = c_speed[j];
    c_speed[j] = tmp;
    tmp = c_start[i];
    c_start[i] = c_start[j];
    c_start[j] = tmp;
}

static void cs_swap_if_greater(__global float* c_start, __global float* c_speed,
    __global float* c_time, const int i, const int j) {
    if (i != j && cs_time_cmp(c_time[i], c_time[j]) > 0) {
        cs_swap(c_start, c_speed, c_time, i, j);
    }
}

// down_heap from retail_sort.rs, the 1-indexed sift-down over the window
// starting at lo.
static void cs_down_heap(__global float* c_start, __global float* c_speed,
    __global float* c_time, int i, const int size, const int lo) {
    const float vt = c_time[lo + i - 1];
    const float vs = c_speed[lo + i - 1];
    const float vx = c_start[lo + i - 1];
    while (2 * i <= size) {
        int child = 2 * i;
        if (child < size && cs_time_cmp(c_time[lo + child - 1], c_time[lo + child]) < 0) {
            child += 1;
        }
        if (cs_time_cmp(vt, c_time[lo + child - 1]) >= 0) {
            break;
        }
        c_time[lo + i - 1] = c_time[lo + child - 1];
        c_speed[lo + i - 1] = c_speed[lo + child - 1];
        c_start[lo + i - 1] = c_start[lo + child - 1];
        i = child;
    }
    c_time[lo + i - 1] = vt;
    c_speed[lo + i - 1] = vs;
    c_start[lo + i - 1] = vx;
}

static void sort_candidates(__global float* c_start, __global float* c_speed,
    __global float* c_time, const int len) {
    if (len <= 1) {
        return;
    }
    int lo_stack[64];
    int hi_stack[64];
    int depth_stack[64];
    int sp = 1;
    lo_stack[0] = 0;
    hi_stack[0] = len - 1;
    depth_stack[0] = 2 * (32 - clz((uint)len));
    while (sp > 0) {
        int lo = lo_stack[sp - 1];
        int hi = hi_stack[sp - 1];
        int depth = depth_stack[sp - 1];
        sp -= 1;
        while (hi > lo) {
            const int size = hi - lo + 1;
            if (size <= 16) {
                if (size == 2) {
                    cs_swap_if_greater(c_start, c_speed, c_time, lo, hi);
                } else if (size == 3) {
                    cs_swap_if_greater(c_start, c_speed, c_time, lo, hi - 1);
                    cs_swap_if_greater(c_start, c_speed, c_time, lo, hi);
                    cs_swap_if_greater(c_start, c_speed, c_time, hi - 1, hi);
                } else {
                    // The insertion-sort port: shift larger elements right.
                    int i = lo;
                    while (i < hi) {
                        const float vt = c_time[i + 1];
                        const float vs = c_speed[i + 1];
                        const float vx = c_start[i + 1];
                        int j = i;
                        while (j >= lo && cs_time_cmp(vt, c_time[j]) < 0) {
                            c_time[j + 1] = c_time[j];
                            c_speed[j + 1] = c_speed[j];
                            c_start[j + 1] = c_start[j];
                            j -= 1;
                        }
                        c_time[j + 1] = vt;
                        c_speed[j + 1] = vs;
                        c_start[j + 1] = vx;
                        i += 1;
                    }
                }
                break;
            }
            if (depth == 0) {
                int i = size / 2;
                while (i >= 1) {
                    cs_down_heap(c_start, c_speed, c_time, i, size, lo);
                    i -= 1;
                }
                i = size;
                while (i >= 2) {
                    cs_swap(c_start, c_speed, c_time, lo, lo + i - 1);
                    cs_down_heap(c_start, c_speed, c_time, 1, i - 1, lo);
                    i -= 1;
                }
                break;
            }
            depth -= 1;
            const int middle = lo + (hi - lo) / 2;
            cs_swap_if_greater(c_start, c_speed, c_time, lo, middle);
            cs_swap_if_greater(c_start, c_speed, c_time, lo, hi);
            cs_swap_if_greater(c_start, c_speed, c_time, middle, hi);
            // The pivot is cloned to registers, swapped to hi-1, then the
            // sentinelled scan from retail_sort.rs runs.
            cs_swap(c_start, c_speed, c_time, middle, hi - 1);
            const float pivot_t = c_time[hi - 1];
            const float pivot_s = c_speed[hi - 1];
            const float pivot_x = c_start[hi - 1];
            int left = lo;
            int right = hi - 1;
            while (left < right) {
                left += 1;
                while (cs_time_cmp(c_time[left], pivot_t) < 0) {
                    left += 1;
                }
                right -= 1;
                while (cs_time_cmp(pivot_t, c_time[right]) < 0) {
                    right -= 1;
                }
                if (left >= right) {
                    break;
                }
                cs_swap(c_start, c_speed, c_time, left, right);
            }
            cs_swap(c_start, c_speed, c_time, left, hi - 1);
            // The Rust tail-call keeps the left range in the loop. Push the
            // right partition, continue with hi = left - 1.
            lo_stack[sp] = left + 1;
            hi_stack[sp] = hi;
            depth_stack[sp] = depth;
            sp += 1;
            hi = left - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// SlopeState from src/retail_states.rs. Slopes are the race's PC_SLOPES
// entries (start, length, slope), n_slopes of them.

static void slope_update(__global uint* si, __global float* sf, const int r,
    const int nslopes, __global const double* pc, const float position) {
    int next = (int)si[SI_SL_NEXT + r];
    while (next < nslopes) {
        const float start = convert_float_rte(pc[PC_SLOPES + 3 * next]);
        if (position < start) {
            break;
        }
        next += 1;
        const float event_per = convert_float_rte(pc[PC_SLOPES + 3 * next - 1]);
        // EventSlopeChange takes the event's gradient but the first
        // same-direction region containing the current position. Both bounds
        // are inclusive, including a shared slope boundary.
        int region = -1;
        const float event_sign = sign(event_per);
        int k = 0;
        while (k < nslopes) {
            const float s_start = convert_float_rte(pc[PC_SLOPES + 3 * k]);
            const float s_len = convert_float_rte(pc[PC_SLOPES + 3 * k + 1]);
            const float s_per = convert_float_rte(pc[PC_SLOPES + 3 * k + 2]);
            if (sign(s_per) == event_sign && s_start <= position
                && position <= s_start + s_len) {
                region = k;
                break;
            }
            k += 1;
        }
        si[SI_SL_REGION + r] = (uint)region;
        if (region >= 0) {
            sf[SF_SL_VALUE + r] = event_per;
            sf[SF_SL_END + r] = convert_float_rte(pc[PC_SLOPES + 3 * region])
                + convert_float_rte(pc[PC_SLOPES + 3 * region + 1]);
        } else {
            sf[SF_SL_VALUE + r] = 0.f;
        }
    }
    si[SI_SL_NEXT + r] = (uint)next;
    // An event fires once. Expiry must not rediscover the same region.
    if ((int)si[SI_SL_REGION + r] >= 0 && position > sf[SF_SL_END + r]) {
        si[SI_SL_REGION + r] = (uint)-1;
        sf[SF_SL_VALUE + r] = 0.f;
    }
}

// ---------------------------------------------------------------------------
// RushingState from src/retail_states.rs. forced_remaining has no writer
// without skills, so the forced branch is absent here.

static float rush_param(__global const double* pdp, const int slot) {
    return convert_float_rte(pdp[slot]);
}

static int rushing_choose_mode(__global uint* si, const int r,
    __global const double* pdp, __global uint* bank) {
    const int style = (int)si[SI_STRATEGY + r];
    if (style == 1 || style == 5) {
        return 4;
    }
    const float roll = (float)r128_integer(bank, 0, 100);
    if (style == 2) {
        return (roll < rush_param(pdp, PR_SENKO_NIGE)) ? 3 : 0;
    }
    if (style == 3) {
        if (roll < rush_param(pdp, PR_SASHI_SENKO)) {
            return 2;
        }
        return (roll < rush_param(pdp, PR_SASHI_SENKO) + rush_param(pdp, PR_SASHI_NIGE))
            ? 3 : 0;
    }
    // style 4
    const float a = rush_param(pdp, PR_OIKOMI_SASHI);
    const float b = rush_param(pdp, PR_OIKOMI_SENKO);
    if (roll < a) {
        return 1;
    }
    if (roll < a + b) {
        return 2;
    }
    return (roll < a + b + rush_param(pdp, PR_OIKOMI_NIGE)) ? 3 : 0;
}

// +1 starts, -1 ends. The starting tick never decrements either clock.
// The section comparison keeps the lottery's section, not a metres product.
static int rushing_step(__global uint* si, __global float* sf, const int r,
    const float distance, const float course_distance, const float dt,
    __global const double* pdp, __global uint* bank) {
    if (!(int)si[SI_RS_ACTIVE + r]) {
        int before_start;
        const int sect = (int)si[SI_SECT + r];
        if (sect >= 0) {
            before_start = distance / (course_distance / 24.0f) < (float)sect;
        } else {
            before_start = distance < INFINITY;
        }
        if ((int)si[SI_RS_USED + r] || before_start) {
            return 0;
        }
        si[SI_RS_ACTIVE + r] = 1u;
        si[SI_RS_USED + r] = 1u;
        si[SI_RS_MODE + r] = (uint)rushing_choose_mode(si, r, pdp, bank);
        sf[SF_RS_CHECK + r] = rush_param(pdp, PR_CHECK_TIME);
        sf[SF_RS_MAXREM + r] = rush_param(pdp, PR_MAXIMUM_TIME);
        return 1;
    }
    int ended;
    sf[SF_RS_MAXREM + r] = fmax(sf[SF_RS_MAXREM + r] - dt, 0.f);
    int natural_end = (sf[SF_RS_MAXREM + r] == 0.f);
    if (!natural_end) {
        float check = fmax(sf[SF_RS_CHECK + r] - dt, 0.f);
        sf[SF_RS_CHECK + r] = check;
        if (check == 0.f) {
            sf[SF_RS_CHECK + r] = rush_param(pdp, PR_CHECK_TIME);
            natural_end = ((float)r128_integer(bank, 0, 100)) < rush_param(pdp, PR_END_PERCENT);
        }
    }
    if (natural_end) {
        si[SI_RS_ACTIVE + r] = 0u;
        si[SI_RS_MODE + r] = 0u;
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// DownhillState from src/retail_states.rs.

static int downhill_update(__global uint* si, __global float* sf, const int r,
    const int downhill, const float wisdom, const float dt,
    __global const double* pdp, __global uint* bank) {
    const int previous = (int)si[SI_DS_ACTIVE + r];
    if (!downhill) {
        if (previous) {
            si[SI_DS_ACTIVE + r] = 0u;
            sf[SF_DS_REM + r] = 1.f;
        }
        return -previous;
    }
    if (sf[SF_DS_REM + r] > 0.f) {
        sf[SF_DS_REM + r] -= dt;
        return 0; // Zero crossing is tested on the following tick.
    }
    sf[SF_DS_REM + r] = 1.f;
    const float roll = r128_real(bank, 0.f, 100.f);

    if (previous) {
        if (roll < rush_param(pdp, PD_END_PERCENT)) {
            si[SI_DS_ACTIVE + r] = 0u;
        }
    } else {
        si[SI_DS_ACTIVE + r] = (uint)(roll < fmin(bounded_stat(wisdom)
            * rush_param(pdp, PD_WISDOM_RATE), 100.f));
    }
    return (int)si[SI_DS_ACTIVE + r] - previous;
}

// ---------------------------------------------------------------------------
// GameHpState retail branch from src/hp_policy.rs and the hp chain of
// src/retail_physics.rs. shl indexes the runner's hp columns.

static float hp_state_rate(__global const double* pd, const int rushing,
    const int competing, const int runaway, const int pace_down, const int downhill) {
    int index;
    if (competing) {
        index = (rushing ? 5 : 3) + runaway;
    } else if (rushing) {
        index = 2;
    } else if (pace_down) {
        index = 1;
    } else {
        index = 0;
    }
    const float rate = phys_f(pd, PT_HP_RATES + index);
    return downhill ? rate * phys_f(pd, PT_HP_RATES + 7) : rate;
}

static float hp_per_second_with_guts_factor(__global const double* pd, const float speed,
    const float base_speed, const float ground, const int has_factor, const float guts_factor,
    const float rate) {
    const float gap = (speed - base_speed) + phys_f(pd, PT_HP_DRAIN);
    // The gap square and first divide stay f32, matching Rust; only the
    // remaining coefficient product widens.
    const float first = ((gap * gap) / phys_f(pd, PT_HP_DRAIN + 1));
    float value = convert_float_rte(
        (((double)first * (double)phys_f(pd, PT_HP_DRAIN + 2) * (double)rate)
            * (double)ground));
    if (has_factor) {
        value *= guts_factor;
    }
    return value;
}

// The synthetic last-leg HP context of get_last_spurt_pair: phase 2 with
// every flag false, so the rate is hp_rates[0] and the guts factor applies.
// guts_factor is the runner's captured hp_guts_factor (SF column SF_HPG).

static float hp_consumption_lastleg(__global const double* pd, __global const double* shl,
    const float guts_factor, const float speed) {
    return hp_per_second_with_guts_factor(pd, speed,
        convert_float_rte(shl[SH_BASESPEED]), convert_float_rte(shl[SH_GROUNDMOD]),
        1, guts_factor, phys_f(pd, PT_HP_RATES));
}

// GameHpState::tick, the retail branch. The HP cost reads the solver's
// current_speed f64, narrowed once for the consumption call.
static void hp_tick(__global const double* pd, __global double* shl, const float guts_factor,
    const int rushing, const int competing, const int runaway, const int pace_down,
    const int downhill, const int phase, const double current_speed, const double dt) {
    if (shl[SH_HP] <= 0.) {
        return;
    }
    const float rate = hp_state_rate(pd, rushing, competing, runaway, pace_down, downhill);
    const float per = hp_per_second_with_guts_factor(pd, convert_float_rte(current_speed),
        convert_float_rte(shl[SH_BASESPEED]), convert_float_rte(shl[SH_GROUNDMOD]),
        phase >= 2, guts_factor, rate);
    const float cost = per * convert_float_rte(dt);
    shl[SH_HP] = (double)fmax(convert_float_rte(shl[SH_HP]) - cost, 0.f);
}

// GameHpState::tick_start_dash, the retail branch. Rust has no hp>0 guard.
static void hp_tick_start_dash(__global const double* pd, __global double* shl,
    const int rushing, const int competing, const int runaway, const int pace_down,
    const int downhill, const float dt) {
    const float rate = hp_state_rate(pd, rushing, competing, runaway, pace_down, downhill);
    const float drain = convert_float_rte(20. * (double)rate * shl[SH_GROUNDMOD]);
    shl[SH_HP] = (double)fmax(convert_float_rte(shl[SH_HP]) - drain * dt, 0.f);
}

// ---------------------------------------------------------------------------
// Spurt search from src/retail_spurt.rs. The candidate arrays live in the
// race scratch as three parallel f32 columns.

// The closure `candidate(speed)` from choose. Slot outputs carry the
// Candidate fields when one is found.
static int spurt_candidate(__global const double* pd, __global const double* shl,
    const float guts_factor, const float remaining, const float distance, const float hp,
    const float base_speed, const float base_cost, const float speed,
    const float latest, float* out_start, float* out_speed, float* out_time) {
    const float cost = hp_consumption_lastleg(pd, shl, guts_factor, speed);
    float sprint_distance = remaining;
    while (sprint_distance > 0.f) {
        const float slow_time = (remaining - sprint_distance) / base_speed;
        const float fast_time = sprint_distance / speed;
        // Store the normal-pace cost as a float before adding the wider
        // sprint cost. The final sum and comparison stay at double precision.
        const float slow_cost = slow_time * base_cost;
        const double needed = (double)slow_cost + (double)fast_time * (double)cost;
        const float start = (distance - 60.f) - sprint_distance;
        if ((double)hp >= needed && start < latest) {
            *out_start = start;
            *out_speed = speed;
            *out_time = slow_time + fast_time;
            return 1;
        }
        sprint_distance -= 30.;
    }
    return 0;
}

// choose from retail_spurt.rs. Writes the Plan fields into the out triple.
static void spurt_choose(__global const double* pd, __global const double* shl,
    const float guts_factor, const float wisdom, const float distance, const float position,
    const float hp, const float max_speed, const float base_speed, __global uint* bank,
    __global float* c_start, __global float* c_speed, __global float* c_time,
    float* out_start, float* out_speed, int* out_result) {
    const float remaining = fmax((distance - position) - 60.f, 0.f);
    const float base_cost = hp_consumption_lastleg(pd, shl, guts_factor, base_speed);
    if (hp < (remaining / base_speed) * base_cost) {
        *out_start = distance;
        *out_speed = max_speed;
        *out_result = 4;
        return;
    }
    if (hp >= convert_float_rte((double)remaining / (double)max_speed
        * (double)hp_consumption_lastleg(pd, shl, guts_factor, max_speed))) {
        *out_start = position;
        *out_speed = max_speed;
        *out_result = 1;
        return;
    }
    const float latest = distance - 100.f;
    const float lower = fmin(base_speed, max_speed);
    float speed = max_speed;
    int found = 0;
    float s0, v0, t0;
    int it = 0;
    while (it < 101) {
        if (spurt_candidate(pd, shl, guts_factor, remaining, distance, hp, base_speed,
                base_cost, speed, latest, &s0, &v0, &t0)) {
            found = 1;
            break;
        }
        speed -= 0.1f;
        if (speed <= lower) {
            *out_start = distance;
            *out_speed = max_speed;
            *out_result = 8;
            return;
        }
        it += 1;
    }
    if (!found) {
        *out_start = distance;
        *out_speed = max_speed;
        *out_result = 8;
        return;
    }
    int len = 0;
    it = 0;
    while (it < 101) {
        if (len < CAND_MAX
            && spurt_candidate(pd, shl, guts_factor, remaining, distance, hp, base_speed,
                base_cost, speed, latest, &s0, &v0, &t0)) {
            c_start[len] = s0;
            c_speed[len] = v0;
            c_time[len] = t0;
            len += 1;
        }
        speed -= 0.1f;
        if (speed <= lower) {
            break;
        }
        it += 1;
    }
    if (len == 0) {
        *out_start = distance;
        *out_speed = max_speed;
        *out_result = 8;
        return;
    }
    sort_candidates(c_start, c_speed, c_time, len);
    const float chance = 15.f + 0.05f * wisdom;
    int k = 0;
    while (k < len) {
        if ((float)r128_integer(bank, 0, 100) < chance) {
            *out_start = fmax(position, c_start[k]);
            *out_speed = c_speed[k];
            *out_result = 2;
            return;
        }
        k += 1;
    }
    *out_start = fmax(position, c_start[len - 1]);
    *out_speed = c_speed[len - 1];
    *out_result = 2;
}

// SpurtState::update from retail_spurt.rs. The plan comes from the caller's
// calculate closure so the RNG draw order stays inside the search.
static int spurt_state_update(__global uint* si, __global float* sf, const int r,
    const float position, const float out_start, const float out_speed, const int out_result) {
    if ((int)si[SI_SP_COUNT + r] == 0 || (int)si[SI_SP_PENDING + r]) {
        si[SI_SP_COUNT + r] += 1u;
        if ((int)si[SI_SP_COUNT + r] == 1 || out_result == 1 || out_result == 2) {
            si[SI_SP_RESULT + r] = (uint)out_result;
            sf[SF_SP_SPEED + r] = out_speed;
            if ((int)si[SI_SP_COUNT + r] == 1 || sf[SF_SP_START + r] < 0.f) {
                sf[SF_SP_START + r] = (out_result == 1 || out_result == 2) ? out_start : -1.f;
            }
        }
        si[SI_SP_PENDING + r] = 0u;
    }
    const int result = (int)si[SI_SP_RESULT + r];
    return (result == 1 || result == 2) && sf[SF_SP_START + r] > 0.f
        && position >= sf[SF_SP_START + r];
}

// ---------------------------------------------------------------------------
// Position-keep (pace) state from src/retail_position_keep.rs.

static float wisdom_logarithm(const float wisdom) {
    const float argument = bounded_stat(wisdom) * 0.1f;
    return convert_float_rte(engine_log10((double)argument));
}

static int pace_base_style(const int style) {
    return (style == 5) ? 1 : style;
}

static void pace_thresholds(__global const double* pk, const int style, const float distance,
    float* low, float* high) {
    if (style < 2 || style > 4) {
        *low = 0.f;
        *high = 0.f;
        return;
    }
    // Store the scale after the wider multiply-and-add.
    const float factor = fmax(convert_float_rte(1.
        + (double)(distance - convert_float_rte(pk[PK_BASE_DISTANCE]))
            * (double)convert_float_rte(pk[PK_DISTANCE_COEF])), 1.f);
    const float l = convert_float_rte(pk[PK_GAPS + 2 * (style - 2)]);
    const float h = convert_float_rte(pk[PK_GAPS + 2 * (style - 2) + 1]);
    *low = (style == 2) ? l : l * factor;
    *high = h * factor;
}

// State::update. Mode numbers: 0 SpeedUp, 1 Overtake, 2 PaceUp, 3 PaceDown,
// 4 Extra, 7 None. The coefficient lands in SF_PS_COEF, and the caller
// mirrors it to pos_keep_speed_coef every tick.
static void pace_update(__global uint* si, __global float* sf,
    const int r, const float dt, __global const double* pk, __global uint* bank,
    const float distance, const float position, const float wisdom, const int strategy,
    const int rushing, const int rushing_mode, const int phase, const int rank,
    const int pacemaker_rank, const int pacemaker_style, const float pacemaker_gap,
    const int pacemaker_is_self, const int later_style_ahead, const int ordinary_nige_count,
    const int has_behind_gap, const float behind_gap, const int has_speed_skill) {
    if (rushing != (int)si[SI_PS_RUSHING + r]) {
        si[SI_PS_RUSHING + r] = (uint)rushing;
        si[SI_PS_MODE + r] = 7u;
        sf[SF_PS_COEF + r] = 1.f;
        sf[SF_PS_COOL + r] = 1.f;
        sf[SF_PS_COOL + r] = 0.f;
        sf[SF_PS_INTVL + r] = 0.f;
    }
    int style;
    if ((int)si[SI_PS_RUSHING + r]) {
        style = (rushing_mode == 1) ? 3 : ((rushing_mode == 2) ? 2 : 1);
    } else {
        style = pace_base_style(strategy);
    }
    si[SI_PS_STYLE + r] = (uint)style;
    float low, high;
    pace_thresholds(pk, style, distance, &low, &high);
    const int runaway = (strategy == 5) ? 1 : 0;
    float speed_end;
    if (runaway == 1) {
        speed_end = convert_float_rte(pk[PK_SPEED_END + 2]);
    } else if (pacemaker_is_self && pace_base_style(strategy) != 1) {
        speed_end = convert_float_rte(pk[PK_SPEED_END]);
    } else {
        speed_end = convert_float_rte(pk[PK_SPEED_END + ((ordinary_nige_count == 1) ? 1 : 0)]);
    }
    // While rushing, having nobody behind does not satisfy the speed-up
    // distance-behind condition (start or end).
    const int speed_gap_met = has_behind_gap ? (behind_gap >= speed_end) : (!rushing);
    const int mode = (int)si[SI_PS_MODE + r];
    if (mode != 7) {
        int end = position >= sf[SF_PS_EEND + r];
        if (!end) {
            switch (mode) {
                case 0: end = has_behind_gap && speed_gap_met; break;
                case 1: end = rank == 1
                    && (!has_behind_gap
                        || behind_gap >= convert_float_rte(pk[PK_OVERTAKE_END + runaway])); break;
                case 2: end = pacemaker_gap <= sf[SF_PS_EXIT + r]; break;
                case 3: end = pacemaker_is_self || has_speed_skill
                    || pacemaker_gap >= sf[SF_PS_EXIT + r]; break;
                default: end = !later_style_ahead; break;
            }
        }
        if (end) {
            si[SI_PS_MODE + r] = 7u;
            sf[SF_PS_COEF + r] = 1.f;
            sf[SF_PS_COOL + r] = convert_float_rte(pk[PK_COOLDOWN]);
        }
        return;
    }
    const float section = floor(distance / 24.);
    if (!(section * (convert_float_rte(pk[PK_START_SECTION]) - 1.f) <= position
        && position < section * convert_float_rte(pk[PK_END_SECTION]))) {
        return;
    }
    // Deliberately sequential: the second clock cannot tick until the first
    // has crossed zero. Negative crossing residue is retained.
    if (sf[SF_PS_COOL + r] > 0.f) {
        sf[SF_PS_COOL + r] -= dt;
        if (sf[SF_PS_COOL + r] > 0.f) {
            return;
        }
    }
    if (sf[SF_PS_INTVL + r] > 0.f) {
        sf[SF_PS_INTVL + r] -= dt;
        if (sf[SF_PS_INTVL + r] > 0.f) {
            return;
        }
    }
    sf[SF_PS_INTVL + r] = convert_float_rte(pk[PK_INTERVAL]);
    // choose from retail_position_keep.rs.
    int chosen = 7;
    if (style == 1) {
        if (rushing) {
            chosen = 0;
        } else if (later_style_ahead) {
            chosen = 4;
        } else if (rank == 1) {
            chosen = 0;
        } else {
            chosen = 1;
        }
    } else if (pacemaker_rank < rank
        && pace_base_style(pacemaker_style) > pace_base_style(strategy)) {
        chosen = 4;
    } else if (pacemaker_gap < low) {
        chosen = pacemaker_is_self ? 7 : 3;
    } else if (pacemaker_gap > high) {
        chosen = 2;
    }
    if (chosen == 7) {
        return;
    }
    if ((chosen == 3 && has_speed_skill) || (chosen == 0 && speed_gap_met)) {
        return;
    }
    if (chosen < 3 && !rushing) {
        const float roll = r128_real(bank, 0.f, 100.f);
        const float limit = wisdom_logarithm(wisdom)
            * convert_float_rte(pk[PK_WISDOM + 2 * chosen + runaway]);
        if (roll >= limit) {
            return;
        }
    }
    si[SI_PS_MODE + r] = (uint)chosen;
    sf[SF_PS_ESTART + r] = position;
    sf[SF_PS_EEND + r] = position + section * convert_float_rte(pk[PK_CONTINUE + runaway]);
    if (chosen == 2 || chosen == 3) {
        if (chosen == 3 && phase == 1) {
            high = low + (high - low) * clamp(convert_float_rte(pk[PK_MIDDLE_GAP]), 0.f, 1.f);
        }
        sf[SF_PS_EXIT + r] = r128_real(bank, low, high);
    }
    const int alternate = (chosen == 3) ? ((phase == 1) ? 1 : 0) : runaway;
    sf[SF_PS_COEF + r] = convert_float_rte(pk[PK_COEFFICIENTS + 2 * chosen + alternate]);
}

// ---------------------------------------------------------------------------
// Lane machine from src/retail_lane.rs. Owner r reads live rows through the
// solver columns and its cached rows from the SCR_CACHED block, where each
// owner keeps n rows at slot = roster index (the owner row is a snapshot of
// itself). Vision stays 20.0: without skills no Vision aux modifier stacks
// onto it. The collect pass scans roster indices ascending with strict
// minimums, the tie order both engine collect paths end in.

#define LANE_VISION 20.0f
#define LANE_EDGE 0.055555559694767f
#define LANE_STRAIGHT 0.0555555559694767f
#define LANE_CORRIDOR 0.04444444924592972f
#define LANE_CROWD 0.1111111119389534f

static float lane_front_lane_limit(const float gap, const float front_distance,
    const float front_lane) {
    // The narrowing coefficient is stored before the lane limit interpolates.
    const float narrowing = (gap / front_distance) * 0.6f;
    return convert_float_rte((1. - (double)narrowing) * (double)front_lane);
}

static int lane_is_visible(const float gap, const float lane_gap, const float vision) {
    if (vision <= 0. || !(gap >= 0. && gap <= vision)) {
        return 0;
    }
    // The width expression stays double until the final f32 store; an
    // intermediate rounding can admit a runner just outside the cone.
    const float widest = convert_float_rte(
        ((double)1. * (double)0.6388888955116272f + (double)0.1111111119389534f) * 0.5);
    const float magnitude = fabs(lane_gap);
    if (magnitude > widest) {
        return 0;
    }
    const float ratio = gap / vision;
    const float width = convert_float_rte(
        ((double)ratio * (double)0.6388888955116272f + (double)0.1111111119389534f) * 0.5);
    return magnitude <= width;
}

static int lane_reachable(const int can_in, const int can_out, const float in_space,
    const float out_space, const float lane, const float candidate) {
    const float delta = candidate - lane;
    if (delta == 0.) {
        return 1;
    }
    if (delta > 0.) {
        return can_out && delta <= out_space;
    }
    return can_in && -delta <= in_space;
}

// A peer is blocked when any qualifying front runner exists. The first match
// stops the scan; tie order is irrelevant here.
static int lane_has_front_blocker(__global const float* cached, const int n,
    const int r, const float position, const float lane, const float front_distance,
    const float front_lane) {
    int j = 0;
    while (j < n) {
        if (j != r) {
            const float gap = cached[4 * (LIST_STRIDE * r + j)] - position;
            const float lane_gap = fabs(cached[4 * (LIST_STRIDE * r + j) + 1] - lane);
            if (gap > 0. && gap < front_distance && lane_gap <= lane_front_lane_limit(
                gap, front_distance, front_lane)) {
                return 1;
            }
        }
        j += 1;
    }
    return 0;
}

// TargetState::collect_cached over the live solver columns, storing the
// horse snapshot rows the later passes keep. The front cache stays in the
// SI/SF columns for the update_targets cache check.
static void lane_collect(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    const int r, const int n, __global const double* pd, const int empty_in_move,
    const int phase, const float max_lane, const float lane, const float position) {
    const float front_distance = convert_float_rte(pd[PT_BLOCKING]);
    const float front_lane_param = convert_float_rte(pd[PL_FRONT_LANE]);
    const float side_distance = convert_float_rte(pd[PL_SIDE_DISTANCE]);
    const float side_lane = convert_float_rte(pd[PL_SIDE_LANE]);
    const float reach = fmax(fmax(LANE_VISION, front_distance), side_distance);
    int visible_len = 0;
    int front = -1;
    float nearest = 0.0f;
    int nearest_set = 0;
    int inner_nearby = 0;
    // SideRoom starts as the open rails and narrows as peers qualify.
    int can_in = lane > 0.0001f;
    int can_out = lane < max_lane;
    float in_space = lane;
    float out_space = max_lane - lane;
    __global uint* visible = lists + LIST_STRIDE * r + RACE_MAXR;
    // The side room's inner blocker is rebuilt fresh every tick.
    si[SI_L_INNER_IDX + r] = (uint)-1;
    sf[SF_L_SIDE_INNER + r] = 0.f;
    int k = 0;
    while (k < n) {
        if (k != r) {
            const float other_position = convert_float_rte(sd[SD_POS + k]);
            const float position_gap = other_position - position;
            if (position_gap > reach) {
                k += 1;
                continue;
            }
            const float other_lane = convert_float_rte(sd[SD_LANE + k]);
            const float lane_gap = other_lane - lane;
            if (lane_is_visible(position_gap, lane_gap, LANE_VISION)) {
                visible[visible_len++] = (uint)k;
            }
            // Ascending scan with a strict minimum keeps the lowest index
            // on a nearest-front tie, as both engine collect paths do.
            if (position_gap > 0. && position_gap < front_distance
                && fabs(lane_gap) <= lane_front_lane_limit(position_gap,
                    front_distance, front_lane_param)
                && (!nearest_set || position_gap < nearest)) {
                front = k;
                nearest = position_gap;
                nearest_set = 1;
            }
            if (fabs(position_gap) < side_distance) {
                if (lane_gap < 0.) {
                    in_space = fmin(in_space, -lane_gap);
                    if (-lane_gap < side_lane) {
                        can_in = 0;
                        si[SI_L_INNER_IDX + r] = (uint)k;
                        sf[SF_L_SIDE_INNER + r] = other_lane;
                    }
                } else if (lane_gap > 0.) {
                    out_space = fmin(out_space, lane_gap);
                    if (lane_gap < side_lane) {
                        can_out = 0;
                    }
                }
            }
            const float inward = lane - other_lane;
            if (fabs(position_gap) < side_distance && inward > 0. && inward < side_lane) {
                inner_nearby = 1;
            }
        }
        cached[4 * (LIST_STRIDE * r + k)] = convert_float_rte(sd[SD_POS + k]);
        cached[4 * (LIST_STRIDE * r + k) + 1] = convert_float_rte(sd[SD_LANE + k]);
        cached[4 * (LIST_STRIDE * r + k) + 2] = sf0[SF_OBSERVED + k];
        cached[4 * (LIST_STRIDE * r + k) + 3] = convert_float_rte(sd[SD_TS + k]);
        k += 1;
    }
    si[SI_L_CANIN + r] = (uint)can_in;
    si[SI_L_CANOUT + r] = (uint)can_out;
    sf[SF_L_SIDE_INS + r] = in_space;
    sf[SF_L_SIDE_OUTS + r] = out_space;
    si[SI_L_VISN + r] = (uint)visible_len;
    si[SI_L_FRONT_IDX + r] = (uint)front;
    sf[SF_L_FRONT_POS + r] = position;
    sf[SF_L_FRONT_LANE + r] = lane;
    sf[SF_FORCE_IN + r] = empty_in_move && phase < 1
        && lane >= convert_float_rte(pd[PL_FORCE_IN_THRESHOLD]) && inner_nearby ? 1.f : 0.f;
}

// update_targets from retail_lane.rs over the collect snapshot and live rows.
static int lane_eligible(__global uint* si, __global float* cached,
    __global const double* sd, __global const float* sf0, const int r, const int n,
    __global const double* pd, const int front, const int j, const float position,
    const float owner_speed, const float owner_target) {
    const int k = j;
    const float gap = cached[4 * (LIST_STRIDE * r + k)]
        - convert_float_rte(sd[SD_POS + r]);
    const float speed_gap = owner_speed - cached[4 * (LIST_STRIDE * r + k) + 2];
    const int directly = (front == k);
    const float other_speed = sf0[SF_OBSERVED + k];
    const float other_target = convert_float_rte(sd[SD_TS + k]);
    const float gap_f = convert_float_rte((double)gap);
    const float speed_gap_f = convert_float_rte((double)speed_gap);
    if (gap_f < -1.f) {
        return 0;
    }
    if (directly) {
        return other_speed < owner_target;
    }
    if (!(gap_f >= 1.f && gap_f <= 20.f) || speed_gap_f <= 0.f
        || gap_f / speed_gap_f > 15.f) {
        return 0;
    }
    const float owner_target_f = convert_float_rte((double)owner_target);
    const int blocked_result = other_speed < owner_target_f;
    const int unblocked_result = other_target < owner_target_f;
    if (blocked_result == unblocked_result) {
        return blocked_result;
    }
    if (lane_has_front_blocker(cached, n, r, convert_float_rte(sd[SD_POS + r]),
        convert_float_rte(sd[SD_LANE + r]), convert_float_rte(pd[PT_BLOCKING]),
        convert_float_rte(pd[PL_FRONT_LANE]))) {
        return blocked_result;
    }
    return unblocked_result;
}

static void lane_update_targets(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    const int r, const int n, __global const double* pd, const float dt,
    const float position, const float lane, const float owner_speed,
    const float owner_target) {
    const float front_distance = convert_float_rte(pd[PT_BLOCKING]);
    const float front_lane_param = convert_float_rte(pd[PL_FRONT_LANE]);
    int front = (int)si[SI_L_FRONT_IDX + r];
    if (sf[SF_L_FRONT_POS + r] != position || sf[SF_L_FRONT_LANE + r] != lane) {
        // The pre-movement front stays valid while owner and parameters match.
        front = -1;
        float nearest = 0.0f;
        int nearest_set = 0;
        int k = 0;
        while (k < n) {
            if (k != r) {
                const float gap = cached[4 * (LIST_STRIDE * r + k)] - position;
                const float lane_gap = fabs(cached[4 * (LIST_STRIDE * r + k) + 1] - lane);
                if (gap > 0. && gap < front_distance
                    && lane_gap <= lane_front_lane_limit(gap, front_distance,
                        front_lane_param)
                    && (!nearest_set || gap < nearest)) {
                    front = k;
                    nearest = gap;
                    nearest_set = 1;
                }
            }
            k += 1;
        }
    }
    __global uint* targets = lists + LIST_STRIDE * r;
    __global const uint* visible = lists + LIST_STRIDE * r + RACE_MAXR;
    int targets_len = (int)si[SI_L_TGTN + r];
    int write = 0;
    int t = 0;
    while (t < targets_len) {
        if ((int)targets[t] < n) {
            targets[write++] = targets[t];
        }
        t += 1;
    }
    targets_len = write;
    const int existing_count = targets_len;
    int v = 0;
    while (v < (int)si[SI_L_VISN + r]) {
        const int j = (int)visible[v];
        int contains = 0;
        t = 0;
        while (t < targets_len) {
            if ((int)targets[t] == j) {
                contains = 1;
                break;
            }
            t += 1;
        }
        if (!contains && lane_eligible(si, cached, sd, sf0, r, n, pd, front, j,
            position, owner_speed, owner_target)) {
            targets[targets_len++] = (uint)j;
        }
        v += 1;
    }
    // Entries retained from the prior tick pass the identical immutable check.
    write = 0;
    t = 0;
    while (t < targets_len) {
        const int ordinal = t + 1;
        if (ordinal > existing_count || lane_eligible(si, cached, sd, sf0, r, n, pd,
            front, (int)targets[t], position, owner_speed, owner_target)) {
            targets[write++] = targets[t];
        }
        t += 1;
    }
    targets_len = write;
    if (front >= 0) {
        int contains = 0;
        t = 0;
        while (t < targets_len) {
            if ((int)targets[t] == front) {
                contains = 1;
                break;
            }
            t += 1;
        }
        if (!contains) {
            targets[targets_len++] = (uint)front;
        }
    }
    si[SI_L_TGTN + r] = (uint)targets_len;
    if (targets_len > 0) {
        sf[SF_L_COOLDOWN + r] = convert_float_rte(pd[PL_COOLDOWN]);
    } else if (sf[SF_L_COOLDOWN + r] > 0.) {
        sf[SF_L_COOLDOWN + r] -= dt;
        if (sf[SF_L_COOLDOWN + r] <= 0.) {
            si[SI_L_FORCE + r] = 1u;
        }
    }
}

// Approximately from retail_lane.rs.
static int lane_approximately(const float a, const float b) {
    return fabs(a - b) < fmax(1e-6f * fmax(fabs(a), fabs(b)), 1.1210387714598537e-44f);
}

// PassingSearch::target_sorted_unchecked with sorted = None. The visible and
// targets lists carry roster indices; live rows read the solver columns.
static float lane_passing_target(__global uint* si, __global uint* lists,
    __global const float* cached, __global const double* sd, __global const float* sf0,
    const int r, const int n,
    const float lane, const float position, const int phase, const float max_lane,
    const float side_lane, const int can_in, const int can_out, const float in_space,
    const float out_space, const int lane_preference) {
    const int targets_len = (int)si[SI_L_TGTN + r];
    const int visible_len = (int)si[SI_L_VISN + r];
    __global const uint* targets = lists + LIST_STRIDE * r;
    __global const uint* visible = lists + LIST_STRIDE * r + RACE_MAXR;
    float best = lane;
    float best_score = INFINITY;
    float best_origin = -INFINITY;
    float best_side = 0.;
    // consider/clear run inline below; the retained-best rules follow
    // retail_lane.rs exactly, including the tie clearance test.
    int t = 0;
    float nearest = INFINITY;
    while (t < targets_len) {
        const int idx = (int)targets[t];
        const float live_pos = convert_float_rte(sd[SD_POS + idx]);
        nearest = fmin(nearest, live_pos);
        int dsign = 0;
        while (dsign < 2) {
            const float direction = (dsign == 0) ? -1.f : 1.f;
            int edge = idx;
            int walk = 1;
            while (walk) {
                const float live_lane = convert_float_rte(sd[SD_LANE + edge]);
                const float live_position = convert_float_rte(sd[SD_POS + edge]);
                const float cached_lane = cached[4 * (LIST_STRIDE * r + edge) + 1];
                const float cached_position = cached[4 * (LIST_STRIDE * r + edge)];
                int next = -1;
                float next_lane = 0.f;
                int j = 0;
                while (j < n) {
                    if (j != r) {
                        const float row_position = cached[4 * (LIST_STRIDE * r + j)];
                        const float position_gap = row_position - cached_position;
                        const float lane_gap = direction
                            * (cached[4 * (LIST_STRIDE * r + j) + 1] - cached_lane);
                        if (position_gap >= 0. && position_gap <= 3.f && lane_gap > 0.f
                            && lane_gap < LANE_CROWD && direction
                            * (convert_float_rte(sd[SD_LANE + j]) - live_lane) > 0.f) {
                            const float cand = direction * convert_float_rte(sd[SD_LANE + j]);
                            if (next < 0 || cand > next_lane
                                || (cand == next_lane && j < next)) {
                                next = j;
                                next_lane = cand;
                            }
                        }
                    }
                    j += 1;
                }
                if (next < 0) {
                    break;
                }
                edge = next;
            }
            const float candidate = convert_float_rte(sd[SD_LANE + edge])
                + direction * LANE_EDGE;
            const float end = convert_float_rte(sd[SD_POS + edge]) + 0.5f;
            int blocked = 0;
            int v = 0;
            if (!(candidate >= 0.f && candidate <= max_lane)
                || !lane_reachable(can_in, can_out, in_space, out_space, lane, candidate)) {
                blocked = 1;
            } else {
                const double low = (double)candidate - (double)LANE_CORRIDOR;
                const double high = (double)candidate + (double)LANE_CORRIDOR;
                while (v < visible_len) {
                    const int w = (int)visible[v];
                    const float row_position = convert_float_rte(sd[SD_POS + w]);
                    const double row_lane = (double)convert_float_rte(sd[SD_LANE + w]);
                    if (position <= row_position && row_position <= end
                        && low <= row_lane && row_lane <= high) {
                        blocked = 1;
                        break;
                    }
                    v += 1;
                }
            }
            if (!blocked) {
                const float origin = cached[4 * (LIST_STRIDE * r + idx)];
                const float weight = (candidate < lane)
                    ? ((phase == 0) ? 100.f : ((phase >= 2) ? 1.149999976158142f : 1.f))
                    : 1.f;
                const float gap = lane - candidate;
                const float skill = ((lane_preference == 1 && gap > 0.)
                    || (lane_preference == 2 && gap < 0.)) ? 1.5f : 1.f;
                const float cost = fabs((gap * weight) * skill);
                const float inward = (in_space < lane) ? fmax(in_space - side_lane, 0.f) : in_space;
                const float outward = (out_space < max_lane - lane)
                    ? fmax(out_space - side_lane, 0.f) : out_space;
                const int delta_c = candidate - lane < 0.f;
                const int fits_c = delta_c ? (lane - candidate <= inward)
                    : (candidate - lane <= outward);
                const int delta_b = best - lane < 0.f;
                const int fits_b = delta_b ? (lane - best <= inward)
                    : (best - lane <= outward);
                if (cost < best_score || (cost == best_score && origin > best_origin
                    && (candidate - lane) * direction > 0.f
                    && (best - lane) * best_side > 0.f && fits_c && !fits_b)) {
                    best = candidate;
                    best_score = cost;
                    best_origin = origin;
                    best_side = direction;
                }
            }
            dsign += 1;
        }
        t += 1;
    }
    const float straight = (phase < 2 && can_in && lane > LANE_STRAIGHT)
        ? lane - LANE_STRAIGHT : lane;
    if (targets_len == 0) {
        nearest = INFINITY;
    }
    int blocked = 0;
    if (!(straight >= 0.f && straight <= max_lane)
        || !lane_reachable(can_in, can_out, in_space, out_space, lane, straight)) {
        blocked = 1;
    } else {
        double low, high;
        if (straight != lane) {
            low = (double)convert_float_rte((double)lane - (double)LANE_CORRIDOR
                - (double)LANE_STRAIGHT);
            high = (double)convert_float_rte((double)lane + (double)LANE_CORRIDOR
                - (double)LANE_STRAIGHT);
        } else {
            low = (double)straight - (double)LANE_CORRIDOR;
            high = (double)straight + (double)LANE_CORRIDOR;
        }
        int v = 0;
        const float end = nearest + 3.f;
        while (v < visible_len) {
            const int w = (int)visible[v];
            const float row_position = convert_float_rte(sd[SD_POS + w]);
            const double row_lane = (double)convert_float_rte(sd[SD_LANE + w]);
            if (position <= row_position && row_position <= end
                && low <= row_lane && row_lane <= high) {
                blocked = 1;
                break;
            }
            v += 1;
        }
    }
    if (!blocked) {
        // The straight fallback keeps lane as its origin, so only a strictly
        // lower cost replaces the best candidate.
        const float weight = (straight < lane)
            ? ((phase == 0) ? 100.f : ((phase >= 2) ? 1.149999976158142f : 1.f)) : 1.f;
        const float gap = lane - straight;
        const float skill = ((lane_preference == 1 && gap > 0.)
            || (lane_preference == 2 && gap < 0.)) ? 1.5f : 1.f;
        const float cost = fabs((gap * weight) * skill);
        if (cost < best_score) {
            best = straight;
        }
    }
    return best;
}

// TargetState::target_sorted. fixed_goal stays 0 and lane_preference 0
// because no skills are active, and has_hp tracks the live HP column.
static float lane_target(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    __global const double* pd, __global uint* bank, const int r, const int n,
    const float position, const float lane, const float owner_speed,
    const float owner_target, const int phase, const int has_hp, const int pace_down) {
    const int can_in = (int)si[SI_L_CANIN + r];
    const int can_out = (int)si[SI_L_CANOUT + r];
    const float in_space = sf[SF_L_SIDE_INS + r];
    const float out_space = sf[SF_L_SIDE_OUTS + r];
    const float side_lane = convert_float_rte(pd[PL_SIDE_LANE]);
    const float max_lane = sf[SF_LANEMAX + r];
    const float previous_target = convert_float_rte(sd[SD_LANET + r]);
    const float delta = previous_target - lane;
    if (!(int)si[SI_L_FORCE + r] && (int)si[SI_L_TGTN + r] == 0
        && fabs(delta) >= 1.0f / 36.0f && (delta > 0.f ? can_out : can_in)) {
        return previous_target;
    }
    si[SI_L_FORCE + r] = 0u;
    if (!(int)si[SI_L_XOP + r] && position >= pd[PC_LAST_CORNER_START]) {
        // Store the random outward bonus after the wider coefficient product.
        const float normalized = clamp(lane / convert_float_rte(pd[PL_OUTWARD_NORMALIZE]),
            0.f, 1.f);
        sf[SF_L_XO + r] = clamp(convert_float_rte(
            (double)normalized * (double)convert_float_rte(pd[PL_OUTWARD_COEFFICIENT])
            + (double)r128_real(bank, 0.f, convert_float_rte(pd[PL_OUTWARD_RANDOM]))),
            0.f, max_lane);
        si[SI_L_XOP + r] = 1u;
    }
    const float first_move = convert_float_rte(pd[PC_FIRST_MOVE_POINT]);
    const int passing = ((int)si[SI_L_TGTN + r] != 0 || sf[SF_L_COOLDOWN + r] > 0.)
        && (phase >= 2
            || !(first_move - 200.f <= position && position <= first_move));
    if (passing) {
        float target = lane_passing_target(si, lists, cached, sd, sf0, r, n, lane,
            position, phase, max_lane, side_lane, can_in, can_out, in_space, out_space, 0);
        if ((int)si[SI_L_XOP + r] && target >= lane && sf[SF_L_XO + r] > target
            && lane_reachable(can_in, can_out, in_space, out_space, lane, sf[SF_L_XO + r])) {
            target = sf[SF_L_XO + r];
        }
        return target;
    }
    if (!has_hp) {
        return lane;
    }
    if (pace_down) {
        return 0.18f;
    }
    if ((int)si[SI_L_XOP + r]) {
        return (sf[SF_L_XO + r] > lane && can_out)
            ? fmin(lane + 0.05f, sf[SF_L_XO + r]) : lane;
    }
    if (phase < 2 && can_in) {
        return lane - 0.05f;
    }
    if (phase == 1 && (int)si[SI_L_INNER_IDX + r] >= 0) {
        const float inner = sf[SF_L_SIDE_INNER + r];
        if (lane - inner < side_lane - (1.0f / 72.0f)) {
            // Normal.Search reads the cached gap, then the blocker's live
            // lane (3432a16..3432a54).
            const float live_inner = convert_float_rte(
                sd[SD_LANE + (int)si[SI_L_INNER_IDX + r]]);
            return fmin(lane + 0.05f, live_inner + side_lane);
        }
    }
    return lane;
}

// MoveState::step, empty modifiers (no lane-move skills). The x87-style
// lateral-speed chain follows retail_lane.rs exactly.
static int lane_move_step(__global uint* si, __global float* sf, const int r,
    __global const double* pd, const int after_goal, const float position,
    const float power, const int phase, const int rank, const float first_move_point,
    const float max_lane, const int can_in, const int can_out, const float dt) {
    sf[SF_MV_TARGET + r] = clamp(sf[SF_MV_TARGET + r], 0.f, max_lane);
    if (lane_approximately(sf[SF_MV_TARGET + r], sf[SF_MV_LANE + r])) {
        sf[SF_MV_TIME + r] = 0.f;
        return 0;
    }
    const int previous_direction = (int)si[SI_MV_DIR + r];
    si[SI_MV_DIR + r] = (uint)(sf[SF_MV_TARGET + r] >= sf[SF_MV_LANE + r]);
    const float lane = sf[SF_MV_LANE + r];
    float speed;
    if (after_goal) {
        // Finished runners use signed dt as their lateral speed.
        speed = (sf[SF_MV_TARGET + r] < lane) ? -dt : dt;
    } else {
        const float rate0 = convert_float_rte(
            ((double)bounded_stat(power) * (double)convert_float_rte(pd[PL_POWER_COEFFICIENT])
                + (double)convert_float_rte(pd[PL_SPEED_ADD]))
            * (double)convert_float_rte(pd[PL_BASE_SPEED]));
        double factor;
        if (phase >= 2) {
            factor = 1. + ((double)rank - 1.) * (double)0.01f;
        } else if (position <= first_move_point) {
            factor = 1. + (double)(lane / max_lane)
                * (double)convert_float_rte(pd[PL_OUTER_COEFFICIENT]);
        } else {
            factor = 1.;
        }
        const float rate = convert_float_rte((double)rate0 * factor);
        const double acceleration = (double)convert_float_rte(pd[PL_ACCELERATION])
            * (double)convert_float_rte(pd[PL_BASE_SPEED]) * (double)dt;
        float speed_next = sf[SF_MV_SPEED + r];
        if (speed_next < rate) {
            speed_next = fmin(convert_float_rte((double)speed_next + acceleration), rate);
        } else {
            speed_next = fmax(convert_float_rte((double)speed_next - acceleration), rate);
        }
        speed = clamp(fmax(speed_next, 0.f), 0.f, 0.6f);
        if (lane_approximately(speed, 0.f)) {
            return 0;
        }
        if (sf[SF_MV_TARGET + r] < lane) {
            speed = convert_float_rte(-(double)speed * (1. + (double)lane));
        }
        sf[SF_MV_SPEED + r] = fmax(speed_next, 0.f);
    }
    float next = convert_float_rte((double)lane + (double)dt * (double)speed);
    next = (sf[SF_MV_TARGET + r] < lane) ? fmax(next, sf[SF_MV_TARGET + r])
        : fmin(next, sf[SF_MV_TARGET + r]);
    next = clamp(next, 0.f, max_lane);
    if ((next < lane && !can_in) || (next > lane && !can_out)) {
        return 0;
    }
    const int moved = (next != lane);
    sf[SF_MV_LANE + r] = next;
    if (previous_direction == (int)si[SI_MV_DIR + r]) {
        sf[SF_MV_TIME + r] += dt;
    }
    if (lane_approximately(next, sf[SF_MV_TARGET + r])) {
        sf[SF_MV_SPEED + r] = 0.f;
    }
    return moved;
}

// resolve_overlaps_with_order over the post-movement rows. For the field of
// one this only clears the force flag; the general pass keeps the loaded
// comparison-sort equal-key permutation and shifts later rows outward.
static void lane_resolve_overlaps(__global uint* si, __global double* sd, const int n) {
    int order[RACE_MAXR];
    int i = 0;
    while (i < n) {
        order[i] = i;
        i += 1;
    }
    // Insertion sort by the truncated lane key. The engine sort's equal-key
    // permutation only matters for overlapping lanes, and overlaps then
    // apply in this scan order.
    int k = 1;
    while (k < n) {
        const int elem = order[k];
        const int key = convert_int_rtz(sd[SD_LANE + elem]);
        int m = k;
        while (m > 0 && convert_int_rtz(sd[SD_LANE + order[m - 1]]) > key) {
            order[m] = order[m - 1];
            m -= 1;
        }
        order[m] = elem;
        k += 1;
    }
    i = 0;
    while (i < n) {
        si[SI_L_FORCE + order[i]] = 0u;
        i += 1;
    }
    const float rail = convert_float_rte(0.4 / 18.);
    i = 0;
    while (i + 1 < n) {
        const int previous = order[i];
        const int current = order[i + 1];
        const float gap_position = convert_float_rte(sd[SD_POS + current])
            - convert_float_rte(sd[SD_POS + previous]);
        const float gap_lane = convert_float_rte(sd[SD_LANE + current])
            - convert_float_rte(sd[SD_LANE + previous]);
        if (fabs(gap_position) < 0.4f && fabs(gap_lane) < rail) {
            sd[SD_LANE + current] =
                (double)(convert_float_rte(sd[SD_LANE + previous]) + rail);
            si[SI_L_FORCE + current] = 1u;
        }
        i += 1;
    }
}

// blocking_sample: the front blocker (index and f32 gap) plus the side flag
// over the post-skill rows, including finishers for the front query.
static void lane_blocking_sample(__global const double* sd, __global const uint* si,
    const int r, const int n, __global const double* pd, int* front, float* front_gap,
    int* side) {
    const float front_distance = convert_float_rte(pd[PT_BLOCKING]);
    const float front_lane_param = convert_float_rte(pd[PL_FRONT_LANE]);
    const float side_distance = convert_float_rte(pd[PL_SIDE_DISTANCE]);
    const float side_lane = convert_float_rte(pd[PL_SIDE_LANE]);
    const double course_distance = pd[PI_DISTANCE];
    const float position = convert_float_rte(sd[SD_POS + r]);
    const float lane = convert_float_rte(sd[SD_LANE + r]);
    *front = -1;
    float nearest = INFINITY;
    *side = 0;
    int i = 0;
    while (i < n) {
        if (i != r) {
            const float gap = convert_float_rte(sd[SD_POS + i]) - position;
            const float lane_gap = convert_float_rte(sd[SD_LANE + i]) - lane;
            if (gap > 0. && gap < front_distance && gap < nearest
                && fabs(lane_gap) <= lane_front_lane_limit(gap, front_distance,
                    front_lane_param)) {
                *front = i;
                *front_gap = gap;
                nearest = gap;
            }
            const int finished = (sd[SD_POS + i] >= course_distance);
            if (!finished && fabs(gap) < side_distance && fabs(lane_gap) < side_lane
                && lane_gap != 0.) {
                *side = 1;
            }
        }
        i += 1;
    }
}