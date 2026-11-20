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

// Course and runner caps. Grid mode admits rosters up to 120
// (src/lib.rs:262-272) but the engine answers larger non-grid rosters, so
// the kernel carries 144 and every runner-indexed region sizes to it.
#define RACE_MAXR 144
// 400 s of race time at dt=0.0666 is under 6007 ticks.
#define RACE_MAX_TICKS 6144
// Slopes per course, from course_data.json entries.
#define RACE_MAX_SLOPES 16

// ---------------------------------------------------------------------------
// Race input, u32 words per race, stride RI_STRIDE.

#define RI_SEED 0            // i32 request seed (RetailRandom seed)
#define RI_N 1               // runners
#define RI_COURSE 2          // geometry table index
#define RI_GROUND 3          // 1..4
#define RI_SURFACE 4         // 1..2
#define RI_PAIRED 5          // 1 = pairedRandomStreams (per-runner RUN banks)
#define RI_STARTUP 6         // 1 = experimentalRetailStartup (dt=0 startup tick)
#define RI_GEOM_ON 7         // 1 = experimentalRetailGeometry
#define RI_HOST_FLAG 8       // nonzero: the host already knows this race needs
                             // a fallback (skills, startup, no geometry)
#define RI_SINGLE 9          // singleMode: the +400 fold in the adjusted stats
#define RI_NSLOPES 10
#define RI_NSTR 11           // straights present
#define RI_NCOR 12           // corners present
#define RI_LOT_MIN 13
#define RI_LOT_MAX 14
#define RI_EMPTY_MOVE 15       // course empty_in_move (first_move_lane_point > 0)
// Debug-only: when nonzero and equal to the frame index just captured, the
// loop writes pace, clock, rush, lane, and lane-list state for each runner
// into the fourteen reserved tail frame slots. Zero in production keeps
// those slots zero. The slot follows the per-runner block below.
// Per-runner block, slot RI_RUN0 + RI_RUN_STRIDE * r.
#define RI_RUN0 16
#define RI_RUN_STRIDE 13
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
#define RR_TEAM 12           // fixed team id, 0 no team (race_field.rs:461 drop)
#define RI_DEBUG_TICK (RI_RUN0 + RI_RUN_STRIDE * RACE_MAXR)
#define RI_FULL (RI_DEBUG_TICK + 14) // 1: write the full-reply channels and exports
#define RI_STRIDE (RI_DEBUG_TICK + 16)

// ---------------------------------------------------------------------------
// Race input, f64 words (raw bits) per race, stride RIL_STRIDE. Float
// parameters are stored as the raw JSON f64 and narrowed once in the kernel,
// mirroring Parameters::from_value's `get_f64 as f32`.

// Stride: the fixed parameter blocks, then the per-runner stats, race
// scalars and per-runner precompute blocks at the tail (see below).
#define RIL_STRIDE (PD_RUN0 + PD_RUN_STRIDE * RACE_MAXR)
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
// 240..: per-runner raw stats, initial lane and start position. Per runner,
// seven slots: speed, stamina, power, guts, wisdom (raw request f64), lane,
// and the start position (0.0 retail, the grid's negative rows when grid
// starts are in play).
#define PI_RUNNER0 240
#define PI_RUN_STRIDE 7
#define PI_RUN_LANE 5
#define PI_RUN_POS 6
// 240 + PI_RUN_STRIDE * RACE_MAXR..: TargetParameters from src/retail_lane.rs,
// the keys of TargetParameters::from_value and MoveParameters::from_value,
// plus the Block-group lane gaps.
#define PL_FRONT_LANE (PI_RUNNER0 + PI_RUN_STRIDE * RACE_MAXR)      // FrontBlockLaneGap
#define PL_SIDE_DISTANCE (PL_FRONT_LANE + 1)   // SideBlockDistanceGap
#define PL_SIDE_LANE (PL_FRONT_LANE + 2)       // SideBlockLaneGap
#define PL_FORCE_IN_THRESHOLD (PL_FRONT_LANE + 3) // forceInMoveLaneThreshold
#define PL_COOLDOWN (PL_FRONT_LANE + 4)        // overTakeCoolDownTime
#define PL_OUTWARD_NORMALIZE (PL_FRONT_LANE + 5) // lastMoveOutLaneNormalizeVal
#define PL_OUTWARD_COEFFICIENT (PL_FRONT_LANE + 6) // lastMoveOutLaneCoef
#define PL_OUTWARD_RANDOM (PL_FRONT_LANE + 7)  // lastMoveOutLaneAddRandom
#define PL_POWER_COEFFICIENT (PL_FRONT_LANE + 8) // laneMoveSpeedPowCoef
#define PL_SPEED_ADD (PL_FRONT_LANE + 9)       // laneMoveSpeedAdd
#define PL_BASE_SPEED (PL_FRONT_LANE + 10)     // laneMoveSpeedBase
#define PL_OUTER_COEFFICIENT (PL_FRONT_LANE + 11) // laneMoveSpeedOutCoef
#define PL_ACCELERATION (PL_FRONT_LANE + 12)   // laneMoveAccelMoveSpeedBaseCoef
// ..: CompeteFight and CompeteTop settings with the per-runner powf
// precomputes. With no live GutsUp these stay prepare-baked race constants;
// with a live one the kernel recomputes the guts-derived words from the
// field at the same points the engine reads them.
#define PD_CF_DIST (PL_FRONT_LANE + 16)        // CompeteFight DistanceGap (raw JSON f64)
#define PD_CF_LANE (PD_CF_DIST + 1)            // CompeteFight LaneGap (raw JSON f64)
#define PD_CF_SPEED (PD_CF_DIST + 2)           // [3] CompeteFight AddParam1Coef1..3
#define PD_CF_ACCEL (PD_CF_DIST + 5)           // [3] CompeteFight AddParam2Coef1..3
#define PD_LIVE_GUTS (PD_CF_DIST + 13)         // 1 when a GutsUp arm can fire mid-race
#define PD_PM_RANGE (PD_CF_DIST + 8)           // PaseMakerChangeForwardRunningStyleRange
#define PD_PM_NOTFWD (PD_CF_DIST + 9)          // PaseMakerChangeTopHorseNotMostForwardStyleCount
#define PD_DTYPE (PD_CF_DIST + 10)             // course distance_type, the asitame coefficient row
#define PD_FC_HALF0 (PD_CF_DIST + 11)          // final_corner_half start, NaN when the course has none
#define PD_FC_HALF1 (PD_CF_DIST + 12)          // final_corner_half end
#define PD_RUN0 (PL_FRONT_LANE + 32)           // per runner, stride PD_RUN_STRIDE
#define PD_RUN_STRIDE 11
#define PRD_POWF 0             // per-runner spurt powf bits as the integer value
#define PRD_DUEL_T 1           // duel target-speed modifier, f64
#define PRD_DUEL_A 2           // duel accel modifier, f64
#define PRD_SPOT_M 3           // spot struggle modifier, f64
#define PRD_SPOT_D 4           // spot struggle duration, f64
#define PRD_BOOST_GUTS 5       // bumped guts field f64, 0 when no bump
#define PRD_BOOST_WIS 6        // bumped wisdom field f64, 0 when no bump
#define PRD_RUSH_BONUS 7       // rushing bonus f32 bits as the integer value
#define PRD_WIS_ACC 8          // bumped wisdom accumulator f64, NaN when no bump
#define PRD_GUTS_ACC 9         // bumped guts accumulator f64, NaN when no bump
#define PRD_SPOT_PROF 10       // strategy proficiency scaling the live spot duration

// ---------------------------------------------------------------------------
// Per-race scratch bases, u32 words at scratch + SCR_* (the im_* blocks) and
// the runner/race state that follows. Runner-state columns: field base + r.

// initialization sub-blocks, sized to the im_* strides so the race kernel
// can call the initialization functions in place. The ulong views need the
// block bases even, and every stride here is even.
#define SCR_STATS_IN 0                 // RACE_MAXR * IIU
#define SCR_STATS_OUT (SCR_STATS_IN + RACE_MAXR * IIU)  // RACE_MAXR * 160, ob 128 then olb
#define SCR_INIT_IN (SCR_STATS_OUT + RACE_MAXR * 160)   // IIU
#define SCR_INIT_OUT (SCR_INIT_IN + IIU)         // OOU
#define SCR_HP_IN (SCR_INIT_OUT + OOU)           // RACE_MAXR * IIU
#define SCR_HP_OUT (SCR_HP_IN + RACE_MAXR * IIU)        // RACE_MAXR * 180, ob 172 then olb
#define SCR_STATE (SCR_HP_OUT + RACE_MAXR * 180)
// Integer state, one RACE_MAXR-wide column per slot (field base + runner index).
#define SI_SLOTS 90
#define SI_STRIDE (SI_SLOTS * RACE_MAXR)
#define SI_SLOT(n) ((n) * RACE_MAXR)
#define SI_STRATEGY SI_SLOT(0)
#define SI_DISTAPT SI_SLOT(1)
#define SI_SUAPT SI_SLOT(2)      // surface aptitude (drives the accel ground factor)
#define SI_RANK SI_SLOT(3)
#define SI_PHASE SI_SLOT(4)      // solver phase 0..2
#define SI_RPHASE SI_SLOT(5)     // retail phase count, unclamped 0..4
#define SI_RMODE SI_SLOT(6)      // rushing_mode 0..4
#define SI_KAKARI SI_SLOT(7)
#define SI_PACEDOWN SI_SLOT(8)
#define SI_DOWNHILL SI_SLOT(9)
#define SI_DASH SI_SLOT(10)
#define SI_PREPARED SI_SLOT(11)
#define SI_EVENTS_READY SI_SLOT(12)
#define SI_CORNER SI_SLOT(13)
#define SI_FINALE SI_SLOT(14)    // final_corner
#define SI_LS_ENTERED SI_SLOT(15) // last_straight_entered
#define SI_FINISHED SI_SLOT(16)
#define SI_BLOCKER SI_SLOT(17)   // front_blocker solver index, -1 none
#define SI_HPEMPTY SI_SLOT(18)
#define SI_DIRTY SI_SLOT(19)     // live_roster_dirty
#define SI_SECT SI_SLOT(20)      // rushing start_section, -1 none
#define SI_RS_ACTIVE SI_SLOT(21)
#define SI_RS_USED SI_SLOT(22)
#define SI_RS_MODE SI_SLOT(23)
#define SI_DS_ACTIVE SI_SLOT(24)
#define SI_SL_NEXT SI_SLOT(25)   // slope next_event
#define SI_SL_REGION SI_SLOT(26) // slope region, -1 none
#define SI_PS_RUSHING SI_SLOT(27)
#define SI_PS_STYLE SI_SLOT(28)
#define SI_PS_MODE SI_SLOT(29)   // pace mode 0..4, 7 none
#define SI_L_FORCE SI_SLOT(30)   // lane force_check
#define SI_L_XOP SI_SLOT(31)     // extra_outward present
#define SI_L_PREV_CANIN SI_SLOT(32) // -1 None, 0 Some(false), 1 Some(true)
#define SI_SP_PENDING SI_SLOT(33)
#define SI_SP_COUNT SI_SLOT(34)
#define SI_SP_RESULT SI_SLOT(35)
#define SI_MV_DIR SI_SLOT(36)    // move continue_direction, -1 initial
#define SI_SEC_CUR SI_SLOT(37)   // SectionIndex current
#define SI_BMIN_BITS SI_SLOT(38) // base_minimum guts bits
#define SI_L_INNER_IDX SI_SLOT(39) // side inner_blocker index, -1 none
#define SI_L_TGTN SI_SLOT(40)    // lane targets list length
#define SI_L_VISN SI_SLOT(41)    // lane visible list length
#define SI_L_FRONT_IDX SI_SLOT(42) // blocking sample front index, -1 none
#define SI_L_CANIN SI_SLOT(43)   // side room can_in
#define SI_L_CANOUT SI_SLOT(44)  // side room can_out
#define SI_L_FORCEIN SI_SLOT(45) // lane force_in
#define SI_RANKCOL SI_SLOT(46)    // field rank_of, 1-based
#define SI_SP_ACTIVE SI_SLOT(47)  // is_last_spurt
#define SI_DUEL_ACTIVE SI_SLOT(48)
#define SI_DUEL_FIGHT SI_SLOT(49) // compete_fight_count
#define SI_DUEL_GROUP SI_SLOT(50) // duel_group, -1 for None
#define SI_DUEL_PRE_FIN SI_SLOT(51) // duel_preupdate_finished
#define SI_DUEL_PRE_STR SI_SLOT(52) // duel_preupdate_final_straight
#define SI_GOAL_SNAP SI_SLOT(53)  // duel_goal_crossing_snapshot
#define SI_SPOT_ACTIVE SI_SLOT(54)
#define SI_SPOT_STARTED SI_SLOT(55)
#define SI_SPOT_EXITED SI_SLOT(56) // spot_struggle_distance_exited
#define SI_SPOT_LATCH SI_SLOT(57)  // spot_struggle_cutoff_latched
#define SI_SPOT_CLOSED SI_SLOT(58) // spot_struggle_entry_closed
#define SI_SPOT_TOKEN SI_SLOT(59)  // retail target token 1 is registered
#define SI_DUEL_TOKEN SI_SLOT(60)  // retail target and accel token 4 is registered
#define SI_CPFLAGS SI_SLOT(62)     // conserve power and asitame bit flags
#define SI_LS_TICK SI_SLOT(61)     // last_straight_entry_tick of this tick
#define SI_TEMPT SI_SLOT(63)       // temptation_count: rush starts (rushingStartCount)
#define SI_ACT0 SI_SLOT(64)        // activate_count[0], bumped at fire
#define SI_ACT1 SI_SLOT(65)        // activate_count[1], bumped at fire
#define SI_BND50 SI_SLOT(66)       // out-50 order band cell: 0 None, 1 timer, 2 NaN
#define SI_L_MOVED SI_SLOT(67)     // moved_lane_this_tick from the previous tick
#define SI_BND20 SI_SLOT(68)       // out-20 order band cell, same encoding
#define SI_BND40 SI_SLOT(69)       // out-40 order band cell, same encoding
#define SI_BND70 SI_SLOT(70)       // out-70 order band cell, same encoding
#define SI_ACTALL SI_SLOT(71)      // cumulative activations, the v75 anyskill count
#define SI_HEAL SI_SLOT(72)        // activate_count_heal, bumped per self recovery
#define SI_POV2 SI_SLOT(73)        // phase_overtakes[2], the end-phase rank gains
#define SI_ACT2 SI_SLOT(74)        // activate_count[2], bumped at fire in phase 2
#define SI_ACTLH SI_SLOT(75)       // activate_count_later_half, bumps past mid-course
#define SI_POV1 SI_SLOT(76)        // phase_overtakes[1], the middle-phase rank gains
#define SI_POVFC SI_SLOT(77)       // final_corner_overtakes, tick-start flag gated
#define SI_BNDIN20 SI_SLOT(78)     // in-20 order band cell, same encoding as out
#define SI_BNDIN40 SI_SLOT(79)     // in-40 order band cell
#define SI_BNDIN50 SI_SLOT(80)     // in-50 order band cell
#define SI_BNDIN80 SI_SLOT(81)     // in-80 order band cell
#define SI_TEAM SI_SLOT(82)        // fixed team id (race_field.rs:461 same-team debuff drop)
#define SI_LCO SI_SLOT(83)         // late_corner_overtakes, tick-start corner flags
#define SI_PENDN SI_SLOT(84)       // pending_overtakes, drained as kind-1 events at prepare
#define SI_GREEN SI_SLOT(85)       // activated_green_skills, distinct green ids fired
#define SI_AANU SI_SLOT(86)        // running skills holding queued activations
#define SI_AARET SI_SLOT(87)       // retiring_additional_skills, set by the retire scan
#define SI_RESRV SI_SLOT(88)       // records of this runner holding a random reservation
#define SI_PPHASE SI_SLOT(89)      // last positioning phase written to the reply channel
#define CP_UNSUPPORTED 1u
#define CP_CUTOFF 2u
#define CP_WASBLK 4u
#define CP_WASRUSH 8u
#define CP_WASSPOT 16u
#define CP_INITSET 32u
#define CP_CLOCK 64u
#define CP_HASTH 128u
#define CP_STRUGGLE 256u
#define CP_KAKARI 512u
#define CP_FIRED 1024u
#define CP_ASIT 2048u
// Integer slots end.
#define SCR_SF (SCR_STATE + SI_STRIDE)
// f32 state (raw-bit floats), one RACE_MAXR-wide column per slot.
#define SF_SLOTS 56
#define SF_STRIDE (SF_SLOTS * RACE_MAXR)
#define SF_SLOT(n) ((n) * RACE_MAXR)
#define SF_LANEMAX SF_SLOT(0)
#define SF_OBSERVED SF_SLOT(1)
#define SF_MIN SF_SLOT(2)         // retail minimum
#define SF_BMIN_VAL SF_SLOT(3)    // base_minimum value
#define SF_ACCEL_BASE SF_SLOT(4)  // adjusted power (the accel cache is inert)
#define SF_FORCE_IN SF_SLOT(5)    // force_in_coefficient
#define SF_PENDING_HP SF_SLOT(6)
#define SF_AFTER_GOAL SF_SLOT(7)
#define SF_DELAY SF_SLOT(8)       // delay_remaining
#define SF_RS_CHECK SF_SLOT(9)
#define SF_RS_MAXREM SF_SLOT(10)
#define SF_DS_REM SF_SLOT(11)
#define SF_SL_END SF_SLOT(12)
#define SF_SL_VALUE SF_SLOT(13)
#define SF_PS_COOL SF_SLOT(14)
#define SF_PS_INTVL SF_SLOT(15)
#define SF_PS_COEF SF_SLOT(16)
#define SF_PS_ESTART SF_SLOT(17)
#define SF_PS_EEND SF_SLOT(18)
#define SF_PS_EXIT SF_SLOT(19)
#define SF_L_COOLDOWN SF_SLOT(20)
#define SF_L_XO SF_SLOT(21)
#define SF_L_SIDE_INNER SF_SLOT(22) // side inner_blocker lane value
#define SF_L_FRONT_POS SF_SLOT(23)  // front_cache owner position
#define SF_L_FRONT_LANE SF_SLOT(24) // front_cache owner lane
#define SF_L_FRONT_DIST SF_SLOT(25) // unused slot (kept spacing)
#define SF_L_FRONT_LIMIT SF_SLOT(26) // aliased below as SF_SPURT_MAX
#define SF_L_FRONT_IDX SF_SLOT(27)  // front_cache front index (float, -1 none)
#define SF_SP_START SF_SLOT(28)
#define SF_SP_SPEED SF_SLOT(29)
#define SF_MV_SPEED SF_SLOT(30)   // move_speed
#define SF_MV_TARGET SF_SLOT(31)  // move target lane
#define SF_MV_TIME SF_SLOT(32)    // move continue_time
#define SF_HPG SF_SLOT(33)
#define SF_HPW SF_SLOT(34)
#define SF_MV_LANE SF_SLOT(35)    // MoveState lane
#define SF_L_SIDE_INS SF_SLOT(36) // side room in_space
#define SF_L_SIDE_OUTS SF_SLOT(37) // side room out_space
#define SF_WORLD_X SF_SLOT(38)    // geometry world cursor x
#define SF_WORLD_Y SF_SLOT(39)    // geometry world cursor y
#define SF_BLOCKER_GAP SF_SLOT(40) // front_blocker gap, f32
#define SF_SPURT_MAX SF_SLOT(26)  // race-constant maximum_spurt_speed (front_cache slot)
#define SF_DUEL_PRE SF_SLOT(44)   // duel_preupdate_speed (observed at tick start)
#define SF_AS_VALUE SF_SLOT(48)   // asitame accel effect modifier
#define SF_CP_GELAP SF_SLOT(49)   // conserve gain_elapsed (f32 accumulator)
#define SF_CP_GTHRESH SF_SLOT(50) // conserve gain_threshold
#define SF_RS_FORCED SF_SLOT(55)  // rushing forced_remaining, -1.0 when None
// RushingScore (stadium_score.rs): per-tick countdown while kakari is held;
// each zero crossing charges one penalty. The count stays an exact integer.
#define SF_RSCORE SF_SLOT(41)     // penalties charged (integer-valued f32)
#define SF_RREM SF_SLOT(42)       // countdown remaining, starts at 1.0
#define SCR_LISTS (SCR_SF + SF_STRIDE)
// Lane lists, SCR_LISTS + LIST_STRIDE * r. The targets list occupies the
// first RACE_MAXR slots, then the visible list (both up to n-1 entries).
#define LIST_STRIDE (2 * RACE_MAXR)
#define SCR_CAND (SCR_LISTS + LIST_STRIDE * RACE_MAXR)
#define CAND_MAX 101
// Race-level integer block after the candidate floats.
#define SCR_CACHED ((SCR_CAND + 3 * CAND_MAX + 63) & ~63)
// Cached horses, SCR_CACHED + 4 * (CACHE_STRIDE * r + j): position, lane,
// speed and target of owner r's view, roster slot j (the owner's own slot
// is never read).
#define CACHE_STRIDE RACE_MAXR
#define SCR_RACE (SCR_CACHED + 4 * CACHE_STRIDE * RACE_MAXR)
#define RS_TICKS 0
#define RS_FRAMES 1
#define RS_FALLBACK 2
#define RS_FB_ARG 3
#define RS_DONE 4
#define RS_ORDER 8            // RACE_MAXR: roster indices sorted by position
#define RS_PREVRANK (RS_ORDER + RACE_MAXR)     // RACE_MAXR: rank from the previous tick
#define RS_PM (RS_PREVRANK + RACE_MAXR)        // section pacemaker, +1 (0 none)
#define RS_PM_SEC (RS_PM + 1)   // pacemaker section at the last handover
#define RS_PM_COUNT (RS_PM + 2)
#define RS_PM_HANDED (RS_PM + 3)
#define RS_PM_HAS (RS_PM + 4)   // 0 before the constructor's first selection
#define RS_COUNTS ((RS_PM + 8 + 7) & ~7)       // 8: exact per-strategy counts
#define RS_STYLE (RS_COUNTS + 8)               // RACE_MAXR: base-style prefix maxima over the order
#define RS_STRAIGHT (RS_STYLE + RACE_MAXR)     // RACE_MAXR: pre-movement final-straight row flags
#define RS_LKEYS (RS_STRAIGHT + RACE_MAXR)     // RACE_MAXR: resolve-overlaps truncated lane keys
#define RS_LORDER (RS_LKEYS + RACE_MAXR)       // RACE_MAXR: resolve-overlaps scan order
#define RS_ADV (RS_LORDER + RACE_MAXR)         // RACE_MAXR: stepped (not gated) flags of the tick
#define SCR_BANKS (SCR_RACE + RS_ADV + RACE_MAXR) // paired mode: one RUN bank per runner
// Skill runtime scratch: one state word block per table record, the three
// per-runner receiver token orders, then the keyed-substream bank. The token
// orders mirror the engine's AdditiveReceiver entry vectors, 0 is the empty
// sentinel and registration order is append order.
#define SCR_SK (SCR_BANKS + RACE_MAXR * BANK_WORDS)
#define SK_SEQN 40
#define SK_TOK_SPOT 1u
#define SK_TOK_DUEL 4u
#define SK_TOK_SKILL 16u
// A skill token packs (record, effect slot): the slot lives in the low three
// bits, so a repeated effect kind registers and expires each entry on its own.
#define SK_TOK_EFF(rec, slot) (SK_TOK_SKILL + ((rec) << 3) + (slot))
#define SK_TOK_REC(tok) (((tok) - SK_TOK_SKILL) >> 3)
#define SK_TOK_SLOT(tok) ((tok) & 7u)
#define SK_TSEQ (SCR_SK + SK_RMAX * SKR_STRIDE)
#define SK_ASEQ (SK_TSEQ + SK_RMAX * SK_SEQN)
#define SK_CSEQ (SK_ASEQ + SK_RMAX * SK_SEQN)
#define SK_LSEQ (SK_CSEQ + SK_RMAX * SK_SEQN)
#define SK_VSEQ (SK_LSEQ + SK_RMAX * SK_SEQN)
#define SK_SUB (SK_VSEQ + SK_RMAX * SK_SEQN)
// A skill duration order full past SK_SEQN flags the race instead of
// silently dropping a receiver.
#define FB_SKILL_SEQ 8u
#define SK_STAGED (SK_SUB + BANK_WORDS)   // chain-staging float values
// The follow-up application pool. A token SK_TOK_AA + idx stands for one
// pending application of an additionalActivate effect, and its kind, value
// and duration live here (not in the record timer chain) so a re-dispatched
// draw or tier reads back exactly what was registered.
#define SK_TOK_AA (SK_TOK_SKILL + (SK_RMAX << 3) + 8)
#define SK_AAPN 128
#define SK_AAPW 6      // kind|slot|rec, value bits, timer bits per entry
#define SK_AAP (SK_STAGED + 2 * SK_RMAX)
// The running-skill lists, in application order, cap 8 per runner.
#define SK_RUNN 8
#define SK_RUN (SK_AAP + SK_AAPN * SK_AAPW)
#define SCR_SK_WORDS (SK_RUN + RACE_MAXR * (1 + SK_RUNN))
// One f64 per effect slot of every record: the value each queued application
// registered, the number every fold reads.
#define SK_VVAL (SCR_SK_WORDS)
// Diagnostics only, free-form per wave: the additionalActivate traffic
// journal (word 0 the used word count, then ten-word records) plus the tick
// mirror race_tick writes. No layout meaning beyond the scratch total that
// tests/layout.rs pins.
#define SK_JRN (SK_VVAL + SK_RMAX * 16)
#define SK_JRN_N 16384
#define SK_JRN_T (SK_JRN + SK_JRN_N)
// Full-reply channels, live only while scr[SCR_RC_ON] is set. Channel bases
// hold a word count; data follows the header word. A cap overflow flags the
// race (FB_RC), which the host answers with the vendored engine.
#define SCR_RC_ON (SK_JRN_T + 1)
#define SCR_RC_SEQ (SCR_RC_ON + 1)   // shared activation ordinal counter
#define SCR_RC_LEAD (SCR_RC_SEQ + 1) // per-tick style leader flags
#define SCR_RC_RPT (SCR_RC_LEAD + RACE_MAXR) // 1 during the constructor repeat
#define SCR_RC_ACT (SCR_RC_RPT + 1)
#define RC_ACT_CAP 24576
#define SCR_RC_TGT (SCR_RC_ACT + 1 + RC_ACT_CAP)
#define RC_TGT_CAP 49152
#define SCR_RC_FAN (SCR_RC_TGT + 1 + RC_TGT_CAP)
#define RC_FAN_CAP 16384
#define SCR_RC_CONS (SCR_RC_FAN + 1 + RC_FAN_CAP)
#define RC_CONS_CAP 768
#define SCR_RC_COMP (SCR_RC_CONS + 1 + RC_CONS_CAP)
#define RC_COMP_CAP 768
#define SCR_RC_PHR (SCR_RC_COMP + 1 + RC_COMP_CAP)
#define RC_PHR_CAP 8192
#define SCR_RC_KAK (SCR_RC_PHR + 1 + RC_PHR_CAP)
#define RC_KAK_CAP 1024
#define SCR_U_WORDS (SCR_RC_KAK + 1 + RC_KAK_CAP)

// Per-race scratch f64, one RACE_MAXR-wide column per slot (field base +
// runner index). Array fields keep RACE_MAXR slots per element:
// base + element * RACE_MAXR + r.
#define SD_SLOTS (2 * RACE_MAXR + 83)
#define SD_STRIDE (SD_SLOTS * RACE_MAXR)
#define SD_SLOT(n) ((n) * RACE_MAXR)
#define SD_POS SD_SLOT(0)
#define SD_LANE SD_SLOT(1)
#define SD_LANET SD_SLOT(2)       // lane_target
#define SD_LCS SD_SLOT(3)         // lane_change_speed
#define SD_LCT SD_SLOT(4)         // move_lane_continue_time
#define SD_CS SD_SLOT(5)          // current_speed
#define SD_TS SD_SLOT(6)          // target_speed
#define SD_TSWSK SD_SLOT(13)      // target speed folded with current-speed skills
#define SD_OFA SD_SLOT(14)        // modifiers_one_frame_accel, the expiry impulse
#define SD_MS SD_SLOT(7)          // min_speed
#define SD_LSS SD_SLOT(8)         // last_spurt_speed
#define SD_BTS SD_SLOT(9)         // base_target_speed[3], element stride RACE_MAXR
#define SD_PKCOEF SD_SLOT(12)     // pos_keep_speed_coef
#define SD_SECMOD SD_SLOT(15)     // section_modifier[25], element stride RACE_MAXR
#define SD_DUEL SD_SLOT(40)       // duel proximity timers, element j stride RACE_MAXR, -1.0 absent
// Duel then overtake cooldown, each a RACE_MAXR-wide pair map, then the
// scalar timers. The duel map spans slots 40..40+RACE_MAXR, the overtake map
// 40+RACE_MAXR..40+2*RACE_MAXR, the scalars follow.
#define SD_OVC SD_SLOT(40 + RACE_MAXR)   // slot SD_OVC + target*RACE_MAXR + i, -1.0 until the pair fires once
#define SD_SPOT_TIMER SD_SLOT(40 + 2 * RACE_MAXR) // spot_struggle_timer, f64 (f32-rounded steps)
#define SD_CP_GAUGE (SD_SPOT_TIMER + RACE_MAXR)   // conserve power clean ledger gauge
#define SD_CP_PREVRATE (SD_SPOT_TIMER + 2 * RACE_MAXR) // conserve previous_rate
#define SD_CP_INITPW (SD_SPOT_TIMER + 3 * RACE_MAXR)  // conserve initial_power
#define SD_CP_ACC (SD_SPOT_TIMER + 4 * RACE_MAXR)     // legacy cp_acc accumulate
#define SD_AS_TIMER (SD_SPOT_TIMER + 5 * RACE_MAXR)   // asitame effect duration timer, counts up to zero
#define SD_RAWPW (SD_SPOT_TIMER + 6 * RACE_MAXR)      // horse raw_power from initialization output 2
#define SD_CLOCK (SD_SPOT_TIMER + 7 * RACE_MAXR)      // accumulatetime race clock, f64 (f32-rounded steps)
#define SD_BSIDE (SD_SPOT_TIMER + 8 * RACE_MAXR)      // blocked_side_continuetime, f64
#define SD_IFNEAR (SD_SPOT_TIMER + 9 * RACE_MAXR)     // infront_near_lane_time, f64
#define SD_BHNEAR (SD_SPOT_TIMER + 10 * RACE_MAXR)    // behind_near_lane_time, f64
#define SD_BHNEAR1 (SD_SPOT_TIMER + 11 * RACE_MAXR)   // behind_near_lane_time_set1, f64
#define SD_HPS (SD_SPOT_TIMER + 12 * RACE_MAXR)       // retail_stat_values[0] accumulator, f64 bits of f32
#define SD_PENDING_HP (SD_SPOT_TIMER + 13 * RACE_MAXR) // recovery HP banked for the next step head
#define SD_OTIME (SD_SPOT_TIMER + 14 * RACE_MAXR)      // overtake_target_time, targeted-by-behind timer
#define SD_BFRONT (SD_SPOT_TIMER + 15 * RACE_MAXR)     // blocked_front_continuetime, f64
#define SD_POSPREV (SD_SPOT_TIMER + 16 * RACE_MAXR)    // positions as published by the last field_update
#define SD_NOUP (SD_SPOT_TIMER + 17 * RACE_MAXR)       // overtake_target_no_order_up_time, f64
#define SD_NODOWN (SD_SPOT_TIMER + 18 * RACE_MAXR)     // overtake_target_no_order_down_time, f64
// The aux-skill lifetimes the target filter and the lane search read. The
// countdowns mirror the engine's new_skill_timer chain (negative runs, the
// SkillTimer::step rounding, retired when the timer reaches zero), and the
// retired state is +inf so an armed exact zero keeps the entry present for
// the tick it fires, exactly like the engine's vector before its sweep.
#define SK_GOALN 4                                     // live SetTargetLane entries per runner
#define SD_IMMUNE (SD_SPOT_TIMER + 19 * RACE_MAXR)     // DebuffImmunity countdown, engine race_solver.rs:2454
#define SD_GOAL_OUT (SD_SPOT_TIMER + 20 * RACE_MAXR)   // PreferOutsideLane presence countdown
#define SD_GOAL_IN (SD_SPOT_TIMER + 21 * RACE_MAXR)    // PreferInsideLane presence countdown
#define SD_GOALV (SD_SPOT_TIMER + 22 * RACE_MAXR)      // SetTargetLane entry lane, element stride RACE_MAXR
#define SD_GOALT (SD_SPOT_TIMER + (22 + SK_GOALN) * RACE_MAXR) // ... and its countdown, +inf free
// additional_turn_floor walks the ordered id pass, the sentinels standing in
// for i64::MIN and i64::MAX (all wire ids are small). The random-skill
// requests are the pending 36/37 effect counts, drained at the p0 pass end.
#define SD_AFLOOR (SD_GOALT + SK_GOALN * RACE_MAXR)
#define SD_RGOLD (SD_AFLOOR + RACE_MAXR)
#define SD_RANY (SD_RGOLD + RACE_MAXR)
#define SD_HPW (SD_RANY + RACE_MAXR)        // horse.wisdom field, f64
#define SD_HPW_A (SD_HPW + RACE_MAXR)       // retail_stat_values[4], f64 of an f32 accumulator
#define SD_HPG (SD_HPW_A + RACE_MAXR)       // horse.guts field, f64
#define SD_HPG_A (SD_HPG + RACE_MAXR)       // retail_stat_values[3], f64 of an f32 accumulator
#define SD_STAM (SD_HPG_A + RACE_MAXR)      // horse.stamina field, f64
#define SD_STAM_A (SD_STAM + RACE_MAXR)     // retail_stat_values[1], f64 of an f32 accumulator
#define SD_SPOT_MOD (SD_STAM_A + RACE_MAXR) // spot struggle modifier, captured at entry
#define SD_DUEL_T (SD_SPOT_MOD + RACE_MAXR) // duel target-speed modifier, captured at entry
#define SD_DUEL_A (SD_DUEL_T + RACE_MAXR)   // duel accel modifier, captured at entry
#define SRL_RUNNER 0
#define SRL_HP SD_STRIDE          // hp columns, SH_STRIDE per runner
#define SH_STRIDE 8
#define SH_MAX 0
#define SH_HP 1
#define SH_BASESPEED 2
#define SH_GROUNDMOD 3
// Race f64 scalars.
#define RD_T (SH_STRIDE * RACE_MAXR + SRL_HP)   // loop time, the engine f64 f32-sum
#define RD_WORDS (RD_T + 32)
// im_* f64 block bases in the same scratch (double indices).
#define SRL_STATS_IN_L RD_WORDS                            // RACE_MAXR * IIL
#define SRL_INIT_IN_L (SRL_STATS_IN_L + RACE_MAXR * IIL)   // IIL
#define SRL_INIT_OUT_L (SRL_INIT_IN_L + IIL)               // OOL
#define SRL_HP_IN_L (SRL_INIT_OUT_L + OOL)                 // RACE_MAXR * IIL
#define SRL_HP_OUT_L (SRL_HP_IN_L + RACE_MAXR * IIL)       // RACE_MAXR * OOL
#define SRL_SKD (SRL_HP_OUT_L + RACE_MAXR * OOL)           // per-record skill doubles
#define SRL_WORDS (SRL_SKD + SK_RMAX * SKD_STRIDE)

// SectionModifier length.
#define SEC_MOD 25

// ---------------------------------------------------------------------------
// Per-race status output, u32 words at status + id * ST_STRIDE.
#define ST_STRIDE (ST_SPFIRST0 + RACE_MAXR)
#define ST_FALLBACK 0          // 0 none, 1 unsupported domain (tick in ST_ARG)
#define ST_ARG 1               // tick or argument of the fallback
#define ST_TICKS 2             // loop iterations (frames carry ST_TICKS+1)
#define ST_FINISH0 8           // per runner: finish_step bits then finish_raw
// Compact-reply exports, per runner in roster order, one word each. The last
// three are f32 bit patterns, widened by the host like the engine's f64 casts.
#define ST_TEMPT0 (ST_FINISH0 + 4 * RACE_MAXR)  // temptation_count (rushingStartCount)
#define ST_RSCORE0 (ST_TEMPT0 + RACE_MAXR)      // RushingScore count (rushingPenaltyCount)
#define ST_HPEMPTY0 (ST_RSCORE0 + RACE_MAXR)    // hp_empty_on_race, 0 or 1
#define ST_DELAY0 (ST_HPEMPTY0 + RACE_MAXR)     // start_delay f32 bits (startDelayTime)
#define ST_LSS0 (ST_DELAY0 + RACE_MAXR)         // SF_SP_START f32 bits (lastSpurtStartDistance)
// Full-reply exports, written only when scr[SCR_RC_ON] is set. The snapshot
// pair is the live guts and wisdom field at the engine's parameter_orders
// point: after every constructor activation, before the startup pass. The
// spurt pair is the clock and position at the last start edge, valid when
// ST_SPFIRST0 is nonzero. ST_SPFIRST0 carries the first tick the runner was
// in its last spurt, plus one (0 none).
#define ST_SNAPG0 (ST_LSS0 + RACE_MAXR)         // horse.guts f64 bits, 2 per runner
#define ST_SNAPW0 (ST_SNAPG0 + 2 * RACE_MAXR)   // horse.wisdom f64 bits
#define ST_SPURTT0 (ST_SNAPW0 + 2 * RACE_MAXR)  // actual spurt clock f64 bits
#define ST_SPURTD0 (ST_SPURTT0 + 2 * RACE_MAXR) // actual spurt distance f64 bits
#define ST_SPFIRST0 (ST_SPURTD0 + 2 * RACE_MAXR)

// Race frames output: frames + (id * RACE_MAX_TICKS + tick) * n_runners * 6
// + runner * 6, in the engine's row order pos, speed, hp, lane, rushing,
// blocker.
#define FR_COLS 6

// Per-race skillEvents log, u32 words at events + id * EV_STRIDE. Word 0 is
// the entry count, then (runner, id) pairs. id is the skill id, or EV_KAKARI
// for a rushing episode that ended. The engine's activate hook stores the
// time and its deactivate hook emits the entry, so one ended episode is one
// entry; an entry per fire reproduces the reply's (horseIndex, id) multiset.
// An overflow flags the race (FB_EVENT) instead of dropping an entry.
#define EV_MAX 512
#define EV_STRIDE (1 + 2 * EV_MAX)
#define EV_KAKARI 0xFFFFFFFFu
#define FB_EVENT 7u

static void race_event(__global uint* ev, __global uint* st, const int runner,
    const uint id) {
    const uint count = ev[0];
    if (count < EV_MAX) {
        ev[1 + 2 * count] = (uint)runner;
        ev[2 + 2 * count] = id;
        ev[0] = count + 1u;
    } else {
        st[ST_FALLBACK] = 1u;
        st[ST_ARG] = FB_EVENT;
    }
}

#define FB_RC 9u

// Reserve len channel words and return the write pointer, or 0 when full
// mode is off. An overflow flags the race once and stops further channel
// writes so the host falls back cleanly instead of reading a torn record.
static __global uint* rc_take(__global uint* scr, __global uint* st,
    const int base, const int cap, const int len) {
    if (scr[SCR_RC_ON] == 0u) {
        return 0;
    }
    const uint used = scr[base];
    if (used + (uint)len > (uint)cap) {
        if (st[ST_FALLBACK] == 0u) {
            st[ST_FALLBACK] = 1u;
            st[ST_ARG] = FB_RC;
        }
        scr[SCR_RC_ON] = 0u;
        return 0;
    }
    scr[base] = used + (uint)len;
    return scr + base + 1 + used;
}

// Return words a reservation did not use. Only the newest reservation of a
// channel may be given back, which the capture sites guarantee.
static void rc_give(__global uint* scr, const int base, const int words) {
    scr[base] -= (uint)words;
}

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
// each runner's PRD_POWF word; the inputs are race constants.
static float phys_maximum_spurt_speed(__global const double* pd, const float speed,
    const int strategy, const float aptitude, const float distance,
    const double powf_word) {
    const float base = convert_float_rte(
        (double)phys_base_target_speed(pd, speed, strategy, 2, aptitude, distance)
            + (double)phys_course_speed(pd, distance) * (double)phys_f(pd, PT_SPURT));
    const double addition = ((double)phys_root(bounded_stat(speed) * phys_f(pd, PT_SPURT + 1))
        * (double)aptitude) * (double)phys_f(pd, PT_ADD_SPEED);
    const double guts_add = (double)as_float((uint)powf_word) * (double)phys_f(pd, PT_SPURT + 4);
    return convert_float_rte(
        ((double)base * (double)phys_f(pd, PT_SPURT + 5) + addition) + guts_add);
}

// Under PD_LIVE_GUTS these replace the prepare-baked PRD words, evaluated at
// the points the engine reads its live horse.guts: the spurt plan closure,
// the duel and spot entry snapshots. Each mirrors the host bake and the
// engine term for term, and gs_pow and gs_powf are the bit-exact ucrtbase
// ports gated in gpu/tests/wave8_powf_gate.rs.
static double sk_powf_word(__global const double* pd, const double field) {
    const float argument = bounded_stat(convert_float_rte(field)) * phys_f(pd, PT_SPURT + 2);
    return (double)as_uint(gs_powf(argument, phys_f(pd, PT_SPURT + 3)));
}

static double sk_duel_modifier(__global const double* pd, const uint cf, const double field) {
    const float argument = convert_float_rte(fmax(field, 1.0)) * phys_f(pd, cf);
    const float power = convert_float_rte(gs_pow((double)argument, (double)phys_f(pd, cf + 1)));
    return (double)(power * phys_f(pd, cf + 2));
}

static double sk_spot_modifier(const double field) {
    const float argument = convert_float_rte(fmax(field, 1.0)) * 500.0f;
    const float power = convert_float_rte(gs_pow((double)argument, (double)0.6f));
    return (double)(power * 0.0001f);
}

static double sk_spot_duration(__global const double* pd, const uint base, const double field) {
    return gs_pow(700.0 * fmax(field, 1.0), 0.5) * 0.012 * pd[base + PRD_SPOT_PROF];
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
// spot and duel are the CompeteTop/CompeteFight late terms (retail token 1
// and 4), zero when no token is registered.
static float phys_target_speed(__global const double* pd, const float base,
    const float section, const float pace, const float force_in, const float power,
    const float slope, const int downhill, const float minimum, const int hp_remaining,
    __global const float* mods, const int nmods, const float spot, const float duel,
    const int moving_lane_skill) {
    float speed = (base + section) * pace + force_in;
    speed = apply_additive_modifiers(speed, mods, nmods);
    if (moving_lane_skill) {
        // The lane-speed skill late term, evaluated right after the skill
        // modifiers and before the slope correction (retail_physics.rs).
        speed = speed + phys_root(bounded_stat(power) * 0.0002f);
    }
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
    speed += spot;
    speed += duel;
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

// acceleration_from_base_iter from retail_physics.rs. Without skills the
// chain is the duel token, then the Conserve Power (asitame) token, applied
// before the start-dash add.
static float phys_acceleration_from_base(__global const double* pd, float base,
    const int start_dash, __global const float* mods, const int nmods,
    const float duel, const int has_duel, const float asit, const int has_asit) {
    base = apply_additive_modifiers(base, mods, nmods);
    if (has_duel) {
        base += duel;
    }
    if (has_asit) {
        base += asit;
    }
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
    const float forced = sf[SF_RS_FORCED + r];
    if (forced >= 0.f) {
        // An armed forced clock owns the rush: the natural clocks freeze and
        // the end-percent lottery spends no draw (retail_states.rs:178-181).
        const float remaining = fmax(forced - dt, 0.f);
        sf[SF_RS_FORCED + r] = remaining;
        ended = remaining == 0.f;
    } else {
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
        ended = natural_end;
    }
    if (ended) {
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
// itself). Vision is the caller's tick-start visible_distance, still holding
// effects that expire this tick, as the engine's collect-before-prepare is.
// The collect pass scans roster indices ascending with strict
// minimums, the tie order both engine collect paths end in.

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
// stops the scan, and tie order is irrelevant. The query is anchored on the
// candidate's cached row; every roster slot is a blocker row.
static int lane_has_front_blocker(__global const float* cached, const int n,
    const int r, const float position, const float lane, const float front_distance,
    const float front_lane) {
    int j = 0;
    while (j < n) {
        const float gap = cached[4 * (CACHE_STRIDE * r + j)] - position;
        const float lane_gap = fabs(cached[4 * (CACHE_STRIDE * r + j) + 1] - lane);
        if (gap > 0. && gap < front_distance && lane_gap <= lane_front_lane_limit(
            gap, front_distance, front_lane)) {
            return 1;
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
    const int phase, const float max_lane, const float lane, const float position,
    const float vision) {
    const float front_distance = convert_float_rte(pd[PT_BLOCKING]);
    const float front_lane_param = convert_float_rte(pd[PL_FRONT_LANE]);
    const float side_distance = convert_float_rte(pd[PL_SIDE_DISTANCE]);
    const float side_lane = convert_float_rte(pd[PL_SIDE_LANE]);
    const float reach = fmax(fmax(vision, front_distance), side_distance);
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
        // The cache covers every roster slot, as the engine's
        // extend_from_slice does. The reach cutoff below only trims the
        // visible/front/side analysis; the crowd walk and front blocker
        // scan read cached rows far behind and far ahead.
        cached[4 * (CACHE_STRIDE * r + k)] = convert_float_rte(sd[SD_POS + k]);
        cached[4 * (CACHE_STRIDE * r + k) + 1] = convert_float_rte(sd[SD_LANE + k]);
        cached[4 * (CACHE_STRIDE * r + k) + 2] = sf0[SF_OBSERVED + k];
        cached[4 * (CACHE_STRIDE * r + k) + 3] = convert_float_rte(sd[SD_TSWSK + k]);
        if (k != r) {
            const float other_position = convert_float_rte(sd[SD_POS + k]);
            const float position_gap = other_position - position;
            if (position_gap > reach) {
                k += 1;
                continue;
            }
            const float other_lane = convert_float_rte(sd[SD_LANE + k]);
            const float lane_gap = other_lane - lane;
            if (lane_is_visible(position_gap, lane_gap, vision)) {
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
    // The flag drives the target add; SF_FORCE_IN keeps the horse's
    // coefficient from the initialization chain.
    si[SI_L_FORCEIN + r] = (uint)(empty_in_move && phase < 1
        && lane >= convert_float_rte(pd[PL_FORCE_IN_THRESHOLD]) && inner_nearby);
}

// update_targets from retail_lane.rs over the collect snapshot and live rows.
static int lane_eligible(__global float* cached, __global const double* sd,
    __global const float* sf0, const int r, const int n,
    __global const double* pd, const int front, const int j, const float position,
    const float owner_speed, const float owner_target) {
    const int k = j;
    const float gap = cached[4 * (CACHE_STRIDE * r + k)]
        - convert_float_rte(sd[SD_POS + r]);
    const float speed_gap = owner_speed - cached[4 * (CACHE_STRIDE * r + k) + 2];
    const int directly = (front == k);
    const float other_speed = sf0[SF_OBSERVED + k];
    const float other_target = convert_float_rte(sd[SD_TSWSK + k]);
    const float gap_f = convert_float_rte((double)gap);
    const float speed_gap_f = convert_float_rte((double)speed_gap);
    int res;
    if (gap_f < -1.f) {
        res = 0;
    } else if (directly) {
        res = other_speed < owner_target;
    } else if (!(gap_f >= 1.f && gap_f <= 20.f) || speed_gap_f <= 0.f
        || gap_f / speed_gap_f > 15.f) {
        res = 0;
    } else {
        const float owner_target_f = convert_float_rte((double)owner_target);
        const int blocked_result = other_speed < owner_target_f;
        const int unblocked_result = other_target < owner_target_f;
        if (blocked_result == unblocked_result) {
            res = blocked_result;
        } else if (lane_has_front_blocker(cached, n, r,
            cached[4 * (CACHE_STRIDE * r + k)], cached[4 * (CACHE_STRIDE * r + k) + 1],
            convert_float_rte(pd[PT_BLOCKING]),
            convert_float_rte(pd[PL_FRONT_LANE]))) {
            res = blocked_result;
        } else {
            res = unblocked_result;
        }
    }
    return res;
}

static void lane_update_targets(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    const int r, const int n, __global const double* pd, const float dt,
    const float position, const float lane, const float owner_speed,
    const float owner_target, const long dbg_row, __global double* fr) {
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
                const float gap = cached[4 * (CACHE_STRIDE * r + k)] - position;
                const float lane_gap = fabs(cached[4 * (CACHE_STRIDE * r + k) + 1] - lane);
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
    uint orig[6];
    int norig = 0;
    while (norig < existing_count && norig < 6) {
        orig[norig] = targets[norig];
        norig += 1;
    }
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
        if (!contains && lane_eligible(cached, sd, sf0, r, n, pd, front, j,
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
        if (ordinal > existing_count || lane_eligible(cached, sd, sf0, r, n, pd,
            front, (int)targets[t], position, owner_speed, owner_target)) {
            targets[write++] = targets[t];
        }
        t += 1;
    }
    targets_len = write;
    if (dbg_row >= 0) {
        int k = 0;
        while (k < 6) {
            double keep = -1.;
            if (k < norig) {
                keep = 0.;
                int q = 0;
                while (q < targets_len) {
                    if ((int)targets[q] == (int)orig[k]) { keep = 1.; }
                    q += 1;
                }
            }
            fr[dbg_row + 6 * (long)r + k] = keep;
            k += 1;
        }
    }
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
    const float out_space, const int lane_preference,
    const long dbg_base, __global double* fr) {
    const int targets_len = (int)si[SI_L_TGTN + r];
    const int visible_len = (int)si[SI_L_VISN + r];
    __global const uint* targets = lists + LIST_STRIDE * r;
    __global const uint* visible = lists + LIST_STRIDE * r + RACE_MAXR;
    // Trace row 3: the walk inputs of the second target, written near the
    // end of the candidate loop. Row 2 (survived-target flags) is written
    // by the target update.
    const long row3 = dbg_base >= 0 ? dbg_base + (long)12 * n : -1;
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
                const float cached_lane = cached[4 * (CACHE_STRIDE * r + edge) + 1];
                const float cached_position = cached[4 * (CACHE_STRIDE * r + edge)];
                int next = -1;
                float next_lane = 0.f;
                int j = 0;
                while (j < n) {
                    if (j != r) {
                        const float row_position = cached[4 * (CACHE_STRIDE * r + j)];
                        const float position_gap = row_position - cached_position;
                        const float lane_gap = direction
                            * (cached[4 * (CACHE_STRIDE * r + j) + 1] - cached_lane);
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
            if (row3 >= 0 && t == 1 && dsign == 1) {
                // Row 3: the crowd-walk inputs for the walk anchored on
                // target 1 (row 3's cached and live lane, row 4's cached
                // and live lane) and the + direction's final edge.
                fr[row3 + 6 * (long)r + 0] =
                    (double)cached[4 * (CACHE_STRIDE * r + 3) + 1];
                fr[row3 + 6 * (long)r + 1] =
                    (double)convert_float_rte(sd[SD_LANE + 3]);
                fr[row3 + 6 * (long)r + 2] =
                    (double)cached[4 * (CACHE_STRIDE * r + 4) + 1];
                fr[row3 + 6 * (long)r + 3] =
                    (double)convert_float_rte(sd[SD_LANE + 4]);
                fr[row3 + 6 * (long)r + 4] = (double)(candidate);
                fr[row3 + 6 * (long)r + 5] = (double)(edge);
            }
            if (!blocked) {
                const float origin = cached[4 * (CACHE_STRIDE * r + idx)];
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
    if (dbg_base >= 0) {
        fr[dbg_base + 6 * (long)r + 0] = (double)nearest;
        fr[dbg_base + 6 * (long)r + 1] = (double)best;
        fr[dbg_base + 6 * (long)r + 2] = (double)straight;
        fr[dbg_base + 6 * (long)r + 3] = (double)blocked;
        fr[dbg_base + 6 * (long)r + 4] = (double)(int)si[SI_L_FRONT_IDX + r];
        fr[dbg_base + 6 * (long)r + 5] = (double)targets_len;
    }
    return best;
}

// TargetState::target_sorted. lane_preference carries the presence of the
// PreferOutsideLane/PreferInsideLane aux flags and fixed_goal the last live
// SetTargetLane lane, both read from the aux lifetimes in the sd block.
static float lane_target(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    __global const double* pd, __global uint* bank, const int r, const int n,
    const float position, const float lane, const float owner_speed,
    const float owner_target, const int phase, const int has_hp, const int pace_down,
    const int lane_preference, const float fixed_goal,
    const long dbg_base, __global double* fr) {
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
            position, phase, max_lane, side_lane, can_in, can_out, in_space, out_space,
            lane_preference, dbg_base, fr);
        // The outward fallback: extra_outward is Some once the random bonus
        // is drawn, else a non-zero fixed_goal takes its place
        // (retail_lane.rs:400-411).
        const int has_xop = (int)si[SI_L_XOP + r];
        const int has_extra = has_xop || fixed_goal != 0.f;
        const float extra = has_xop ? sf[SF_L_XO + r] : fixed_goal;
        if (has_extra && target >= lane && extra > target
            && lane_reachable(can_in, can_out, in_space, out_space, lane, extra)) {
            target = extra;
        }
        return target;
    }
    if (fixed_goal != 0.f) {
        return fixed_goal > lane && can_out ? fmin(fixed_goal, lane + 0.05f)
            : (fixed_goal < lane && can_in ? fmax(fixed_goal, lane - 0.05f) : lane);
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

// MoveState::step with the lane-speed skill modifiers. The x87-style
// lateral-speed chain follows retail_lane.rs exactly.
static int lane_move_step(__global uint* si, __global float* sf, const int r,
    __global const double* pd, const int after_goal, const float position,
    const float power, const int phase, const int rank, const float first_move_point,
    const float max_lane, const int can_in, const int can_out, const float dt,
    __global const float* mods, const int nmods) {
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
        speed = clamp(apply_additive_modifiers(fmax(speed_next, 0.f), mods, nmods),
            0.f, 0.6f);
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

// comparison_sort from src/retail_sort.rs over a local (key, order) pair.
// The engine sorts roster indices by the truncated lane key, and the
// introsort's equal-key permutation is observable, so this mirrors the
// same structure as sort_candidates with an int comparator.
static int ro_cmp(const int a, const int b) {
    return (a > b) ? 1 : ((a < b) ? -1 : 0);
}

static void ro_swap(__global uint* key, __global uint* order, const int i, const int j) {
    if (i == j) {
        return;
    }
    uint tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
    tmp = key[i];
    key[i] = key[j];
    key[j] = tmp;
}

static void ro_swap_if_greater(__global uint* key, __global uint* order,
    const int i, const int j) {
    if (i != j && ro_cmp((int)key[i], (int)key[j]) > 0) {
        ro_swap(key, order, i, j);
    }
}

static void ro_down_heap(__global uint* key, __global uint* order, int i,
    const int size, const int lo) {
    const int vt = (int)key[lo + i - 1];
    const uint vx = order[lo + i - 1];
    while (2 * i <= size) {
        int child = 2 * i;
        if (child < size && ro_cmp((int)key[lo + child - 1], (int)key[lo + child]) < 0) {
            child += 1;
        }
        if (ro_cmp(vt, (int)key[lo + child - 1]) >= 0) {
            break;
        }
        key[lo + i - 1] = key[lo + child - 1];
        order[lo + i - 1] = order[lo + child - 1];
        i = child;
    }
    key[lo + i - 1] = (uint)vt;
    order[lo + i - 1] = vx;
}

static void ro_sort(__global uint* key, __global uint* order, const int len) {
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
                    ro_swap_if_greater(key, order, lo, hi);
                } else if (size == 3) {
                    ro_swap_if_greater(key, order, lo, hi - 1);
                    ro_swap_if_greater(key, order, lo, hi);
                    ro_swap_if_greater(key, order, hi - 1, hi);
                } else {
                    int i = lo;
                    while (i < hi) {
                        const int vt = (int)key[i + 1];
                        const uint vx = order[i + 1];
                        int j = i;
                        while (j >= lo && ro_cmp(vt, (int)key[j]) < 0) {
                            key[j + 1] = key[j];
                            order[j + 1] = order[j];
                            j -= 1;
                        }
                        key[j + 1] = (uint)vt;
                        order[j + 1] = vx;
                        i += 1;
                    }
                }
                break;
            }
            if (depth == 0) {
                int i = size / 2;
                while (i >= 1) {
                    ro_down_heap(key, order, i, size, lo);
                    i -= 1;
                }
                i = size;
                while (i >= 2) {
                    ro_swap(key, order, lo, lo + i - 1);
                    ro_down_heap(key, order, 1, i - 1, lo);
                    i -= 1;
                }
                break;
            }
            depth -= 1;
            const int middle = lo + (hi - lo) / 2;
            ro_swap_if_greater(key, order, lo, middle);
            ro_swap_if_greater(key, order, lo, hi);
            ro_swap_if_greater(key, order, middle, hi);
            ro_swap(key, order, middle, hi - 1);
            const int pivot = (int)key[hi - 1];
            int left = lo;
            int right = hi - 1;
            while (left < right) {
                left += 1;
                while (ro_cmp((int)key[left], pivot) < 0) {
                    left += 1;
                }
                right -= 1;
                while (ro_cmp(pivot, (int)key[right]) < 0) {
                    right -= 1;
                }
                if (left >= right) {
                    break;
                }
                ro_swap(key, order, left, right);
            }
            ro_swap(key, order, left, hi - 1);
            lo_stack[sp] = left + 1;
            hi_stack[sp] = hi;
            depth_stack[sp] = depth;
            sp += 1;
            hi = left - 1;
        }
    }
}

// resolve_overlaps_with_order over the post-movement rows. The engine's
// comparison_sort permutes equal keys non-stably, and the outward shifts
// then apply in that order.
static void lane_resolve_overlaps(__global uint* si, __global double* sd, const int n,
    __global uint* keys, __global uint* order) {
    int i = 0;
    while (i < n) {
        keys[i] = (uint)convert_int_rtz(sd[SD_LANE + i]);
        order[i] = (uint)i;
        i += 1;
    }
    ro_sort(keys, order, n);
    i = 0;
    while (i < n) {
        si[SI_L_FORCE + (int)order[i]] = 0u;
        i += 1;
    }
    const float rail = convert_float_rte(0.4 / 18.);
    i = 0;
    while (i + 1 < n) {
        const int previous = (int)order[i];
        const int current = (int)order[i + 1];
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

// ---------------------------------------------------------------------------
// Multi-runner field state: RaceField::update, the section pacemaker machine
// and the duel and spot competition state machines.

static int field_style(const int strategy) {
    return (strategy == 5) ? 1 : strategy;
}

// update_pacemaker (race_field.rs). It reads the order published by the
// previous update with live positions, once per field_update call.
static void field_pacemaker(__global uint* scr, __global const uint* si,
    __global const double* sd, const int n, const float dist_f, const float end_f,
    const double course_distance, const float range, const int threshold) {
    __global uint* rs = scr + SCR_RACE;
    __global const uint* order = rs + RS_ORDER;
    int overall = -1;
    for (int k = 0; k < n; k++) {
        const int i = (int)order[k];
        if (sd[SD_POS + i] < course_distance) { overall = i; break; }
    }
    if (overall < 0) {
        return;
    }
    const float pos_overall = convert_float_rte(sd[SD_POS + overall]);
    const int section = phys_section(pos_overall, dist_f);
    const int has = (int)rs[RS_PM_HAS];
    const int handed = (int)rs[RS_PM_HANDED];
    if (has && (float)section >= end_f && !handed) {
        return;
    }
    int active[RACE_MAXR];
    int na = 0;
    for (int k = 0; k < n; k++) {
        const int i = (int)order[k];
        if (sd[SD_POS + i] < course_distance) active[na++] = i;
    }
    int forward = 64;
    for (int k = 0; k < na; k++) {
        const int st = field_style((int)si[SI_STRATEGY + active[k]]);
        if (st < forward) forward = st;
    }
    int leader = active[0];
    for (int k = 0; k < na; k++) {
        if (field_style((int)si[SI_STRATEGY + active[k]]) == forward) { leader = active[k]; break; }
    }
    int selected = has ? (int)rs[RS_PM] - 1 : leader;
    const int old_section = has ? (int)rs[RS_PM_SEC] : 0;
    if (!has) {
        rs[RS_PM] = (uint)(leader + 1);
        rs[RS_PM_SEC] = 0u;
        rs[RS_PM_HAS] = 1u;
    }
    int pm_count = (int)rs[RS_PM_COUNT];
    int pm_handed = handed;
    const int style_overall = field_style((int)si[SI_STRATEGY + overall]);
    const int style_leader = field_style((int)si[SI_STRATEGY + leader]);
    const int entering = section > old_section && (float)section < end_f;
    if (entering) {
        if (style_overall == forward && !pm_handed) {
            selected = leader;
        }
        if ((forward > 1 || pm_handed)
            && pos_overall - convert_float_rte(sd[SD_POS + leader]) >= range) {
            int best = -1;
            for (int k = 0; k < na; k++) {
                const int i = active[k];
                if (pos_overall - convert_float_rte(sd[SD_POS + i]) < range
                    && (best < 0 || field_style((int)si[SI_STRATEGY + i])
                        < field_style((int)si[SI_STRATEGY + best]))) {
                    best = i;
                }
            }
            selected = best < 0 ? overall : best;
            pm_count = 0;
        }
        if (style_overall != forward) {
            pm_count += 1;
            const int below = pm_count < threshold;
            if (!below || pm_handed) {
                if (!below) pm_count = 0;
                int candidate = overall;
                if (style_overall >= 3) {
                    candidate = -1;
                    for (int k = 0; k < na; k++) {
                        const int i = active[k];
                        if (field_style((int)si[SI_STRATEGY + i]) < style_overall
                            && pos_overall - convert_float_rte(sd[SD_POS + i]) <= range) {
                            candidate = i;
                            break;
                        }
                    }
                    if (candidate < 0) candidate = overall;
                }
                if (!below || (field_style((int)si[SI_STRATEGY + candidate]) == forward
                    && forward <= 1)) {
                    selected = candidate;
                    if (field_style((int)si[SI_STRATEGY + candidate]) != forward) {
                        pm_count = 0;
                        pm_handed = 1;
                    }
                }
            }
        }
        rs[RS_PM] = (uint)(selected + 1);
        rs[RS_PM_SEC] = (uint)section;
    }
    if (!entering && (pm_handed || pm_count >= threshold)) {
        int pos_leader = -1, pos_selected = -1;
        for (int k = 0; k < na; k++) {
            if (active[k] == leader) pos_leader = k;
            if (active[k] == selected) pos_selected = k;
        }
        if (style_overall == forward
            || (leader != selected && pos_leader < pos_selected)) {
            int returning = -1;
            for (int k = 0; k < na && active[k] != selected; k++) {
                if (field_style((int)si[SI_STRATEGY + active[k]]) == forward) {
                    returning = active[k];
                }
            }
            rs[RS_PM] = (uint)((returning < 0 ? leader : returning) + 1);
            rs[RS_PM_SEC] = (uint)section;
            pm_count = 0;
            pm_handed = 0;
        }
    }
    rs[RS_PM_COUNT] = (uint)pm_count;
    rs[RS_PM_HANDED] = (uint)pm_handed;
}

// RaceField::update: prev rank copy, the (position descending, index
// ascending) rank pass, style prefix maxima and the pacemaker pass. The
// overtake cooldown block and order-band tracking feed skill state. The
// compiler refuses the only other rank-gain consumers: duration_usage 4,
// additionalActivate and value_usage 11.
static void field_update(__global uint* scr, __global uint* si,
    __global double* sd, const int n, const float dist_f, const float end_f,
    const double course_distance, const float range, const int threshold) {
    __global uint* rs = scr + SCR_RACE;
    __global uint* order = rs + RS_ORDER;
    for (int s = 0; s < 8; s++) rs[RS_COUNTS + s] = 0u;
    for (int i = 0; i < n; i++) rs[RS_COUNTS + (int)si[SI_STRATEGY + i]] += 1u;
    for (int i = 0; i < n; i++) rs[RS_PREVRANK + i] = si[SI_RANKCOL + i];
    for (int i = 0; i < n; i++) order[i] = (uint)i;
    for (int k = 1; k < n; k++) {
        const int i = (int)order[k];
        int m = k;
        while (m > 0) {
            const int p = (int)order[m - 1];
            const int shift = (sd[SD_POS + i] > sd[SD_POS + p])
                || (sd[SD_POS + i] == sd[SD_POS + p] && p > i);
            if (!shift) {
                break;
            }
            order[m] = order[m - 1];
            m -= 1;
        }
        order[m] = (uint)i;
    }
    // Order-band "continue" tracking runs on the previous rank pass, before
    // the new rank publishes. The rate in the filter's name picks the cell,
    // and track_order_bands updates all eight cells every tick: the out
    // bands take rank >= threshold, the in bands rank <= threshold, and the
    // in-80 cell uses the 0.8 threshold track_order_bands passes as [4].
    const double band_thr[5] = {round(0.2 * (double)n), round(0.4 * (double)n),
        round(0.5 * (double)n), round(0.7 * (double)n), round(0.8 * (double)n)};
    const int band_out[4] = {SI_BND20, SI_BND40, SI_BND50, SI_BND70};
    const int band_in[4] = {SI_BNDIN20, SI_BNDIN40, SI_BNDIN50, SI_BNDIN80};
    const int band_in_thr[4] = {0, 1, 2, 4};
    for (int r = 0; r < n; r++) {
        if (sd[SD_CLOCK + r] < 5.0) {
            continue;
        }
        const double prev_rank = (double)(int)rs[RS_PREVRANK + r];
        for (int b = 0; b < 4; b++) {
            uint cell = si[band_out[b] + r];
            const int in_band = prev_rank >= band_thr[b];
            if (cell == 2u) {
            } else if (!in_band) {
                si[band_out[b] + r] = 2u;
            } else if (cell == 0u) {
                si[band_out[b] + r] = 1u;
            }
            cell = si[band_in[b] + r];
            if (cell == 2u) {
            } else if (prev_rank > band_thr[band_in_thr[b]]) {
                si[band_in[b] + r] = 2u;
            } else if (cell == 0u) {
                si[band_in[b] + r] = 1u;
            }
        }
    }
    for (int r = 0; r < n; r++) si[SI_RANKCOL + (int)order[r]] = (uint)(r + 1);
    int highest = 0;
    for (int k = 0; k < n; k++) {
        const int i = (int)order[k];
        rs[RS_STYLE + i] = (uint)highest;
        const int st = field_style((int)si[SI_STRATEGY + i]);
        if (st > highest) highest = st;
    }
    // The rank-gain pass: improvement against the published previous rank,
    // gated by a 6 s cooldown per (runner, just-passed) pair. Runners already
    // past the line at the previous publish keep their counters frozen.
    for (int i = 0; i < n; i++) {
        if (sd[SD_POSPREV + i] < course_distance) {
            const int rank = (int)si[SI_RANKCOL + i];
            if ((int)rs[RS_PREVRANK + i] > rank) {
                int event = 1;
                if (rank < n) {
                    const int target = (int)order[rank];
                    const double now = sd[SD_CLOCK + i];
                    __global double* cd = sd + SD_OVC + target * RACE_MAXR + i;
                    if (now < *cd) {
                        event = 0;
                    } else {
                        *cd = now + 6.0;
                    }
                }
                if (event) {
                    // record_overtakes counts the pass into the current phase,
                    // and the final-corner history reads the course-event flag
                    // of the tick start, prepared before the movement above.
                    const int ph = (int)si[SI_PHASE + i];
                    if (ph == 1) {
                        si[SI_POV1 + i] += 1u;
                    } else if (ph == 2) {
                        si[SI_POV2 + i] += 1u;
                    }
                    if (si[SI_FINALE + i] != 0u) {
                        si[SI_POVFC + i] += 1u;
                    }
                    // pending_overtakes feeds the kind-1 activations at the
                    // next prepare; late_corner_overtakes gates the
                    // value_usage 11 tier on the tick-start phase and corner.
                    si[SI_PENDN + i] += 1u;
                    if ((int)si[SI_RPHASE + i] >= 2 && (int)si[SI_CORNER + i] != 0) {
                        si[SI_LCO + i] += 1u;
                    }
                    sd[SD_NOUP + i] = 0.0;
                }
            }
            if ((int)rs[RS_PREVRANK + i] < rank) {
                // The engine resets the down timer on any rank drop, whether
                // or not this pass passed anybody.
                sd[SD_NODOWN + i] = 0.0;
            }
        }
        sd[SD_POSPREV + i] = sd[SD_POS + i];
    }
    field_pacemaker(scr, si, sd, n, dist_f, end_f, course_distance, range, threshold);
}

// collect_retail_duel_proximity: the preupdate snapshots and the proximity
// timer map (SD_DUEL, absent slots are -1.0). straight_flags hold the
// final-straight row flags of the pre-movement rows.
static void duel_collect(__global uint* si, __global float* sf, __global double* sd,
    const int r, const int n, __global const double* pd, const float dt,
    const double course_distance, const int any_straight,
    __global const uint* straight_flags) {
    sf[SF_DUEL_PRE + r] = sf[SF_OBSERVED + r];
    const int crossing = (int)si[SI_GOAL_SNAP + r];
    si[SI_GOAL_SNAP + r] = 0u;
    const float position = convert_float_rte(sd[SD_POS + r]);
    const float lane = convert_float_rte(sd[SD_LANE + r]);
    const int straight = (int)straight_flags[r];
    si[SI_DUEL_PRE_FIN + r] =
        (uint)(position >= convert_float_rte(course_distance) && !crossing);
    si[SI_DUEL_PRE_STR + r] = (uint)(straight || crossing);
    for (int j = 0; j < n; j++) {
        __global double* slot = sd + SD_DUEL + j * RACE_MAXR + r;
        if (!straight && !any_straight) {
            *slot = -1.0;
            continue;
        }
        const int ok = (j != r)
            && fabs(convert_float_rte(sd[SD_POS + j]) - position)
                <= convert_float_rte(pd[PD_CF_DIST])
            && fabs(convert_float_rte(sd[SD_LANE + j]) - lane)
                <= convert_float_rte(pd[PD_CF_LANE])
            && (straight || (int)straight_flags[j]);
        if (!ok) {
            *slot = -1.0;
            continue;
        }
        const double v = *slot;
        *slot = (v >= 0.) ? (double)(convert_float_rte(v) + dt) : 0.0;
    }
}

static void duel_activate(__global uint* scr, __global uint* st,
    __global uint* si, __global double* sd, __global const double* pd, const int x) {
    si[SI_DUEL_ACTIVE + x] = 1u;
    si[SI_DUEL_FIGHT + x] += 1u;
    si[SI_DUEL_TOKEN + x] = 1u;
    if ((int)pd[PD_LIVE_GUTS]) {
        // activate_retail_duel snapshots both modifiers from the live field.
        sd[SD_DUEL_T + x] = sk_duel_modifier(pd, PD_CF_SPEED, sd[SD_HPG + x]);
        sd[SD_DUEL_A + x] = sk_duel_modifier(pd, PD_CF_ACCEL, sd[SD_HPG + x]);
    }
    // The engine pushes the CompeteFight record right after activation.
    __global uint* rc = rc_take(scr, st, SCR_RC_COMP, RC_COMP_CAP, 3);
    if (rc) {
        rc[0] = scr[SK_JRN_T];
        rc[1] = 5u;
        rc[2] = (uint)x;
    }
}

// update_dueling, the retail path. duel_low_hp uses the plain f64 ratio,
// eligible_straight uses the preupdate snapshots with the corner state, and
// the duel draws no RNG.
static void duel_update(__global uint* scr, __global uint* st,
    __global uint* si, __global const float* sf,
    __global double* sd, __global double* shl, __global const double* pd, const int r, const int n,
    const double course_distance, const int has_corner_events) {
    if (!has_corner_events) {
        return;
    }
    if ((int)si[SI_DUEL_ACTIVE + r]) {
        int nearby_participant = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) {
                continue;
            }
            if ((int)si[SI_DUEL_FIGHT + j] > 0
                && (int)si[SI_DUEL_GROUP + j] == (int)si[SI_DUEL_GROUP + r]
                && fabs(sd[SD_POS + j] - sd[SD_POS + r]) <= 5.0) {
                nearby_participant = 1;
                break;
            }
        }
        const double self_ratio =
            fmax(shl[r * SH_STRIDE + SH_HP] / shl[r * SH_STRIDE + SH_MAX], 0.0);
        if (self_ratio < 0.05 || !nearby_participant) {
            si[SI_DUEL_ACTIVE + r] = 0u;
            si[SI_DUEL_TOKEN + r] = 0u;
        }
        return;
    }
    const int eligible_straight = (int)si[SI_DUEL_PRE_STR + r]
        || ((int)si[SI_DUEL_PRE_FIN + r] && (int)si[SI_FINALE + r]
            && (int)si[SI_CORNER + r] == 0);
    if (!eligible_straight || (int)si[SI_DUEL_FIGHT + r] > 0) {
        return;
    }
    const int rank_half = ((n + 1) / 2 > 1) ? (n + 1) / 2 : 1;
    const int top_half = (int)si[SI_RANKCOL + r] <= rank_half;
    const double self_ratio =
        fmax(shl[r * SH_STRIDE + SH_HP] / shl[r * SH_STRIDE + SH_MAX], 0.0);
    if (self_ratio < 0.15) {
        return;
    }
    int nearby = -1, eligible = -1, ready = -1;
    for (int j = 0; j < n; j++) {
        if (j == r) {
            continue;
        }
        const double timer = sd[SD_DUEL + j * RACE_MAXR + r];
        if (timer < 0.) {
            continue;
        }
        if (nearby < 0) {
            nearby = j;
        }
        const double other_ratio =
            fmax(shl[j * SH_STRIDE + SH_HP] / shl[j * SH_STRIDE + SH_MAX], 0.0);
        const int qualifies = other_ratio >= 0.15
            && fabs((double)sf[SF_DUEL_PRE + j] - (double)sf[SF_DUEL_PRE + r])
                <= (double)0.6f
            && (top_half || (int)si[SI_RANKCOL + j] <= rank_half);
        if (qualifies) {
            if (eligible < 0) eligible = j;
            if (timer >= 2.0 && (ready < 0 || timer > sd[SD_DUEL + ready * RACE_MAXR + r])) {
                ready = j;
            }
        }
    }
    const int target = ready >= 0 ? ready : (eligible >= 0 ? eligible : nearby);
    if (ready < 0) {
        return;
    }
    const int j = target;
    if ((int)si[SI_DUEL_FIGHT + j] > 0 && !(int)si[SI_DUEL_ACTIVE + j]) {
        return;
    }
    const int peer_group = (int)si[SI_DUEL_GROUP + j];
    const int own_group = (int)si[SI_DUEL_GROUP + r];
    const int group = peer_group >= 0 ? peer_group
        : (own_group >= 0 ? own_group : (r < j ? r : j));
    if (own_group < 0) si[SI_DUEL_GROUP + r] = (uint)group;
    if (peer_group < 0) si[SI_DUEL_GROUP + j] = (uint)group;
    duel_activate(scr, st, si, sd, pd, r);
    if (!(int)si[SI_DUEL_ACTIVE + j] && (int)si[SI_DUEL_FIGHT + j] == 0) {
        duel_activate(scr, st, si, sd, pd, j);
    }
}

// latch_active_spot_cutoff plus update_spot_struggle, the Nige/Oonige state
// machine. Timers were advanced in prepare, the token bit drives the target
// chain, and no part of it draws RNG.
static void spot_update(__global uint* scr, __global uint* st, __global uint* si, __global double* sd,
    const int r, const int n, __global const double* pd, const double section_length) {
    const int strategy = (int)si[SI_STRATEGY + r];
    if (strategy != 1 && strategy != 5) {
        return;
    }
    const double pos = sd[SD_POS + r];
    const double lane = sd[SD_LANE + r];
    const double timer = sd[SD_SPOT_TIMER + r];
    __global uint* rs = scr + SCR_RACE;
    if ((int)si[SI_SPOT_STARTED + r]) {
        if ((int)si[SI_SPOT_ACTIVE + r] && !(int)si[SI_SPOT_EXITED + r]) {
            int distance_exit = 1;
            for (int j = 0; j < n; j++) {
                if (j == r) {
                    continue;
                }
                const int holds = (int)si[SI_STRATEGY + j] == strategy
                    && (int)si[SI_SPOT_STARTED + j]
                    && fabs(sd[SD_POS + j] - pos) <= 5.0
                    && fabs(sd[SD_LANE + j] - lane) < 0.416;
                if (holds) {
                    distance_exit = 0;
                    break;
                }
            }
            if (distance_exit) si[SI_SPOT_EXITED + r] = 1u;
        }
        if ((int)si[SI_SPOT_ACTIVE + r] && timer < 0.0
            && pos >= section_length * 8.0) {
            si[SI_SPOT_LATCH + r] = 1u;
        }
        int group_cutoff = (int)si[SI_SPOT_LATCH + r];
        if (!group_cutoff) {
            for (int j = 0; j < n; j++) {
                if (j != r && (int)si[SI_SPOT_STARTED + j] && (int)si[SI_SPOT_LATCH + j]) {
                    group_cutoff = 1;
                    break;
                }
            }
        }
        if ((int)si[SI_SPOT_ACTIVE + r]
            && ((int)si[SI_SPOT_EXITED + r] || timer >= 0.0 || group_cutoff)) {
            si[SI_SPOT_ACTIVE + r] = 0u;
            si[SI_SPOT_TOKEN + r] = 0u;
        }
        return;
    }
    if ((int)si[SI_SPOT_CLOSED + r] || (int)rs[RS_COUNTS + strategy] < 2) {
        return;
    }
    const double entry_end = section_length * 6.0;
    double race_front = -INFINITY;
    int eligible_front_rank = 0x7fffffff;
    int front_idx = -1;
    double front_pos = -INFINITY, front_lane = 0.0;
    int group_len = 0, group_late = 1;
    for (int j = 0; j < n; j++) {
        const int j_strategy = (int)si[SI_STRATEGY + j];
        const int j_rank = (int)si[SI_RANKCOL + j];
        if (j_rank == 1) race_front = sd[SD_POS + j];
        if (strategy == 5 || j_strategy != 5) {
            if (j_rank < eligible_front_rank) eligible_front_rank = j_rank;
        }
        if (j_strategy == strategy) {
            if ((int)si[SI_SPOT_STARTED + j]) {
                return;
            }
            group_len += 1;
            if (sd[SD_POS + j] < entry_end) group_late = 0;
            if (front_idx < 0 || j_rank < (int)si[SI_RANKCOL + front_idx]) {
                front_pos = sd[SD_POS + j];
                front_lane = sd[SD_LANE + j];
                front_idx = j;
            }
        }
    }
    if (group_late) {
        si[SI_SPOT_CLOSED + r] = 1u;
        return;
    }
    if (group_len < 2 || race_front < 150.0) {
        return;
    }
    if ((int)si[SI_RANKCOL + front_idx] != eligible_front_rank) {
        return;
    }
    int count = 0, early = 0, started[RACE_MAXR];
    for (int j = 0; j < n; j++) {
        // Every slot needs a value: the start loop below reads all j, and a
        // skipped other-style row would otherwise read stack garbage.
        started[j] = 0;
        if ((int)si[SI_STRATEGY + j] != strategy) {
            continue;
        }
        const int invited = front_pos - sd[SD_POS + j] <= 3.75
            && fabs(front_lane - sd[SD_LANE + j]) <= 0.165
            && (j == front_idx || sd[SD_POS + j] < entry_end);
        started[j] = invited;
        if (invited) {
            count += 1;
            if (sd[SD_POS + j] < entry_end) early = 1;
        }
    }
    if (count < 2 || !early) {
        return;
    }
    // The engine records the leader before its partners, partners in roster
    // order; the field reorders these slots before serialization.
    __global uint* rc = rc_take(scr, st, SCR_RC_COMP, RC_COMP_CAP, 3);
    if (rc) {
        rc[0] = scr[SK_JRN_T];
        rc[1] = 4u;
        rc[2] = (uint)front_idx;
    }
    for (int j = 0; j < n; j++) {
        if (!started[j] || j == front_idx) {
            continue;
        }
        rc = rc_take(scr, st, SCR_RC_COMP, RC_COMP_CAP, 3);
        if (rc) {
            rc[0] = scr[SK_JRN_T];
            rc[1] = 4u;
            rc[2] = (uint)j;
        }
    }
    for (int j = 0; j < n; j++) {
        if (!started[j]) {
            continue;
        }
        const uint base = (uint)(PD_RUN0 + j * PD_RUN_STRIDE);
        si[SI_SPOT_ACTIVE + j] = 1u;
        si[SI_SPOT_STARTED + j] = 1u;
        si[SI_SPOT_TOKEN + j] = 1u;
        if ((int)pd[PD_LIVE_GUTS]) {
            // start_spot_struggle snapshots the modifier and the timer from
            // the live field.
            const double field = sd[SD_HPG + j];
            sd[SD_SPOT_MOD + j] = sk_spot_modifier(field);
            sd[SD_SPOT_TIMER + j] = -sk_spot_duration(pd, base, field);
        } else {
            sd[SD_SPOT_TIMER + j] = -pd[base + PRD_SPOT_D];
        }
    }
}

// ---------------------------------------------------------------------------
// The Conserve Power ledger (conserve_power.rs) and the asitame release the
// builder attaches to every retail runner (race_solver.rs, race_solver_builder.rs).

// ConservePowerState::duration when supported; the clean-coefficient table
// returns zero for unlisted distance types.
static double cp_clean_duration(__global const double* sd, const int r,
    const double distance, const int dtype) {
    double coef;
    if (dtype == 1) {
        coef = 0.45;
    } else if (dtype == 2) {
        coef = 1.;
    } else if (dtype == 3) {
        coef = 0.875;
    } else if (dtype == 4) {
        coef = 0.8;
    } else {
        return 0.;
    }
    return (double)convert_float_rte(
        sd[SD_CP_GAUGE + r] * sqrt(distance) * coef / 1450.);
}

// asitame_release_duration: K times the distance-type activity coefficient
// times the legacy gauge.
static double cp_legacy_duration(const double acc, const int dtype) {
    const double dtc = (dtype == 1) ? 0.45
        : ((dtype == 2) ? 1. : ((dtype == 3) ? 0.875 : 0.8));
    return (double)convert_float_rte(0.0185 * dtc * acc);
}

// conserve_release_duration: the clean ledger when supported, else legacy.
static double cp_release_duration(__global const uint* si, __global const double* sd,
    const int r, const double distance, const int dtype) {
    const uint flags = si[SI_CPFLAGS + r];
    if ((flags & CP_INITSET) && !(flags & CP_UNSUPPORTED)) {
        return cp_clean_duration(sd, r, distance, dtype);
    }
    return cp_legacy_duration(sd[SD_CP_ACC + r], dtype);
}

// ConservePowerState::update, minus the trace-only now clock.
static void cp_update(__global uint* si, __global float* sf, __global double* sd,
    const int r, const double distance, const int strategy, const float dt,
    const int blocked, const int pace_down, const int rushing, const int spot) {
    uint flags = si[SI_CPFLAGS + r];
    if (flags & CP_UNSUPPORTED) {
        return;
    }
    if (flags & CP_CLOCK) {
        sf[SF_CP_GELAP + r] += dt;
    }
    const int accumulating = sd[SD_POS + r] < floor(distance / 24.0) * 10.0;
    if (accumulating) {
        const double power = sd[SD_RAWPW + r];
        if (power <= 1200. || strategy < 1 || strategy > 5
            || ((flags & CP_INITSET) && sd[SD_CP_INITPW + r] != power)) {
            si[SI_CPFLAGS + r] = flags | CP_UNSUPPORTED;
            return;
        }
        if (!(flags & CP_INITSET)) {
            sd[SD_CP_INITPW + r] = power;
            sf[SF_CP_GELAP + r] = 0.f;
            sf[SF_CP_GTHRESH + r] = 1.f;
            flags |= CP_INITSET | CP_CLOCK | CP_HASTH;
        }
    }
    if (!(flags & CP_CUTOFF)) {
        while ((flags & CP_HASTH)
            && sf[SF_CP_GELAP + r] >= sf[SF_CP_GTHRESH + r]) {
            sd[SD_CP_GAUGE + r] += sd[SD_CP_PREVRATE + r];
            sf[SF_CP_GTHRESH + r] += 1.f;
        }
    }
    if (rushing && !(flags & CP_WASRUSH)) {
        sd[SD_CP_GAUGE + r] *= 0.8;
    }
    if (spot && !(flags & CP_WASSPOT)) {
        sd[SD_CP_GAUGE + r] *= 0.95;
    }
    flags = (flags & ~(CP_WASRUSH | CP_WASSPOT))
        | (rushing ? CP_WASRUSH : 0u) | (spot ? CP_WASSPOT : 0u);
    if (!accumulating) {
        si[SI_CPFLAGS + r] = flags | CP_CUTOFF;
        return;
    }
    sd[SD_CP_PREVRATE + r] = (pace_down ? 6.7 : 4.2)
        * ((strategy == 2) ? 0.75 : ((strategy == 4) ? 0.7 : 1.0));
    if (blocked || rushing || spot) {
        flags |= CP_WASBLK;
        flags &= ~CP_HASTH;
    } else if (flags & CP_WASBLK) {
        sf[SF_CP_GELAP + r] = 0.f;
        sf[SF_CP_GTHRESH + r] = 2.5f;
        flags |= CP_HASTH;
        flags &= ~CP_WASBLK;
    }
    si[SI_CPFLAGS + r] = flags;
}

// activate_skill for the injected asitame Accel effect, from the release
// path. The modifier follows the release-conserve parameters and the effect
// timer counts up from the negated scaled duration.
static void asitame_release(__global uint* scr, __global uint* st,
    __global uint* si, __global float* sf,
    __global double* sd, const int r, __global const double* pd,
    const double distance, const int strategy) {
    uint flags = si[SI_CPFLAGS + r];
    if (flags & CP_FIRED) {
        return;
    }
    const double pos = sd[SD_POS + r];
    if (!(pos >= distance * 2.0 / 3.0 && pos < distance)) {
        return;
    }
    const double power = sd[SD_RAWPW + r];
    if (!(power > 1200.)) {
        return;
    }
    const int dtype = (int)pd[PD_DTYPE];
    if ((flags & CP_INITSET) && !(flags & CP_UNSUPPORTED)
        && !(cp_clean_duration(sd, r, distance, dtype)
            >= (double)convert_float_rte(0.2f))) {
        return;
    }
    double coef;
    if (dtype == 1) {
        coef = (strategy == 2) ? 0.7 : ((strategy == 3) ? 0.75
            : ((strategy == 4) ? 0.7 : ((strategy == 0) ? 0. : 1.)));
    } else if (dtype == 2) {
        coef = (strategy == 2) ? 0.8 : ((strategy == 3) ? 0.7
            : ((strategy == 4) ? 0.75 : ((strategy == 0) ? 0. : 1.)));
    } else if (dtype == 3) {
        coef = (strategy == 2) ? 0.9 : ((strategy == 3) ? 0.875
            : ((strategy == 4) ? 0.86 : ((strategy == 0) ? 0. : 1.)));
    } else if (dtype == 4) {
        coef = (strategy == 2) ? 0.9 : ((strategy == 3) ? 1.
            : ((strategy == 4) ? 0.9 : ((strategy == 0) ? 0. : 1.)));
    } else {
        coef = 0.;
    }
    const float arg = fmax((convert_float_rte(power) - 1200.f) * 130.f, 0.f);
    const float root = convert_float_rte(sqrt((double)arg));
    double modifier = (double)convert_float_rte((double)root
        * (double)convert_float_rte(0.001f)
        * (double)convert_float_rte((float)coef));
    if (flags & CP_STRUGGLE) {
        modifier *= 0.98;
    }
    if (flags & CP_KAKARI) {
        modifier *= 0.8;
    }
    flags |= CP_FIRED | CP_ASIT;
    sf[SF_AS_VALUE + r] = convert_float_rte(modifier);
    sd[SD_AS_TIMER + r] = -cp_release_duration(si, sd, r, distance, dtype);
    si[SI_CPFLAGS + r] = flags;
    // The engine's activate hook records the conserve event with the clock.
    const ulong cbits = as_ulong(sd[SD_CLOCK + r]);
    __global uint* rc = rc_take(scr, st, SCR_RC_CONS, RC_CONS_CAP, 3);
    if (rc) {
        rc[0] = (uint)r;
        rc[1] = (uint)cbits;
        rc[2] = (uint)(cbits >> 32);
    }
}

// ---------------------------------------------------------------------------
// The skill runtime (handoff step 7): the timer step, effect expiry, the
// retail cooldown machine, the whole-field priority scan with its fire gate,
// and the receiver token orders the physics chains fold. skills.cl carries the
// table layout, and the engine counterparts are race_solver.rs (scan gate,
// fire_pending_skill, cooldowns, apply_activation) and race_solver_retail.rs
// (timer step, receiver folds).

static __global uint* sk_seq(__global uint* scr, const int kind) {
    // Recovery, the runtime no-op and the raw stat bump hold no duration
    // token: their fire writes state directly.
    return scr + (kind == SK_EFF_TARGET ? SK_TSEQ
        : (kind == SK_EFF_ACCEL ? SK_ASEQ
            : (kind == SK_EFF_LANESPEED ? SK_LSEQ
                : (kind == SK_EFF_VISION ? SK_VSEQ : SK_CSEQ))));
}

static int sk_queued_kind(const int kind) {
    return kind == SK_EFF_TARGET || kind == SK_EFF_ACCEL || kind == SK_EFF_CURRENT
        || kind == SK_EFF_CURRENT_NAT || kind == SK_EFF_LANESPEED
        || kind == SK_EFF_VISION;
}

// is_skill_demerit (race_field.rs:55-63): the debuff class the target filter
// and the immune self-skip gate on. Kind 13 is a demerit whatever its value.
static int sk_is_debuff(const int kind, const double modifier) {
    if (kind == 13 || kind == 14) {
        return 1;
    }
    if (kind == 10) {
        return modifier > 1.0;
    }
    if (kind == 29) {
        return !(modifier < 0.0);
    }
    if ((kind >= 1 && kind <= 5) || kind == 8 || kind == 9 || kind == 21
        || kind == 22 || kind == 27 || kind == 28 || kind == 30 || kind == 31
        || kind == 32) {
        return modifier < 0.0;
    }
    return 0;
}

// Arm one aux presence cell from a fire: the engine keeps one entry per
// fire and presence reads any of them, so the single cell takes the newest
// entry's countdown only when it outlives what is already armed.
static void sk_aux_arm(__global double* cell, const double scaled) {
    const double dur = convert_float_rte(scaled);
    *cell = fmin(*cell, dur < 0.0 ? -INFINITY : -dur);
}

// Append a SetTargetLane entry (lane value plus its countdown) to the
// runner's live list. 0 when all SK_GOALN slots are busy, the rare case
// that flags the race.
static int sk_goal_push(__global double* sd, const int j, const float lane,
    const double scaled) {
    for (int e = 0; e < SK_GOALN; e++) {
        if (sd[SD_GOALT + e * RACE_MAXR + j] >= 0.0) {
            sd[SD_GOALV + e * RACE_MAXR + j] = (double)lane;
            const double dur = convert_float_rte(scaled);
            sd[SD_GOALT + e * RACE_MAXR + j] = dur < 0.0 ? -INFINITY : -dur;
            return 1;
        }
    }
    return 0;
}

// The effect word a token was registered for, the (record, slot) it packs.
static __global const uint* sk_tok_effect(__global const uint* sk,
    __global const uint* scr, const uint tok) {
    const uint t = tok >= SK_TOK_AA ? scr[SK_AAP + SK_AAPW * (int)(tok - SK_TOK_AA)] : tok;
    const int rec = (int)SK_TOK_REC(t);
    const int slot = (int)SK_TOK_SLOT(t);
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    return sk + SK_EFFECTS + SK_ESTRIDE * ((int)rd[6] + slot);
}

// The value one record-slot application registered at its fire. The engine's
// active-skill entry carries the value_usage factor applied (the roll, the
// counter tier), and the pace gate, the target column and the physics chains
// all read that entry, not the wire modifier (race_solver.rs:2485-2503, 2592).
static void sk_tok_stage(__global uint* scr, const uint tok, const double value) {
    const ulong bits = as_ulong(value);
    const int i = SK_VVAL + 2 * (int)(tok - SK_TOK_SKILL);
    scr[i] = (uint)bits;
    scr[i + 1] = (uint)(bits >> 32);
}

static double sk_tok_staged(__global const uint* scr, const uint tok) {
    const int i = SK_VVAL + 2 * (int)(tok - SK_TOK_SKILL);
    return as_double((ulong)scr[i] | ((ulong)scr[i + 1] << 32));
}

// active_lane_speed_skills non-empty, the moving-lane late term's gate.
static int sk_lane_active(__global const uint* scr, const int r) {
    return (scr + SK_LSEQ + r * SK_SEQN)[0] != 0u;
}

static int sk_seq_find(__global const uint* q, const uint tok) {
    for (int i = 0; i < SK_SEQN; i++) {
        if (q[i] == tok) {
            return i;
        }
    }
    return -1;
}

// 0 when the fixed 40-entry order is full, the rare case that flags the race.
static int sk_seq_push(__global uint* q, const uint tok) {
    for (int i = 0; i < SK_SEQN; i++) {
        if (q[i] == 0u) {
            q[i] = tok;
            return 1;
        }
    }
    return 0;
}

static void sk_seq_remove(__global uint* q, const uint tok) {
    for (int i = 0; i < SK_SEQN; i++) {
        if (q[i] == tok) {
            for (int j = i; j + 1 < SK_SEQN; j++) {
                q[j] = q[j + 1];
            }
            q[SK_SEQN - 1] = 0u;
            return;
        }
    }
}

// The f64 modifier a token carries. A pool token carries the runtime value
// registered at application time (a re-drawn roll or a counter tier), which
// is exactly what the engine keeps in its active-skill entry.
static double sk_tok_modifier(__global const uint* sk, __global const uint* scr, const uint tok) {
    if (tok >= SK_TOK_AA) {
        const __global uint* e = scr + SK_AAP + SK_AAPW * (int)(tok - SK_TOK_AA);
        return as_double((ulong)e[2] | ((ulong)e[3] << 32));
    }
    return sk_tok_staged(scr, tok);
}

// The registered value of a record token, f32-rounded the way the engine
// registers modifier as f32. A token whose effect is not this kind (the
// natural and ordinary current-speed groups share one order) reads as zero,
// the same filter the old record scan gave.
static float sk_tok_value(__global const uint* sk, __global const uint* scr,
    const uint tok, const int kind) {
    if (tok >= SK_TOK_AA) {
        const __global uint* e = scr + SK_AAP + SK_AAPW * (int)(tok - SK_TOK_AA);
        if ((int)e[1] != kind) {
            return 0.f;
        }
        return convert_float_rte(as_double((ulong)e[2] | ((ulong)e[3] << 32)));
    }
    const __global const uint* ef = sk_tok_effect(sk, scr, tok);
    if ((int)ef[0] != kind) {
        return 0.f;
    }
    return convert_float_rte(sk_tok_staged(scr, tok));
}

// visible_distance over the runner's vision aux tokens, the engine's f64 sum
// order (registration order) before the retail floor and the f32 read.
static double sk_vision(__global const uint* sk, __global const uint* scr, const int r) {
    double acc = 20.0;
    const __global uint* q = scr + SK_VSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        acc += sk_tok_modifier(sk, scr, tok);
    }
    return fmax(acc, 0.0);
}

// A value the engine reports as unavailable: every comparison against it
// answers false, which the isnan guard in sk_feat_cmp enforces.
#define SK_NAN ((double)as_float(0x7fc00000u))

// The scan gate's live predicate values, one FEAT_* code per dynamic leaf.
// `arg` is the comparison argument word so FEAT_OUTBAND can pick its band
// cell, and `wd` gives FEAT_ANYACT the record's observe_anyskill state.
static double sk_feat_value(const uint code, const double arg,
    __global const double* wd, __global const uint* si,
    __global const double* shl, __global const double* sd, __global const double* pd,
    __global const float* sf, __global const uint* rs, __global const uint* sk,
    __global const uint* sks, const int r, const int n,
    const double distance) {
    const uint feat = code >> 3;
    switch (feat) {
    case FEAT_ORDER:
    case FEAT_ORDER_RATE:
        return (double)(int)si[SI_RANKCOL + r];
    case FEAT_HP_PER: {
        const __global const double* hr = shl + r * SH_STRIDE;
        const float ratio = convert_float_rte(hr[SH_HP]) / convert_float_rte(hr[SH_MAX]);
        return (double)(int)((double)ratio * 100.0);
    }
    case FEAT_IS_OVERTAKE:
        return (int)si[SI_L_TGTN + r] != 0 ? 1.0 : 0.0;
    case FEAT_IS_TEMPTATION:
        return (int)si[SI_KAKARI + r] != 0 ? 1.0 : 0.0;
    case FEAT_LS_ENTRY:
        return (int)si[SI_LS_TICK + r] != 0 ? 1.0 : 0.0;
    case FEAT_PHASE:
        return (double)(int)si[SI_PHASE + r];
    case FEAT_RPHASE:
        return (double)(int)si[SI_RPHASE + r];
    case FEAT_CORNER:
        return (double)(int)si[SI_CORNER + r];
    case FEAT_FINALE:
        return (double)(int)si[SI_FINALE + r];
    case FEAT_FC_HALF: {
        if (isnan(pd[PD_FC_HALF0])) return SK_NAN;
        const float pos = convert_float_rte(sd[SD_POS + r]);
        return (pos >= (float)pd[PD_FC_HALF0] && pos <= (float)pd[PD_FC_HALF1])
            ? 1.0 : 0.0;
    }
    case FEAT_DIST_RATE:
        return (double)(int)sd[SD_POS + r];
    case FEAT_REMAIN:
        return distance - floor(sd[SD_POS + r]);
    case FEAT_VREMAIN:
        return distance - floor(sd[SD_POS + (int)sk[SK_VIDX]]);
    case FEAT_CLOCK:
        return floor((double)convert_float_rte((float)sd[SD_CLOCK + r]));
    case FEAT_TEMPT:
        return (double)(int)si[SI_TEMPT + r];
    case FEAT_ACT0:
        return (double)(int)si[SI_ACT0 + r];
    case FEAT_ACT1:
        return (double)(int)si[SI_ACT1 + r];
    case FEAT_BSIDE:
        return trunc(sd[SD_BSIDE + r]);
    case FEAT_IFNEAR:
        return floor(sd[SD_IFNEAR + r]);
    case FEAT_BHNEAR:
        return floor(sd[SD_BHNEAR + r]);
    case FEAT_BHNEAR1:
        return floor(sd[SD_BHNEAR1 + r]);
    case FEAT_ORDER_DELTA:
        return (double)((int)si[SI_RANKCOL + r] - (int)rs[RS_PREVRANK + r]);
    case FEAT_OUTBAND: {
        // The engine's order check is the band cell being set and not NaN,
        // and the out filter only accepts ==1 of that boolean.
        const int rate = (int)arg;
        const uint cell = rate == 20 ? si[SI_BND20 + r]
            : rate == 40 ? si[SI_BND40 + r]
            : rate == 70 ? si[SI_BND70 + r] : si[SI_BND50 + r];
        return cell == 1u ? 1.0 : 0.0;
    }
    case FEAT_ANYACT:
        return wd[SKD_ANY_RESULT];
    case FEAT_NEAR: {
        const double pos = sd[SD_POS + r];
        const double lane = sd[SD_LANE + r];
        const double lane_lim = (double)0.16666f;
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) continue;
            if (fabs(sd[SD_POS + j] - pos) <= 3.0
                && fabs(sd[SD_LANE + j] - lane) <= lane_lim) {
                c++;
            }
        }
        return (double)c;
    }
    case FEAT_BASH_BEHIND:
    case FEAT_BASH_INFRONT: {
        const int rank = (int)si[SI_RANKCOL + r];
        if (feat == FEAT_BASH_INFRONT ? rank <= 1 : rank >= n) return SK_NAN;
        const int j =
            (int)rs[RS_ORDER + (feat == FEAT_BASH_INFRONT ? rank - 2 : rank)];
        const double d = feat == FEAT_BASH_INFRONT
            ? fmax(sd[SD_POS + j] - sd[SD_POS + r], 0.0)
            : fmax(sd[SD_POS + r] - sd[SD_POS + j], 0.0);
        return d / 2.5;
    }
    case FEAT_LASTSPURT: {
        if ((int)si[SI_SP_ACTIVE + r]) return 1.0;
        const int res = (int)si[SI_SP_RESULT + r];
        if ((int)si[SI_PHASE + r] < 2 || (res != 1 && res != 2)
            || (int)si[SI_SP_PENDING + r] || !(sf[SF_SP_START + r] > 0.0f)) {
            return 0.0;
        }
        return convert_float_rte(sd[SD_POS + r]) >= sf[SF_SP_START + r] ? 1.0 : 0.0;
    }
    case FEAT_SLOPE:
        return sf[SF_SL_VALUE + r] > 0.f
            ? 1.0 : (sf[SF_SL_VALUE + r] < 0.f ? 2.0 : 0.0);
    case FEAT_OTIME:
        return floor(sd[SD_OTIME + r]);
    case FEAT_DDTOP: {
        // ranked_distance_bounds with the leader substitution: my own live
        // position when I lead, and the f32 gap floored before widening.
        const int lead = (int)rs[RS_ORDER];
        const float gap = convert_float_rte(sd[SD_POS + lead])
            - convert_float_rte(sd[SD_POS + r]);
        return (double)floor(gap);
    }
    case FEAT_HEAL:
        return (double)(int)si[SI_HEAL + r];
    case FEAT_CHGUP_END:
        return (double)(int)si[SI_POV2 + r];
    case FEAT_LANETYPE: {
        // HorseRaceInfo.get_Lane: independent f32 bands on the live lane.
        const float lane = convert_float_rte(sd[SD_LANE + r]);
        return (double)((lane > 0.2f ? 1 : 0) + (lane > 0.4f ? 1 : 0)
            + (lane > 0.6f ? 1 : 0));
    }
    case FEAT_FIGHT:
        return (double)(int)si[SI_DUEL_FIGHT + r];
    case FEAT_BHIN: {
        // lane_gap_to_behind: last place reports None, rejecting every
        // comparison, and the sign of the lane gap becomes the 0/1 value.
        const int rank = (int)si[SI_RANKCOL + r];
        if (rank >= n) {
            return SK_NAN;
        }
        const int j = (int)rs[RS_ORDER + rank];
        return sd[SD_LANE + r] - sd[SD_LANE + j] > 0.0 ? 1.0 : 0.0;
    }
    case FEAT_MOVELANE:
        return sf[SF_MV_TIME + r] >= 1.f
            ? (double)(1 + ((int)si[SI_MV_DIR + r] != 0 ? 1 : 0)) : 0.0;
    case FEAT_BFRONT:
        return trunc(sd[SD_BFRONT + r]);
    case FEAT_INBAND: {
        const int rate = (int)arg;
        const uint cell = rate == 20 ? si[SI_BNDIN20 + r]
            : rate == 40 ? si[SI_BNDIN40 + r]
            : rate == 80 ? si[SI_BNDIN80 + r] : si[SI_BNDIN50 + r];
        return cell == 1u ? 1.0 : 0.0;
    }
    case FEAT_NEARIF: {
        // near_infront_count: the near corridor's forward half, in f64 gaps.
        const double pos = sd[SD_POS + r];
        const double lane = sd[SD_LANE + r];
        const double lane_lim = (double)0.16666f;
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) continue;
            const double gap = sd[SD_POS + j] - pos;
            if (gap > 0.0 && gap <= 3.0 && fabs(sd[SD_LANE + j] - lane) <= lane_lim) {
                c++;
            }
        }
        return (double)c;
    }
    case FEAT_VISN:
        return (double)(int)si[SI_L_VISN + r];
    case FEAT_USED: {
        // used_skills holds exactly the skill ids this runner activated, and
        // the per-record used bit is set on its first fire.
        const uint sid = (uint)arg;
        const int nrec = (int)sk[SK_NREC];
        for (int i = 0; i < nrec; i++) {
            const __global const uint* ru = sk + SK_RECORDS + SK_RSTRIDE * i;
            if (ru[0] == sid && (int)ru[1] == r
                && sks[SKR_STRIDE * i + SKR_USED] != 0u) {
                return 1.0;
            }
        }
        return 0.0;
    }
    case FEAT_ACT2:
        return (double)(int)si[SI_ACT2 + r];
    case FEAT_ACTLH:
        return (double)(int)si[SI_ACTLH + r];
    case FEAT_ACTALL:
        return (double)(int)si[SI_ACTALL + r];
    case FEAT_NOTOP:
        return floor(sd[SD_NOUP + r]);
    case FEAT_NOTOD:
        return floor(sd[SD_NODOWN + r]);
    case FEAT_CHGUP_MID:
        return (double)(int)si[SI_POV1 + r];
    case FEAT_CHGUP_FC:
        return (double)(int)si[SI_POVFC + r];
    case FEAT_BALL:
        return trunc(fmin(sd[SD_BFRONT + r], sd[SD_BSIDE + r]));
    case FEAT_BFRONTB: {
        // The engine's blocked_front rebuilds front_blocker from live solver
        // rows at evaluation time, so the scan recomputes the sample rather
        // than reading the field-update tick's stored one.
        int fb = -1;
        float fg = 0.f;
        int sb = 0;
        lane_blocking_sample(sd, si, r, n, pd, &fb, &fg, &sb);
        return fb >= 0 ? 1.0 : 0.0;
    }
    case FEAT_SURR: {
        // is_surrounded: three-direction membership on f32-rounded gaps.
        const double pos = sd[SD_POS + r];
        const double lane = sd[SD_LANE + r];
        int out = 0;
        int front = 0;
        int behind = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) continue;
            const double dg = (double)convert_float_rte(sd[SD_POS + j] - pos);
            const double lg = (double)convert_float_rte(sd[SD_LANE + j] - lane);
            if (fabs(dg) <= 1.5 && lg >= 0.0 && lg <= (double)0.16666f) out = 1;
            if (dg > 0.0 && dg <= 3.0 && fabs(lg) <= (double)0.08333f) front = 1;
            if (dg >= -3.0 && dg < 0.0 && fabs(lg) <= (double)0.08333f) behind = 1;
            if (out && front && behind) break;
        }
        return out && front && behind ? 1.0 : 0.0;
    }
    case FEAT_TEMPT_B:
    case FEAT_TEMPT_F: {
        // Published-order slices around the runner, counting kakari rows.
        const int rank = (int)si[SI_RANKCOL + r];
        int c = 0;
        if (feat == FEAT_TEMPT_F) {
            for (int k = 0; k < rank - 1; k++) {
                if ((int)si[SI_KAKARI + (int)rs[RS_ORDER + k]] != 0) c++;
            }
        } else {
            for (int k = rank; k < n; k++) {
                if ((int)si[SI_KAKARI + (int)rs[RS_ORDER + k]] != 0) c++;
            }
        }
        return (double)c;
    }
    case FEAT_RTEMP:
    case FEAT_RTEMP + 1:
    case FEAT_RTEMP + 2:
    case FEAT_RTEMP + 3:
    case FEAT_RTEMP_OPP:
    case FEAT_RTEMP_OPP + 1:
    case FEAT_RTEMP_OPP + 2:
    case FEAT_RTEMP_OPP + 3: {
        // running_style_temptation_count_{nige,senko,sashi,oikomi}: the
        // base-style group's kakari rows. The plain arm counts the querying
        // runner when it is in the group and rushing, the _opponent_ arm
        // drops it (race_field.rs:945).
        const int opp = feat >= FEAT_RTEMP_OPP;
        const int group = 1 + (int)(feat - (opp ? FEAT_RTEMP_OPP : FEAT_RTEMP));
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) continue;
            if (field_style((int)si[SI_STRATEGY + j]) == group
                && (int)si[SI_KAKARI + j] != 0) {
                c++;
            }
        }
        if (!opp && field_style((int)si[SI_STRATEGY + r]) == group
            && (int)si[SI_KAKARI + r] != 0) {
            c++;
        }
        return (double)c;
    }
    case FEAT_DDRATE: {
        // distance_diff_rate: the ranked endpoints with self substitution,
        // then the f32 spread percentage, Approximately-zero exact.
        const int lead = (int)rs[RS_ORDER];
        const int last = (int)rs[RS_ORDER + n - 1];
        const float first = convert_float_rte(sd[SD_POS + lead]);
        const float tail = convert_float_rte(sd[SD_POS + last]);
        const float own = convert_float_rte(sd[SD_POS + r]);
        const float spread = first - tail;
        if (fabs(spread) < fmax(1e-6f * fabs(spread), 8.0f * as_float(0x00000001u))) {
            return 0.0;
        }
        return (double)floor(((first - own) / spread) * 100.0f);
    }
    case FEAT_DDTOPF: {
        const int lead = (int)rs[RS_ORDER];
        const float gap = convert_float_rte(sd[SD_POS + lead])
            - convert_float_rte(sd[SD_POS + r]);
        return (double)((floor(gap * 10.0f) / 10.0f));
    }
    default:
        return 0.0;
    }
}

// The order_rate threshold comes from the field size, the engine's
// `(n as f64 * (rate / 100.0)).round()` with half-away rounding.
static int sk_feat_cmp(const uint code, const double v, const double arg, const int n) {
    if (isnan(v)) {
        return 0;
    }
    double b = arg;
    const uint feat = code >> 3;
    if (feat == FEAT_ORDER_RATE) {
        b = round((double)n * (arg / 100.0));
    } else if (feat == FEAT_OUTBAND || feat == FEAT_INBAND || feat == FEAT_USED) {
        // The argument word carries the band rate or the skill id, and the
        // engine's filter only accepts the ==1 form of its boolean.
        b = 1.0;
    }
    switch (code & 7u) {
    case 0u: return v == b;
    case 1u: return v != b;
    case 2u: return v < b;
    case 3u: return v <= b;
    case 4u: return v > b;
    default: return v >= b;
    }
}

// One record's postfix condition program, evaluated over the live leaves by
// the scan gate. Programs hold AND markers only; the '@' structure lives in
// the arms table, so sk_eval_arms folds the membership and program results.
static int sk_eval(__global const uint* sk, const int prog_base, const int prog_len,
    __global const double* wd, __global const uint* si, __global const double* shl,
    __global const double* sd, __global const double* pd, __global const float* sf,
    __global const uint* rs, __global const uint* sks, const int r, const int n,
    const double distance) {
    int top = 0;
    char stack[64];
    for (int i = 0; i < prog_len; i++) {
        const __global const uint* nd = sk + SK_PROGRAMS + SK_PSTRIDE * (prog_base + i);
        const uint op = nd[0];
        if (op == OP_AND) {
            stack[top - 2] = (char)(stack[top - 2] && stack[top - 1]);
            top -= 1;
        } else if (op == OP_TRUE) {
            stack[top++] = (char)(nd[1] != 0u);
        } else {
            const double arg = as_double((ulong)nd[2] | ((ulong)nd[3] << 32));
            const double v = sk_feat_value(nd[1], arg, wd, si, shl, sd, pd, sf, rs,
                sk, sks, r, n, distance);
            stack[top++] = (char)sk_feat_cmp(nd[1], v, arg, n);
        }
    }
    return top > 0 && stack[top - 1];
}

// Membership in a runtime window list, the engine's region any() check.
static int sk_win_any(__global const double* wd, const int base, const int m,
    const double pos) {
    for (int i = 0; i < m; i++) {
        if (wd[base + 2 * i] <= pos && pos < wd[base + 2 * i + 1]) {
            return 1;
        }
    }
    return 0;
}

static double sk_word_f64(__global const uint* w) {
    return as_double((ulong)w[0] | ((ulong)w[1] << 32));
}

// The Or closure the engine's apply builds for a multi-arm condition: one
// arm passes when the position sits in its runtime chain and its program
// holds. Single records keep the plain program path (the engine's
// extra_condition has no membership closures without an Or node).
static int sk_eval_arms(__global const uint* sk, __global const uint* rd,
    __global double* wd, __global uint* sks, const int rec,
    __global const uint* si, __global const double* shl, __global const double* sd,
    __global const double* pd, __global const float* sf, __global const uint* rs,
    const int r, const int n, const double distance) {
    const int narms = (int)rd[20];
    for (int a = 0; a < narms; a++) {
        const __global uint* ar = sk + SK_ARMS + SK_ASTRIDE * ((int)rd[21] + a);
        const int nwin = (int)sks[SKR_STRIDE * rec + SKR_AWIN0 + a];
        if (nwin == 0) {
            continue;
        }
        const double pos = sd[SD_POS + r];
        if (!sk_win_any(wd, SKD_ARMS + 2 * SK_WINMAX * a, nwin, pos)) {
            continue;
        }
        if (sk_eval(sk, (int)ar[1], (int)ar[0], wd, si, shl, sd, pd, sf, rs, sks, r,
            n, distance)) {
            return 1;
        }
    }
    return 0;
}

// The prerequisite watch: membership in the compiled precondition region
// list plus the precondition program, latched once true for the record's
// lifetime, exactly like the engine's per-tick watch loop.
static int sk_prereq_check(__global const uint* sk, __global const uint* rd,
    __global const double* wd, __global const uint* si, __global const double* shl,
    __global const double* sd, __global const double* pd, __global const float* sf,
    __global const uint* rs, __global const uint* sks, const int r, const int n,
    const double distance) {
    const double pos = sd[SD_POS + r];
    const int npre = (int)rd[22];
    const __global uint* rg = sk + SK_REGIONS + SK_GSTRIDE * (int)rd[23];
    int member = 0;
    for (int i = 0; i < npre; i++) {
        const double s = sk_word_f64(rg + SK_GSTRIDE * i);
        const double e = sk_word_f64(rg + SK_GSTRIDE * i + 2);
        if (s <= pos && pos < e) {
            member = 1;
            break;
        }
    }
    if (!member) {
        return 0;
    }
    return sk_eval(sk, (int)rd[25], (int)rd[24], wd, si, shl, sd, pd, sf, rs, sks, r,
        n, distance);
}

// SkillEffect::duration_coefficient: the leader-gap ramp, the HP tiers, or
// the flat one. hp is the frozen pre-activation ledger value.
static double sk_coeff(const uint du, const double hp, const double gap) {
    if (du == 2u) {
        const float ratio = convert_float_rte(fmax(gap, 0.0)) / 62.5f;
        return (double)fmin(0.8f + ratio, 1.6f);
    }
    if (du == 3u) {
        if (hp < 2000.0) { return 1.; }
        if (hp < 2400.0) { return 1.5; }
        if (hp < 2600.0) { return 2.; }
        if (hp < 2800.0) { return 2.200000047683716; }
        if (hp < 3000.0) { return 2.5; }
        if (hp < 3200.0) { return 3.; }
        if (hp < 3500.0) { return 3.5; }
        return 4.;
    }
    if (du == 7u) {
        if (hp < 1500.0) { return 1.; }
        if (hp < 1800.0) { return 1.5; }
        if (hp < 2000.0) { return 2.; }
        if (hp < 2100.0) { return 2.5; }
        return 3.;
    }
    return 1.;
}

// prepare_retail_tick's timer pass plus expire_skill_effects plus
// The current-speed expiries, drained from the tail of the activation order so
// a natural effect's impulse sums in the engine's reverse list order. The
// token leaves the order once, so the impulse lands exactly once.
static void skills_expire_current(__global const uint* sk, __global uint* scr,
    __global double* skd, __global double* sd, const int n) {
    for (int r = 0; r < n; r++) {
        __global uint* q = sk_seq(scr, SK_EFF_CURRENT) + r * SK_SEQN;
        for (int i = SK_SEQN - 1; i >= 0; i--) {
            const uint tok = q[i];
            if (tok == 0u) {
                continue;
            }
            // A pool token keeps its timer in the pool entry, not in the
            // record chain. The kind gate keeps follow-ups out of this
            // order, so the branch only keeps the decode honest.
            const __global uint* pool = scr + SK_AAP + SK_AAPW * (int)(tok - SK_TOK_AA);
            const uint meta = tok >= SK_TOK_AA ? pool[0] : tok;
            const int rec = (int)SK_TOK_REC(meta);
            const double tm = tok >= SK_TOK_AA
                ? as_double((ulong)pool[4] | ((ulong)pool[5] << 32))
                : skd[SKD_STRIDE * rec + SKD_EFF_T + (int)SK_TOK_SLOT(meta)];
            if (tm < 0.0) {
                continue;
            }
            const __global const uint* ef = sk_tok_effect(sk, scr, tok);
            if ((int)ef[0] == SK_EFF_CURRENT_NAT) {
                // expire_skill_effects hands a natural current speed to
                // modifiers_one_frame_accel, a one-tick speed offset.
                sd[SD_OFA + r] += sk_tok_modifier(sk, scr, tok);
            }
            sk_seq_remove(q, tok);
        }
    }
}

// The value_usage fold at an activation's read point. The retail roll usages
// draw once per application, so every follow-up re-draws, and the counter
// usages read the live per-runner tallies (race_solver.rs:296-301, 2485-2503).
static double sk_vu_factor(__global uint* bank, __global uint* scr,
    __global const uint* si, const int r, const int kind, const int vu, const int paired) {
    if (vu == 11) {
        const double c = (double)(int)si[SI_LCO + r];
        return c < 2.0 ? 1.0
            : (c < 3.0 ? 1.100000023841858 : (c < 4.0 ? 1.2000000476837158 : 1.25));
    }
    if (kind == SK_EFF_NOOP) {
        return 1.0;
    }
    if (vu == 8 || vu == 9) {
        __global uint* ab = paired ? scr + SCR_BANKS + r * BANK_WORDS : bank;
        const double roll = (double)r128_integer(ab, 0, 100);
        return roll <= 60.0 ? 0.0
            : (roll <= 90.0 ? 0.019999999552965164 : 0.03999999910593033);
    }
    if (vu == 14) {
        const double c = (double)(int)si[SI_GREEN + r];
        return c < 3.0 ? 0.0 : (c < 5.0 ? 1.0 : (c < 6.0 ? 2.0 : 3.0));
    }
    return 1.0;
}

// A running instance with a kind-2 or kind-3 slot keeps its ordered turn and
// survives past its timer until that turn comes (race_solver.rs:1810-1814).
static int aa_has23(__global const uint* sk, const int rec) {
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (int)rd[6];
    for (int e = 0; e < (int)rd[5]; e++) {
        const int aa = (int)ef[e * SK_ESTRIDE + 5];
        if (aa == 2 || aa == 3) {
            return 1;
        }
    }
    return 0;
}

// The fire's extend_on_overtake flag: any effect of the record reads
// duration_usage 4 (race_solver.rs:2652).
static int aa_has4(__global const uint* sk, const int rec) {
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (int)rd[6];
    for (int e = 0; e < (int)rd[5]; e++) {
        if (ef[e * SK_ESTRIDE + 3] == 4u) {
            return 1;
        }
    }
    return 0;
}

// Running-list append in application order, and in-place removal that keeps
// the order the engine's Vec::remove keeps.
static int aa_push(__global uint* scr, const int r, const int rec) {
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    int len = (int)list[0];
    if (len >= SK_RUNN) {
        return 0;
    }
    list[++len] = (uint)rec;
    list[0] = (uint)len;
    return 1;
}

static void aa_remove(__global uint* scr, const int r, const int slot) {
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    const int len = (int)list[0];
    for (int k = slot; k < len; k++) {
        list[k] = list[k + 1];
    }
    list[len] = 0u;
    list[0] = (uint)(len - 1);
}

// Journal one additionalActivate pool event: op tag, the tick mirror, four
// uint fields, two f64 fields. Appends stop silently once the fixed window
// is full, so diagnostics never change a race or a fallback.
#define SK_JRN_FIRE 1u   // a=rec b=runner c=n_eff d=aa_any|2*ext4, e1=run timer e2=clock
#define SK_JRN_FAN 2u    // a=pidx b=host c=recv d=meta, e1=value e2=timer
#define SK_JRN_DIE 3u    // a=pidx b=holder (all when none) c=kind, e1=timer
#define SK_JRN_RETIRE 4u // a=rec b=runner, e1=run timer
#define SK_JRN_FLIP 5u   // a=rec b=runner, e1=cooldown timer
#define SK_JRN_EXT 6u    // a=pidx b=owner c=take, e1=new timer
static void sk_jrn(__global uint* scr, const uint op, const uint a, const uint b,
    const uint c, const uint d, const double e1, const double e2) {
    const uint w = scr[SK_JRN];
    if (w + 11u > (uint)SK_JRN_N) {
        return;
    }
    __global uint* at = scr + SK_JRN + 1u + w;
    at[0] = op;
    at[1] = scr[SK_JRN_T];
    at[2] = a;
    at[3] = b;
    at[4] = c;
    at[5] = d;
    const ulong v1 = as_ulong(e1);
    const ulong v2 = as_ulong(e2);
    at[6] = (uint)v1;
    at[7] = (uint)(v1 >> 32);
    at[8] = (uint)v2;
    at[9] = (uint)(v2 >> 32);
    scr[SK_JRN] = w + 10u;
}

// retire_finished_running_skills over one runner, scanning the same way the
// engine's index loop does while removals shift the tail.
static int aa_retire(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* si, __global uint* scr,
    __global double* sd, const int r) {
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    const int len = (int)list[0];
    int retiring = 0;
    int retired = 0;
    for (int k = len; k >= 1; k--) {
        const int rec = (int)list[k];
        if (skd[SKD_STRIDE * rec + SKD_RUN_T] < 0.0) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((double)rd[0] > sd[SD_AFLOOR + r] && aa_has23(sk, rec)) {
            retiring = 1;
            continue;
        }
        __global uint* rs = sks + SKR_STRIDE * rec;
        rs[SKR_AA_RUN] = 0u;
        if (rs[SKR_AA_UNAPPLIED] != 0u) {
            si[SI_AANU + r] -= 1u;
            rs[SKR_AA_UNAPPLIED] = 0u;
        }
        sk_jrn(scr, SK_JRN_RETIRE, (uint)rec, (uint)r, 0xFFFFFFFFu, 0xFFFFFFFFu,
            skd[SKD_STRIDE * rec + SKD_RUN_T], 0.0);
        aa_remove(scr, r, k);
        retired = 1;
    }
    si[SI_AARET + r] = (uint)retiring;
    return retired;
}

// start_expired_skill_cooldowns for one runner: a waiting record starts its
// cooldown once the instance duration ran out and no running activation of
// it is still open (race_solver.rs:2418-2435).
static void aa_flip(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, const int r, const float dist_f) {
    const int nrec = (int)sk[SK_NREC];
    for (int rec = 0; rec < nrec; rec++) {
        __global uint* rs = sks + SKR_STRIDE * rec;
        if (rs[SKR_STATE] != (uint)SK_S_WAITING || rs[SKR_AA_RUN] != 0u) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((int)rd[1] != r) {
            continue;
        }
        __global double* wd = skd + SKD_STRIDE * rec;
        if (wd[SKD_INSTANCE] < 0.0) {
            continue;
        }
        const double cd = as_double((ulong)rd[13] | ((ulong)rd[14] << 32));
        wd[SKD_COOLDOWN] = -(double)(convert_float_rte(cd) * (dist_f / 1000.0f));
        rs[SKR_STATE] = (uint)SK_S_COOLDOWN;
        sk_jrn(scr, SK_JRN_FLIP, (uint)rec, (uint)r, 0xFFFFFFFFu, 0xFFFFFFFFu,
            wd[SKD_COOLDOWN], 0.0);
    }
}

static int aa_pool_alloc(__global uint* scr) {
    for (int p = 0; p < SK_AAPN; p++) {
        if (scr[SK_AAP + SK_AAPW * p] == 0u) {
            return p;
        }
    }
    return -1;
}

static int sk_targets(__global const uint* si, __global const uint* rs,
    __global const uint* scr, const int r, const int target, const int limit,
    const int n, int* outs);

// Registering one delayed application. The effect re-runs its value_usage
// dispatch (a fresh roll for the draw usages) and lands with the collected
// duration as the override, so the coefficient never applies. The follow-up
// path skips counters, the green insert and the instance and cooldown write
// (race_solver.rs:2504-2539, 2642-2648). A cross-target slot re-draws the fan
// at its turn and takes the owner's value, which is exact for the kinds whose
// delivery is one timer and one value in the receiver's order.
static void aa_apply(__global const uint* sk, __global uint* scr, __global uint* si,
    __global const uint* rs, __global double* sd, __global double* skd, __global uint* st,
    __global uint* bank,
    const int paired, const int n, const int r, const int rec, const int slot,
    const double duration, const float dt) {
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * ((int)rd[6] + slot);
    const int kind = (int)ef[0];
    const int target = (int)ef[1];
    const double wire = as_double((ulong)ef[6] | ((ulong)ef[7] << 32));
    const double modifier = wire * sk_vu_factor(bank, scr, si, r, kind, (int)ef[2], paired);
    // skill_targets classifies the demerit from the effect's own modifier,
    // before the activation's draw multiplies it (race_solver.rs:2482).
    const int debuff = sk_is_debuff(kind, wire);
    // The engine's apply_activation still records the follow-up event with
    // its selection, even when the self debuff skips delivery (the immune
    // runner's own entry was filtered out of the selection).
    __global uint* fan = rc_take(scr, st, SCR_RC_FAN, RC_FAN_CAP, 6 + RACE_MAXR);
    if (fan != 0) {
        const ulong fbits = as_ulong(sd[SD_CLOCK + r]);
        fan[0] = (uint)r;
        fan[1] = rd[0];
        fan[2] = rd[26];
        fan[3] = (uint)fbits;
        fan[4] = (uint)(fbits >> 32);
    }
    if (target == 1 && debuff && sd[SD_IMMUNE + r] <= 0.0) {
        if (fan != 0) {
            fan[5] = 1u;
            rc_give(scr, SCR_RC_FAN, RACE_MAXR);
        }
        return;
    }
    // The follow-up application also leaves the host a timed entry for the
    // slot, the override duration taking the scaled one's place
    // (race_solver.rs:2531-2534, 2540-2548, 2592-2616).
    __global double* wda = skd + SKD_STRIDE * rec;
    const double adur = convert_float_rte(duration);
    if (target != 1 ? adur > 0.0
        : (sk_queued_kind(kind) || kind == SK_EFF_SETLANE
            || kind == SK_EFF_GOAL_OUT || kind == SK_EFF_GOAL_IN || kind == SK_EFF_IMMUNE)) {
        wda[SKD_EFF_T + slot] = adur < 0.0 ? -INFINITY : -adur;
    }
    int outs[RACE_MAXR];
    int nrecv = 1;
    outs[0] = r;
    // Noop and Runaway skip target selection, and the follow-up event
    // reports null for them, while the effect still delivers to self
    // (race_solver.rs:2479).
    const int null_sel = target == 1 && (kind == SK_EFF_NOOP || kind == SK_EFF_RUNAWAY);
    if (target != 1) {
        nrecv = sk_targets(si, rs, scr, r, target, (int)ef[4], n, outs);
        const uint owner_team = si[SI_TEAM + r];
        int keep = 0;
        for (int q = 0; q < nrecv; q++) {
            const int j = outs[q];
            if (debuff && sd[SD_IMMUNE + j] <= 0.0) {
                continue;
            }
            if (debuff && owner_team != 0u && j != r && si[SI_TEAM + j] == owner_team) {
                continue;
            }
            outs[keep++] = j;
        }
        nrecv = keep;
    }
    if (fan != 0) {
        if (null_sel) {
            fan[5] = 0u;
            rc_give(scr, SCR_RC_FAN, RACE_MAXR);
        } else {
            fan[5] = (uint)(nrecv + 1);
            for (int q = 0; q < nrecv; q++) {
                fan[6 + q] = (uint)outs[q];
            }
            rc_give(scr, SCR_RC_FAN, RACE_MAXR - nrecv);
        }
    }
    const ulong vbits = as_ulong(modifier);
    // new_skill_timer rounds the override to f32 first and maps a negative
    // duration to the permanent sentinel.
    const double d = (double)convert_float_rte(duration);
    for (int q = 0; q < nrecv; q++) {
        const int j = outs[q];
        double t = d < 0.0 ? -INFINITY : -d;
        if (j > r) {
            // Retail delivery lands in the receiver's timer map while the
            // host's own history runs (race_field.rs:472-477). A receiver
            // with a higher roster index has not had its prepare pass step
            // that map yet this tick, so the copy takes its first f32 timer
            // step on the delivery tick itself (race_field_retail.rs:221-225,
            // race_solver_retail.rs:825-827).
            t = (double)(convert_float_rte((float)t) + dt);
        }
        const ulong tbits = as_ulong(t);
        const int pidx = aa_pool_alloc(scr);
        if (pidx < 0) {
            st[ST_FALLBACK] = 1u;
            st[ST_ARG] = FB_SKILL_SEQ;
            return;
        }
        __global uint* pool = scr + SK_AAP + SK_AAPW * pidx;
        pool[0] = SK_TOK_EFF((uint)rec, (uint)slot);
        pool[1] = (uint)kind;
        pool[2] = (uint)vbits;
        pool[3] = (uint)(vbits >> 32);
        pool[4] = (uint)tbits;
        pool[5] = (uint)(tbits >> 32);
        if (!sk_seq_push(sk_seq(scr, kind) + j * SK_SEQN, (uint)(SK_TOK_AA + pidx))) {
            pool[0] = 0u;
            st[ST_FALLBACK] = 1u;
            st[ST_ARG] = FB_SKILL_SEQ;
        } else {
            sk_jrn(scr, SK_JRN_FAN, (uint)pidx, (uint)r, (uint)j,
                SK_TOK_EFF((uint)rec, (uint)slot), modifier, t);
        }
    }
}

// additional_skill_event(1) for one overtake: every running skill with a
// live instance delivers its kind-1 slots, three at most per slot, with the
// durations read before any of them is applied (race_solver.rs:1857-1875).
static void aa_event1(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si, __global const uint* rs,
    __global double* sd, __global uint* st, __global uint* bank, const int paired,
    const int n, const int r, const float dt) {
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    const int len = (int)list[0];
    int a_rec[16];
    int a_slot[16];
    double a_dur[16];
    int na = 0;
    for (int k = 1; k <= len; k++) {
        const int rec = (int)list[k];
        const double timer = skd[SKD_STRIDE * rec + SKD_RUN_T];
        if (timer >= 0.0) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        __global uint* rs = sks + SKR_STRIDE * rec;
        const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (int)rd[6];
        for (int e = 0; e < (int)rd[5]; e++) {
            if ((int)ef[e * SK_ESTRIDE + 5] != 1) {
                continue;
            }
            __global uint* cnt = rs + SKR_AA_COUNT + e;
            if ((int)*cnt >= 3) {
                continue;
            }
            *cnt += 1u;
            if (na >= 16) {
                st[ST_FALLBACK] = 1u;
                st[ST_ARG] = FB_SKILL_SEQ;
                continue;
            }
            a_rec[na] = rec;
            a_slot[na] = e;
            a_dur[na] = -timer + (1.0 / 15.0);
            na++;
        }
    }
    for (int a = 0; a < na; a++) {
        aa_apply(sk, scr, si, rs, sd, skd, st, bank, paired, n, r, a_rec[a], a_slot[a], a_dur[a],
            dt);
    }
}

// apply_unapplied_additional: a running skill's turn comes when its id first
// crosses the floor the p0 walk raises, and each kind-2 or kind-3 slot
// delivers up to its cap, three for kind 2 and two for kind 3
// (race_solver.rs:1901-1941).
static void aa_turns(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si, __global const uint* rs,
    __global double* sd, __global uint* st, __global uint* bank, const int paired,
    const int n, const float dist_f, const int r, const double upto) {
    const double old_floor = sd[SD_AFLOOR + r];
    if (upto <= old_floor) {
        return;
    }
    sd[SD_AFLOOR + r] = upto;
    if ((int)si[SI_AANU + r] == 0 && (int)si[SI_AARET + r] == 0) {
        return;
    }
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    const int len = (int)list[0];
    int a_rec[16];
    int a_slot[16];
    double a_dur[16];
    int na = 0;
    for (int k = 1; k <= len; k++) {
        const int rec = (int)list[k];
        __global uint* rs = sks + SKR_STRIDE * rec;
        const uint n = rs[SKR_AA_UNAPPLIED];
        if (n == 0u) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((double)rd[0] <= old_floor || (double)rd[0] > upto) {
            continue;
        }
        rs[SKR_AA_UNAPPLIED] = 0u;
        si[SI_AANU + r] -= 1u;
        const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (int)rd[6];
        const double dur = -skd[SKD_STRIDE * rec + SKD_RUN_T] + (1.0 / 15.0);
        for (int e = 0; e < (int)rd[5]; e++) {
            const int aa = (int)ef[e * SK_ESTRIDE + 5];
            if (aa != 2 && aa != 3) {
                continue;
            }
            __global uint* cnt = rs + SKR_AA_COUNT + e;
            const int cap = aa == 3 ? 2 : 3;
            const int have = (int)*cnt < cap ? (int)*cnt : cap;
            const uint room = (uint)(cap - have);
            const uint take = n < room ? n : room;
            *cnt += take;
            for (uint q = 0; q < take; q++) {
                if (na >= 16) {
                    st[ST_FALLBACK] = 1u;
                    st[ST_ARG] = FB_SKILL_SEQ;
                    break;
                }
                a_rec[na] = rec;
                a_slot[na] = e;
                a_dur[na] = dur;
                na++;
            }
        }
    }
    for (int a = 0; a < na; a++) {
        // Turn deliveries run after every runner's prepare pass for the tick,
        // so no receiver's timer map is still un-stepped in front of them.
        aa_apply(sk, scr, si, rs, sd, skd, st, bank, paired, n, r, a_rec[a], a_slot[a], a_dur[a],
            0.0f);
    }
    if (aa_retire(sk, sks, skd, si, scr, sd, r)) {
        aa_flip(sk, sks, skd, scr, r, dist_f);
    }
}

// The pool timers step on the prepare's timer pass, and an entry whose timer
// reached zero leaves every receiver order the same way a record token does
// (expire_skill_effects over the aux and speed lists).
static void aa_pool_step(__global uint* scr, const float dt, const int n) {
    for (int p = 0; p < SK_AAPN; p++) {
        __global uint* pool = scr + SK_AAP + SK_AAPW * p;
        if (pool[0] == 0u) {
            continue;
        }
        const uint tok = (uint)(SK_TOK_AA + p);
        double t = as_double((ulong)pool[4] | ((ulong)pool[5] << 32));
        if (t >= 0.0) {
            t = 0.0;
        } else {
            t = (double)(convert_float_rte((float)t) + dt);
        }
        if (t < 0.0) {
            const ulong bits = as_ulong(t);
            pool[4] = (uint)bits;
            pool[5] = (uint)(bits >> 32);
            continue;
        }
        const int kind = (int)pool[1];
        int gone = 0;
        for (int j = 0; j < n; j++) {
            if (sk_seq_find(sk_seq(scr, kind) + j * SK_SEQN, tok) >= 0) {
                sk_seq_remove(sk_seq(scr, kind) + j * SK_SEQN, tok);
                sk_jrn(scr, SK_JRN_DIE, (uint)p, (uint)j, (uint)kind, 0xFFFFFFFFu,
                    t, 0.0);
                gone = 1;
            }
        }
        if (gone == 0) {
            sk_jrn(scr, SK_JRN_DIE, (uint)p, 0xFFFFFFFFu, (uint)kind, 0xFFFFFFFFu,
                t, 0.0);
        }
        pool[0] = 0u;
    }
}

// One aux presence countdown: step while the engine timer runs, retire to
// the free sentinel once its sweep would drop the entry
// (race_solver.rs:1990-1996, new_skill_timer chain).
static void sk_aux_step(__global double* cell, const float dt) {
    if (*cell < 0.0) {
        const double next = (double)(convert_float_rte((float)*cell) + dt);
        *cell = next < 0.0 ? next : INFINITY;
    } else {
        *cell = INFINITY;
    }
}

// extend_running_skills_on_overtake for one overtake of one runner: three
// extensions per running instance, each course-distance/1000 of f64 timer off
// the instance and off every live entry that runner owns under the same skill
// id (race_solver.rs:1824-1839). The engine walks its five active lists, so a
// copy delivered to another runner carries the Other perspective and stays, and
// a sweep already ran this tick, so a live entry has a negative timer. The
// engine runs this walk once per pending overtake, so a batch subtracts the
// same extension once per overtake it still has room for, and nothing else
// touches these timers between two overtakes of one batch.
static void aa_extend(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, const int nrec, const int r,
    const int np, const double distance) {
    __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
    const int len = (int)list[0];
    if (len == 0 || np <= 0) {
        return;
    }
    const double extension = distance / 1000.0;
    for (int k = 1; k <= len; k++) {
        const int rec = (int)list[k];
        if (!aa_has4(sk, rec)) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        __global double* wd = skd + SKD_STRIDE * rec;
        const int room = 3 - (int)wd[SKD_AEXT];
        const int take = np < room ? np : room;
        if (take <= 0) {
            continue;
        }
        wd[SKD_AEXT] += (double)take;
        for (int q = 0; q < take; q++) {
            wd[SKD_RUN_T] -= extension;
        }
        for (int q = 0; q < 5; q++) {
            const uint base = q == 0 ? SK_TSEQ
                : (q == 1 ? SK_ASEQ : (q == 2 ? SK_CSEQ : (q == 3 ? SK_LSEQ : SK_VSEQ)));
            __global uint* ord = scr + base + r * SK_SEQN;
            for (int i = 0; i < SK_SEQN; i++) {
                const uint tok = ord[i];
                if (tok == 0u) {
                    continue;
                }
                const int pidx = tok >= SK_TOK_AA ? (int)(tok - SK_TOK_AA) : -1;
                const uint meta = pidx >= 0 ? scr[SK_AAP + SK_AAPW * pidx] : tok;
                if (meta == 0u) {
                    continue;
                }
                const int er = (int)SK_TOK_REC(meta);
                __global const uint* erd = sk + SK_RECORDS + SK_RSTRIDE * er;
                if ((int)erd[1] != r || erd[0] != rd[0]) {
                    continue;
                }
                const int slot = (int)SK_TOK_SLOT(meta);
                const __global const uint* ef = sk + SK_EFFECTS
                    + SK_ESTRIDE * ((int)erd[6] + slot);
                if ((int)ef[1] != 1) {
                    continue;
                }
                if (pidx >= 0) {
                    __global uint* pool = scr + SK_AAP + SK_AAPW * pidx;
                    double t = as_double((ulong)pool[4] | ((ulong)pool[5] << 32));
                    if (t < 0.0) {
                        for (int w = 0; w < take; w++) {
                            t -= extension;
                        }
                        const ulong bits = as_ulong(t);
                        pool[4] = (uint)bits;
                        pool[5] = (uint)(bits >> 32);
                        sk_jrn(scr, SK_JRN_EXT, (uint)pidx, (uint)r, (uint)take,
                            0xFFFFFFFFu, t, 0.0);
                    }
                } else {
                    __global double* ewd = skd + SKD_STRIDE * er;
                    for (int w = 0; w < take && ewd[SKD_EFF_T + slot] < 0.0; w++) {
                        ewd[SKD_EFF_T + slot] -= extension;
                    }
                }
            }
        }
        // The cooldown key keeps one waiting entry, keyed by the firing
        // record's (skill id, detail), so a sibling that fire parked flips on
        // this instance's running timer, which an overtake extends
        // (race_solver.rs:2341, 2387, 2421-2424).
        for (int pr = 0; pr < nrec; pr++) {
            if (pr == rec || (int)sks[SKR_STRIDE * pr + SKR_STATE] != (uint)SK_S_WAITING) {
                continue;
            }
            __global const uint* srd = sk + SK_RECORDS + SK_RSTRIDE * pr;
            if ((int)srd[1] != r || srd[0] != rd[0]) {
                continue;
            }
            __global double* swd = skd + SKD_STRIDE * pr;
            for (int w = 0; w < take && swd[SKD_INSTANCE] < 0.0; w++) {
                swd[SKD_INSTANCE] -= extension;
            }
        }
    }
}

// start_expired_skill_cooldowns for one whole field, after the per-runner
// prepare loop. Dead records keep stepping because the engine's timer map is
// keyed by the firing, not by the pending record's life.
static void skills_advance(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si, __global const uint* rs,
    __global double* sd, __global uint* st, const float dt, const int n,
    const float dist_f, const double distance, __global uint* bank, const int paired) {
    const int nrec = (int)sk[SK_NREC];
    // The aux lifetimes the target filter and the lane search read, stepped
    // and swept like the engine's active_aux_skills retain, which keeps
    // vector order for the SetTargetLane last-live read.
    for (int r = 0; r < n; r++) {
        sk_aux_step(sd + SD_IMMUNE + r, dt);
        sk_aux_step(sd + SD_GOAL_OUT + r, dt);
        sk_aux_step(sd + SD_GOAL_IN + r, dt);
        int keep = 0;
        for (int e = 0; e < SK_GOALN; e++) {
            double t = sd[SD_GOALT + e * RACE_MAXR + r];
            if (t >= 0.0) {
                continue;
            }
            t = (double)(convert_float_rte((float)t) + dt);
            if (t >= 0.0) {
                continue;
            }
            sd[SD_GOALV + keep * RACE_MAXR + r] = sd[SD_GOALV + e * RACE_MAXR + r];
            sd[SD_GOALT + keep * RACE_MAXR + r] = t;
            keep++;
        }
        for (int e = keep; e < SK_GOALN; e++) {
            sd[SD_GOALT + e * RACE_MAXR + r] = INFINITY;
        }
    }
    for (int rec = 0; rec < nrec; rec++) {
        const uint state = sks[SKR_STRIDE * rec + SKR_STATE];
        if (state < (uint)SK_S_WAITING) {
            continue;
        }
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        __global double* wd = skd + SKD_STRIDE * rec;
        for (int w = SKD_EFF_T; w < SKD_STRIDE; w++) {
            wd[w] = (double)(convert_float_rte((float)wd[w]) + dt);
        }
        wd[SKD_INSTANCE] = (double)(convert_float_rte((float)wd[SKD_INSTANCE]) + dt);
        wd[SKD_RUN_T] = (double)(convert_float_rte((float)wd[SKD_RUN_T]) + dt);
        wd[SKD_COOLDOWN] = (double)(convert_float_rte((float)wd[SKD_COOLDOWN]) + dt);
        // expire_skill_effects: an active token whose duration timer reached
        // zero leaves every receiver's order, the caster's own included. The
        // current-speed kinds drain in their own reverse pass, which owns the
        // natural expiry impulse.
        const int eff_base = (int)rd[6];
        const int n_eff = (int)rd[5];
        for (int e = 0; e < n_eff; e++) {
            const int kind = (int)(sk + SK_EFFECTS)[SK_ESTRIDE * (uint)(eff_base + e)];
            if (!sk_queued_kind(kind) || kind == SK_EFF_CURRENT
                || kind == SK_EFF_CURRENT_NAT) {
                continue;
            }
            if (wd[SKD_EFF_T + e] >= 0.0) {
                const uint tok = SK_TOK_EFF((uint)rec, (uint)e);
                for (int j = 0; j < n; j++) {
                    sk_seq_remove(sk_seq(scr, kind) + j * SK_SEQN, tok);
                }
            }
        }
    }
    skills_expire_current(sk, scr, skd, sd, n);
    aa_pool_step(scr, dt, n);
    // process_skill_history: the running lists retire first, then each
    // rank pass collected since the last prepare fires its kind-1
    // activations, and start_expired_skill_cooldowns runs last per runner
    // (race_solver.rs:849-850, 1976-1997).
    for (int r = 0; r < n; r++) {
        const int np = (int)si[SI_PENDN + r];
        // The retail history runs the extensions before the retire sweep, so an
        // instance extended past its expiry this tick stays in the list
        // (race_solver.rs:1779-1797). One pass carries the whole batch, since
        // the engine repeats the same walk per pending overtake.
        aa_extend(sk, sks, skd, scr, nrec, r, np, distance);
        aa_retire(sk, sks, skd, si, scr, sd, r);
        si[SI_PENDN + r] = 0u;
        for (int q = 0; q < np; q++) {
            aa_event1(sk, sks, skd, scr, si, rs, sd, st, bank, paired, n, r, dt);
        }
    }
    for (int r = 0; r < n; r++) {
        aa_flip(sk, sks, skd, scr, r, dist_f);
    }
    if (scr[SCR_RC_ON] != 0u) {
        // The deactivate hook of lib.rs:688-723 pushes the pending tuple when
        // the last tracked entry of a (runner, id) leaves. The scan mirrors
        // that with the record timers, the kernel's live-entry ledger.
        for (int rec = 0; rec < nrec; rec++) {
            __global uint* ps = sks + SKR_STRIDE * rec;
            if (ps[SKR_PEND] == 0u) {
                continue;
            }
            __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            const int pr = (int)rd[1];
            int live = 0;
            for (int q = 0; q < nrec && live == 0; q++) {
                __global const uint* qd = sk + SK_RECORDS + SK_RSTRIDE * q;
                if ((int)qd[1] != pr || qd[0] != rd[0]) {
                    continue;
                }
                __global const double* qd_ = skd + SKD_STRIDE * q;
                for (int e = 0; e < (int)qd[5]; e++) {
                    const double t = qd_[SKD_EFF_T + e];
                    if (t < 0.0 && isfinite(t)) {
                        live = 1;
                        break;
                    }
                }
            }
            if (live != 0) {
                continue;
            }
            __global uint* act = rc_take(scr, st, SCR_RC_ACT, RC_ACT_CAP, 10);
            if (act == 0) {
                return;
            }
            act[0] = (uint)pr;
            act[1] = rd[0];
            act[2] = 0u;
            act[3] = 0u;
            act[4] = 0u;
            act[5] = 0u;
            act[6] = 0u;
            act[7] = ps[SKR_ORD];
            act[8] = 4u;
            act[9] = scr[SK_JRN_T];
            ps[SKR_PEND] = 0u;
            ps[SKR_ORD] = 0u;
        }
    }
}

// skill_targets for the rank-window codes and the kakari group code: the
// first `limit` candidates in the engine's order. The ahead code (9) slices
// the published order outward ahead of self, the behind code (10) includes
// self then walks behind, the visible code (4) filters the order by this
// tick's visible column, and the group code (21) takes every runner whose
// strategy equals limit and who is currently rushing, uncapped
// (race_field.rs:439-445). The debuff same-team drop is applied by the
// caller on this capped list, matching race_field.rs:458-461.
static int sk_targets(__global const uint* si, __global const uint* rs,
    __global const uint* scr, const int r, const int target, const int limit,
    const int n, int* outs) {
    const int rank = (int)si[SI_RANKCOL + r];
    const int vis_base = (int)(SCR_LISTS + (uint)r * LIST_STRIDE + RACE_MAXR);
    const int vis_len = (int)si[SI_L_VISN + r];
    const int cap = target == 21 ? RACE_MAXR : limit;
    int cnt = 0;
    for (int k = 0; k < n && cnt < cap; k++) {
        int j;
        if (target == 9) {
            if (k >= rank - 1) {
                break;
            }
            j = (int)rs[RS_ORDER + rank - 2 - k];
        } else if (target == 10) {
            if (rank + k >= n) {
                break;
            }
            j = (int)rs[RS_ORDER + rank + k];
        } else if (target == 21) {
            j = (int)rs[RS_ORDER + k];
            if (j == r || !(int)si[SI_KAKARI + j]
                || field_style((int)si[SI_STRATEGY + j]) != (limit == 5 ? 1 : limit)) {
                continue;
            }
        } else {
            j = (int)rs[RS_ORDER + k];
            if (j == r) {
                continue;
            }
            int seen = 0;
            for (int v = 0; v < vis_len; v++) {
                if ((int)scr[vis_base + v] == j) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                continue;
            }
        }
        outs[cnt++] = j;
    }
    return cnt;
}

// The engine's legacy base_target_speed (race_solver.rs:43-60). One
// refresh_stat_effects call replaces the field enable_retail_runtime had set
// to the retail form (race_solver_retail.rs:647) with these f64 values.
static double phys_legacy_base_target_speed(const double distance, const int strategy,
    const int phase, const double speed, const int distapt) {
    // Strategy rows are 1-indexed, row 0 the engine's unused zero row.
    const double coeff[6][3] = {
        { 0.0, 0.0, 0.0 },
        { 1.0, 0.98, 0.962 },
        { 0.978, 0.991, 0.975 },
        { 0.938, 0.998, 0.994 },
        { 0.931, 1.0, 1.0 },
        { 1.063, 0.962, 0.95 }
    };
    const double prof[8] = { 1.05, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1 };
    const double base_speed = 20.0 - (distance - 2000.0) / 1000.0;
    const double sel = (phase == 2) ? 1.0 : 0.0;
    return base_speed * coeff[strategy][phase]
        + (((sel * sqrt(500.0 * speed)) * prof[distapt]) * 0.002);
}

// The live writes of refresh_stat_effects (race_solver.rs:1330-1340): the
// target base flips to the legacy form from the clamped speed field, HP goes
// through the init-plus-spend re-round and the hp context recaptures wisdom.
// The min_speed and last_spurt_speed rewrites are dead in retail: the step
// head recomputes both before any consumer (race_solver_retail.rs:961 and
// 1007-1034), and the legacy last_spurt_speed carries an f64 guts powf whose
// only pre-phase-2 reader is the unmodeled staminasyoubu gate.
static void sk_stat_refresh(__global const uint* si, __global double* sd,
    __global double* shl, __global float* sf, const double distance, const int r) {
    const double speed = fmin(fmax((double)(float)sd[SD_HPS + r], 1.0), 2000.0);
    const int strategy = (int)si[SI_STRATEGY + r];
    const int distapt = (int)si[SI_DISTAPT + r];
    for (int ph = 0; ph < 3; ph++) {
        sd[SD_BTS + ph * RACE_MAXR + r] = phys_legacy_base_target_speed(
            distance, strategy, ph, speed, distapt);
    }
    // initial_hp and hp_guts_factor rewrite max_hp and SF_HPG with
    // bit-identical values: strategy, distance and their stat inputs are
    // unchanged, and hp_guts_factor is an init-only quantity the engine
    // never recomputes from the mid-race guts field.
    const double max_hp = shl[r * SH_STRIDE + SH_MAX];
    const double spent = max_hp - shl[r * SH_STRIDE + SH_HP];
    shl[r * SH_STRIDE + SH_HP] = (double)fmin(
        fmax((float)max_hp - (float)spent, 0.f), (float)max_hp);
    // hp.init recaptures retail.wisdom from the live horse.wisdom field.
    sf[SF_HPW + r] = convert_float_rte(sd[SD_HPW + r]);
}

// The hp half of refresh_stat_effects when the stamina field moved: hp.init
// rebuilds max_hp from the field with the im_hp chain (initialization.cl:
// 601-604), and the spend restore keeps the same f32 clamp as GameHpState.
static void sk_hp_stamina_reset(__global const uint* si, __global double* sd,
    __global double* shl, __global const double* pd, const float dist_f, const int r) {
    const float bs = clamp_1_2000_f32(convert_float_rte(sd[SD_STAM + r]));
    const int strategy = (int)si[SI_STRATEGY + r];
    const float hp_initial = convert_float_rte(pd[PT_HP_INITIAL]);
    const float hp_style = convert_float_rte(pd[PT_HP_STYLE + strategy - 1]);
    const double max_new = (double)convert_float_rte(
        ((double)bs * (double)hp_initial) * (double)hp_style + (double)dist_f);
    const double spent = shl[r * SH_STRIDE + SH_MAX] - shl[r * SH_STRIDE + SH_HP];
    shl[r * SH_STRIDE + SH_MAX] = max_new;
    shl[r * SH_STRIDE + SH_HP] = (double)fmin(
        fmax((float)max_new - convert_float_rte(spent), 0.f), (float)max_new);
}

// hp_guts_factor (retail_physics.rs:442), the value GameHpState::init
// captures from the current guts whenever hp.init runs.
static float sk_hp_guts_factor(__global const double* pd, const float guts) {
    const float scaled = clamp_1_2000_f32(guts) * convert_float_rte(pd[PT_HP_DRAIN + 3]);
    const float root = convert_float_rte(sqrt((double)scaled));
    return convert_float_rte(1.0 + pd[PT_HP_DRAIN + 4] / (double)root);
}

// The fire bookkeeping of fire_pending_skill and apply_activation for one
// record: the duration-timer kinds register a receiver token, recovery and
// the raw stat bump write state directly, the largest scaled duration becomes
// the (skill, detail) instance timer, and the phase counter bumps. cd <= 0
// marks the record dead forever (engine: cooldown map entry +INF and the
// pending record removed).
static void skills_fire(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si,
    __global double* shl, __global double* sd, __global const double* pd,
    __global float* sf, __global const uint* rs, const int rec, const int r,
    const double hp, const double gap, const float dist_f, const int init_pass,
    const int n, const double distance, __global uint* ev, __global uint* st,
    __global uint* bank, const int paired) {
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    // The engine's activation hook emits one skillEvents entry per Own-side
    // fire (an immediate one, or a pending entry the deactivate hook or the
    // race-end finalizer writes once). rd[0] is the wire skill id.
    race_event(ev, st, r, rd[0]);
    // additional_skill_event(2) runs before any effect draws: every running
    // skill that still owns an open instance or a later turn queues one more
    // activation, its own skill id excluded (race_solver.rs:1842-1855, 2472).
    {
        __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
        const double floor = sd[SD_AFLOOR + r];
        for (int k = 1; k <= (int)list[0]; k++) {
            const int rr = (int)list[k];
            __global const uint* ard = sk + SK_RECORDS + SK_RSTRIDE * rr;
            if (ard[0] == rd[0] || !aa_has23(sk, rr)) {
                continue;
            }
            __global uint* ars = sks + SKR_STRIDE * rr;
            if (skd[SKD_STRIDE * rr + SKD_RUN_T] >= 0.0
                && (double)ard[0] <= floor) {
                continue;
            }
            if (ars[SKR_AA_UNAPPLIED] == 0u) {
                si[SI_AANU + r] += 1u;
            }
            ars[SKR_AA_UNAPPLIED] += 1u;
        }
    }
    __global double* wd = skd + SKD_STRIDE * rec;
    const int eff_base = (int)rd[6];
    const int n_eff = (int)rd[5];
    double runtime = 0.0;
    int aa_any = 0;
    int ext4 = 0;
    // The skillTargetEvents record, appended per effect inside the loop.
    // Slot encoding: 0 is null, k is a list of k-1 indices.
    __global uint* tgtp = 0;
    int tgt_res = 0;
    __global uint* tgt0 = 0;
    if (scr[SCR_RC_ON] != 0u) {
        tgt_res = 6 + n_eff * (1 + RACE_MAXR);
        __global uint* tgt = rc_take(scr, st, SCR_RC_TGT, RC_TGT_CAP, tgt_res);
        if (tgt != 0) {
            const ulong cbits = as_ulong(sd[SD_CLOCK + r]);
            tgt[0] = (uint)r;
            tgt[1] = rd[0];
            tgt[2] = rd[26];
            tgt[3] = (uint)cbits;
            tgt[4] = (uint)(cbits >> 32);
            tgt[5] = (uint)n_eff;
            tgt0 = tgt;
            tgtp = tgt + 6;
        }
    }
    for (int e = 0; e < n_eff; e++) {
        const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (eff_base + e);
        const int kind = (int)ef[0];
        const double modifier = as_double((ulong)ef[6] | ((ulong)ef[7] << 32))
            * sk_vu_factor(bank, scr, si, r, kind, (int)ef[2], paired);
        sk_tok_stage(scr, SK_TOK_EFF((uint)rec, (uint)e), modifier);
        if (tgtp != 0) {
            // apply_activation selects per effect before any of its delivery
            // side effects, so the capture reads the same state the engine's
            // skill_targets call sees (race_solver.rs:2478-2484, 2527-2530).
            const int tgt_target = (int)ef[1];
            if (init_pass != 0) {
                tgtp[0] = 0u;
                tgtp += 1;
            } else if (tgt_target == 1
                && (kind == SK_EFF_NOOP || kind == SK_EFF_RUNAWAY)) {
                tgtp[0] = 0u;
                tgtp += 1;
            } else if (tgt_target == 1) {
                if (sk_is_debuff(kind, modifier) && sd[SD_IMMUNE + r] <= 0.0) {
                    tgtp[0] = 1u;
                    tgtp += 1;
                } else {
                    tgtp[0] = 2u;
                    tgtp[1] = (uint)r;
                    tgtp += 2;
                }
            } else {
                int outs2[RACE_MAXR];
                const int cnt2 = sk_targets(si, rs, scr, r, tgt_target, (int)ef[4], n, outs2);
                const uint team2 = si[SI_TEAM + r];
                const int deb2 = sk_is_debuff(kind, modifier);
                int keep2 = 0;
                for (int q = 0; q < cnt2; q++) {
                    const int j = outs2[q];
                    if (deb2 && sd[SD_IMMUNE + j] <= 0.0) {
                        continue;
                    }
                    if (deb2 && team2 != 0u && j != r && si[SI_TEAM + j] == team2) {
                        continue;
                    }
                    outs2[keep2++] = j;
                }
                tgtp[0] = (uint)(keep2 + 1);
                for (int q = 0; q < keep2; q++) {
                    tgtp[1 + q] = (uint)outs2[q];
                }
                tgtp += 1 + keep2;
            }
        }
        const double base = as_double((ulong)ef[8] | ((ulong)ef[9] << 32));
        const double coeff = sk_coeff(ef[3], hp, gap);
        const float m1 = convert_float_rte(base) * (dist_f / 1000.0f);
        const float m2 = m1 * convert_float_rte(coeff);
        const double scaled = (double)(m2 * 1.0f);
        runtime = fmax(runtime, scaled);
        if (ef[3] == 4u) {
            ext4 = 1;
        }
        if (ef[5] != 0u) {
            // An additionalActivate effect waits for its turn in a running
            // skill instead of applying now, after its duration has already
            // counted into the instance (race_solver.rs:2504-2525).
            aa_any = 1;
            continue;
        }
        const int target = (int)ef[1];
        if (target != 1) {
            // The constructor pass has no field to deliver to, while the
            // effect's duration still feeds the instance timer above.
            if (init_pass == 0) {
                if (sk_queued_kind(kind)) {
                    const double dur = convert_float_rte(scaled);
                    wd[SKD_EFF_T + e] = dur < 0.0 ? -INFINITY : -dur;
                } else if (scaled > 0.0) {
                    // A delivered effect also leaves the caster a timed aux
                    // entry holding the skill id, which counts toward that
                    // id's remaining duration (race_solver.rs:2531-2534).
                    wd[SKD_EFF_T + e] = -(double)convert_float_rte(scaled);
                }
                int outs[RACE_MAXR];
                const int cnt = sk_targets(si, rs, scr, r, target, (int)ef[4], n, outs);
                const uint tok = SK_TOK_EFF((uint)rec, (uint)e);
                const uint owner_team = si[SI_TEAM + r];
                const int debuff = sk_is_debuff(kind, modifier);
                for (int q = 0; q < cnt; q++) {
                    const int j = outs[q];
                    // The skill_targets debuff filter (race_field.rs:458-461):
                    // after the cap an immune receiver drops out, then a
                    // same-team receiver on the owner's non-zero team, no
                    // backfill.
                    if (debuff && sd[SD_IMMUNE + j] <= 0.0) {
                        continue;
                    }
                    if (debuff && owner_team != 0u && j != r && si[SI_TEAM + j] == owner_team) {
                        continue;
                    }
                    if (kind == SK_EFF_RECOVERY) {
                        // Delivered perspective is never Self: HP lands from
                        // the receiver's max, the heal count stays with the
                        // caster.
                        sd[SD_PENDING_HP + j] += (double)(convert_float_rte(
                            shl[j * SH_STRIDE + SH_MAX]) * convert_float_rte(modifier));
                    }
                    if (kind == SK_EFF_GOAL_OUT || kind == SK_EFF_GOAL_IN
                        || kind == SK_EFF_IMMUNE) {
                        sk_aux_arm(sd + (kind == SK_EFF_GOAL_OUT ? SD_GOAL_OUT
                            : (kind == SK_EFF_GOAL_IN ? SD_GOAL_IN : SD_IMMUNE)) + j, scaled);
                    }
                    if (kind == SK_EFF_SETLANE
                        && !sk_goal_push(sd, j, convert_float_rte(modifier), scaled)) {
                        st[ST_FALLBACK] = 1u;
                        st[ST_ARG] = FB_SKILL_SEQ;
                    }
                    if (kind == SK_EFF_EXTKAKARI && (int)si[SI_KAKARI + j] != 0) {
                        // set_forced_end replaces the rush's forced clock
                        // (retail_states.rs:154-158, race_solver.rs:2584-2590).
                        sf[SF_RS_FORCED + j] = fmax(convert_float_rte(modifier), 0.f);
                    }
                    if (kind == SK_EFF_SPEEDUP || kind == SK_EFF_STAMINUP
                        || kind == SK_EFF_WISDOMUP || kind == SK_EFF_GUTSUP) {
                        // apply_external_effect with the effect rewritten to
                        // the self target (race_field.rs:467-477): the stat
                        // write, then the follow-up path's refresh, per
                        // receiver (race_solver.rs:2644-2648).
                        if (kind == SK_EFF_SPEEDUP) {
                            const float value =
                                (float)sd[SD_HPS + j] + convert_float_rte(modifier);
                            sd[SD_HPS + j] = (double)value;
                            const float speed =
                                convert_float_rte(fmin(fmax((double)value, 1.0), 2000.0));
                            const int table[8] =
                                {10500, 10000, 9000, 8000, 6000, 4000, 2000, 1000};
                            const float aptitude =
                                convert_float_rte((float)table[(int)si[SI_DISTAPT + j]]) * 0.0001f;
                            sf[SF_SPURT_MAX + j] = convert_float_rte(
                                phys_maximum_spurt_speed(pd, speed, (int)si[SI_STRATEGY + j],
                                    aptitude, dist_f,
                                    (int)pd[PD_LIVE_GUTS] ? sk_powf_word(pd, sd[SD_HPG + j])
                                        : pd[PD_RUN0 + j * PD_RUN_STRIDE + PRD_POWF]));
                        } else if (kind == SK_EFF_WISDOMUP) {
                            const float value =
                                (float)sd[SD_HPW_A + j] + convert_float_rte(modifier);
                            sd[SD_HPW_A + j] = (double)value;
                            sd[SD_HPW + j] = fmin(fmax((double)value, 1.0), 2000.0);
                        } else if (kind == SK_EFF_STAMINUP) {
                            const float value =
                                (float)sd[SD_STAM_A + j] + convert_float_rte(modifier);
                            sd[SD_STAM_A + j] = (double)value;
                            sd[SD_STAM + j] = fmin(fmax((double)value, 1.0), 2000.0);
                        } else {
                            const float value =
                                (float)sd[SD_HPG_A + j] + convert_float_rte(modifier);
                            sd[SD_HPG_A + j] = (double)value;
                            sd[SD_HPG + j] = fmin(fmax((double)value, 1.0), 2000.0);
                            // The step head recomputes minimum and the spurt
                            // plan's maximum from horse.guts every tick, so
                            // both captured caches move with the field now.
                            sf[SF_BMIN_VAL + j] = phys_minimum_speed(pd,
                                convert_float_rte(sd[SD_HPG + j]), dist_f);
                            const float speed = convert_float_rte(
                                fmin(fmax(sd[SD_HPS + j], 1.0), 2000.0));
                            const int table[8] =
                                {10500, 10000, 9000, 8000, 6000, 4000, 2000, 1000};
                            const float aptitude =
                                convert_float_rte((float)table[(int)si[SI_DISTAPT + j]]) * 0.0001f;
                            sf[SF_SPURT_MAX + j] = convert_float_rte(
                                phys_maximum_spurt_speed(pd, speed, (int)si[SI_STRATEGY + j],
                                    aptitude, dist_f, sk_powf_word(pd, sd[SD_HPG + j])));
                        }
                        if (kind == SK_EFF_STAMINUP) {
                            sk_hp_stamina_reset(si, sd, shl, pd, dist_f, j);
                        } else {
                            sk_stat_refresh(si, sd, shl, sf, distance, j);
                        }
                    }
                    if (sk_queued_kind(kind)) {
                        __global uint* qq = sk_seq(scr, kind) + j * SK_SEQN;
                        if (sk_seq_find(qq, tok) < 0 && !sk_seq_push(qq, tok)) {
                            st[ST_FALLBACK] = 1u;
                            st[ST_ARG] = FB_SKILL_SEQ;
                        }
                    }
                }
            }
            continue;
        }
        // apply_activation's own-side guard (race_solver.rs:2526): an immune
        // runner skips a debuff-class effect aimed at self.
        if (sk_is_debuff(kind, modifier) && sd[SD_IMMUNE + r] <= 0.0) {
            continue;
        }
        if (kind == 36 || kind == 37) {
            // A random-activation effect only files its request count; the
            // draw happens at the end of the priority-0 pass, and never
            // during the constructor pass (race_solver.rs:2677-2684).
            if (init_pass == 0) {
                if (kind == 36) {
                    sd[SD_RANY + r] = modifier;
                } else {
                    sd[SD_RGOLD + r] = modifier;
                }
            }
            continue;
        }
        if (kind == SK_EFF_RECOVERY) {
            if (modifier > 0.0) {
                // The engine counts a heal only from the self perspective.
                si[SI_HEAL + r] += 1u;
            }
            sd[SD_PENDING_HP + r] += (double)(convert_float_rte(shl[r * SH_STRIDE + SH_MAX])
                * convert_float_rte(modifier));
            continue;
        }
        if (kind == SK_EFF_SPEEDUP) {
            // add_stat_modifier(0): the f32 accumulator never clamps, the
            // horse field sees the clamped value, and the spurt plan then
            // samples maximum speed from it at its next recalculation.
            const float value = (float)sd[SD_HPS + r] + convert_float_rte(modifier);
            sd[SD_HPS + r] = (double)value;
            const float speed = convert_float_rte(fmin(fmax((double)value, 1.0), 2000.0));
            const int table[8] = {10500, 10000, 9000, 8000, 6000, 4000, 2000, 1000};
            const float aptitude =
                convert_float_rte((float)table[(int)si[SI_DISTAPT + r]]) * 0.0001f;
            sf[SF_SPURT_MAX + r] = convert_float_rte(phys_maximum_spurt_speed(pd, speed,
                (int)si[SI_STRATEGY + r], aptitude, dist_f,
                (int)pd[PD_LIVE_GUTS] ? sk_powf_word(pd, sd[SD_HPG + r])
                    : pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_POWF]));
            continue;
        }
        if (kind == SK_EFF_STAMINUP) {
            // add_stat_modifier(1) moves the f32 accumulator and the clamped
            // field. The engine rebuilds max hp only after the constructor
            // pass (hp.init follows the constructor fires) and on fan
            // delivery refresh (race_solver.rs:2644-2648); a mid-race self
            // fire never touches it. A wisdom-gated constructor fire is
            // already baked into the init seeds and hp, so it writes nothing.
            if ((rd[2] & SK_FLAG_INIT_BUMP) == 0) {
                const float value = (float)sd[SD_STAM_A + r] + convert_float_rte(modifier);
                sd[SD_STAM_A + r] = (double)value;
                sd[SD_STAM + r] = fmin(fmax((double)value, 1.0), 2000.0);
                if (init_pass != 0) {
                    sk_hp_stamina_reset(si, sd, shl, pd, dist_f, r);
                }
            }
            continue;
        }
        if (kind == SK_EFF_WISDOMUP) {
            // A static fire is already inside the PRD-baked init inputs. Any
            // other fire, init pass included, moves the accumulator and the
            // live field the pace and downhill draws read; a wisdom-gated
            // static fire lands in the init pass, exactly behind the engine's
            // retail init draws.
            if ((rd[2] & SK_FLAG_BUMPED) == 0) {
                const float value = (float)sd[SD_HPW_A + r] + convert_float_rte(modifier);
                sd[SD_HPW_A + r] = (double)value;
                sd[SD_HPW + r] = fmin(fmax((double)value, 1.0), 2000.0);
                if (init_pass != 0) {
                    // The retail init's hp.init (hp_policy.rs:208) captures
                    // retail.wisdom from the field after the constructor fire.
                    sf[SF_HPW + r] = convert_float_rte(sd[SD_HPW + r]);
                }
            }
            continue;
        }
        if (kind == SK_EFF_GUTSUP) {
            // A static fire is already inside the PRD-baked init inputs. Any
            // other fire moves the accumulator and the live field; the
            // minimum and spurt caches follow it, since the engine's step
            // head recomputes both from horse.guts every tick.
            if ((rd[2] & SK_FLAG_BUMPED) == 0) {
                const float value = (float)sd[SD_HPG_A + r] + convert_float_rte(modifier);
                sd[SD_HPG_A + r] = (double)value;
                sd[SD_HPG + r] = fmin(fmax((double)value, 1.0), 2000.0);
                sf[SF_BMIN_VAL + r] = phys_minimum_speed(pd,
                    convert_float_rte(sd[SD_HPG + r]), dist_f);
                const float speed =
                    convert_float_rte(fmin(fmax(sd[SD_HPS + r], 1.0), 2000.0));
                const int table[8] = {10500, 10000, 9000, 8000, 6000, 4000, 2000, 1000};
                const float aptitude =
                    convert_float_rte((float)table[(int)si[SI_DISTAPT + r]]) * 0.0001f;
                sf[SF_SPURT_MAX + r] = convert_float_rte(phys_maximum_spurt_speed(pd, speed,
                    (int)si[SI_STRATEGY + r], aptitude, dist_f, sk_powf_word(pd, sd[SD_HPG + r])));
                if (init_pass != 0) {
                    // The retail init's hp.init (hp_policy.rs:207) captures
                    // guts_factor from the field after the constructor fire.
                    sf[SF_HPG + r] = sk_hp_guts_factor(pd, convert_float_rte(sd[SD_HPG + r]));
                }
            }
            continue;
        }
        if (kind == SK_EFF_GOAL_OUT || kind == SK_EFF_GOAL_IN || kind == SK_EFF_IMMUNE) {
            // The engine keeps a timed aux entry for these kinds, and its
            // duration counts toward the id's remaining time
            // (race_solver.rs:2540-2548).
            const double dur = convert_float_rte(scaled);
            wd[SKD_EFF_T + e] = dur < 0.0 ? -INFINITY : -dur;
            sk_aux_arm(sd + (kind == SK_EFF_GOAL_OUT ? SD_GOAL_OUT
                : (kind == SK_EFF_GOAL_IN ? SD_GOAL_IN : SD_IMMUNE)) + r, scaled);
            continue;
        }
        if (kind == SK_EFF_SETLANE) {
            // SetTargetLane lives in the engine's aux list while it runs,
            // so its duration counts toward the id's remaining time
            // (race_solver.rs:2545-2548).
            const double dur = convert_float_rte(scaled);
            wd[SKD_EFF_T + e] = dur < 0.0 ? -INFINITY : -dur;
            if (!sk_goal_push(sd, r, convert_float_rte(modifier), scaled)) {
                st[ST_FALLBACK] = 1u;
                st[ST_ARG] = FB_SKILL_SEQ;
            }
            continue;
        }
        if (kind == SK_EFF_EXTKAKARI) {
            if ((int)si[SI_KAKARI + r] != 0) {
                sf[SF_RS_FORCED + r] = fmax(convert_float_rte(modifier), 0.f);
            }
            continue;
        }
        if (!sk_queued_kind(kind)) {
            continue;
        }
        const double dur = convert_float_rte(scaled);
        wd[SKD_EFF_T + e] = dur < 0.0 ? -INFINITY : -dur;
        __global uint* q1 = sk_seq(scr, kind) + r * SK_SEQN;
        const uint tok = SK_TOK_EFF((uint)rec, (uint)e);
        if (sk_seq_find(q1, tok) < 0 && !sk_seq_push(q1, tok)) {
            st[ST_FALLBACK] = 1u;
            st[ST_ARG] = FB_SKILL_SEQ;
        }
    }
    if (tgtp != 0) {
        rc_give(scr, SCR_RC_TGT, tgt_res - (int)(tgtp - tgt0));
    }
    // The activation-hook record for skillEvents: a pending fire reuses the
    // live (runner, id) slot's ordinal, else takes the next sequence number
    // (lib.rs:660-682).
    __global uint* act = 0;
    if (scr[SCR_RC_ON] != 0u) {
        act = rc_take(scr, st, SCR_RC_ACT, RC_ACT_CAP, 10);
    }
    if (act != 0) {
        // The activate hook at lib.rs:660-682 reads the id's live self-entry
        // timers after this fire's own entries landed, so the ledger slots,
        // not the instance timer, decide pending (race_solver.rs:731-740).
        const int nrec = (int)sk[SK_NREC];
        double remaining = 0.0;
        for (int q = 0; q < nrec; q++) {
            __global const uint* qd = sk + SK_RECORDS + SK_RSTRIDE * q;
            if ((int)qd[1] != r || qd[0] != rd[0]) {
                continue;
            }
            __global const double* qt = skd + SKD_STRIDE * q;
            for (int e2 = 0; e2 < (int)qd[5]; e2++) {
                const double t2 = qt[SKD_EFF_T + e2];
                if (t2 < 0.0 && isfinite(t2) && -t2 > remaining) {
                    remaining = -t2;
                }
            }
        }
        int reused = 0;
        if (remaining > 0.0) {
            // The engine's insert overwrites a live (horse, id) slot and
            // keeps its ordinal, so the holding record retires here.
            for (int q = 0; q < nrec; q++) {
                __global const uint* qd = sk + SK_RECORDS + SK_RSTRIDE * q;
                if ((int)qd[1] != r || qd[0] != rd[0]) {
                    continue;
                }
                __global uint* qs = sks + SKR_STRIDE * q;
                if (qs[SKR_PEND] == 0u) {
                    continue;
                }
                if (reused == 0) {
                    reused = (int)qs[SKR_ORD];
                }
                qs[SKR_PEND] = 0u;
                qs[SKR_ORD] = 0u;
            }
        }
        int ord = reused != 0 ? reused - 1 : (int)scr[SCR_RC_SEQ];
        if (reused == 0) {
            scr[SCR_RC_SEQ] = (uint)ord + 1u;
        }
        __global uint* rs2 = sks + SKR_STRIDE * rec;
        if (remaining > 0.0) {
            rs2[SKR_ORD] = (uint)(ord + 1);
            rs2[SKR_PEND] = 1u;
        }
        const ulong cbits = as_ulong(sd[SD_CLOCK + r]);
        const ulong dbits = as_ulong(remaining);
        act[0] = (uint)r;
        act[1] = rd[0];
        act[2] = rd[26];
        act[3] = (uint)cbits;
        act[4] = (uint)(cbits >> 32);
        act[5] = (uint)dbits;
        act[6] = (uint)(dbits >> 32);
        act[7] = (uint)(ord + 1);
        act[8] = (uint)((remaining > 0.0 ? 1 : 0) | (scr[SCR_RC_RPT] != 0u ? 2 : 0));
        act[9] = scr[SK_JRN_T];
    }
    wd[SKD_INSTANCE] = runtime > 0.0 ? -runtime : 0.0;
    wd[SKD_RUN_T] = wd[SKD_INSTANCE];
    wd[SKD_AEXT] = 0.0;
    if ((aa_any || ext4) && runtime > 0.0) {
        // The running instance exists once the fire left a positive duration
        // behind with a delayed effect or a duration_usage 4 effect, and it
        // carries its own timer, which an overtake extends
        // (race_solver.rs:2650-2660).
        __global uint* rs = sks + SKR_STRIDE * rec;
        rs[SKR_AA_RUN] = 1u;
        // Each new instance carries fresh zero delivery counters, so the cap
        // applies per instance and the counters must be zeroed here, the one
        // point where the kernel knows the record belongs to this runner
        // (race_solver.rs:2474, 2525). The pending slot words live past the
        // counters and survive a new instance.
        for (int w = SKR_AA_COUNT; w < SKR_ORD; w++) {
            rs[w] = 0u;
        }
        if (!aa_push(scr, r, rec)) {
            st[ST_FALLBACK] = 1u;
            st[ST_ARG] = FB_SKILL_SEQ;
        }
        sk_jrn(scr, SK_JRN_FIRE, (uint)rec, (uint)r, (uint)n_eff,
            (uint)(aa_any + 2 * ext4), wd[SKD_RUN_T], sd[SD_CLOCK + r]);
    }
    const int phase = (int)si[SI_PHASE + r];
    if (phase == 0) {
        si[SI_ACT0 + r] += 1u;
    } else if (phase == 1) {
        si[SI_ACT1 + r] += 1u;
    } else {
        si[SI_ACT2 + r] += 1u;
    }
    // apply_activation counts a fire past the course's distance half into
    // activate_count_later_half, and marks the skill id used, all unconditionally
    // for a self-perspective activation.
    if (sd[SD_POS + r] >= distance * 0.5) {
        si[SI_ACTLH + r] += 1u;
    }
    sks[SKR_STRIDE * rec + SKR_USED] = 1u;
    // activated_green_skills holds each green-tagged own skill id once per
    // runner, and the value_usage 14 tiers read its size. The flag covers
    // every record of the (runner, id) group, fired alternatives included.
    if ((rd[2] & SK_FLAG_GREEN) != 0u && sks[SKR_STRIDE * rec + SKR_GREEN] == 0u) {
        si[SI_GREEN + r] += 1u;
        for (int other = 0; other < (int)sk[SK_NREC]; other++) {
            __global const uint* od = sk + SK_RECORDS + SK_RSTRIDE * other;
            if ((int)od[1] == r && od[0] == rd[0]) {
                sks[SKR_STRIDE * other + SKR_GREEN] = 1u;
            }
        }
    }
    const double cd = as_double((ulong)rd[13] | ((ulong)rd[14] << 32));
    if (cd <= 0.0) {
        sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DEAD;
    } else if (init_pass != 0) {
        // The engine has no retail timer map in the constructor pass, so that
        // fire's cooldown starts at once as an absolute ready time.
        wd[SKD_COOLDOWN] = -(double)(convert_float_rte(cd) * (dist_f / 1000.0f));
        sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_COOLDOWN;
    } else {
        sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_WAITING;
    }
    // v75 anyskill count: every own activation counts, and the engine's
    // excluded skill ids are the non-numeric legacy ones, never a record.
    si[SI_ACTALL + r] += 1u;
    // The shared cooldown key covers the whole skill id
    // (race_solver_builder.rs:589), so the fire's park lowers every live
    // record of the (runner, id) group. Each ready sibling inherits the
    // leader's state and timers: a READY one and a Cooldown one whose timer
    // already ran up, which the engine's key map also reads as ready. A
    // wisdom-failed retained primary stays pending only for reservations
    // (race_solver_builder.rs:1002-1005), so it keeps its state but shares
    // the key's timers. DEAD siblings left the pending list and stay put.
    __global uint* ls = sks + SKR_STRIDE * rec;
    const uint lead_state = ls[SKR_STATE];
    for (int other = 0; other < (int)sk[SK_NREC]; other++) {
        if (other == rec) {
            continue;
        }
        const __global const uint* od = sk + SK_RECORDS + SK_RSTRIDE * other;
        if ((int)od[0] != (int)rd[0] || (int)od[1] != r) {
            continue;
        }
        __global uint* os = sks + SKR_STRIDE * other;
        __global double* ow = skd + SKD_STRIDE * other;
        const uint ost = os[SKR_STATE];
        if (ost == (uint)SK_S_DISABLED) {
            if (!(od[2] & SK_FLAG_RETAIN)) {
                continue;
            }
            ow[SKD_INSTANCE] = wd[SKD_INSTANCE];
            ow[SKD_COOLDOWN] = wd[SKD_COOLDOWN];
            continue;
        }
        const int ready = ost == (uint)SK_S_READY
            || (ost == (uint)SK_S_COOLDOWN && ow[SKD_COOLDOWN] >= 0.0);
        if (!ready) {
            continue;
        }
        os[SKR_USED] = 1u;
        os[SKR_STATE] = lead_state;
        ow[SKD_INSTANCE] = wd[SKD_INSTANCE];
        ow[SKD_COOLDOWN] = wd[SKD_COOLDOWN];
    }
}

// The end-of-priority-0 reservation pass. A kind-37 effect requests a
// gold-or-evolution pick and kind-36 also allows white; every pick draws one
// index from the live candidate list and removes it, so the draws depend on
// the eligibility order (race_solver.rs:2105-2119, 2696-2737).
static void aa_select(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si, __global double* sd,
    __global uint* bank, const int paired, const int r) {
    const int nrec = (int)sk[SK_NREC];
    for (int rec = 0; rec < nrec; rec++) {
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((int)rd[1] == r) {
            sks[SKR_STRIDE * rec + SKR_RESERVED] = 0u;
        }
    }
    si[SI_RESRV + r] = 0u;
    const double gold = sd[SD_RGOLD + r];
    const double any = sd[SD_RANY + r];
    sd[SD_RGOLD + r] = 0.0;
    sd[SD_RANY + r] = 0.0;
    if (gold <= 0.0 && any <= 0.0) {
        return;
    }
    __global uint* abank = paired ? scr + SCR_BANKS + r * BANK_WORDS : bank;
    int picked[16];
    int gold_n = 0;
    for (int pass = 0; pass < 2; pass++) {
        const double want = pass == 0 ? gold : any;
        if (want <= 0.0) {
            continue;
        }
        // Gold or evolution always; white joins only the any-request pool
        // (race_solver.rs:2705).
        const uint rar_bits = pass == 0 ? SK_FLAG_RANGOLD
            : (SK_FLAG_RANGOLD | SK_FLAG_RANWHITE);
        int cand[16];
        int nc = 0;
        for (int rec = 0; rec < nrec && nc < 16; rec++) {
            __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            if ((int)rd[1] != r
                || (rd[2] & (SK_FLAG_PRIMARY | SK_FLAG_GENERAL | SK_FLAG_RSELIG))
                    != (SK_FLAG_PRIMARY | SK_FLAG_GENERAL | SK_FLAG_RSELIG)
                || (rd[2] & rar_bits) == 0u) {
                continue;
            }
            int skip = 0;
            for (int q = 0; q < gold_n; q++) {
                if (sk[SK_RECORDS + SK_RSTRIDE * picked[q]] == rd[0]) {
                    skip = 1;
                    break;
                }
            }
            if (skip != 0) {
                continue;
            }
            const uint state = sks[SKR_STRIDE * rec + SKR_STATE];
            if (state == (uint)SK_S_DEAD) {
                continue;
            }
            // A wisdom-failed record stays pending only as a retained
            // primary, where a reservation overrides the failed lottery
            // (race_solver_builder.rs:1002-1005).
            if (state == (uint)SK_S_DISABLED && !(rd[2] & SK_FLAG_RETAIN)) {
                continue;
            }
            __global double* wd = skd + SKD_STRIDE * rec;
            if (((state == (uint)SK_S_COOLDOWN || state == (uint)SK_S_DISABLED)
                && wd[SKD_COOLDOWN] < 0.0)
                || wd[SKD_INSTANCE] < 0.0) {
                continue;
            }
            __global uint* list = scr + SK_RUN + r * (1 + SK_RUNN);
            for (int k = 1; k <= (int)list[0]; k++) {
                const int rr = (int)list[k];
                if (sk[SK_RECORDS + SK_RSTRIDE * rr] == rd[0]) {
                    skip = 1;
                    break;
                }
            }
            if (skip != 0) {
                continue;
            }
            // The id is distinct once: a later alternative replaces the
            // earlier entry at the same position, keeping the id's first
            // order slot.
            int dup = -1;
            for (int q = 0; q < nc; q++) {
                if (sk[SK_RECORDS + SK_RSTRIDE * cand[q]] == rd[0]) {
                    dup = q;
                    break;
                }
            }
            if (dup >= 0) {
                cand[dup] = rec;
            } else {
                cand[nc++] = rec;
            }
        }
        int take = (int)want < nc ? (int)want : nc;
        for (int t = 0; t < take; t++) {
            const int j = (int)r128_integer(abank, 0, nc - t);
            const int rec = cand[j];
            for (int q = j; q + 1 < nc - t; q++) {
                cand[q] = cand[q + 1];
            }
            sks[SKR_STRIDE * rec + SKR_RESERVED] = 1u;
            si[SI_RESRV + r] = 1u;
            if (pass == 0 && gold_n < 16) {
                picked[gold_n++] = rec;
            }
        }
    }
}

// process_skill_priority for one priority pass: the whole field in roster
// order, records in wire order, with the compact scan gate. An out-of-window
// visit that the engine's run-skip covers takes its ordered turn as a floor
// bump only; the reservation set bypasses the window, the conditions and the
// arms (race_solver.rs:2188-2320).
static void skills_scan(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si,
    __global double* sd, __global double* shl, __global const uint* rs,
    __global const double* pd, __global float* sf, const double distance,
    const int p, const int n, const float dist_f, const int init_pass,
    __global uint* ev, __global uint* st, __global uint* bank, const int paired) {
    const int nrec = (int)sk[SK_NREC];
    // The prerequisite watch pass runs before the gate, over the records of
    // this priority only, latching each one the first time it holds.
    for (int rec = 0; rec < nrec; rec++) {
        const __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((int)rd[3] != p || rd[22] == 0u || sks[SKR_STRIDE * rec + SKR_LATCH] != 0u) {
            continue;
        }
        const uint state = sks[SKR_STRIDE * rec + SKR_STATE];
        if (state == (uint)SK_S_DEAD || state == (uint)SK_S_DISABLED) {
            continue;
        }
        const int r = (int)rd[1];
        if (sk_prereq_check(sk, rd, skd + SKD_STRIDE * rec, si, shl, sd, pd, sf, rs,
            sks, r, n, distance)) {
            sks[SKR_STRIDE * rec + SKR_LATCH] = 1u;
        }
    }
    for (int r = 0; r < n; r++) {
        const double pos = sd[SD_POS + r];
        if (p == 0) {
            // The ordered turn walk restarts with every priority-0 pass
            // (race_solver.rs:2192-2194).
            sd[SD_AFLOOR + r] = -1e300;
        }
        for (int rec = 0; rec < nrec; rec++) {
            __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            if ((int)rd[1] != r || (int)rd[3] != p) {
                continue;
            }
            uint state = sks[SKR_STRIDE * rec + SKR_STATE];
            const int reserved = sks[SKR_STRIDE * rec + SKR_RESERVED] != 0u;
            if ((state == (uint)SK_S_DEAD || state == (uint)SK_S_DISABLED) && !reserved) {
                continue;
            }
            __global double* wd = skd + SKD_STRIDE * rec;
            if (p == 0) {
                const int out = !(pos >= wd[SKD_TRIG0] && pos < wd[SKD_TRIG1]);
                if (out && ((rd[2] & SK_FLAG_RETAIN) != 0u || !(pos >= wd[SKD_TRIG1]))
                    && (rd[2] & SK_FLAG_ANYACT) == 0u && (int)si[SI_AANU + r] == 0
                    && (int)si[SI_RESRV + r] == 0) {
                    if ((double)rd[0] > sd[SD_AFLOOR + r]) {
                        sd[SD_AFLOOR + r] = (double)rd[0];
                    }
                    continue;
                }
                aa_turns(sk, sks, skd, scr, si, rs, sd, st, bank, paired, n, dist_f, r,
                    (double)rd[0]);
            }
            if (rd[2] & SK_FLAG_ANYACT) {
                // observe_anyskill runs on every visit before the gate: the
                // engine reads the activation count whenever the record is
                // checkable, and a visit that is not checkable leaves the
                // (last, trues) state untouched and reports false.
                const int ready = state == (uint)SK_S_READY
                    || (state == (uint)SK_S_COOLDOWN && wd[SKD_COOLDOWN] >= 0.0);
                const int checkable =
                    ready && (rd[22] == 0u || sks[SKR_STRIDE * rec + SKR_LATCH] != 0u);
                if (!checkable) {
                    wd[SKD_ANY_RESULT] = 0.0;
                } else {
                    const double value =
                        (double)(int)si[SI_ACTALL + r] - wd[SKD_ANY_TRUES];
                    const int changed = value > wd[SKD_ANY_LAST];
                    wd[SKD_ANY_LAST] = value;
                    if (changed) {
                        wd[SKD_ANY_TRUES] += 1.0;
                    }
                    wd[SKD_ANY_RESULT] = (double)changed;
                }
            }
            if (pos >= wd[SKD_TRIG1] && !reserved) {
                if (!(rd[2] & SK_FLAG_RETAIN)) {
                    sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DEAD;
                }
                continue;
            }
            if (state == (uint)SK_S_COOLDOWN && wd[SKD_COOLDOWN] >= 0.0) {
                // A reservation-only record returns to its inert state.
                state = sks[SKR_STRIDE * rec + SKR_WISFAIL] != 0u
                    ? (uint)SK_S_DISABLED : (uint)SK_S_READY;
                sks[SKR_STRIDE * rec + SKR_STATE] = state;
            }
            if (reserved) {
                // The reservation fire bypasses the arms, the windows and the
                // extra condition, so even a wisdom-failed retained primary
                // fires once its shared key is ready (race_solver.rs:2263-2277).
                const int key_ready = state == (uint)SK_S_READY
                    || (state == (uint)SK_S_DISABLED && wd[SKD_COOLDOWN] >= 0.0);
                if (key_ready) {
                    const double gap = sd[SD_POS + (int)rs[RS_ORDER + 0]] - pos;
                    skills_fire(sk, sks, skd, scr, si, shl, sd, pd, sf, rs, rec, r,
                        shl[r * SH_STRIDE + SH_HP], fmax(gap, 0.0), dist_f, init_pass,
                        n, distance, ev, st, bank, paired);
                }
                continue;
            }
            if (state != (uint)SK_S_READY || pos < wd[SKD_TRIG0]) {
                continue;
            }
            if (rd[22] != 0u && sks[SKR_STRIDE * rec + SKR_LATCH] == 0u) {
                continue;
            }
            if (!sks[SKR_STRIDE * rec + SKR_SINGLE] && !sk_win_any(wd, SKD_WINDOWS,
                (int)sks[SKR_STRIDE * rec + SKR_NWIN], pos)) {
                continue;
            }
            if (!sk_eval_arms(sk, rd, wd, sks, rec, si, shl, sd, pd, sf, rs, r, n,
                distance)) {
                continue;
            }
            if (init_pass != 0 && (rd[2] & SK_FLAG_INIT_DEFER) != 0u) {
                continue;
            }
            const double gap = sd[SD_POS + (int)rs[RS_ORDER + 0]] - pos;
            skills_fire(sk, sks, skd, scr, si, shl, sd, pd, sf, rs, rec, r,
                shl[r * SH_STRIDE + SH_HP], fmax(gap, 0.0), dist_f, init_pass, n,
                distance, ev, st, bank, paired);
        }
        if (p == 0) {
            // The pass end gives every running skill still short of a turn
            // its turn, then the random-skill reservations are recomputed
            // (race_solver.rs:2105-2119, 2318-2320).
            aa_turns(sk, sks, skd, scr, si, rs, sd, st, bank, paired, n, dist_f, r, 1e300);
            aa_select(sk, sks, skd, scr, si, sd, bank, paired, r);
        }
    }
}

// update_near_lane_timers: a rank change resets the three near-lane timers,
// otherwise the published order's neighbors accumulate them with the plain
// f64 dt. Runs on post-movement rows, before the competition pass.
static void near_lane_timers(__global uint* si, __global double* sd,
    __global const uint* rs, const int r, const int n, const double dt) {
    if ((int)si[SI_RANKCOL + r] != (int)rs[RS_PREVRANK + r]) {
        sd[SD_IFNEAR + r] = 0.0;
        sd[SD_BHNEAR + r] = 0.0;
        sd[SD_BHNEAR1 + r] = 0.0;
        return;
    }
    const double pos = sd[SD_POS + r];
    const double lane = sd[SD_LANE + r];
    const int rank = (int)si[SI_RANKCOL + r];
    if (rank > 1) {
        const int j = (int)rs[RS_ORDER + rank - 2];
        const double gap = fmax(sd[SD_POS + j] - pos, 0.0);
        const double lgap = fabs(sd[SD_LANE + j] - lane);
        sd[SD_IFNEAR + r] =
            (gap <= 2.5 && lgap <= (double)0.0556f) ? sd[SD_IFNEAR + r] + dt : 0.0;
    } else {
        sd[SD_IFNEAR + r] = 0.0;
    }
    if (rank < n) {
        const int j = (int)rs[RS_ORDER + rank];
        const double gap = fmax(pos - sd[SD_POS + j], 0.0);
        const double lgap = fabs(lane - sd[SD_LANE + j]);
        sd[SD_BHNEAR + r] =
            (gap <= 2.5 && lgap <= (double)0.0556f) ? sd[SD_BHNEAR + r] + dt : 0.0;
        sd[SD_BHNEAR1 + r] =
            (gap <= 5.0 && lgap <= (double)0.15f) ? sd[SD_BHNEAR1 + r] + dt : 0.0;
    } else {
        sd[SD_BHNEAR + r] = 0.0;
        sd[SD_BHNEAR1 + r] = 0.0;
    }
}

// Duel and spot competition register or drop their fixed receiver tokens
// during the field-condition passes; the token orders catch up once per tick
// after them, duel first, matching the engine's duel-start registration
// before the spot pass. A cleared flag skips its token in every fold even
// before this sync, so the same-tick expiry reads stay exact.
static void sk_token_sync(__global uint* scr, __global const uint* si, const int n) {
    for (int r = 0; r < n; r++) {
        __global uint* tq = scr + SK_TSEQ + r * SK_SEQN;
        __global uint* aq = scr + SK_ASEQ + r * SK_SEQN;
        if ((int)si[SI_DUEL_TOKEN + r]) {
            if (sk_seq_find(tq, SK_TOK_DUEL) < 0) {
                sk_seq_push(tq, SK_TOK_DUEL);
            }
            if (sk_seq_find(aq, SK_TOK_DUEL) < 0) {
                sk_seq_push(aq, SK_TOK_DUEL);
            }
        } else {
            sk_seq_remove(tq, SK_TOK_DUEL);
            sk_seq_remove(aq, SK_TOK_DUEL);
        }
        if ((int)si[SI_SPOT_TOKEN + r]) {
            if (sk_seq_find(tq, SK_TOK_SPOT) < 0) {
                sk_seq_push(tq, SK_TOK_SPOT);
            }
        } else {
            sk_seq_remove(tq, SK_TOK_SPOT);
        }
    }
}

// retail_target_speed_with_current_skills: the live Horse row target, the
// non-natural current-speed values folded over f32 target_speed in push
// order. Written after every skill pass and after each runner's physics
// update, read by the lane caches, the blocking sample and the lane calls.
static void sk_columns(__global const uint* sk, __global const uint* scr,
    __global double* sd, const int n) {
    for (int r = 0; r < n; r++) {
        float acc = convert_float_rte(sd[SD_TS + r]);
        const __global uint* q = scr + SK_CSEQ + r * SK_SEQN;
        for (int i = 0; i < SK_SEQN; i++) {
            const uint tok = q[i];
            if (tok == 0u) {
                break;
            }
            acc = acc + sk_tok_value(sk, scr, tok, SK_EFF_CURRENT);
        }
        sd[SD_TSWSK + r] = (double)acc;
    }
}

static void sk_column_one(__global const uint* sk, __global const uint* scr,
    __global double* sd, const int r) {
    float acc = convert_float_rte(sd[SD_TS + r]);
    const __global uint* q = scr + SK_CSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        acc = acc + sk_tok_value(sk, scr, tok, SK_EFF_CURRENT);
    }
    sd[SD_TSWSK + r] = (double)acc;
}

// pace_cancel_current_speed() > 0. || retail_target_modifiers.apply(0.) > 0.
// The fixed tokens fold at their pd values while their flags stay set; the
// engine's receiver dropped an expired token before this context is built
// mid-race_step, which is the same instant the flag clears.
static int sk_has_speed(__global const uint* sk, __global const uint* scr,
    __global const uint* si, const int r, __global const double* pd,
    __global const double* sd) {
    float ordinary = 0.f;
    const __global uint* cq = scr + SK_CSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = cq[i];
        if (tok == 0u) {
            break;
        }
        ordinary = ordinary + sk_tok_value(sk, scr, tok, SK_EFF_CURRENT);
    }
    float target = 0.f;
    const __global uint* tq = scr + SK_TSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = tq[i];
        if (tok == 0u) {
            break;
        }
        if (tok == SK_TOK_SPOT) {
            if ((int)si[SI_SPOT_TOKEN + r]) {
                target = target + convert_float_rte((int)pd[PD_LIVE_GUTS]
                    ? sd[SD_SPOT_MOD + r] : pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_SPOT_M]);
            }
        } else if (tok == SK_TOK_DUEL) {
            if ((int)si[SI_DUEL_TOKEN + r]) {
                target = target + convert_float_rte((int)pd[PD_LIVE_GUTS]
                    ? sd[SD_DUEL_T + r] : pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_DUEL_T]);
            }
        } else {
            target = target + sk_tok_value(sk, scr, tok, SK_EFF_TARGET);
        }
    }
    return ((double)ordinary > 0. || target > 0.f) ? 1 : 0;
}

// The ordinary (token-excluded) receiver values for one physics chain, in
// registration order, so the existing mods arguments keep the engine chain.
static int sk_mods(__global const uint* sk, __global const uint* scr, const int r,
    const int kind, __global float* out) {
    const __global uint* q = scr + (kind == SK_EFF_TARGET ? SK_TSEQ
        : (kind == SK_EFF_ACCEL ? SK_ASEQ
            : (kind == SK_EFF_LANESPEED ? SK_LSEQ : SK_CSEQ))) + r * SK_SEQN;
    int m = 0;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        if (tok == SK_TOK_SPOT || tok == SK_TOK_DUEL) {
            continue;
        }
        out[m++] = sk_tok_value(sk, scr, tok, kind);
    }
    return m;
}

// retail_current_speed: the natural group folds first, then the ordinary
// group, the engine's apply_additive order over the two receiver vecs.
static float sk_current_fold(__global const uint* sk, __global const uint* scr,
    const int r) {
    float natural = 0.f;
    float ordinary = 0.f;
    const __global uint* q = scr + SK_CSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        natural = natural + sk_tok_value(sk, scr, tok, SK_EFF_CURRENT_NAT);
    }
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        ordinary = ordinary + sk_tok_value(sk, scr, tok, SK_EFF_CURRENT);
    }
    return natural + ordinary;
}