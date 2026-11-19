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
#define RI_PAIRED 5          // 1 = pairedRandomStreams (per-runner RUN banks)
#define RI_STARTUP 6         // 1 = experimentalRetailStartup (fallback)
#define RI_GEOM_ON 7         // 1 = experimentalRetailGeometry
#define RI_HOST_FLAG 8       // nonzero: the host already knows this race needs
                             // a fallback (skills, startup, no geometry)
#define RI_HOST_ARG 9        // which domain the host flag stands for
#define RI_NSLOPES 10
#define RI_NSTR 11           // straights present
#define RI_NCOR 12           // corners present
#define RI_LOT_MIN 13
#define RI_LOT_MAX 14
#define RI_EMPTY_MOVE 15       // course empty_in_move (first_move_lane_point > 0)
// Debug-only: when nonzero and equal to the frame index just captured, the
// loop writes pace, clock, rush, lane, and lane-list state for each runner
// into the fourteen reserved tail frame slots. Zero in production keeps
// those slots zero.
#define RI_DEBUG_TICK 304
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

#define RIL_STRIDE 640
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
// 400..: CompeteFight and CompeteTop settings with the per-runner powf
// precomputes. The duel and spot modifiers are race constants derived from
// guts, so the host supplies them like PT_POWF (no powf in the kernel).
#define PD_CF_DIST 400         // CompeteFight DistanceGap (raw JSON f64)
#define PD_CF_LANE 401         // CompeteFight LaneGap (raw JSON f64)
#define PD_CF_SPEED 402        // [3] CompeteFight AddParam1Coef1..3
#define PD_CF_ACCEL 405        // [3] CompeteFight AddParam2Coef1..3
#define PD_PM_RANGE 408        // PaseMakerChangeForwardRunningStyleRange
#define PD_PM_NOTFWD 409
#define PD_DTYPE 410               // course distance_type, the asitame coefficient row       // PaseMakerChangeTopHorseNotMostForwardStyleCount
#define PD_FC_HALF0 411            // final_corner_half start, NaN when the course has none
#define PD_FC_HALF1 412            // final_corner_half end
#define PD_RUN0 448            // per runner, stride 5
#define PD_RUN_STRIDE 8
#define PRD_POWF 0             // per-runner spurt powf bits as the integer value
#define PRD_DUEL_T 1           // duel target-speed modifier, f64
#define PRD_DUEL_A 2           // duel accel modifier, f64
#define PRD_SPOT_M 3           // spot struggle modifier, f64
#define PRD_SPOT_D 4           // spot struggle duration, f64
#define PRD_BOOST_GUTS 5       // bumped guts field f64, 0 when no bump
#define PRD_BOOST_WIS 6        // bumped wisdom field f64, 0 when no bump
#define PRD_RUSH_BONUS 7       // rushing bonus f32 bits as the integer value

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
// Integer state, one 24-wide column per slot (field base + runner index).
#define SI_SLOTS 82
#define SI_STRIDE (SI_SLOTS * 24)
#define SI_SLOT(n) ((n) * 24)
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
// f32 state (raw-bit floats), one 24-wide column per slot.
#define SF_SLOTS 56
#define SF_STRIDE (SF_SLOTS * 24)
#define SF_SLOT(n) ((n) * 24)
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
// RushingScore (stadium_score.rs): per-tick countdown while kakari is held;
// each zero crossing charges one penalty. The count stays an exact integer.
#define SF_RSCORE SF_SLOT(41)     // penalties charged (integer-valued f32)
#define SF_RREM SF_SLOT(42)       // countdown remaining, starts at 1.0
#define SCR_LISTS (SCR_SF + SF_STRIDE)
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
#define RS_PM 56              // section pacemaker, +1 (0 none)
#define RS_PM_SEC 57          // pacemaker section at the last handover
#define RS_PM_COUNT 58
#define RS_PM_HANDED 59
#define RS_PM_HAS 60          // 0 before the constructor's first selection
#define RS_COUNTS 64          // 8: exact per-strategy counts
#define RS_STYLE 72           // 24: base-style prefix maxima over the order
#define RS_STRAIGHT 96        // 24: pre-movement final-straight row flags
#define RS_LKEYS 120          // 24: resolve-overlaps truncated lane keys
#define RS_LORDER 144         // 24: resolve-overlaps scan order
#define RS_ADV 168            // 24: stepped (not gated) flags of the tick
#define SCR_BANKS (SCR_RACE + 192) // paired mode: one RUN bank per runner
// Skill runtime scratch: one state word block per table record, the three
// per-runner receiver token orders, then the keyed-substream bank. The token
// orders mirror the engine's AdditiveReceiver entry vectors, 0 is the empty
// sentinel and registration order is append order.
#define SCR_SK (SCR_BANKS + 24 * BANK_WORDS)
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
#define SCR_SK_WORDS (SK_STAGED + 2 * SK_RMAX)
#define SCR_U_WORDS SCR_SK_WORDS

// Per-race scratch f64, one 24-wide column per slot (field base + runner
// index). Array fields keep 24 slots per element: base + element * 24 + r.
#define SD_SLOTS 107
#define SD_STRIDE (SD_SLOTS * 24)
#define SD_SLOT(n) ((n) * 24)
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
#define SD_BTS SD_SLOT(9)         // base_target_speed[3], element stride 24
#define SD_PKCOEF SD_SLOT(12)     // pos_keep_speed_coef
#define SD_SECMOD SD_SLOT(15)     // section_modifier[25], element stride 24
#define SD_DUEL SD_SLOT(40)       // duel proximity timers, element j stride 24, -1.0 absent
#define SD_SPOT_TIMER SD_SLOT(64) // spot_struggle_timer, f64 (f32-rounded steps)
#define SD_CP_GAUGE SD_SLOT(65)   // conserve power clean ledger gauge
#define SD_CP_PREVRATE SD_SLOT(66) // conserve previous_rate
#define SD_CP_INITPW SD_SLOT(67)  // conserve initial_power
#define SD_CP_ACC SD_SLOT(68)     // legacy cp_acc accumulate
#define SD_AS_TIMER SD_SLOT(69)   // asitame effect duration timer, counts up to zero
#define SD_RAWPW SD_SLOT(70)      // horse raw_power from initialization output 2
#define SD_CLOCK SD_SLOT(71)      // accumulatetime race clock, f64 (f32-rounded steps)
#define SD_BSIDE SD_SLOT(72)      // blocked_side_continuetime, f64
#define SD_IFNEAR SD_SLOT(73)     // infront_near_lane_time, f64
#define SD_BHNEAR SD_SLOT(74)     // behind_near_lane_time, f64
#define SD_BHNEAR1 SD_SLOT(75)    // behind_near_lane_time_set1, f64
#define SD_HPS SD_SLOT(76)        // retail_stat_values[0] accumulator, f64 bits of f32
#define SD_PENDING_HP SD_SLOT(77) // recovery HP banked for the next step head
#define SD_OTIME SD_SLOT(78)      // overtake_target_time, targeted-by-behind timer
#define SD_BFRONT SD_SLOT(79)     // blocked_front_continuetime, f64
#define SD_POSPREV SD_SLOT(80)    // positions as published by the last field_update
// Overtake cooldown, pair-keyed like the engine's map: slot SD_OVC + target
// * 24 + i holds the earliest next rank-gain time for runner i passing target.
#define SD_OVC SD_SLOT(81)        // 24 slots, -1.0 until the pair fires once
#define SD_NOUP SD_SLOT(105)      // overtake_target_no_order_up_time, f64
#define SD_NODOWN SD_SLOT(106)    // overtake_target_no_order_down_time, f64
#define SRL_RUNNER 0
#define SRL_HP SD_STRIDE          // hp columns, SH_STRIDE per runner
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
#define SRL_SKD (SRL_HP_OUT_L + 24 * OOL)          // per-record skill doubles
#define SRL_WORDS (SRL_SKD + SK_RMAX * SKD_STRIDE)

// SectionModifier length.
#define SEC_MOD 25

// ---------------------------------------------------------------------------
// Per-race status output, u32 words at status + id * ST_STRIDE.
#define ST_STRIDE 232
#define ST_FALLBACK 0          // 0 none, 1 unsupported domain (tick in ST_ARG)
#define ST_ARG 1               // tick or argument of the fallback
#define ST_TICKS 2             // loop iterations (frames carry ST_TICKS+1)
#define ST_FINISH0 8           // per runner: finish_step bits then finish_raw
// Compact-reply exports, per runner in roster order, one word each. The last
// three are f32 bit patterns, widened by the host like the engine's f64 casts.
#define ST_TEMPT0 112          // temptation_count (rushingStartCount)
#define ST_RSCORE0 136         // RushingScore count (rushingPenaltyCount)
#define ST_HPEMPTY0 160        // hp_empty_on_race, 0 or 1
#define ST_DELAY0 184          // start_delay f32 bits (startDelayTime)
#define ST_LSS0 208            // SF_SP_START f32 bits (lastSpurtStartDistance)

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
        const float gap = cached[4 * (LIST_STRIDE * r + j)] - position;
        const float lane_gap = fabs(cached[4 * (LIST_STRIDE * r + j) + 1] - lane);
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
        cached[4 * (LIST_STRIDE * r + k)] = convert_float_rte(sd[SD_POS + k]);
        cached[4 * (LIST_STRIDE * r + k) + 1] = convert_float_rte(sd[SD_LANE + k]);
        cached[4 * (LIST_STRIDE * r + k) + 2] = sf0[SF_OBSERVED + k];
        cached[4 * (LIST_STRIDE * r + k) + 3] = convert_float_rte(sd[SD_TSWSK + k]);
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
    const float gap = cached[4 * (LIST_STRIDE * r + k)]
        - convert_float_rte(sd[SD_POS + r]);
    const float speed_gap = owner_speed - cached[4 * (LIST_STRIDE * r + k) + 2];
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
            cached[4 * (LIST_STRIDE * r + k)], cached[4 * (LIST_STRIDE * r + k) + 1],
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
            if (row3 >= 0 && t == 1 && dsign == 1) {
                // Row 3: the crowd-walk inputs for the walk anchored on
                // target 1 (row 3's cached and live lane, row 4's cached
                // and live lane) and the + direction's final edge.
                fr[row3 + 6 * (long)r + 0] =
                    (double)cached[4 * (LIST_STRIDE * r + 3) + 1];
                fr[row3 + 6 * (long)r + 1] =
                    (double)convert_float_rte(sd[SD_LANE + 3]);
                fr[row3 + 6 * (long)r + 2] =
                    (double)cached[4 * (LIST_STRIDE * r + 4) + 1];
                fr[row3 + 6 * (long)r + 3] =
                    (double)convert_float_rte(sd[SD_LANE + 4]);
                fr[row3 + 6 * (long)r + 4] = (double)(candidate);
                fr[row3 + 6 * (long)r + 5] = (double)(edge);
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

// TargetState::target_sorted. fixed_goal stays 0 and lane_preference 0
// because no skills are active, and has_hp tracks the live HP column.
static float lane_target(__global uint* si, __global float* sf, __global uint* lists,
    __global float* cached, __global const double* sd, __global const float* sf0,
    __global const double* pd, __global uint* bank, const int r, const int n,
    const float position, const float lane, const float owner_speed,
    const float owner_target, const int phase, const int has_hp, const int pace_down,
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
            position, phase, max_lane, side_lane, can_in, can_out, in_space, out_space, 0,
            dbg_base, fr);
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

static void duel_activate(__global uint* si, const int x) {
    si[SI_DUEL_ACTIVE + x] = 1u;
    si[SI_DUEL_FIGHT + x] += 1u;
    si[SI_DUEL_TOKEN + x] = 1u;
}

// update_dueling, the retail path. duel_low_hp uses the plain f64 ratio,
// eligible_straight uses the preupdate snapshots with the corner state, and
// the duel draws no RNG.
static void duel_update(__global uint* si, __global const float* sf,
    __global const double* sd, __global double* shl, const int r, const int n,
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
    duel_activate(si, r);
    if (!(int)si[SI_DUEL_ACTIVE + j] && (int)si[SI_DUEL_FIGHT + j] == 0) {
        duel_activate(si, j);
    }
}

// latch_active_spot_cutoff plus update_spot_struggle, the Nige/Oonige state
// machine. Timers were advanced in prepare, the token bit drives the target
// chain, and no part of it draws RNG.
static void spot_update(__global uint* scr, __global uint* si, __global double* sd,
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
    for (int j = 0; j < n; j++) {
        if (!started[j]) {
            continue;
        }
        const uint base = (uint)(PD_RUN0 + j * PD_RUN_STRIDE);
        si[SI_SPOT_ACTIVE + j] = 1u;
        si[SI_SPOT_STARTED + j] = 1u;
        si[SI_SPOT_TOKEN + j] = 1u;
        sd[SD_SPOT_TIMER + j] = -pd[base + PRD_SPOT_D];
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
static void asitame_release(__global uint* si, __global float* sf,
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

// The effect word a token was registered for, the (record, slot) it packs.
static __global const uint* sk_tok_effect(__global const uint* sk, const uint tok) {
    const int rec = (int)SK_TOK_REC(tok);
    const int slot = (int)SK_TOK_SLOT(tok);
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    return sk + SK_EFFECTS + SK_ESTRIDE * ((int)rd[6] + slot);
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

// The f64 modifier a token carries, straight from the table.
static double sk_tok_modifier(__global const uint* sk, const uint tok) {
    const __global const uint* ef = sk_tok_effect(sk, tok);
    return as_double((ulong)ef[6] | ((ulong)ef[7] << 32));
}

// The registered value of a record token, f32-rounded the way the engine
// registers modifier as f32. A token whose effect is not this kind (the
// natural and ordinary current-speed groups share one order) reads as zero,
// the same filter the old record scan gave.
static float sk_tok_value(__global const uint* sk, const uint tok, const int kind) {
    const __global const uint* ef = sk_tok_effect(sk, tok);
    if ((int)ef[0] != kind) {
        return 0.f;
    }
    return convert_float_rte(as_double((ulong)ef[6] | ((ulong)ef[7] << 32)));
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
        acc += sk_tok_modifier(sk, tok);
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
    case FEAT_RTEMP: {
        // running_style_temptation_count: the base-style group's kakari rows,
        // the querying runner counted through the caller's own kakari flag.
        const int group = (int)arg;
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (j == r) continue;
            if (field_style((int)si[SI_STRATEGY + j]) == group
                && (int)si[SI_KAKARI + j] != 0) {
                c++;
            }
        }
        if (field_style((int)si[SI_STRATEGY + r]) == group
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
            const int rec = (int)SK_TOK_REC(tok);
            if (skd[SKD_STRIDE * rec + SKD_EFF_T + (int)SK_TOK_SLOT(tok)] < 0.0) {
                continue;
            }
            const __global const uint* ef = sk_tok_effect(sk, tok);
            if ((int)ef[0] == SK_EFF_CURRENT_NAT) {
                // expire_skill_effects hands a natural current speed to
                // modifiers_one_frame_accel, a one-tick speed offset.
                sd[SD_OFA + r] += sk_tok_modifier(sk, tok);
            }
            sk_seq_remove(q, tok);
        }
    }
}

// start_expired_skill_cooldowns for one whole field, after the per-runner
// prepare loop. Dead records keep stepping because the engine's timer map is
// keyed by the firing, not by the pending record's life.
static void skills_advance(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global double* sd, const float dt,
    const int n, const float dist_f) {
    const int nrec = (int)sk[SK_NREC];
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
        // start_expired_skill_cooldowns: the waiting cooldown starts once the
        // (skill, detail) instance duration ran out.
        if (state == (uint)SK_S_WAITING && wd[SKD_INSTANCE] >= 0.0) {
            const double cd = as_double((ulong)rd[13] | ((ulong)rd[14] << 32));
            wd[SKD_COOLDOWN] = -(double)(convert_float_rte(cd) * (dist_f / 1000.0f));
            sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_COOLDOWN;
        }
    }
    skills_expire_current(sk, scr, skd, sd, n);
}

// skill_targets for the three rank-window codes. The ahead and behind codes
// slice the published order outward from self, the visible code (4) also
// demands a spot in this tick's lane cache. Same-team drops are compiler
// gated (all-zero teams), and debuff immunity needs the refused kind 38.
static int sk_targets(__global const uint* si, __global const uint* rs,
    __global const uint* scr, const int r, const int target, const int limit,
    const int n, int* outs) {
    const int rank = (int)si[SI_RANKCOL + r];
    const int vis_base = (int)(SCR_LISTS + (uint)r * LIST_STRIDE + RACE_MAXR);
    const int vis_len = (int)si[SI_L_VISN + r];
    int cnt = 0;
    for (int k = 0; k < n && cnt < limit; k++) {
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

// The fire bookkeeping of fire_pending_skill and apply_activation for one
// record: the duration-timer kinds register a receiver token, recovery and
// the raw stat bump write state directly, the largest scaled duration becomes
// the (skill, detail) instance timer, and the phase counter bumps. cd <= 0
// marks the record dead forever (engine: cooldown map entry +INF and the
// pending record removed).
static void skills_fire(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si,
    __global const double* shl, __global double* sd, __global const double* pd,
    __global float* sf, __global const uint* rs, const int rec, const int r,
    const double hp, const double gap, const float dist_f, const int init_pass,
    const int n, const double distance, __global uint* ev, __global uint* st) {
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    // The engine's activation hook emits one skillEvents entry per Own-side
    // fire (an immediate one, or a pending entry the deactivate hook or the
    // race-end finalizer writes once). rd[0] is the wire skill id.
    race_event(ev, st, r, rd[0]);
    __global double* wd = skd + SKD_STRIDE * rec;
    const int eff_base = (int)rd[6];
    const int n_eff = (int)rd[5];
    double runtime = 0.0;
    for (int e = 0; e < n_eff; e++) {
        const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * (eff_base + e);
        const int kind = (int)ef[0];
        const double modifier = as_double((ulong)ef[6] | ((ulong)ef[7] << 32));
        const double base = as_double((ulong)ef[8] | ((ulong)ef[9] << 32));
        const double coeff = sk_coeff(ef[3], hp, gap);
        const float m1 = convert_float_rte(base) * (dist_f / 1000.0f);
        const float m2 = m1 * convert_float_rte(coeff);
        const double scaled = (double)(m2 * 1.0f);
        runtime = fmax(runtime, scaled);
        const int target = (int)ef[1];
        if (target != 1) {
            // The constructor pass has no field to deliver to, while the
            // effect's duration still feeds the instance timer above.
            if (init_pass == 0) {
                if (sk_queued_kind(kind)) {
                    const double dur = convert_float_rte(scaled);
                    wd[SKD_EFF_T + e] = dur < 0.0 ? -INFINITY : -dur;
                }
                int outs[RACE_MAXR];
                const int cnt = sk_targets(si, rs, scr, r, target, (int)ef[4], n, outs);
                const uint tok = SK_TOK_EFF((uint)rec, (uint)e);
                for (int q = 0; q < cnt; q++) {
                    const int j = outs[q];
                    if (kind == SK_EFF_RECOVERY) {
                        // Delivered perspective is never Self: HP lands from
                        // the receiver's max, the heal count stays with the
                        // caster.
                        sd[SD_PENDING_HP + j] += (double)(convert_float_rte(
                            shl[j * SH_STRIDE + SH_MAX]) * convert_float_rte(modifier));
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
                pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_POWF]));
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
    wd[SKD_INSTANCE] = runtime > 0.0 ? -runtime : 0.0;
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
}

// process_skill_priority for one priority pass: the whole field in roster
// order, records in wire order, with the compact scan gate. The engine
// removal for pos past the trigger only fires for non-retained primaries,
// and the reserved and pending-removal paths belong to rejected constructs.
static void skills_scan(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* scr, __global uint* si,
    __global double* sd, __global const double* shl, __global const uint* rs,
    __global const double* pd, __global float* sf, const double distance,
    const int p, const int n, const float dist_f, const int init_pass,
    __global uint* ev, __global uint* st) {
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
        for (int rec = 0; rec < nrec; rec++) {
            __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            if ((int)rd[1] != r || (int)rd[3] != p) {
                continue;
            }
            uint state = sks[SKR_STRIDE * rec + SKR_STATE];
            if (state == (uint)SK_S_DEAD || state == (uint)SK_S_DISABLED) {
                continue;
            }
            __global double* wd = skd + SKD_STRIDE * rec;
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
            if (pos >= wd[SKD_TRIG1]) {
                if (!(rd[2] & SK_FLAG_RETAIN)) {
                    sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DEAD;
                }
                continue;
            }
            if (state == (uint)SK_S_COOLDOWN && wd[SKD_COOLDOWN] >= 0.0) {
                state = (uint)SK_S_READY;
                sks[SKR_STRIDE * rec + SKR_STATE] = state;
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
                distance, ev, st);
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
            acc = acc + sk_tok_value(sk, tok, SK_EFF_CURRENT);
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
        acc = acc + sk_tok_value(sk, tok, SK_EFF_CURRENT);
    }
    sd[SD_TSWSK + r] = (double)acc;
}

// pace_cancel_current_speed() > 0. || retail_target_modifiers.apply(0.) > 0.
// The fixed tokens fold at their pd values while their flags stay set; the
// engine's receiver dropped an expired token before this context is built
// mid-race_step, which is the same instant the flag clears.
static int sk_has_speed(__global const uint* sk, __global const uint* scr,
    __global const uint* si, const int r, const __global double* pd) {
    float ordinary = 0.f;
    const __global uint* cq = scr + SK_CSEQ + r * SK_SEQN;
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = cq[i];
        if (tok == 0u) {
            break;
        }
        ordinary = ordinary + sk_tok_value(sk, tok, SK_EFF_CURRENT);
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
                target = target + convert_float_rte(
                    pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_SPOT_M]);
            }
        } else if (tok == SK_TOK_DUEL) {
            if ((int)si[SI_DUEL_TOKEN + r]) {
                target = target + convert_float_rte(
                    pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_DUEL_T]);
            }
        } else {
            target = target + sk_tok_value(sk, tok, SK_EFF_TARGET);
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
        out[m++] = sk_tok_value(sk, tok, kind);
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
        natural = natural + sk_tok_value(sk, tok, SK_EFF_CURRENT_NAT);
    }
    for (int i = 0; i < SK_SEQN; i++) {
        const uint tok = q[i];
        if (tok == 0u) {
            break;
        }
        ordinary = ordinary + sk_tok_value(sk, tok, SK_EFF_CURRENT);
    }
    return natural + ordinary;
}