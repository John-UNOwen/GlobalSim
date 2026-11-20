// Classic (non-retail) physics core. The float arithmetic mirrors the engine
// line for line (race_solver.rs, race_field.rs, hp_policy.rs, lane_selection.rs,
// overtaking.rs, starting_grid.rs cited inline). Construction (stats, rng
// seeding, section and duel constants) is host side in classic_exec.rs, so this
// kernel runs no powf or log10.

#define CLS_MAXR 120
#define CLS_MAX_TICKS 6144
#define CLS_FRAME_COLS 14
#define CLS_MAX_SLOPES 16
#define CLS_DT (1.0 / 15.0)
#define CLS_MAX_RACE_SECONDS 400.0
#define CLS_HORSE_LANE (1.0 / 18.0)
#define CLS_LANE (1.0 / 18.0)

// Input words, one block per race.
#define CLS_RI_SEED 0
#define CLS_RI_N 1
#define CLS_RI_NSLOPES 2
#define CLS_RI_TRACK 3
#define CLS_RI_SURFACE 4
#define CLS_RI_GROUND 5
#define CLS_RI_FLAGS 6
#define CLS_RI_DUELON 7
#define CLS_RI_NMAX 8
#define CLS_RI_STRIDE (16 + 48 * CLS_MAXR)
#define CLS_RI_RUN0 16
#define CLS_RI_RUN_STRIDE 48
#define R_STRATEGY 0
#define R_DISTAPT 1
#define R_SURFAPT 2
#define R_STAPT 3
#define R_SOLVER_HI 6
#define R_SOLVER_LO 7
#define R_HP_HI 8
#define R_HP_LO 9
#define R_PACE_HI 10
#define R_PACE_LO 11
#define R_HILL0 12
#define R_SLOPE_IDX 44
#define R_INIT_DOWNHILL 46
#define R_CAPFLAG 47

// Input doubles, one block per race. All constructor values, engine-rounded.
#define CLS_RD_RUN0 16
#define CLS_RD_RUN_STRIDE 64
#define D_DISTANCE 0
#define D_MAXLANE 1
#define D_FIRSTMOVE 2
#define D_SECTION 3
#define D_LASTSTR_START 4
#define D_LASTSTR_END 5
#define D_LASTCORNER_START 6
#define D_BASESPEED 7
#define D_GROUNDMOD 9
#define CLS_RD_SLOPES (16 + 64 * CLS_MAXR)
#define CLS_RD_STRIDE (CLS_RD_SLOPES + 48)
#define F_LANE0 0
#define F_SPEED_STAT 1
#define F_STAMINA 2
#define F_POWER 3
#define F_GUTS 4
#define F_WISDOM 5
#define F_DELAY 6
#define F_SECT0 7
#define F_KSTART 32
#define F_KDUR 33
#define F_BTS0 34
#define F_LSS 37
#define F_MINSPEED 38
#define F_BACC0 39
#define F_MAXHP 45
#define F_GUTSMOD 46
#define F_PKMIN 47
#define F_PKMAX 48
#define F_PKEND 49
#define F_PKCONT 50
#define F_PCT15 51
#define F_PCT20 52
#define F_DTS 53
#define F_DA 54
#define F_SPOTM 55
#define F_SPOTD 56
#define F_SUBPAR 57
#define F_SLOPE0 58
#define F_STYLECNT 59
#define F_NIGECNT 60

// Status words, one block per race: fallback, ticks, loop end, per runner
// finish times as raw f64 bits.
#define CLS_ST_STRIDE (16 + 17 * CLS_MAXR)
#define S_FALLBACK 0
#define S_TICKS 1
#define S_END 2
#define CLS_ST_RUN0 16
#define CLS_ST_RUN_STRIDE 17
#define S_FIN0 16
// Full-reply exports: last-spurt transition, the actual last-spurt start
// pair (race_solver.rs:1652-1653), temptation and rushing-score counts
// (lib.rs:1000-1001) and the first last-spurt capture tick (lib.rs:920).
#define S_LSTTRANS 6
#define S_SPURTT 8
#define S_SPURTD 10
#define S_TEMPT 12
#define S_RSCNT 13
#define S_SPURTTICK 14

// Reply record streams, one region per race: competition starts as
// (tick, kind, runner) triples in engine push order (lib.rs:899-918) and
// positioning phase records as (runner, phase) pairs (lib.rs:886).
#define CLS_RP_COMPLEN 0
#define CLS_RP_PHRCOUNT 1
#define CLS_RP_SKLEN 2
#define CLS_RP_COMP0 8
#define CLS_RP_COMP_CAP 2400
#define CLS_RP_PHR0 (CLS_RP_COMP0 + CLS_RP_COMP_CAP)
#define CLS_RP_PHR_CAP 800
#define CLS_RP_SK0 (CLS_RP_PHR0 + CLS_RP_PHR_CAP)
// One kakari record per runner at most (latched temptation_count), five
// words each: runner plus the start and duration double pair.
#define CLS_RP_SK_CAP 640
#define CLS_RP_STRIDE (CLS_RP_SK0 + CLS_RP_SK_CAP)

// Scratch words per race. Field tables first, then per-runner state, then the
// next_overtake_times bitmap.
#define U_PMVALID 0
#define U_PMIDX 1
#define U_PPRES 2
#define U_CAP 3
#define U_EIM 4
#define U_DUELON 5
#define U_RANK 16
#define U_PREVRANK 136
#define U_ORDER 256
#define U_STYLEA 376
#define U_SU_RUN0 512
#define U_SU_RUN_STRIDE 72
#define UR_STRATEGY 0
#define UR_PHASE 1
#define UR_RUSH 2
#define UR_PKMODE 3
#define UR_DTARGET 4
#define UR_DGROUP 5
#define UR_FIGHT 6
#define UR_MOVEDIR 7
#define UR_EXTRA 8
#define UR_MLCT 9
#define UR_SLOPEIDX 10
#define UR_FIN 11
#define UR_KAKARI 12
#define UR_PACEDOWN 13
#define UR_DOWNHILL 14
#define UR_SPURT 15
#define UR_DASH 16
#define UR_MOVELANE 17
#define UR_DUELACT 18
#define UR_SPOTACT 19
#define UR_SPOTSTART 20
#define UR_ENTCLOSED 21
#define UR_DISTEXIT 22
#define UR_CUTOFF 23
#define UR_PKRUSH 24
#define UR_SOLVER_HI 25
#define UR_SOLVER_LO 26
#define UR_HP_HI 27
#define UR_HP_LO 28
#define UR_PACE_HI 29
#define UR_PACE_LO 30
#define UR_HILL0 31
#define UR_STARTCNT 63
#define UR_ENDCNT 64
#define UR_TEMPT 65
// Full-reply mirrors: the stadium-score countdown (stadium_score.rs:51-67),
// its count, the positioning previous phase (stadium_score.rs:33-47), and
// the per-tick spot and duel event baselines (lib.rs:846-847, 899-918).
#define UR_RSREM 66
#define UR_RSCNT 67
#define UR_PREVPH 68
#define UR_PREVFIGHT 69
#define UR_PREVSPOT 70
#define U_TGTMAP 9168
#define U_BLOCK 23568
#define U_TLIST 23696
#define U_OMAP 23824
// Ranks as cached at the loop-pass start (lib.rs:879-880) and the per-tick
// style-group leader scan (stadium_score.rs:20-30).
#define U_TICKRANK 23952
#define U_GLEADER 24080
#define CLS_SU_STRIDE 24400
// Pos-keep mode codes, PosKeepMode in race_solver.rs.
#define PK_NONE 0
#define PK_PACEDOWN 1
#define PK_PACEUP 2
#define PK_PACEUPEX 3
#define PK_SPEEDUP 4
#define PK_OVERTAKE 5

// Scratch doubles per race.
#define UDB_PREVPOS 16
#define UDB_OVERT 152
#define UDB_COOL 512
#define UDB_DPARK 14912
#define UDB_PROXT 29312
#define UDB_OTH0 29440
#define UDB_OTHPOS UDB_OTH0
#define UDB_OTHLANE (UDB_OTH0 + CLS_MAXR)
#define UDB_OTHSPEED (UDB_OTH0 + 2 * CLS_MAXR)
#define UDB_OTHTGT (UDB_OTH0 + 3 * CLS_MAXR)
#define UDB_CAND 29920
#define UDB_CKEY 30240
#define UDB_SPURTC 30560
#define UDB_SPURTV 31072
#define UDB_SPURTN 31584
#define CLS_SD_RUN0 41216
#define CLS_SD_RUN_STRIDE 112
#define CLS_SD_STRIDE 54784
#define DF_POS 0
#define DF_SPEED 1
#define DF_ACCEL 2
#define DF_TARGET 3
#define DF_LANE 4
#define DF_LANETARGET 5
#define DF_HP 6
#define DF_MAXHP 7
#define DF_ACCUM 8
#define DF_DOWNHT 9
#define DF_KTIMER 10
#define DF_SPOTT 11
#define DF_DELAY 12
#define DF_KSTART 13
#define DF_KDUR 14
#define DF_TSA 15
#define DF_TSE 16
#define DF_CSA 17
#define DF_CSE 18
#define DF_ACA 19
#define DF_ACE 20
#define DF_ONEFRAME 21
#define DF_LSS 22
#define DF_LSTTRANS 23
#define DF_MINSPEED 24
#define DF_BTS0 25
#define DF_BACC0 28
#define DF_SECT0 34
#define DF_PKINT 59
#define DF_PKCOOL 60
#define DF_PKEFFEND 61
#define DF_PKEFFEXIT 62
#define DF_PKMIN 63
#define DF_PKMAX 64
#define DF_PKEND 65
#define DF_PKCONT 66
#define DF_PKCOEF 67
#define DF_LCSPEED 68
#define DF_EXTRA 69
#define DF_BFRONT 70
#define DF_BSIDE 71
#define DF_BALL 72
#define DF_NEARA 73
#define DF_NEARB 74
#define DF_SET1 75
#define DF_DCONT 76
#define DF_WISDOM 77
#define DF_POWER 78
#define DF_PCT15 79
#define DF_PCT20 80
#define DF_SUBPAR 81
#define DF_SLOPEPER 82
#define DF_GROUNDMOD 83
#define DF_GUTSMOD 84
#define DF_FINRAW 85
#define DF_FINSTEP 86
#define DF_DTS 87
#define DF_DA 88
#define DF_SPOTM 89
#define DF_SPOTD 90
#define DF_BASESPEED 91
#define DF_SECTION 92
#define DF_PREVPOS 93
#define DF_EFF 94
#define DF_STYLECNT 95
#define DF_NEXTPHASE 96
#define DF_ENDCNT 97
#define DF_NIGECNT 98
// Captured at the last-spurt flip (race_solver.rs:1651-1653): the runner
// clock, the position, and the capture tick (lib.rs:920-922).
#define DF_SPURTT 99
#define DF_SPURTD 100
#define DF_SPURTTICK 101
// The runner clock at kakari activation, measured for the kakari
// skillEvents entry the deactivate hook writes (lib.rs:653, 690-701).
#define DF_KSTARTTIME 102

typedef struct cc_ctx {
    __global unsigned int* su;
    __global double* sd;
    __global const double* cd;
    __global unsigned int* rp;
    int n;
    int nhills;
    int rp_over;
} Cc;

#define SR(f) (c->sd[CLS_SD_RUN0 + (r) * CLS_SD_RUN_STRIDE + (f)])
#define SO(f, j) (c->sd[CLS_SD_RUN0 + (j) * CLS_SD_RUN_STRIDE + (f)])
#define SU(f) (c->su[U_SU_RUN0 + (r) * U_SU_RUN_STRIDE + (f)])
#define SUO(f, j) (c->su[U_SU_RUN0 + (j) * U_SU_RUN_STRIDE + (f)])

static void cc_update_spot(Cc* c, const int r);
static void cc_update_dueling(Cc* c, const int r, const double dt);
static void cc_emit_kakari(Cc* c, const unsigned int r,
    const double start_t, const double end_t);

// KBN compensated add, CompensatedAccumulator::add (race_solver.rs:184-208).
static void cc_add(__global double* acc, __global double* err, const double v) {
    const double t = *acc + v;
    if (fabs(*acc) >= fabs(v)) {
        *err += (*acc - t) + v;
    } else {
        *err += (v - t) + *acc;
    }
    *acc = t;
}

// Engine rng consumers on one stream stored in two scratch words. The rng
// helpers take private pointers, so the state round-trips through locals.
static double cc_random(__global unsigned int* hi, __global unsigned int* lo) {
    uint h = *hi;
    uint l = *lo;
    const double v = r30_random(&h, &l);
    *hi = h;
    *lo = l;
    return v;
}

static int cc_uniform_cap(Cc* c, __global unsigned int* hi,
    __global unsigned int* lo, const double upper) {
    uint h = *hi;
    uint l = *lo;
    int found = 1;
    const int v = r30_uniform(&h, &l, upper, &found);
    *hi = h;
    *lo = l;
    if (!found) {
        c->su[U_CAP] = 1u;
    }
    return v;
}

static int cc_strategy(Cc* c, const int j) {
    return (int)c->su[U_SU_RUN0 + j * U_SU_RUN_STRIDE + UR_STRATEGY];
}

// strategy_matches (horse_types.rs:101-105): Nige and Oonige match each other.
static int cc_style_matches(const int s1, const int s2) {
    return s1 == s2 || (s1 == 1 && s2 == 5) || (s1 == 5 && s2 == 1);
}

// base_style (race_field.rs:65) and select_mode / pacing_style (rushing.rs).
static int cc_base_style(const int s) {
    return s == 5 ? 1 : s;
}

static int cc_select_mode(const int strategy, const int roll) {
    if (strategy == 1 || strategy == 5) {
        return 4;
    }
    if (strategy == 2) {
        return 3;
    }
    if (strategy == 3) {
        return roll < 25 ? 2 : 3;
    }
    return roll < 10 ? 1 : (roll < 30 ? 2 : 3);
}

static int cc_pacing_style(const int original, const int mode) {
    switch (mode) {
        case 1: return 3;
        case 2: return 2;
        case 3: case 4: return 1;
        default: return original;
    }
}

static int cc_rank(Cc* c, const int j) {
    return (int)c->su[U_RANK + j];
}

static int cc_prev_rank(Cc* c, const int j) {
    return (int)c->su[U_PREVRANK + j];
}

// front_blocker classic branch (race_field.rs:590-610), unfinished others,
// 0 < gap < 2 and a lane tolerance shrinking with the gap, min gap wins.
static int cc_front_blocker(Cc* c, const int r, const double pos,
    const double lane, double* gap_out) {
    const double dist = c->cd[D_DISTANCE];
    int best = -1;
    double best_gap = 0.0;
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        const double opos = SO(DF_POS, j);
        if (opos >= dist) {
            continue;
        }
        const double gap = opos - pos;
        if (!(gap > 0.0 && gap < 2.0)) {
            continue;
        }
        const double lg = fabs(SO(DF_LANE, j) - lane);
        if (lg > (1.0 - 0.6 * gap / 2.0) * 0.75 * CLS_HORSE_LANE) {
            continue;
        }
        if (best < 0 || gap < best_gap) {
            best = j;
            best_gap = gap;
        }
    }
    *gap_out = best_gap;
    return best;
}

// side_room classic branch (race_field.rs:632-661).
static void cc_side_room(Cc* c, const int r, const double pos,
    const double lane, int* inner, int* outer) {
    const double dist = c->cd[D_DISTANCE];
    const double tol = 2.0 * CLS_HORSE_LANE;
    int in_b = 0;
    int out_b = 0;
    *inner = 0;
    *outer = 0;
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        const double opos = SO(DF_POS, j);
        if (opos >= dist) {
            continue;
        }
        const double d = opos - pos;
        if (!(-1.05 < d && d < 1.05)) {
            continue;
        }
        const double lg = SO(DF_LANE, j) - lane;
        if (!(-tol < lg && lg < tol)) {
            continue;
        }
        if (lg < 0.0) {
            in_b = 1;
            if (out_b) {
                break;
            }
        } else if (lg > 0.0) {
            out_b = 1;
            if (in_b) {
                break;
            }
        }
    }
    *inner = in_b;
    *outer = out_b;
}

static void cc_ranked_neighbors(Cc* c, const int r, int* ahead, int* behind) {
    const int rank = cc_rank(c, r);
    *ahead = rank > 1 ? (int)c->su[U_ORDER + rank - 2] : -1;
    *behind = rank < c->n ? (int)c->su[U_ORDER + rank] : -1;
}

// distance_to_behind (race_field.rs:541-548), -1.0 means None.
static double cc_distance_to_behind(Cc* c, const int r) {
    const int rank = cc_rank(c, r);
    if (rank >= c->n) {
        return -1.0;
    }
    const int b = (int)c->su[U_ORDER + rank];
    return fmax(SO(DF_POS, r) - SO(DF_POS, b), 0.0);
}

// leading_eligible (race_field.rs:854-868) over the group table, cached per
// update like pacemaker_valid (race_field.rs:736-747).
static int cc_pacemaker(Cc* c) {
    if (c->su[U_PMVALID]) {
        return (int)c->su[U_PMIDX] - 1;
    }
    int result = -1;
    for (int g = 0; g < 4 && result < 0; g++) {
        for (int k = 0; k < c->n && result < 0; k++) {
            const int j = (int)c->su[U_ORDER + k];
            const int s = cc_strategy(c, j);
            const int in_group = (g == 0 && (s == 1 || s == 5)) ||
                (g == 1 && s == 2) || (g == 2 && s == 3) || (g == 3 && s == 4);
            if (in_group) {
                result = j;
            }
        }
    }
    c->su[U_PMVALID] = 1u;
    c->su[U_PMIDX] = (unsigned int)(result + 1);
    return result;
}

static double cc_gap_to_pacemaker(Cc* c, const int r) {
    const int pm = cc_pacemaker(c);
    if (pm < 0 || pm == r) {
        return 0.0;
    }
    return SO(DF_POS, pm) - SO(DF_POS, r);
}

static int cc_later_style_ahead(Cc* c, const int r) {
    const int sa = (int)c->su[U_STYLEA + r];
    return sa > cc_base_style(cc_strategy(c, r));
}

static int cc_pacemaker_is_later_style_ahead(Cc* c, const int r) {
    const int pm = cc_pacemaker(c);
    if (pm < 0) {
        return 0;
    }
    return cc_rank(c, pm) < cc_rank(c, r) &&
        cc_base_style(cc_strategy(c, pm)) > cc_base_style(cc_strategy(c, r));
}

// visible (lane_selection.rs:6-9).
static int cc_visible(const double pos, const double lane, const double vision,
    const double p, const double l) {
    const double gap = p - pos;
    return vision > 0.0 && gap >= 0.0 && gap <= vision &&
        fabs(l - lane) <= (gap / vision * 11.5 + 2.0) * CLS_LANE / 2.0;
}

// is_target_with_block_check (overtaking.rs:10-24), f32 casts as written.
static int cc_is_target(const double gap, const double speed_gap,
    const double owner_target, const double other_target,
    const double other_speed, const int directly_blocking,
    const int other_blocked) {
    const float g = convert_float_rte(gap);
    const float sg = convert_float_rte(speed_gap);
    if (g < -1.0f) {
        return 0;
    }
    if (directly_blocking) {
        return convert_float_rte(other_speed) < convert_float_rte(owner_target);
    }
    if (!(g >= 1.0f && g <= 20.0f) || sg <= 0.0f || g / sg > 15.0f) {
        return 0;
    }
    const int blocked_result =
        convert_float_rte(other_speed) < convert_float_rte(owner_target);
    const int unblocked_result =
        convert_float_rte(other_target) < convert_float_rte(owner_target);
    if (blocked_result == unblocked_result) {
        return blocked_result;
    }
    return other_blocked ? blocked_result : unblocked_result;
}

// Build the others row set for runner r (update_lane: unfinished others in
// index order). Returns the row count and fills the map.
static int cc_build_others(Cc* c, const int r) {
    const double dist = c->cd[D_DISTANCE];
    int m = 0;
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        if (SO(DF_POS, j) >= dist) {
            continue;
        }
        c->sd[UDB_OTHPOS + m] = SO(DF_POS, j);
        c->sd[UDB_OTHLANE + m] = SO(DF_LANE, j);
        c->sd[UDB_OTHSPEED + m] = SO(DF_SPEED, j);
        c->sd[UDB_OTHTGT + m] = SO(DF_TARGET, j);
        c->su[U_OMAP + m] = (unsigned int)j;
        m++;
    }
    return m;
}

// blocked predicate of collect_targets (lane_selection.rs:13-17), rows are
// others plus the owner row.
static int cc_blocked_by(Cc* c, const int m, const double pos, const double lane,
    const double other_pos, const double other_lane) {
    for (int k = 0; k < m; k++) {
        const double gap = c->sd[UDB_OTHPOS + k] - other_pos;
        if (gap > 0.0 && gap < 2.0 &&
            fabs(c->sd[UDB_OTHLANE + k] - other_lane) <=
                (1.0 - 0.3 * gap) * 0.75 * CLS_LANE) {
            return 1;
        }
    }
    const double gap = pos - other_pos;
    return gap > 0.0 && gap < 2.0 &&
        fabs(lane - other_lane) <= (1.0 - 0.3 * gap) * 0.75 * CLS_LANE;
}

// collect_targets (lane_selection.rs:11-32). Writes others-row indices into
// U_TLIST and returns the count.
static int cc_collect_targets(Cc* c, const int r, const int m) {
    const double pos = SR(DF_POS);
    const double lane = SR(DF_LANE);
    const double speed = SR(DF_SPEED);
    const double target = SR(DF_TARGET);
    const double vision = 20.0;
    int nearest = -1;
    double best_gap = 0.0;
    for (int i = 0; i < m; i++) {
        const double gap = c->sd[UDB_OTHPOS + i] - pos;
        if (gap > 0.0 && gap < 2.0 &&
            fabs(c->sd[UDB_OTHLANE + i] - lane) <=
                (1.0 - 0.3 * gap) * 0.75 * CLS_LANE) {
            if (nearest < 0 || gap < best_gap) {
                nearest = i;
                best_gap = gap;
            }
        }
    }
    int t = 0;
    for (int i = 0; i < m; i++) {
        const double p = c->sd[UDB_OTHPOS + i];
        const double l = c->sd[UDB_OTHLANE + i];
        const double v = c->sd[UDB_OTHSPEED + i];
        const double w = c->sd[UDB_OTHTGT + i];
        const double gap = p - pos;
        const int is_t = nearest == i ||
            (cc_visible(pos, lane, vision, p, l) &&
             cc_is_target(gap, speed - v, target, w, v, 0,
                 cc_blocked_by(c, m, pos, lane, p, l)));
        if (is_t) {
            c->su[U_TLIST + t] = (unsigned int)i;
            t++;
        }
    }
    return t;
}

// clear predicate of passing_lane (lane_selection.rs:37-47).
static int cc_pass_clear(Cc* c, const int m, const double pos, const double lane,
    const double vision, const double candidate, const double end) {
    if (candidate < 0.0 || candidate > c->cd[D_MAXLANE]) {
        return 0;
    }
    for (int k = 0; k < m; k++) {
        const double p = c->sd[UDB_OTHPOS + k];
        const double l = c->sd[UDB_OTHLANE + k];
        if (cc_visible(pos, lane, vision, p, l) && pos <= p && p <= end &&
            fabs(l - candidate) <= 0.8 * CLS_LANE) {
            return 0;
        }
        if (fabs(p - pos) < 1.05) {
            if (l > lane && candidate > lane && candidate > l - 2.0 * CLS_LANE) {
                return 0;
            }
            if (l < lane && candidate < lane && candidate < l + 2.0 * CLS_LANE) {
                return 0;
            }
        }
    }
    return 1;
}

// passing_lane (lane_selection.rs:34-79). Writes *out; returns has-some.
static int cc_passing_lane(Cc* c, const int r, const int m, const int nt,
    double* out) {
    if (nt == 0) {
        return 0;
    }
    const double pos = SR(DF_POS);
    const double lane = SR(DF_LANE);
    const int phase = (int)SU(UR_PHASE);
    const double vision = 20.0;
    const double max_lane = c->cd[D_MAXLANE];
    int ncand = 0;
    for (int ti = 0; ti < nt; ti++) {
        const int index = (int)c->su[U_TLIST + ti];
        for (int d = 0; d < 2; d++) {
            const double direction = d == 0 ? -1.0 : 1.0;
            int edge = index;
            for (;;) {
                const double p = c->sd[UDB_OTHPOS + edge];
                const double l = c->sd[UDB_OTHLANE + edge];
                int next = edge;
                for (int j = 0; j < m; j++) {
                    const double q = c->sd[UDB_OTHPOS + j];
                    const double mm = c->sd[UDB_OTHLANE + j];
                    if (q - p >= 0.0 && q - p <= 3.0 &&
                        direction * (mm - l) > 0.0 &&
                        direction * (mm - l) < 2.0 * CLS_LANE &&
                        direction * mm > direction * c->sd[UDB_OTHLANE + next]) {
                        next = j;
                    }
                }
                if (next == edge) {
                    break;
                }
                edge = next;
            }
            const double candidate = c->sd[UDB_OTHLANE + edge] + direction * CLS_LANE;
            if (ncand < 256 &&
                cc_pass_clear(c, m, pos, lane, vision, candidate,
                    c->sd[UDB_OTHPOS + edge] + 0.5)) {
                c->sd[UDB_CAND + ncand] = candidate;
                ncand++;
            }
        }
    }
    int can_move_in = lane > 0.0001;
    if (can_move_in) {
        for (int k = 0; k < m; k++) {
            const double dl = lane - c->sd[UDB_OTHLANE + k];
            if (fabs(c->sd[UDB_OTHPOS + k] - pos) < 1.05 && dl > 0.0 &&
                dl < 2.0 * CLS_LANE) {
                can_move_in = 0;
                break;
            }
        }
    }
    const double straight =
        (phase < 2 && can_move_in && lane > CLS_LANE) ? lane - CLS_LANE : lane;
    double end = 1e300;
    for (int ti = 0; ti < nt; ti++) {
        const double p = c->sd[UDB_OTHPOS + (int)c->su[U_TLIST + ti]];
        end = fmin(end, p);
    }
    end += 3.0;
    if (ncand < 256 && cc_pass_clear(c, m, pos, lane, vision, straight, end)) {
        c->sd[UDB_CAND + ncand] = straight;
        ncand++;
    }
    double best = lane;
    double best_score = 1e300;
    for (int k = 0; k < ncand; k++) {
        const double candidate = c->sd[UDB_CAND + k];
        double weight = 1.0;
        if (candidate < lane) {
            weight = phase == 0 ? 100.0 : (phase >= 2 ? 1.149999976158142 : 1.0);
        }
        const double gap = lane - candidate;
        const double sc = fabs(gap * weight);
        if (sc < best_score) {
            best_score = sc;
            best = candidate;
        }
    }
    *out = best;
    return 1;
}
// update_blocking (race_solver.rs:1246-1261), classic branch: the blocker
// speed cap applies to classic only.
static void cc_update_blocking(Cc* c, const int r, const double dt) {
    double gap = 0.0;
    const int b = cc_front_blocker(c, r, SR(DF_POS), SR(DF_LANE), &gap);
    if (b >= 0) {
        const double cap = (0.988 + 0.012 * (gap / 2.0)) * SO(DF_SPEED, b);
        SR(DF_SPEED) = fmin(SR(DF_SPEED), cap);
        SR(DF_BFRONT) = SR(DF_BFRONT) + dt;
    } else {
        SR(DF_BFRONT) = 0.0;
    }
    int inner = 0;
    int outer = 0;
    cc_side_room(c, r, SR(DF_POS), SR(DF_LANE), &inner, &outer);
    const int side = inner || outer;
    SR(DF_BSIDE) = side ? SR(DF_BSIDE) + dt : 0.0;
    SR(DF_BALL) = (b >= 0 && side) ? SR(DF_BALL) + dt : 0.0;
}

// update_lane_target (race_solver.rs:1272-1308).
static void cc_update_lane_target(Cc* c, const int r, const int m, const int nt) {
    if (SR(DF_HP) <= 0.0) {
        SR(DF_LANETARGET) = SR(DF_LANE);
        return;
    }
    if (SU(UR_PACEDOWN)) {
        SR(DF_LANETARGET) = 0.18;
        return;
    }
    if (c->cd[D_LASTCORNER_START] >= 0.0 && SU(UR_EXTRA) == 0u &&
        SR(DF_POS) >= c->cd[D_LASTCORNER_START]) {
        const double l = SR(DF_LANE);
        SR(DF_EXTRA) = fmin(fmax(l / 0.1, 0.0), 1.0) * 0.5 +
            cc_random(&SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO)) * 0.1;
        SU(UR_EXTRA) = 1u;
    }
    int inner = 0;
    int outer = 0;
    cc_side_room(c, r, SR(DF_POS), SR(DF_LANE), &inner, &outer);
    double passing = 0.0;
    if (cc_passing_lane(c, r, m, nt, &passing)) {
        SR(DF_LANETARGET) = passing;
        return;
    }
    const int blocked_direction =
        SR(DF_LANETARGET) > SR(DF_LANE) ? outer : inner;
    if (fabs(SR(DF_LANETARGET) - SR(DF_LANE)) > 0.5 * CLS_HORSE_LANE &&
        !blocked_direction) {
        return;
    }
    const int on_final = c->cd[D_LASTSTR_START] >= 0.0 &&
        SR(DF_POS) >= c->cd[D_LASTSTR_START];
    if (on_final && SU(UR_EXTRA) && SR(DF_EXTRA) > SR(DF_LANE) && !outer) {
        SR(DF_LANETARGET) = SR(DF_EXTRA);
    } else if ((int)SU(UR_PHASE) < 2 && !inner) {
        SR(DF_LANETARGET) = fmax(SR(DF_LANE) - 0.05, 0.0);
    } else {
        SR(DF_LANETARGET) = SR(DF_LANE);
    }
}

// update_lane (race_solver.rs:1342-1414).
static void cc_update_lane(Cc* c, const int r, const double dt) {
    const int m = cc_build_others(c, r);
    const int nt = cc_collect_targets(c, r, m);
    if (c->su[U_EIM]) {
        cc_update_lane_target(c, r, m, nt);
    }
    const double max_lane = c->cd[D_MAXLANE];
    SR(DF_LANETARGET) = fmin(fmax(SR(DF_LANETARGET), 0.0), max_lane);
    const double lane_distance = fabs(SR(DF_LANETARGET) - SR(DF_LANE));
    if (lane_distance < 1e-9) {
        SR(DF_LCSPEED) = 0.0;
        SU(UR_MOVELANE) = 0u;
        SU(UR_MLCT) = 0u;
        return;
    }
    double lane_modifier;
    if ((int)SU(UR_PHASE) >= 2) {
        lane_modifier = 1.0 + (double)cc_rank(c, r) * 0.01;
    } else if (SR(DF_POS) <= c->cd[D_FIRSTMOVE]) {
        lane_modifier = 1.0 + SR(DF_LANE) / fmax(max_lane, 1e-9) * 0.05;
    } else {
        lane_modifier = 1.0;
    }
    const int previous_direction = (int)SU(UR_MOVEDIR);
    SU(UR_MOVEDIR) = SR(DF_LANETARGET) >= SR(DF_LANE) ? 1u : 0u;
    const double power = fmin(fmax(SR(DF_POWER), 1.0), 2000.0);
    const double target_lane_change_speed =
        0.02 * (0.3 + 0.001 * power) * lane_modifier;
    const double lane_change_accel = 0.02 * 1.5;
    if (SR(DF_LCSPEED) < target_lane_change_speed) {
        SR(DF_LCSPEED) = fmin(target_lane_change_speed,
            SR(DF_LCSPEED) + lane_change_accel * dt);
    } else if (SR(DF_LCSPEED) > target_lane_change_speed) {
        SR(DF_LCSPEED) = fmax(target_lane_change_speed,
            SR(DF_LCSPEED) - lane_change_accel * dt);
    }
    double actual_speed = fmin(fmax(SR(DF_LCSPEED), 0.0), 0.6);
    const double direction = SR(DF_LANETARGET) > SR(DF_LANE) ? 1.0 : -1.0;
    if (direction < 0.0) {
        actual_speed *= 1.0 + SR(DF_LANE);
    }
    const double step = actual_speed * dt * direction;
    const double previous_lane = SR(DF_LANE);
    if (fabs(step) >= lane_distance) {
        SR(DF_LANE) = SR(DF_LANETARGET);
        SR(DF_LCSPEED) = 0.0;
    } else {
        SR(DF_LANE) += step;
    }
    SR(DF_LANE) = fmin(fmax(SR(DF_LANE), 0.0), max_lane);
    const double dist = c->cd[D_DISTANCE];
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        if (SO(DF_POS, j) >= dist || fabs(SO(DF_POS, j) - SR(DF_POS)) >= 1.05) {
            continue;
        }
        if (previous_lane < SO(DF_LANE, j) && SR(DF_LANE) > previous_lane) {
            SR(DF_LANE) = fmin(SR(DF_LANE),
                fmax(previous_lane, SO(DF_LANE, j) - 2.0 * CLS_HORSE_LANE));
        } else if (previous_lane > SO(DF_LANE, j) && SR(DF_LANE) < previous_lane) {
            SR(DF_LANE) = fmax(SR(DF_LANE),
                fmin(previous_lane, SO(DF_LANE, j) + 2.0 * CLS_HORSE_LANE));
        }
    }
    SU(UR_MOVELANE) = fabs(SR(DF_LANE) - previous_lane) > 1e-9 ? 1u : 0u;
    if (SU(UR_MOVELANE) && previous_direction == (int)SU(UR_MOVEDIR)) {
        const unsigned int old = SU(UR_MLCT);
        SU(UR_MLCT) =
            convert_uint_rte(convert_float_rte(as_float(old) + (float)dt));
    }
}
// update_near_lane_timers (race_solver.rs:1416-1450). The three timers feed
// only activation_conditions with zero skills, kept for loop fidelity.
static void cc_update_near_lane_timers(Cc* c, const int r, const double dt) {
    if (cc_rank(c, r) != cc_prev_rank(c, r)) {
        SR(DF_NEARA) = 0.0;
        SR(DF_NEARB) = 0.0;
        SR(DF_SET1) = 0.0;
        return;
    }
    int ahead = -1;
    int behind = -1;
    cc_ranked_neighbors(c, r, &ahead, &behind);
    if (ahead >= 0) {
        const double gap = fmax(SO(DF_POS, ahead) - SR(DF_POS), 0.0);
        const double lg = fabs(SO(DF_LANE, ahead) - SR(DF_LANE));
        SR(DF_NEARA) =
            (fabs(gap) <= 2.5 && lg <= convert_double(0.0556f))
            ? SR(DF_NEARA) + dt : 0.0;
    } else {
        SR(DF_NEARA) = 0.0;
    }
    if (behind >= 0) {
        const double gap = fmax(SR(DF_POS) - SO(DF_POS, behind), 0.0);
        const double lg = fabs(SR(DF_LANE) - SO(DF_LANE, behind));
        SR(DF_NEARB) =
            (fabs(gap) <= 2.5 && lg <= convert_double(0.0556f))
            ? SR(DF_NEARB) + dt : 0.0;
        SR(DF_SET1) =
            (fabs(gap) <= 5.0 && lg <= convert_double(0.15f))
            ? SR(DF_SET1) + dt : 0.0;
    } else {
        SR(DF_NEARB) = 0.0;
        SR(DF_SET1) = 0.0;
    }
}

// update_kakari (race_solver.rs:1452-1465). temptation_count latches at the
// first pass, so a completed rush never re-lotteries.
static void cc_update_kakari(Cc* c, const int r) {
    const int strategy = (int)SU(UR_STRATEGY);
    if ((int)SU(UR_TEMPT) == 0 && SR(DF_POS) >= SR(DF_KSTART)) {
        int roll = 0;
        if (!(strategy == 1 || strategy == 5)) {
            roll = cc_uniform_cap(c, &SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO), 100.0);
        }
        SU(UR_RUSH) = (unsigned int)cc_select_mode(strategy, roll);
        SU(UR_KAKARI) = 1u;
        SU(UR_TEMPT) = 1u;
        SR(DF_KTIMER) = -SR(DF_KDUR);
        SR(DF_KSTARTTIME) = SR(DF_ACCUM);
    } else if (SU(UR_KAKARI) && SR(DF_KTIMER) >= 0.0) {
        SU(UR_KAKARI) = 0u;
        SU(UR_RUSH) = 0u;
        cc_emit_kakari(c, (unsigned int)r, SR(DF_KSTARTTIME), SR(DF_ACCUM));
    }
}

// position_keep_wisdom_check (race_solver.rs:1527-1531), pct precomputed.
static int cc_wisdom_check(Cc* c, const int r, const double pct) {
    if (SU(UR_KAKARI)) {
        return 1;
    }
    return cc_random(&SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO)) <
        fmin(pct, 100.0) / 100.0;
}

// clear_position_keep (race_solver.rs:1520-1525).
static void cc_clear_pk(Cc* c, const int r) {
    SU(UR_PKMODE) = 0u;
    SU(UR_PACEDOWN) = 0u;
    SR(DF_PKCOEF) = 1.0;
    SR(DF_PKCOOL) = 1.0;
}

// update_position_keep (race_solver.rs:1468-1518), with both branches.
static void cc_update_position_keep(Cc* c, const int r, const double dt) {
    if ((int)SU(UR_PKRUSH) != (int)SU(UR_KAKARI)) {
        SU(UR_PKRUSH) = SU(UR_KAKARI);
        cc_clear_pk(c, r);
        SR(DF_PKCOOL) = 0.0;
        SR(DF_PKINT) = 0.0;
    }
    const int was_active = SU(UR_PKMODE) != 0u;
    if (was_active && SR(DF_POS) >= SR(DF_PKEFFEND)) {
        cc_clear_pk(c, r);
        return;
    }
    if (!was_active) {
        if (SR(DF_POS) < 0.0 || SR(DF_POS) >= SR(DF_PKEND)) {
            return;
        }
        if (SR(DF_PKCOOL) > 0.0) {
            SR(DF_PKCOOL) -= dt;
            if (SR(DF_PKCOOL) > 0.0) {
                return;
            }
        }
        if (SR(DF_PKINT) > 0.0) {
            SR(DF_PKINT) -= dt;
            if (SR(DF_PKINT) > 0.0) {
                return;
            }
        }
        SR(DF_PKINT) = 2.0;
    }
    const int strategy = (int)SU(UR_STRATEGY);
    const int style = cc_pacing_style(strategy,
        SU(UR_KAKARI) ? (int)SU(UR_RUSH) : 0);
    if ((int)SU(UR_PKMODE) == PK_PACEUPEX) {
        if (!cc_later_style_ahead(c, r)) {
            cc_clear_pk(c, r);
        }
    } else if (!was_active &&
        ((cc_style_matches(style, 1) && !SU(UR_KAKARI) &&
            cc_later_style_ahead(c, r)) ||
         (!cc_style_matches(style, 1) &&
            cc_pacemaker_is_later_style_ahead(c, r)))) {
        SU(UR_PKMODE) = PK_PACEUPEX;
        SU(UR_PACEDOWN) = 0u;
        SR(DF_PKCOEF) = 2.0;
    } else if (cc_style_matches(style, 1)) {
        // update_position_keep_front_runner (race_solver.rs:1596-1637).
        const int is_leader = cc_rank(c, r) == 1;
        const int choose_speed_up = is_leader ||
            (SU(UR_KAKARI) && ((int)SU(UR_RUSH) == 3 || (int)SU(UR_RUSH) == 4));
        const double gap_2nd = cc_distance_to_behind(c, r);
        const int solo = (int)SR(DF_NIGECNT) == 1;
        const double speed_up_threshold =
            strategy == 5 ? 17.5 : (solo ? 12.5 : 4.5);
        const double overtake_exit_threshold = strategy == 5 ? 27.5 : 10.0;
        if ((int)SU(UR_PKMODE) == PK_SPEEDUP) {
            if (gap_2nd < 0.0 || gap_2nd >= speed_up_threshold) {
                SU(UR_PKMODE) = 0u;
                SR(DF_PKCOEF) = 1.0;
            }
        } else if ((int)SU(UR_PKMODE) == PK_OVERTAKE) {
            if (is_leader && (gap_2nd < 0.0 ||
                gap_2nd >= overtake_exit_threshold)) {
                SU(UR_PKMODE) = 0u;
                SR(DF_PKCOEF) = 1.0;
            }
        } else if (choose_speed_up && gap_2nd >= 0.0 &&
            gap_2nd < speed_up_threshold) {
            if (cc_wisdom_check(c, r, SR(DF_PCT20))) {
                SU(UR_PKMODE) = PK_SPEEDUP;
                SR(DF_PKCOEF) = 1.04;
            }
        } else if (!choose_speed_up && cc_wisdom_check(c, r, SR(DF_PCT20))) {
            SU(UR_PKMODE) = PK_OVERTAKE;
            SR(DF_PKCOEF) = 1.05;
        }
    } else {
        // update_position_keep_non_nige (race_solver.rs:1540-1594).
        const int pm = cc_pacemaker(c);
        if (pm == r) {
            SU(UR_PKMODE) = 0u;
            SU(UR_PACEDOWN) = 0u;
            SR(DF_PKCOEF) = 1.0;
        } else {
            const double gap = cc_gap_to_pacemaker(c, r);
            const double dist = c->cd[D_DISTANCE];
            const double course_factor =
                fmax(0.0008 * (dist - 1000.0) + 1.0, 1.0);
            const double base_min =
                style == 2 ? 3.0 : (style == 3 ? 6.5 : (style == 4 ? 7.5 : 0.0));
            const double base_max =
                style == 2 ? 5.0 : (style == 3 ? 7.0 : (style == 4 ? 8.0 : 0.0));
            const double lo =
                base_min * (style == 2 ? 1.0 : course_factor);
            const double hi = base_max * course_factor;
            if ((int)SU(UR_PKMODE) == PK_PACEDOWN) {
                if (gap >= SR(DF_PKEFFEXIT)) {
                    SU(UR_PKMODE) = 0u;
                    SU(UR_PACEDOWN) = 0u;
                    SR(DF_PKCOEF) = 1.0;
                }
            } else if ((int)SU(UR_PKMODE) == PK_PACEUP) {
                if (gap <= SR(DF_PKEFFEXIT)) {
                    SU(UR_PKMODE) = 0u;
                    SR(DF_PKCOEF) = 1.0;
                }
            } else if (gap < lo) {
                SU(UR_PKMODE) = PK_PACEDOWN;
                SU(UR_PACEDOWN) = 1u;
                const double high = (int)SU(UR_PHASE) == 1
                    ? lo + 0.5 * (hi - lo) : hi;
                SR(DF_PKEFFEXIT) = lo +
                    cc_random(&SU(UR_PACE_HI), &SU(UR_PACE_LO)) * (high - lo);
                SR(DF_PKCOEF) = (int)SU(UR_PHASE) == 1 ? 0.945 : 0.915;
            } else if (gap > hi && cc_wisdom_check(c, r, SR(DF_PCT15))) {
                SU(UR_PKMODE) = PK_PACEUP;
                SR(DF_PKEFFEXIT) = lo +
                    cc_random(&SU(UR_PACE_HI), &SU(UR_PACE_LO)) * (hi - lo);
                SR(DF_PKCOEF) = 1.04;
            }
        }
    }
    if (was_active && SU(UR_PKMODE) == 0u) {
        cc_clear_pk(c, r);
    } else if (!was_active && SU(UR_PKMODE) != 0u) {
        SR(DF_PKEFFEND) = SR(DF_POS) + SR(DF_PKCONT);
    }
}

// get_status_modifier (hp_policy.rs:218-229).
static double cc_hp_status(const int compete_top, const int kakari,
    const int runaway, const int pace_down, const int downhill) {
    double modifier;
    if (compete_top) {
        modifier = kakari ? (runaway ? 7.7 : 3.6) : (runaway ? 3.5 : 1.4);
    } else if (kakari) {
        modifier = 1.6;
    } else if (pace_down) {
        modifier = 0.6;
    } else {
        modifier = 1.0;
    }
    if (downhill) {
        modifier *= 0.4;
    }
    return modifier;
}

// GameHpState::hp_per_second classic (hp_policy.rs:231-243). Rust squares
// first (powi), so the square must not fold into a multiply by 20.
static double cc_hp_rate(Cc* c, const int r, const double velocity,
    const int phase) {
    const double x = velocity - SR(DF_BASESPEED) + 12.0;
    const double sq = x * x;
    return 20.0 * sq / 144.0 *
        cc_hp_status((int)SU(UR_SPOTACT), (int)SU(UR_KAKARI),
            (int)SU(UR_STRATEGY) == 5, (int)SU(UR_PACEDOWN),
            (int)SU(UR_DOWNHILL)) *
        SR(DF_GROUNDMOD) * (phase >= 2 ? SR(DF_GUTSMOD) : 1.0);
}

// GameHpState::tick + spend classic (hp_policy.rs:245-254, 273-277).
static void cc_hp_tick(Cc* c, const int r, const double dt) {
    if (SR(DF_HP) <= 0.0) {
        return;
    }
    const double amount =
        cc_hp_rate(c, r, SR(DF_SPEED), (int)SU(UR_PHASE)) * dt;
    SR(DF_HP) = fmin(fmax(SR(DF_HP) - amount, 0.0), SR(DF_MAXHP));
}

// get_last_spurt_pair classic (hp_policy.rs:291-336), on the hp stream. The
// last-leg context has no flags, so the status modifier is 1.0.
static double cc_last_spurt_rate(Cc* c, const int r, const double v) {
    const double x = v - SR(DF_BASESPEED) + 12.0;
    const double sq = x * x;
    return 20.0 * sq / 144.0 * SR(DF_GROUNDMOD) * SR(DF_GUTSMOD);
}

static void cc_last_spurt_pair(Cc* c, const int r, const double pos,
    double* trans_out, double* speed_out) {
    const double dist = c->cd[D_DISTANCE];
    const double max_speed = SR(DF_LSS);
    const double base2 = SR(DF_BTS0 + 2);
    const double max_dist = dist - dist * 2.0 / 3.0;
    const double s = (max_dist - 60.0) / max_speed;
    if (SR(DF_HP) >= cc_last_spurt_rate(c, r, max_speed) * s) {
        *trans_out = -1.0;
        *speed_out = max_speed;
        return;
    }
    const double hp_at_base = cc_last_spurt_rate(c, r, base2);
    const double remain = dist - 60.0 - pos;
    int count = 0;
    double speed = max_speed - 0.1;
    while (speed >= base2 && count < 512) {
        const double hp_at_speed = cc_last_spurt_rate(c, r, speed);
        const double denom = base2 * hp_at_speed - hp_at_base * speed;
        const double numer = base2 * SR(DF_HP) - hp_at_base * remain;
        const double duration = fmin(remain / speed, fmax(numer / denom, 0.0));
        c->sd[UDB_SPURTC + count] = dist - (duration * speed + 60.0);
        c->sd[UDB_SPURTV + count] = speed;
        count++;
        speed -= 0.1;
    }
    if (count == 0) {
        *trans_out = dist;
        *speed_out = max_speed;
        return;
    }
    // Stable insertion sort by finish_time (hp_policy.rs:328-329).
    for (int k = 1; k < count; k++) {
        const double cs = c->sd[UDB_SPURTC + k];
        const double cv = c->sd[UDB_SPURTV + k];
        const double key = (cs - pos) / base2 + (dist - cs) / cv;
        int mm = k;
        while (mm > 0) {
            const double ps = c->sd[UDB_SPURTC + mm - 1];
            const double pv = c->sd[UDB_SPURTV + mm - 1];
            if ((ps - pos) / base2 + (dist - ps) / pv > key) {
                c->sd[UDB_SPURTC + mm] = ps;
                c->sd[UDB_SPURTV + mm] = pv;
                mm--;
            } else {
                break;
            }
        }
        c->sd[UDB_SPURTC + mm] = cs;
        c->sd[UDB_SPURTV + mm] = cv;
    }
    for (int k = 0; k < count; k++) {
        const long draw = (long)cc_uniform_cap(c, &SU(UR_HP_HI),
            &SU(UR_HP_LO), 100000.0);
        if (draw <= (long)SR(DF_SUBPAR)) {
            *trans_out = c->sd[UDB_SPURTC + k];
            *speed_out = c->sd[UDB_SPURTV + k];
            return;
        }
    }
    *trans_out = c->sd[UDB_SPURTC + count - 1];
    *speed_out = c->sd[UDB_SPURTV + count - 1];
}

// update_last_spurt_state (race_solver.rs:1639-1655).
static void cc_update_last_spurt(Cc* c, const int r) {
    if (SU(UR_SPURT) || (int)SU(UR_PHASE) < 2) {
        return;
    }
    if (SR(DF_LSTTRANS) == -1.0) {
        double trans;
        double speed;
        cc_last_spurt_pair(c, r, SR(DF_POS), &trans, &speed);
        SR(DF_LSTTRANS) = trans;
        SR(DF_LSS) = speed;
    }
    if (SR(DF_POS) >= SR(DF_LSTTRANS)) {
        SU(UR_SPURT) = 1u;
        if (SR(DF_SPURTT) < 0.0) {
            SR(DF_SPURTT) = SR(DF_ACCUM);
            SR(DF_SPURTD) = SR(DF_POS);
        }
    }
}

// update_target_speed (race_solver.rs:1657-1688).
static void cc_update_target_speed(Cc* c, const int r) {
    if (SU(UR_SPURT)) {
        SR(DF_TARGET) = SR(DF_LSS);
    } else {
        int idx = (int)(SR(DF_POS) / SR(DF_SECTION));
        if (idx > 24) {
            idx = 24;
        }
        SR(DF_TARGET) = SR(DF_BTS0 + (int)SU(UR_PHASE)) + SR(DF_SECT0 + idx);
    }
    SR(DF_TARGET) *= SR(DF_PKCOEF);
    SR(DF_TARGET) += SR(DF_TSA) + SR(DF_TSE);
    const double slope_per = SR(DF_SLOPEPER);
    if (SU(UR_DOWNHILL) && slope_per < 0.0) {
        SR(DF_TARGET) += 0.3 + fabs(slope_per) / 100000.0;
    } else if ((int)SU(UR_SLOPEIDX) > 0 && slope_per > 0.0) {
        SR(DF_TARGET) -= slope_per / 10000.0 * 200.0 /
            fmin(fmax(SR(DF_POWER), 1.0), 2000.0);
    }
    if (SR(DF_HP) <= 0.0) {
        SR(DF_TARGET) = SR(DF_MINSPEED);
    }
    SR(DF_TARGET) = fmin(fmax(SR(DF_TARGET), 13.0), 30.0);
}

// apply_forces (race_solver.rs:1690-1701).
static void cc_apply_forces(Cc* c, const int r) {
    if (SR(DF_HP) <= 0.0) {
        SR(DF_ACCEL) = -1.2;
        return;
    }
    if (SR(DF_SPEED) > SR(DF_TARGET)) {
        // PHASE_DECELERATION (race_solver.rs:90).
        SR(DF_ACCEL) = SU(UR_PACEDOWN) ? -0.5
            : ((int)SU(UR_PHASE) == 0 ? -1.2
                : ((int)SU(UR_PHASE) == 1 ? -0.8 : -1.0));
        return;
    }
    const int uphill = SR(DF_SLOPEPER) > 0.0 ? 3 : 0;
    SR(DF_ACCEL) = SR(DF_BACC0 + uphill + (int)SU(UR_PHASE)) +
        SR(DF_ACA) + SR(DF_ACE);
}

// downhill_check (race_solver.rs:1703-1708).
static void cc_downhill_check(Cc* c, const int r, const double roll) {
    if (SR(DF_SLOPEPER) < 0.0 &&
        roll < SR(DF_WISDOM) * 0.0004) {
        SU(UR_DOWNHILL) = 1u;
    }
}

// update_hills (race_solver.rs:1710-1741). Slopes sit ascending in the course
// block; the entry counter plays both pop stacks like the engine's vec.
static void cc_update_hills(Cc* c, const int r) {
    const int nh = c->nhills;
    if ((int)SU(UR_SLOPEIDX) == 0 && (int)SU(UR_STARTCNT) > 0) {
        const int idx = nh - (int)SU(UR_STARTCNT);
        if (SR(DF_POS) >= c->cd[CLS_RD_SLOPES + 3 * idx]) {
            SU(UR_SLOPEIDX) = (unsigned int)(idx + 1);
            SR(DF_SLOPEPER) = c->cd[CLS_RD_SLOPES + 3 * idx + 2];
            SR(DF_DOWNHT) = 0.0;
            const double roll = cc_random(
                &SU(UR_HILL0 + 2 * idx), &SU(UR_HILL0 + 2 * idx + 1));
            cc_downhill_check(c, r, roll);
            SU(UR_STARTCNT) -= 1u;
        }
    } else if ((int)SU(UR_SLOPEIDX) > 0 && (int)SU(UR_ENDCNT) > 0) {
        const int idx = nh - (int)SU(UR_ENDCNT);
        if (SR(DF_POS) > c->cd[CLS_RD_SLOPES + 3 * idx + 1]) {
            SU(UR_SLOPEIDX) = 0u;
            SR(DF_SLOPEPER) = 0.0;
            SU(UR_ENDCNT) -= 1u;
            SU(UR_DOWNHILL) = 0u;
        }
    }
    if (SR(DF_DOWNHT) >= 1.0 && (int)SU(UR_SLOPEIDX) > 0) {
        const int idx = (int)SU(UR_SLOPEIDX) - 1;
        const double roll =
            cc_random(&SU(UR_HILL0 + 2 * idx), &SU(UR_HILL0 + 2 * idx + 1));
        if (SU(UR_DOWNHILL) && roll > 0.8) {
            SU(UR_DOWNHILL) = 0u;
        } else if (!SU(UR_DOWNHILL)) {
            cc_downhill_check(c, r, roll);
        }
        SR(DF_DOWNHT) = 0.0;
    }
}

// update_phase (race_solver.rs:1743-1751).
static void cc_update_phase(Cc* c, const int r) {
    const double dist = c->cd[D_DISTANCE];
    if (SR(DF_POS) >= SR(DF_NEXTPHASE) && (int)SU(UR_PHASE) < 2) {
        SU(UR_PHASE) += 1u;
        SR(DF_NEXTPHASE) = (int)SU(UR_PHASE) == 1 ? dist * 2.0 / 3.0
            : dist * 5.0 / 6.0;
    }
}

static double cc_get_max_speed(Cc* c, const int r) {
    if (SU(UR_DASH)) {
        return fmin(SR(DF_TARGET), 0.85 * SR(DF_BASESPEED));
    }
    if (SR(DF_SPEED) + SR(DF_ONEFRAME) > SR(DF_TARGET)) {
        return 9999.0;
    }
    return SR(DF_TARGET);
}

static void cc_sweep(Cc* c, const int r, const double d) {
    SR(DF_ACCUM) += d;
    SR(DF_DOWNHT) += d;
    SR(DF_KTIMER) += d;
    SR(DF_SPOTT) += d;
}

// step_forward (race_solver.rs:1175-1224). Timers swept here are
// accumulatetime, downhill, kakari and spot (race_solver.rs:978-984, the
// rest belong to skills). Returns the effective moving dt.
static double cc_step_forward(Cc* c, const int r, double dt) {
    if (SR(DF_ACCUM) < SR(DF_DELAY)) {
        const double partial_frame = SR(DF_DELAY) - SR(DF_ACCUM);
        if (partial_frame < dt) {
            cc_sweep(c, r, partial_frame);
            dt -= partial_frame;
        } else {
            cc_sweep(c, r, dt);
            return 0.0;
        }
    }
    const double halfv =
        fmin(SR(DF_SPEED) + 0.5 * dt * SR(DF_ACCEL), cc_get_max_speed(c, r));
    const double displacement = SR(DF_CSA) + SR(DF_CSE);
    SR(DF_POS) += (halfv + displacement) * dt;
    cc_hp_tick(c, r, dt);
    cc_sweep(c, r, dt);
    cc_update_hills(c, r);
    cc_update_phase(c, r);
    cc_update_kakari(c, r);
    cc_update_position_keep(c, r, dt);
    cc_update_last_spurt(c, r);
    cc_update_target_speed(c, r);
    cc_apply_forces(c, r);
    SR(DF_SPEED) = fmin(halfv + 0.5 * dt * SR(DF_ACCEL) + SR(DF_ONEFRAME),
        cc_get_max_speed(c, r));
    if (!SU(UR_DASH) && SR(DF_SPEED) < SR(DF_MINSPEED)) {
        SR(DF_SPEED) = SR(DF_MINSPEED);
    } else if (SU(UR_DASH) &&
        SR(DF_SPEED) >= 0.85 * SR(DF_BASESPEED)) {
        SU(UR_DASH) = 0u;
        cc_add(&SR(DF_ACA), &SR(DF_ACE), -24.0);
    }
    SR(DF_ONEFRAME) = 0.0;
    return dt;
}

static void cc_step_lateral(Cc* c, const int r, const double dt) {
    cc_update_blocking(c, r, dt);
    cc_update_lane(c, r, dt);
}

static void cc_step_field_conditions(Cc* c, const int r, const double dt) {
    cc_update_near_lane_timers(c, r, dt);
    // update_competition_mechanics (race_solver.rs:2746-2758) with the
    // stamina keep and compete gates off: spot struggle, then duel.
    cc_update_spot(c, r);
    cc_update_dueling(c, r, dt);
}
// start_spot_struggle classic (race_solver.rs:2928-2946); the modifier and
// duration come precomputed from the host (powf of adjusted guts).
static void cc_start_spot(Cc* c, const int j) {
    SUO(UR_SPOTACT, j) = 1u;
    SUO(UR_SPOTSTART, j) = 1u;
    SO(DF_SPOTT, j) = -SO(DF_SPOTD, j);
    cc_add(&SO(DF_TSA, j), &SO(DF_TSE, j), SO(DF_SPOTM, j));
}

// update_spot_struggle (race_solver.rs:2792-2926), Nige/Oonige only.
static void cc_update_spot(Cc* c, const int r) {
    const int strategy = (int)SU(UR_STRATEGY);
    if (strategy != 1 && strategy != 5) {
        return;
    }
    const double dist = c->cd[D_DISTANCE];
    if (SU(UR_SPOTSTART)) {
        const double pos = SR(DF_POS);
        const double lane = SR(DF_LANE);
        if (SU(UR_SPOTACT) && !SU(UR_DISTEXIT)) {
            int distance_exit = 1;
            for (int j = 0; j < c->n; j++) {
                if (j == r) {
                    continue;
                }
                const int same = cc_strategy(c, j) == strategy &&
                    SUO(UR_SPOTSTART, j) &&
                    fabs(SO(DF_POS, j) - pos) <= 5.0 &&
                    fabs(SO(DF_LANE, j) - lane) < 0.416;
                if (same) {
                    distance_exit = 0;
                    break;
                }
            }
            if (distance_exit) {
                SU(UR_DISTEXIT) = 1u;
            }
        }
        if (SU(UR_SPOTACT) && SR(DF_SPOTT) < 0.0 &&
            SR(DF_POS) >= SR(DF_SECTION) * 8.0) {
            SU(UR_CUTOFF) = 1u;
        }
        int group_cutoff = (int)SU(UR_CUTOFF);
        for (int j = 0; j < c->n && !group_cutoff; j++) {
            if (j != r && SUO(UR_SPOTSTART, j) && SUO(UR_CUTOFF, j)) {
                group_cutoff = 1;
            }
        }
        if (SU(UR_SPOTACT) &&
            (SU(UR_DISTEXIT) || SR(DF_SPOTT) >= 0.0 || group_cutoff)) {
            SU(UR_SPOTACT) = 0u;
            cc_add(&SR(DF_TSA), &SR(DF_TSE), -SR(DF_SPOTM));
        }
        return;
    }
    if (SU(UR_ENTCLOSED) || (int)SR(DF_STYLECNT) < 2) {
        return;
    }
    const double pos = SR(DF_POS);
    const double lane = SR(DF_LANE);
    int group_len = 0;
    double race_front = -1e300;
    int eligible_front_rank = 0x7fffffff;
    int front_idx = -1;
    double front_pos = -1e300;
    double front_lane = 0.0;
    const double entry_end = SR(DF_SECTION) * 6.0;
    int group_late = 1;
    for (int j = 0; j < c->n; j++) {
        const int js = cc_strategy(c, j);
        const double jp = j == r ? pos : SO(DF_POS, j);
        const double jl = j == r ? lane : SO(DF_LANE, j);
        const int jrank = cc_rank(c, j);
        if (jrank == 1) {
            race_front = jp;
        }
        if (strategy == 5 || js != 5) {
            eligible_front_rank =
                jrank < eligible_front_rank ? jrank : eligible_front_rank;
        }
        if (js == strategy) {
            if (j == r ? (int)SU(UR_SPOTSTART) : (int)SUO(UR_SPOTSTART, j)) {
                return;
            }
            group_len++;
            if (jp < entry_end) {
                group_late = 0;
            }
            if (front_idx < 0 || jrank < cc_rank(c, front_idx)) {
                front_pos = jp;
                front_lane = jl;
                front_idx = j;
            }
        }
    }
    if (group_late) {
        SU(UR_ENTCLOSED) = 1u;
        return;
    }
    if (group_len < 2 || race_front < 150.0) {
        return;
    }
    if (cc_rank(c, front_idx) != eligible_front_rank) {
        return;
    }
    int count = 0;
    int early = 0;
    for (int j = 0; j < c->n; j++) {
        if (cc_strategy(c, j) != strategy) {
            continue;
        }
        const double jp = j == r ? pos : SO(DF_POS, j);
        const double jl = j == r ? lane : SO(DF_LANE, j);
        if (front_pos - jp <= 3.75 && fabs(front_lane - jl) <= 0.165 &&
            (j == front_idx || jp < entry_end)) {
            count++;
            if (jp < entry_end) {
                early = 1;
            }
        }
    }
    if (count < 2 || !early) {
        return;
    }
    for (int j = 0; j < c->n; j++) {
        if (cc_strategy(c, j) != strategy) {
            continue;
        }
        const double jp = j == r ? pos : SO(DF_POS, j);
        const double jl = j == r ? lane : SO(DF_LANE, j);
        if (front_pos - jp <= 3.75 && fabs(front_lane - jl) <= 0.165 &&
            (j == front_idx || jp < entry_end)) {
            cc_start_spot(c, j);
        }
    }
    (void)dist;
}

// activate_retail_duel classic branch (race_solver.rs:3194-3214).
static void cc_activate_duel(Cc* c, const int j) {
    SUO(UR_DUELACT, j) = 1u;
    SUO(UR_FIGHT, j) += 1u;
    cc_add(&SO(DF_TSA, j), &SO(DF_TSE, j), SO(DF_DTS, j));
    cc_add(&SO(DF_ACA, j), &SO(DF_ACE, j), SO(DF_DA, j));
}

// update_dueling (race_solver.rs:3061-3192), classic proximity and entry.
static void cc_update_dueling(Cc* c, const int r, const double dt) {
    if (!c->su[U_DUELON]) {
        return;
    }
    const double dist = c->cd[D_DISTANCE];
    const double pos = SR(DF_POS);
    const double lane = SR(DF_LANE);
    const int in_final =
        c->cd[D_LASTSTR_START] <= pos && pos < c->cd[D_LASTSTR_END];
    if (SU(UR_DUELACT)) {
        int nearby = 0;
        for (int j = 0; j < c->n; j++) {
            if (j == r) {
                continue;
            }
            if (SUO(UR_FIGHT, j) != 0u && SUO(UR_DGROUP, j) != 0u &&
                SUO(UR_DGROUP, j) == SU(UR_DGROUP) &&
                SO(DF_POS, j) < dist &&
                fabs(SO(DF_POS, j) - pos) <= 5.0) {
                nearby = 1;
                break;
            }
        }
        const int low_hp =
            fmax(SR(DF_HP) / SR(DF_MAXHP), 0.0) < 0.05;
        if (!in_final || low_hp || !nearby) {
            SU(UR_DUELACT) = 0u;
            SR(DF_DCONT) = 0.0;
            SU(UR_DTARGET) = 0u;
            cc_add(&SR(DF_TSA), &SR(DF_TSE), -SR(DF_DTS));
            cc_add(&SR(DF_ACA), &SR(DF_ACE), -SR(DF_DA));
        }
        return;
    }
    if (!in_final || (int)SU(UR_FIGHT) > 0) {
        SR(DF_DCONT) = 0.0;
        SU(UR_DTARGET) = 0u;
        for (int k = 0; k < c->n; k++) {
            c->sd[UDB_DPARK + r * CLS_MAXR + k] = 0.0;
        }
        return;
    }
    const int top_count = ((c->n + 1) / 2) > 1 ? (c->n + 1) / 2 : 1;
    const int top_half = cc_rank(c, r) <= top_count;
    const double self_ratio = fmax(SR(DF_HP) / SR(DF_MAXHP), 0.0);
    for (int k = 0; k < c->n; k++) {
        c->sd[UDB_PROXT + k] = 0.0;
    }
    int nearby_i = -1;
    int eligible_i = -1;
    int ready = -1;
    double ready_timer = 0.0;
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        if (SO(DF_POS, j) >= dist ||
            (SUO(UR_FIGHT, j) != 0u && !SUO(UR_DUELACT, j))) {
            continue;
        }
        const double dist_gap = SO(DF_POS, j) - pos;
        const double lane_gap = SO(DF_LANE, j) - lane;
        if (!(fabs(dist_gap) <= 3.0 && fabs(lane_gap) <= 0.25)) {
            continue;
        }
        const double old = c->sd[UDB_DPARK + r * CLS_MAXR + j];
        const double timer = fmin(old + dt, 2.0);
        c->sd[UDB_PROXT + j] = timer;
        if (nearby_i < 0) {
            nearby_i = j;
        }
        const double other_ratio = fmax(SO(DF_HP, j) / SO(DF_MAXHP, j), 0.0);
        const int qualifies = self_ratio >= 0.15 && other_ratio >= 0.15 &&
            fabs(SO(DF_SPEED, j) - SR(DF_SPEED)) <= 0.6000000238418579 &&
            (top_half || cc_rank(c, j) <= top_count);
        if (qualifies) {
            if (eligible_i < 0) {
                eligible_i = j;
            }
            if (timer >= 2.0 && (ready < 0 || timer > ready_timer)) {
                ready = j;
                ready_timer = timer;
            }
        }
    }
    const int target_index =
        ready >= 0 ? ready : (eligible_i >= 0 ? eligible_i : nearby_i);
    SU(UR_DTARGET) = (unsigned int)(target_index + 1);
    SR(DF_DCONT) = target_index >= 0 ? c->sd[UDB_PROXT + target_index] : 0.0;
    for (int k = 0; k < c->n; k++) {
        c->sd[UDB_DPARK + r * CLS_MAXR + k] = c->sd[UDB_PROXT + k];
    }
    if (ready < 0) {
        return;
    }
    // Classic entry (race_solver.rs:3180): group = peer.duel_group
    // .get_or_insert(i); both entrants then get_or_insert(group).
    const int j = ready;
    int group;
    if (SUO(UR_DGROUP, j)) {
        group = (int)SUO(UR_DGROUP, j) - 1;
    } else {
        SUO(UR_DGROUP, j) = (unsigned int)(r + 1);
        group = r;
    }
    if (!SU(UR_DGROUP)) {
        SU(UR_DGROUP) = (unsigned int)(group + 1);
    }
    if (!SUO(UR_DGROUP, j)) {
        SUO(UR_DGROUP, j) = (unsigned int)(group + 1);
    }
    cc_activate_duel(c, r);
}

// next_overtake_times (race_field.rs:339-365). The clocks are dead state with
// zero skills but stay ported for the loop order.
static void cc_next_overtake(Cc* c, const double dt) {
    const double dist = c->cd[D_DISTANCE];
    for (int i = 0; i < c->n; i++) {
        double gap = 0.0;
        const int b = cc_front_blocker(c, i, SO(DF_POS, i), SO(DF_LANE, i), &gap);
        c->su[U_BLOCK + i] = (unsigned int)(b + 1);
    }
    for (int i = 0; i < c->n; i++) {
        int row_count = 0;
        for (int j = 0; j < c->n; j++) {
            int is_t = 0;
            if (j != i && SO(DF_POS, j) < dist) {
                const int direct = ((int)c->su[U_BLOCK + i] - 1) == j;
                is_t = direct || cc_is_target(
                    SO(DF_POS, j) - SO(DF_POS, i),
                    SO(DF_SPEED, i) - SO(DF_SPEED, j),
                    SO(DF_TARGET, i), SO(DF_TARGET, j), SO(DF_SPEED, j),
                    direct, c->su[U_BLOCK + j] != 0u);
            }
            c->su[U_TGTMAP + i * CLS_MAXR + j] = (unsigned int)is_t;
            row_count += is_t;
        }
        (void)row_count;
    }
    for (int i = 0; i < c->n; i++) {
        int targeted = 0;
        int row_nonempty = 0;
        for (int j = 0; j < c->n; j++) {
            if (cc_rank(c, j) > cc_rank(c, i) &&
                c->su[U_TGTMAP + j * CLS_MAXR + i] != 0u) {
                targeted = 1;
            }
            if (c->su[U_TGTMAP + i * CLS_MAXR + j] != 0u) {
                row_nonempty = 1;
            }
        }
        const double fdt = (double)convert_float_rte(dt);
        double v = c->sd[UDB_OVERT + i * 3];
        c->sd[UDB_OVERT + i * 3] = targeted
            ? (double)convert_float_rte(convert_float_rte(v) +
                convert_float_rte(dt)) : 0.0;
        v = c->sd[UDB_OVERT + i * 3 + 1];
        c->sd[UDB_OVERT + i * 3 + 1] = row_nonempty
            ? (double)convert_float_rte(convert_float_rte(v) +
                convert_float_rte(dt)) : 0.0;
        v = c->sd[UDB_OVERT + i * 3 + 2];
        c->sd[UDB_OVERT + i * 3 + 2] = row_nonempty
            ? (double)convert_float_rte(convert_float_rte(v) +
                convert_float_rte(dt)) : 0.0;
    }
}

// RaceField::update (race_field.rs:205-287), classic branch.
static void cc_field_update(Cc* c) {
    const double dist = c->cd[D_DISTANCE];
    for (int j = 0; j < c->n; j++) {
        c->su[U_PREVRANK + j] = c->su[U_RANK + j];
    }
    c->su[U_PMVALID] = 0u;
    __global unsigned int* idx = c->su + U_ORDER;
    for (int k = 0; k < c->n; k++) {
        idx[k] = (unsigned int)k;
    }
    for (int k = 1; k < c->n; k++) {
        const unsigned int item = idx[k];
        const double p = SO(DF_POS, (int)item);
        int m = k;
        while (m > 0) {
            const unsigned int prev = idx[m - 1];
            const double q = SO(DF_POS, (int)prev);
            const int shift = q < p || (q == p && prev > item);
            if (!shift) {
                break;
            }
            idx[m] = prev;
            m--;
        }
        idx[m] = item;
    }
    for (int k = 0; k < c->n; k++) {
        c->su[U_RANK + (int)idx[k]] = (unsigned int)(k + 1);
    }
    int highest = 0;
    for (int k = 0; k < c->n; k++) {
        const int j = (int)c->su[U_ORDER + k];
        c->su[U_STYLEA + j] = (unsigned int)highest;
        const int bs = cc_base_style(cc_strategy(c, j));
        highest = bs > highest ? bs : highest;
    }
    if (c->su[U_PPRES]) {
        for (int i = 0; i < c->n; i++) {
            if (c->sd[UDB_PREVPOS + i] >= dist) {
                continue;
            }
            const int improved =
                (int)c->su[U_PREVRANK + i] > (int)c->su[U_RANK + i];
            int event = improved;
            if (event) {
                const int target = (int)c->su[U_ORDER + (int)c->su[U_RANK + i]];
                const double now = SO(DF_ACCUM, i);
                __global double* cool = &c->sd[UDB_COOL + i * CLS_MAXR + target];
                event = now >= *cool;
                if (event) {
                    *cool = now + 6.0;
                }
            }
            if (event) {
                c->sd[UDB_OVERT + i * 3 + 1] = 0.0;
            }
            if ((int)c->su[U_PREVRANK + i] < (int)c->su[U_RANK + i]) {
                c->sd[UDB_OVERT + i * 3 + 2] = 0.0;
            }
        }
    }
    for (int j = 0; j < c->n; j++) {
        c->sd[UDB_PREVPOS + j] = SO(DF_POS, j);
    }
    c->su[U_PPRES] = 1u;
}
// RaceSolver::new (race_solver.rs:838-952) plus the initial field bookkeeping
// (RaceField::new). The host constructs every value, this copies it into the
// working layout.
static void cc_prepare(Cc* c, __global const unsigned int* ri) {
    const int n = c->n;
    const int nh = c->nhills;
    __global const double* rd = c->cd;
    for (int i = 0; i < n; i++) {
        c->su[U_RANK + i] = 0u;
        c->su[U_PREVRANK + i] = 0u;
        for (int j = 0; j < n; j++) {
            c->sd[UDB_COOL + i * CLS_MAXR + j] = -1.0;
            c->sd[UDB_DPARK + i * CLS_MAXR + j] = 0.0;
        }
        for (int k = 0; k < 3; k++) {
            c->sd[UDB_OVERT + i * 3 + k] = 0.0;
        }
    }
    for (int r = 0; r < n; r++) {
        const int rb = CLS_RI_RUN0 + r * CLS_RI_RUN_STRIDE;
        const int db = CLS_RD_RUN0 + r * CLS_RD_RUN_STRIDE;
        __global unsigned int* u =
            c->su + U_SU_RUN0 + r * U_SU_RUN_STRIDE;
        __global double* d = c->sd + CLS_SD_RUN0 + r * CLS_SD_RUN_STRIDE;
        u[UR_STRATEGY] = ri[rb + R_STRATEGY];
        u[UR_PHASE] = 0u;
        u[UR_RUSH] = 0u;
        u[UR_PKMODE] = 0u;
        u[UR_DTARGET] = 0u;
        u[UR_DGROUP] = 0u;
        u[UR_FIGHT] = 0u;
        u[UR_MOVEDIR] = 0u;
        u[UR_EXTRA] = 0u;
        u[UR_MLCT] = 0u;
        u[UR_SLOPEIDX] = ri[rb + R_SLOPE_IDX];
        u[UR_FIN] = 0u;
        u[UR_KAKARI] = 0u;
        u[UR_PACEDOWN] = 0u;
        u[UR_DOWNHILL] = ri[rb + R_INIT_DOWNHILL];
        u[UR_SPURT] = 0u;
        u[UR_DASH] = 1u;
        u[UR_MOVELANE] = 0u;
        u[UR_DUELACT] = 0u;
        u[UR_SPOTACT] = 0u;
        u[UR_SPOTSTART] = 0u;
        u[UR_ENTCLOSED] = 0u;
        u[UR_DISTEXIT] = 0u;
        u[UR_CUTOFF] = 0u;
        u[UR_PKRUSH] = 0u;
        u[UR_TEMPT] = 0u;
        u[UR_RSREM] = 0x3f800000u;
        u[UR_RSCNT] = 0u;
        u[UR_PREVPH] = 0u;
        u[UR_PREVFIGHT] = 0u;
        u[UR_PREVSPOT] = 0u;
        u[UR_SOLVER_HI] = ri[rb + R_SOLVER_HI];
        u[UR_SOLVER_LO] = ri[rb + R_SOLVER_LO];
        u[UR_HP_HI] = ri[rb + R_HP_HI];
        u[UR_HP_LO] = ri[rb + R_HP_LO];
        u[UR_PACE_HI] = ri[rb + R_PACE_HI];
        u[UR_PACE_LO] = ri[rb + R_PACE_LO];
        for (int k = 0; k < 2 * CLS_MAX_SLOPES; k++) {
            u[UR_HILL0 + k] = ri[rb + R_HILL0 + k];
        }
        u[UR_STARTCNT] =
            (unsigned int)(nh - ((int)u[UR_SLOPEIDX] > 0 ? 1 : 0));
        u[UR_ENDCNT] = (unsigned int)nh;
        d[DF_POS] = 0.0;
        d[DF_SPEED] = 3.0;
        d[DF_ACCEL] = 0.0;
        d[DF_TARGET] = 0.85 * rd[D_BASESPEED];
        d[DF_LANE] = rd[db + F_LANE0];
        d[DF_LANETARGET] = rd[db + F_LANE0];
        d[DF_HP] = rd[db + F_MAXHP];
        d[DF_MAXHP] = rd[db + F_MAXHP];
        d[DF_ACCUM] = 0.0;
        d[DF_DOWNHT] = 0.0;
        d[DF_KTIMER] = 0.0;
        d[DF_SPOTT] = 0.0;
        d[DF_DELAY] = rd[db + F_DELAY];
        d[DF_KSTART] = rd[db + F_KSTART];
        d[DF_KDUR] = rd[db + F_KDUR];
        for (int k = DF_TSA; k <= DF_ACE; k++) {
            d[k] = 0.0;
        }
        d[DF_ACA] = 24.0;
        d[DF_ONEFRAME] = 0.0;
        d[DF_LSS] = rd[db + F_LSS];
        d[DF_LSTTRANS] = -1.0;
        d[DF_MINSPEED] = rd[db + F_MINSPEED];
        for (int k = 0; k < 3; k++) {
            d[DF_BTS0 + k] = rd[db + F_BTS0 + k];
            d[DF_BACC0 + k] = rd[db + F_BACC0 + k];
            d[DF_BACC0 + 3 + k] = rd[db + F_BACC0 + 3 + k];
        }
        for (int k = 0; k < 25; k++) {
            d[DF_SECT0 + k] = rd[db + F_SECT0 + k];
        }
        d[DF_PKINT] = 0.0;
        d[DF_PKCOOL] = 0.0;
        d[DF_PKEFFEND] = 0.0;
        d[DF_PKEFFEXIT] = 0.0;
        d[DF_PKEND] = rd[db + F_PKEND];
        d[DF_PKCONT] = rd[db + F_PKCONT];
        d[DF_PKCOEF] = 1.0;
        d[DF_LCSPEED] = 0.0;
        d[DF_EXTRA] = 0.0;
        for (int k = DF_BFRONT; k <= DF_DCONT; k++) {
            d[k] = 0.0;
        }
        d[DF_WISDOM] = rd[db + F_WISDOM];
        d[DF_POWER] = rd[db + F_POWER];
        d[DF_PCT15] = rd[db + F_PCT15];
        d[DF_PCT20] = rd[db + F_PCT20];
        d[DF_SUBPAR] = rd[db + F_SUBPAR];
        d[DF_SLOPEPER] = rd[db + F_SLOPE0];
        d[DF_GROUNDMOD] = rd[D_GROUNDMOD];
        d[DF_GUTSMOD] = rd[db + F_GUTSMOD];
        d[DF_FINRAW] = 0.0;
        d[DF_FINSTEP] = 0.0;
        d[DF_DTS] = rd[db + F_DTS];
        d[DF_DA] = rd[db + F_DA];
        d[DF_SPOTM] = rd[db + F_SPOTM];
        d[DF_SPOTD] = rd[db + F_SPOTD];
        d[DF_BASESPEED] = rd[D_BASESPEED];
        d[DF_SECTION] = rd[D_SECTION];
        d[DF_PREVPOS] = 0.0;
        d[DF_EFF] = 0.0;
        d[DF_STYLECNT] = rd[db + F_STYLECNT];
        d[DF_NEXTPHASE] = rd[D_DISTANCE] * 1.0 / 6.0;
        d[DF_ENDCNT] = 0.0;
        d[DF_NIGECNT] = rd[db + F_NIGECNT];
        d[DF_SPURTT] = -1.0;
        d[DF_SPURTD] = 0.0;
        d[DF_SPURTTICK] = 0.0;
    }
    c->rp[CLS_RP_COMPLEN] = 0u;
    c->rp[CLS_RP_PHRCOUNT] = 0u;
    c->su[U_CAP] = 0u;
    c->su[U_EIM] = ri[CLS_RI_FLAGS] & 1u;
    c->su[U_DUELON] = ri[CLS_RI_DUELON] & 1u;
    c->su[U_PMVALID] = 0u;
    c->su[U_PMIDX] = 0u;
    c->su[U_PPRES] = 0u;
    for (int k = 0; k < CLS_MAXR; k++) {
        c->su[U_BLOCK + k] = 0u;
        c->su[U_TLIST + k] = 0u;
        c->su[U_OMAP + k] = 0u;
    }
    cc_field_update(c);
}

// Capture row (lib.rs:936-960). Finished runners hold frozen state, the hp
// column clamps at zero and the blocker is recomputed at capture time.
// Columns 6-13 are debug state the CPU replay does not expose, used to
// localise divergences and to prove the hill, pos-keep, spot and duel paths
// actually run.
static void cc_capture(Cc* c, __global double* fr, const int tick,
    const int nmax) {
    for (int r = 0; r < c->n; r++) {
        __global double* row = fr + ((long)tick * nmax + r) * CLS_FRAME_COLS;
        row[0] = SO(DF_POS, r);
        row[1] = SO(DF_SPEED, r);
        row[2] = fmax(SO(DF_HP, r), 0.0);
        row[3] = SO(DF_LANE, r);
        if (SUO(UR_FIN, r)) {
            row[4] = 0.0;
            row[5] = -1.0;
        } else {
            row[4] = (double)SUO(UR_RUSH, r);
            double gap = 0.0;
            const int b = cc_front_blocker(c, r, SO(DF_POS, r),
                SO(DF_LANE, r), &gap);
            row[5] = (double)b;
        }
        row[6] = SO(DF_TARGET, r);
        row[7] = SO(DF_LANETARGET, r);
        row[8] = SR(DF_PKCOEF);
        row[9] = (double)SU(UR_PKMODE);
        row[10] = (double)SU(UR_SPOTACT);
        row[11] = (double)SU(UR_DUELACT);
        row[12] = SO(DF_SLOPEPER, r);
        row[13] = (double)SUO(UR_DOWNHILL, r);
    }
}

static void cc_bits(__global unsigned int* st, const int slot, const double v) {
    const ulong b = as_ulong(v);
    st[slot] = (unsigned int)(b & 0xffffffffu);
    st[slot + 1] = (unsigned int)(b >> 32u);
}

// Reply channel emits. An overflow drops the record and flags the race.
static void cc_emit_comp(Cc* c, const unsigned int tick,
    const unsigned int kind, const unsigned int r) {
    const unsigned int len = c->rp[CLS_RP_COMPLEN];
    if (len + 3u > (unsigned int)CLS_RP_COMP_CAP) {
        c->rp_over = 1;
        return;
    }
    c->rp[CLS_RP_COMP0 + len] = tick;
    c->rp[CLS_RP_COMP0 + len + 1] = kind;
    c->rp[CLS_RP_COMP0 + len + 2] = r;
    c->rp[CLS_RP_COMPLEN] = len + 3u;
}

static void cc_emit_phr(Cc* c, const unsigned int r, const unsigned int phase) {
    const unsigned int len = c->rp[CLS_RP_PHRCOUNT];
    if (len + 2u > (unsigned int)CLS_RP_PHR_CAP) {
        c->rp_over = 1;
        return;
    }
    c->rp[CLS_RP_PHR0 + len] = r;
    c->rp[CLS_RP_PHR0 + len + 1] = phase;
    c->rp[CLS_RP_PHRCOUNT] = len + 2u;
}

// The kakari skillEvents entry the deactivate hook writes (lib.rs:695-701):
// runner, activation clock, measured duration.
static void cc_emit_kakari(Cc* c, const unsigned int r,
    const double start_t, const double end_t) {
    const unsigned int len = c->rp[CLS_RP_SKLEN];
    if (len + 5u > (unsigned int)CLS_RP_SK_CAP) {
        c->rp_over = 1;
        return;
    }
    const ulong t = as_ulong(start_t);
    const ulong d = as_ulong(end_t - start_t);
    c->rp[CLS_RP_SK0 + len] = r;
    c->rp[CLS_RP_SK0 + len + 1] = (unsigned int)(t & 0xffffffffu);
    c->rp[CLS_RP_SK0 + len + 2] = (unsigned int)(t >> 32u);
    c->rp[CLS_RP_SK0 + len + 3] = (unsigned int)(d & 0xffffffffu);
    c->rp[CLS_RP_SK0 + len + 4] = (unsigned int)(d >> 32u);
    c->rp[CLS_RP_SKLEN] = len + 5u;
}

// The race driver (lib.rs:876-970), one work item per race.
static void cc_run(Cc* c, __global const unsigned int* ri,
    __global unsigned int* st, __global double* fr) {
    const int nmax = (int)ri[CLS_RI_NMAX];
    const double dist = c->cd[D_DISTANCE];
    cc_prepare(c, ri);
    // Classic has no pre-loop frame. Each loop pass captures one frame after
    // advancing t, exactly like the engine (lib.rs:939-960, no capture at :855
    // for non-retail), so the replay holds `ticks` frames, not `ticks + 1`.
    double t = 0.0;
    int ticks = 0;
    c->rp_over = 0;
    for (;;) {
        int active = 0;
        for (int r = 0; r < c->n; r++) {
            if (!SUO(UR_FIN, r)) {
                active = 1;
                SO(DF_PREVPOS, r) = SO(DF_POS, r);
            }
            c->su[U_TICKRANK + r] = c->su[U_RANK + r];
        }
        if (!active || t >= CLS_MAX_RACE_SECONDS) {
            break;
        }
        cc_next_overtake(c, CLS_DT);
        for (int r = 0; r < c->n; r++) {
            if (!SUO(UR_FIN, r)) {
                SO(DF_EFF, r) = cc_step_forward(c, r, CLS_DT);
            }
        }
        cc_field_update(c);
        for (int r = 0; r < c->n; r++) {
            if (!SUO(UR_FIN, r) && SO(DF_EFF, r) > 0.0) {
                cc_step_lateral(c, r, SO(DF_EFF, r));
            }
        }
        for (int r = 0; r < c->n; r++) {
            if (!SUO(UR_FIN, r) && SO(DF_EFF, r) > 0.0) {
                cc_step_field_conditions(c, r, SO(DF_EFF, r));
            }
        }
        // Reply observables the engine records after step_all (lib.rs:883
        // and lib.rs:899-922): the style-group leaders over the tick-start
        // ranks, the rushing countdown, positioning phases, and the spot
        // and duel starts in roster order.
        for (int bs = 0; bs < 6; bs++) {
            c->su[U_GLEADER + bs] = 0xFFFFFFFFu;
        }
        for (int r = 0; r < c->n; r++) {
            const int bs = cc_base_style(cc_strategy(c, r));
            const unsigned int cur = c->su[U_GLEADER + bs];
            if (cur == 0xFFFFFFFFu ||
                c->su[U_TICKRANK + r] < c->su[U_TICKRANK + (int)cur]) {
                c->su[U_GLEADER + bs] = (unsigned int)r;
            }
        }
        for (int r = 0; r < c->n; r++) {
            if (SO(DF_SPURTT, r) >= 0.0 && SO(DF_SPURTTICK, r) == 0.0) {
                SO(DF_SPURTTICK, r) = (double)(ticks + 1);
            }
            if (SUO(UR_FIN, r)) {
                continue;
            }
            if (SUO(UR_KAKARI, r)) {
                float rem = as_float(SUO(UR_RSREM, r));
                rem -= (float)(1.0 / 15.0);
                if (rem <= 0.0f) {
                    rem = 1.0f;
                    SUO(UR_RSCNT, r) += 1u;
                }
                SUO(UR_RSREM, r) = as_uint(rem);
            }
            const unsigned int ph = SUO(UR_PHASE, r);
            if (ph != SUO(UR_PREVPH, r)) {
                SUO(UR_PREVPH, r) = ph;
                const int bs = cc_base_style((int)SUO(UR_STRATEGY, r));
                if (c->su[U_GLEADER + bs] == (unsigned int)r) {
                    cc_emit_phr(c, (unsigned int)r, ph);
                }
            }
        }
        for (int r = 0; r < c->n; r++) {
            if (!SUO(UR_FIN, r) && SUO(UR_SPOTSTART, r) &&
                !SUO(UR_PREVSPOT, r)) {
                SUO(UR_PREVSPOT, r) = 1u;
                cc_emit_comp(c, (unsigned int)ticks, 4u, (unsigned int)r);
            }
        }
        for (int r = 0; r < c->n; r++) {
            if (SUO(UR_FIN, r)) {
                continue;
            }
            while (SUO(UR_PREVFIGHT, r) < SUO(UR_FIGHT, r)) {
                SUO(UR_PREVFIGHT, r) += 1u;
                cc_emit_comp(c, (unsigned int)ticks, 5u, (unsigned int)r);
            }
        }
        for (int r = 0; r < c->n; r++) {
            if (SUO(UR_FIN, r) || SO(DF_POS, r) < dist) {
                continue;
            }
            const double prev = SO(DF_PREVPOS, r);
            const double span = SO(DF_POS, r) - prev;
            const double frac = span > 0.0 ? (dist - prev) / span : 0.0;
            SO(DF_FINRAW, r) = t + CLS_DT * frac;
            SO(DF_FINSTEP, r) = t + CLS_DT;
            SUO(UR_FIN, r) = 1u;
        }
        t += CLS_DT;
        cc_capture(c, fr, ticks, nmax);
        ticks++;
        if (ticks >= CLS_MAX_TICKS) {
            st[S_FALLBACK] = 2u;
            break;
        }
    }
    for (int r = 0; r < c->n; r++) {
        if (!SUO(UR_FIN, r)) {
            SO(DF_FINRAW, r) = t;
            SO(DF_FINSTEP, r) = t;
        }
        const int rb = S_FIN0 + r * CLS_ST_RUN_STRIDE;
        cc_bits(st, rb, SO(DF_FINRAW, r));
        cc_bits(st, rb + 2, SO(DF_FINSTEP, r));
        cc_bits(st, rb + S_LSTTRANS, SO(DF_LSTTRANS, r));
        cc_bits(st, rb + S_SPURTT,
            SO(DF_SPURTT, r) < 0.0 ? 0.0 : SO(DF_SPURTT, r));
        cc_bits(st, rb + S_SPURTD,
            SO(DF_SPURTT, r) < 0.0 ? 0.0 : SO(DF_SPURTD, r));
        st[rb + S_TEMPT] = SUO(UR_TEMPT, r);
        st[rb + S_RSCNT] = SUO(UR_RSCNT, r);
        st[rb + S_SPURTTICK] = (unsigned int)SO(DF_SPURTTICK, r);
    }
    st[S_TICKS] = (unsigned int)ticks;
    cc_bits(st, S_END, t);
    if (c->rp_over && !st[S_FALLBACK]) {
        st[S_FALLBACK] = 4u;
    }
}
