// Classic (non-retail) physics core. The float arithmetic mirrors the engine
// line for line (race_solver.rs, race_field.rs, hp_policy.rs, lane_selection.rs,
// overtaking.rs, starting_grid.rs cited inline). Construction (stats, rng
// seeding, section and duel constants) is host side in classic_exec.rs; the
// flag-only competition systems run here on the prepended powf and log10
// ports.

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
// Free slot, kept clear of the slope rng bank R_HILL0..R_HILL0+31.
#define R_TEAM 4

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
// The experimentalLaterStaminaMechanics switch (race_solver.rs:2751-2756).
#define D_LATER 10
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
#define F_SPOTP 61
#define F_POS0 62
// The raw stamina seed: wire stamina times the mood coefficient, before the
// stat adjustment (race_solver_builder.rs:278-298).
#define F_RAWST 63

// Status words, one block per race: fallback, ticks, loop end, per runner
// finish times as raw f64 bits.
#define CLS_ST_STRIDE (16 + 20 * CLS_MAXR)
#define S_FALLBACK 0
#define S_TICKS 1
#define S_END 2
#define CLS_ST_RUN0 16
#define CLS_ST_RUN_STRIDE 20
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
// Diagnostic activation counts for the flag-only competition systems, mirror
// of the scratch counters at runner 0.
#define S_STAT0 15
#define S_STAT1 16
#define S_STAT2 17
#define S_STAT3 18
#define S_STAT4 19

// Reply record streams, one region per race: competition starts as
// (tick, kind, runner) triples in engine push order (lib.rs:899-918) and
// positioning phase records as (runner, phase) pairs (lib.rs:886).
#define CLS_RP_COMPLEN 0
#define CLS_RP_PHRCOUNT 1
#define CLS_RP_SKLEN 2
#define CLS_RP_DETLEN 3
#define CLS_RP_TGTLEN 4
#define CLS_RP_CNSLEN 5
#define CLS_RP_COMP0 8
#define CLS_RP_COMP_CAP 2400
#define CLS_RP_PHR0 (CLS_RP_COMP0 + CLS_RP_COMP_CAP)
#define CLS_RP_PHR_CAP 800
#define CLS_RP_SK0 (CLS_RP_PHR0 + CLS_RP_PHR_CAP)
// Unified skillEvents stream in engine push order. Tag 0 is a skill entry,
// nine words: tag, ordinal, runner, sid, unique, t bits, duration bits.
// Tag 1 is a kakari entry, six words: tag, runner, t bits, duration bits.
#define CLS_RP_SK_CAP 9216
#define CLS_RP_DET0 (CLS_RP_SK0 + CLS_RP_SK_CAP)
#define CLS_RP_DET_CAP 2560
#define CLS_RP_TGT0 (CLS_RP_DET0 + CLS_RP_DET_CAP)
#define CLS_RP_TGT_CAP 6144
#define CLS_RP_CNS0 (CLS_RP_TGT0 + CLS_RP_TGT_CAP)
#define CLS_RP_CNS_CAP 768
#define CLS_RP_STRIDE (CLS_RP_CNS0 + CLS_RP_CNS_CAP)

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
// The three flag-only competition latches (race_solver.rs:634, 631-635), one
// bit each in the one free runner slot.
#define UR_KEEP 71
#define UR_KEEP_KEEP 1u
#define UR_KEEP_CMP 2u
#define UR_KEEP_SEC 4u
#define U_TGTMAP 9168
#define U_BLOCK 23568
#define U_TLIST 23696
#define U_OMAP 23824
// Ranks as cached at the loop-pass start (lib.rs:879-880) and the per-tick
// style-group leader scan (stadium_score.rs:20-30).
#define U_TICKRANK 23952
#define U_GLEADER 24080
// Per-race activation counts for the flag-only competition systems (host
// diagnostics, never part of a reply).
#define U_STAT_KEEP 24096
#define U_STAT_CMP 24097
#define U_STAT_SEC 24098
#define U_STAT_GATE 24099
#define U_STAT_SYB 24100
// Compiled skill table, then the classic skill runtime state (all per race).
#define CLS_SK_TABLE 24400
#define CLS_SKD (CLS_SK_TABLE + SK_STRIDE)
#define CLS_SKD_ORD CLS_SKD
#define CLS_SKD_QLEN (CLS_SKD + 1)
#define CLS_SKD_CONS (CLS_SKD + 16)
#define CLS_SKD_PEND (CLS_SKD + 976)
#define CLS_SKD_SLEN (CLS_SKD + 3856)
#define CLS_SKD_SEQ (CLS_SKD + 4456)
#define CLS_SKD_QMETA (CLS_SKD + 23656)
// Per-record prerequisite latches, set once the engine's per-tick watch holds.
#define CLS_SKD_PRE (CLS_SKD + 25704)
#define CLS_SU_STRIDE (CLS_SKD + 26664)
#define SK_PEND_SLOT 8
#define SK_CAT 32
#define SK_QMAX 2048
#define SK_WINMAXP 16
#define SK_FLAG_ASITAME 32768
#define SK_FLAG_DYN 131072
#define SK_FLAG_UNIQUE 65536
// The injected staminasyoubu conserve skill (race_solver_builder.rs:817-848):
// the scan gate reads live raw stamina and speed, and the TargetSpeed effect
// carries its live activation value.
#define SK_FLAG_SYOUBU 262144
// Instance token words: global effect table index (below 65536), the
// runner-local skill slot, the activation perspective, the source-side Noop
// placeholder, and the natural-deceleration flag.
#define SKT_E(e) (((unsigned int)(e)) & 0xffffu)
#define SKT_S(s) (((((unsigned int)(s)) & 0xfu) << 16))
#define SKT_GET_E(t) ((int)((t) & 0xffffu))
#define SKT_GET_S(t) ((int)(((t) >> 16) & 0xfu))
#define SKT_PERSP ((uint)1 << 20)
#define SKT_PLACE ((uint)1 << 21)
#define SKT_NAT ((uint)1 << 22)
// Skill instance categories, in the engine's expire walk order
// (race_solver.rs:1949-2000): target, current, accel, lane, aux.
#define SKC_TGT 0
#define SKC_CUR 1
#define SKC_ACC 2
#define SKC_LAN 3
#define SKC_AUX 4
#define SKC_COUNT 5
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
#define CLS_SD_RUN_STRIDE 120
// Skill runtime regions. Per-record windows plus the merged trigger and the
// runner-local skill slot, then per-slot cooldown clocks, per-instance timer
// values, the pending skillEvents pairs, and the flush queue durations.
#define UDB_SKW 55808
#define SKW_STRIDE 40
#define SKW_TRIG0 0
#define SKW_TRIG1 1
#define SKW_NWIN 2
#define SKW_SLOT 3
#define SKW_WIN0 8
#define UDB_SKCOOL (55808 + SKW_STRIDE * SK_RMAX)
#define UDB_SKMR (UDB_SKCOOL + SK_RMAX)
#define UDB_SKPEND (UDB_SKMR + CLS_MAXR * SKC_COUNT * SK_CAT)
#define UDB_SKQDUR (UDB_SKPEND + CLS_MAXR * SK_PEND_SLOT * 2)
#define SKW(rec) (c->sd + UDB_SKW + SKW_STRIDE * (rec))
#define SKCOOL(r, s) (c->sd[UDB_SKCOOL + (r) * SK_PEND_SLOT + (s)])
#define SKMR(r, cat, k) (c->sd[UDB_SKMR + ((r) * SKC_COUNT + (cat)) * SK_CAT + (k)])
#define SKSEQ(r, cat, k) (c->su[CLS_SKD_SEQ + ((r) * SKC_COUNT + (cat)) * SK_CAT + (k)])
#define SKSLEN(r, cat) (c->su[CLS_SKD_SLEN + (r) * SKC_COUNT + (cat)])
#define SKPEND(r, s) (c->su + CLS_SKD_PEND + ((r) * SK_PEND_SLOT + (s)) * 3)
#define SKPEND_T(r, s) (c->sd[UDB_SKPEND + ((r) * SK_PEND_SLOT + (s)) * 2])
#define SKPEND_D(r, s) (c->sd[UDB_SKPEND + ((r) * SK_PEND_SLOT + (s)) * 2 + 1])
#define CLS_SD_STRIDE (UDB_SKQDUR + SK_QMAX)
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
// Live horse stats. The engine's add_stat_modifier writes move these fields
// and the readers below consume them live (race_solver.rs:695-709, refresh
// at :1330-1340). DF_SPEEDST/DF_STAMINA/DF_GUTS start from the adjusted
// stats; DF_WISDOM and DF_POWER already carry the live readers.
#define DF_SPEEDST 103
#define DF_STAMINA 104
#define DF_GUTS 105
#define DF_SPOTP 106
// 107, not 106: sharing SPOTP's slot zeroes the spot duration for any
// runner whose distance aptitude reads as zero.
#define DF_DISTAPT 107
// The flag-only competition systems' per-runner state (race_solver.rs:2948-
// 3043): the captured target modifier, the countdown and the reuse wait.
#define DF_CMPM 108
#define DF_CMPT 109
#define DF_CMPCL 110
#define DF_SECM 111
#define DF_SECT 112
#define DF_SECCL 113
// The raw stamina the engine keeps beside the adjusted stat: it seeds the
// staminasyoubu gate and value, and kind 2 effects move it live
// (race_solver.rs:2558).
#define DF_RAWST 114

typedef struct cc_ctx {
    __global unsigned int* su;
    __global double* sd;
    __global const double* cd;
    __global unsigned int* rp;
    __global const unsigned int* ri;
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
static void cc_sk_pass(Cc* c, const int r);
static void cc_sk_scan(Cc* c, const int r, const int fieldless);
static void cc_sk_init(Cc* c);
static void cc_sk_flush(Cc* c);
static void cc_sk_finalize_pending(Cc* c);
static double cc_sk_vision(Cc* c, const int r);
static int cc_sk_lane_pref(Cc* c, const int r);
static int cc_sk_setlane(Cc* c, const int r, double* out);
static double cc_sk_lane_speed_sum(Cc* c, const int r);

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
    const double vision = cc_sk_vision(c, r);
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
    const double vision = cc_sk_vision(c, r);
    const int pref = cc_sk_lane_pref(c, r);
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
        const double skill =
            ((pref == 1 && gap > 0.0) || (pref == 2 && gap < 0.0)) ? 1.5 : 1.0;
        const double sc = fabs((gap * weight) * skill);
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
        double fixed;
        if (cc_sk_setlane(c, r, &fixed)) {
            SR(DF_LANETARGET) = fixed;
        } else {
            cc_update_lane_target(c, r, m, nt);
        }
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
    double actual_speed = fmin(fmax(SR(DF_LCSPEED) + cc_sk_lane_speed_sum(c, r),
        0.0), 0.6);
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
    SR(DF_CMPT) += d;
    SR(DF_CMPCL) += d;
    SR(DF_SECT) += d;
    SR(DF_SECCL) += d;
    for (int cat = 0; cat < SKC_COUNT; cat++) {
        const int len = (int)SKSLEN(r, cat);
        for (int k = 0; k < len; k++) {
            SKMR(r, cat, k) += d;
        }
    }
}

// step_forward (race_solver.rs:1175-1224). Timers swept here are
// accumulatetime, downhill, kakari, spot and the compete and secure lead
// clocks (race_solver.rs:978-982; the rest belong to skills).
// Returns the effective moving dt.
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
    cc_sk_pass(c, r);
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

// competition_distance_coefficient (race_solver.rs:142-156), shared by both
// stamina-cost formulas.
static double cc_competition_dist_coef(const double distance) {
    if (distance < 1801.0) {
        return 0.3;
    }
    if (distance < 2101.0) {
        return 0.5;
    }
    if (distance < 2201.0) {
        return 0.8;
    }
    if (distance < 2401.0) {
        return 1.0;
    }
    if (distance < 2601.0) {
        return 1.1;
    }
    return 1.2;
}

// is_solo_front_runner (race_field.rs:694-704): true when no other runner
// carries a Nige-family strategy.
static int cc_solo_front(Cc* c, const int r) {
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        const int s = cc_strategy(c, j);
        if (s == 1 || s == 5) {
            return 0;
        }
    }
    return 1;
}

// update_stamina_keep (race_solver.rs:2767-2781). The doc's own author calls
// this activation formula speculative; the port only mirrors it.
static void cc_stamina_keep(Cc* c, const int r, const double dt) {
    if ((SU(UR_KEEP) & UR_KEEP_KEEP) != 0u || c->cd[D_DISTANCE] <= 0.0) {
        return;
    }
    const double remaining_frac =
        fmax((c->cd[D_DISTANCE] - SR(DF_POS)) / c->cd[D_DISTANCE], 0.0);
    if (fmax(SR(DF_HP) / SR(DF_MAXHP), 0.0) >= remaining_frac * 1.04) {
        return;
    }
    const double wisdom = fmax(SR(DF_WISDOM), 1.0);
    const double pct = 30.0 * (wisdom / 1000.0 + gs_pow(wisdom, 0.03));
    const double per_tick =
        1.0 - gs_pow(1.0 - fmin(pct, 100.0) / 100.0, dt / 2.0);
    if (cc_random(&SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO)) < per_tick) {
        SU(UR_KEEP) |= UR_KEEP_KEEP;
        c->su[U_STAT_KEEP] += 1u;
    }
}

// update_compete_before_spurt (race_solver.rs:2948-2992). The near_count
// window mirrors race_field.rs:878-891, the leader gap race_field.rs:511-515.
static void cc_compete_before_spurt(Cc* c, const int r, const double dt) {
    if ((SU(UR_KEEP) & UR_KEEP_CMP) != 0u) {
        if (SR(DF_CMPT) >= 0.0) {
            SU(UR_KEEP) &= ~UR_KEEP_CMP;
            SR(DF_CMPCL) = -1.0;
            cc_add(&SR(DF_TSA), &SR(DF_TSE), -SR(DF_CMPM));
        }
        return;
    }
    const double section = SR(DF_SECTION);
    const double pos = SR(DF_POS);
    if (!(section * 10.0 <= pos && pos < section * 15.0) || SR(DF_CMPCL) < 0.0) {
        return;
    }
    if ((SU(UR_KEEP) & UR_KEEP_KEEP) != 0u) {
        c->su[U_STAT_GATE] += 1u;
        return;
    }
    const int strategy = (int)SU(UR_STRATEGY);
    const double threshold = (strategy == 1 || strategy == 5) ? 0.0
        : (strategy == 2 ? 2.5 : (strategy == 3 ? 5.0 : 10.0));
    const int leader = (int)c->su[U_ORDER];
    const double leader_pos = leader == r ? pos : SO(DF_POS, leader);
    const int far = fmax(leader_pos - pos, 0.0) > threshold;
    const double lane = SR(DF_LANE);
    const double tol = (double)(float)0.16666;
    int near = 0;
    for (int j = 0; j < c->n; j++) {
        if (j == r) {
            continue;
        }
        if (fabs(SO(DF_POS, j) - pos) <= 3.0 && fabs(SO(DF_LANE, j) - lane) <= tol) {
            near = 1;
        }
    }
    if (!(far || near)) {
        return;
    }
    // roll_competition_wisdom (race_solver.rs:1534-1538) draws either way.
    const double wisdom = fmax(SR(DF_WISDOM), 1.0);
    const double pct = fmax(15.0 * engine_log10(wisdom * 0.1), 0.0);
    const double per_tick =
        1.0 - gs_pow(1.0 - fmin(pct, 100.0) / 100.0, dt / 2.0);
    if (!(cc_random(&SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO)) < per_tick)) {
        return;
    }
    const double strat_coef = strategy == 5 ? 0.2 : (strategy == 1 ? 0.8 : 1.0);
    double solo_bonus = 1.0;
    if ((strategy == 1 || strategy == 5) && cc_solo_front(c, r)) {
        solo_bonus = strategy == 5 ? 2.0 : 1.1;
    }
    const double modifier = (gs_pow(fmax(SR(DF_POWER), 0.0) / 1500.0, 0.5) * 2.0
        + gs_pow(fmax(SR(DF_GUTS), 0.0) / 3000.0, 0.2))
        * 0.1 * strat_coef * solo_bonus;
    const double stamina_strat_coef = strategy == 5 ? 1.5 : (strategy == 1 ? 1.2 : 1.0);
    const double near_factor = (near && !far) ? 0.5 : 0.0;
    const double stamina_cost = 20.0
        * (stamina_strat_coef * cc_competition_dist_coef(c->cd[D_DISTANCE]) + near_factor)
        * 2.0;
    SU(UR_KEEP) |= UR_KEEP_CMP;
    c->su[U_STAT_CMP] += 1u;
    SR(DF_CMPM) = modifier;
    SR(DF_CMPT) = -2.0;
    cc_add(&SR(DF_TSA), &SR(DF_TSE), modifier);
    SR(DF_HP) = fmin(fmax(SR(DF_HP) - stamina_cost, 0.0), SR(DF_MAXHP));
}

// update_secure_lead (race_solver.rs:2998-3043), flat lead coefficient.
static void cc_secure_lead(Cc* c, const int r, const double dt) {
    if ((SU(UR_KEEP) & UR_KEEP_SEC) != 0u) {
        if (SR(DF_SECT) >= 0.0) {
            SU(UR_KEEP) &= ~UR_KEEP_SEC;
            SR(DF_SECCL) = -1.0;
            cc_add(&SR(DF_TSA), &SR(DF_TSE), -SR(DF_SECM));
        }
        return;
    }
    const double section = SR(DF_SECTION);
    const double pos = SR(DF_POS);
    if (!(section * 10.0 <= pos && pos < section * 15.0) || SR(DF_SECCL) < 0.0) {
        return;
    }
    if ((SU(UR_KEEP) & UR_KEEP_KEEP) != 0u) {
        c->su[U_STAT_GATE] += 1u;
        return;
    }
    const double gap_behind = cc_distance_to_behind(c, r);
    if (gap_behind < 0.0) {
        return;
    }
    const double desirable = 3.0 + 0.0003 * (c->cd[D_DISTANCE] + 1000.0);
    if (gap_behind >= desirable) {
        return;
    }
    if (cc_random(&SU(UR_SOLVER_HI), &SU(UR_SOLVER_LO)) >= 0.20 * (dt / 2.0)) {
        return;
    }
    const int strategy = (int)SU(UR_STRATEGY);
    const double strat_coef = strategy == 5 ? 0.2
        : (strategy == 3 ? 0.8 : 1.0);
    double solo_bonus = 1.0;
    if ((strategy == 1 || strategy == 5) && cc_solo_front(c, r)) {
        solo_bonus = strategy == 5 ? 7.0 : 2.0;
    }
    const double modifier = gs_pow(fmax(SR(DF_GUTS), 0.0) / 2000.0, 0.5)
        * 0.3 * strat_coef * solo_bonus;
    const double stamina_strat_coef = strategy == 5 ? 1.2 : (strategy == 1 ? 1.0 : 0.8);
    const double stamina_cost = 20.0 * stamina_strat_coef
        * cc_competition_dist_coef(c->cd[D_DISTANCE]) * 2.0;
    SU(UR_KEEP) |= UR_KEEP_SEC;
    c->su[U_STAT_SEC] += 1u;
    SR(DF_SECM) = modifier;
    SR(DF_SECT) = -2.0;
    cc_add(&SR(DF_TSA), &SR(DF_TSE), modifier);
    SR(DF_HP) = fmin(fmax(SR(DF_HP) - stamina_cost, 0.0), SR(DF_MAXHP));
}

static void cc_step_field_conditions(Cc* c, const int r, const double dt) {
    cc_update_near_lane_timers(c, r, dt);
    // update_competition_mechanics (race_solver.rs:2746-2758). The engine
    // gates the three stamina systems on the request flag, spot and duel
    // always run.
    const int later = c->cd[D_LATER] != 0.0;
    if (later) {
        cc_stamina_keep(c, r, dt);
    }
    cc_update_spot(c, r);
    if (later) {
        cc_compete_before_spurt(c, r, dt);
        cc_secure_lead(c, r, dt);
    }
    cc_update_dueling(c, r, dt);
}
// start_spot_struggle classic (race_solver.rs:2928-2946); the modifier and
// duration read live guts at entry, the strategy aptitude coefficient rides
// DF_SPOTP. The recomputed pair also feeds the deactivation subtract.
static void cc_start_spot(Cc* c, const int j) {
    const double guts = fmax(SO(DF_GUTS, j), 1.0);
    SO(DF_SPOTM, j) = gs_pow(500.0 * guts, 0.6) * 0.0001;
    SO(DF_SPOTD, j) = gs_pow(700.0 * guts, 0.5) * 0.012 * SO(DF_SPOTP, j);
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

// activate_retail_duel classic branch (race_solver.rs:3194-3214). The duel
// modifiers read live guts at entry, so a mid-race GutsUp moves them; the
// recomputed pair also feeds the deactivation subtract at :1672-1673.
static void cc_activate_duel(Cc* c, const int j) {
    const double guts = fmax(SO(DF_GUTS, j), 1.0);
    SO(DF_DTS, j) = gs_pow(200.0 * guts, 0.708) * 0.0001;
    SO(DF_DA, j) = gs_pow(160.0 * guts, 0.59) * 0.0001;
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
        u[UR_KEEP] = 0u;
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
        d[DF_POS] = rd[db + F_POS0];
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
        d[DF_SPEEDST] = rd[db + F_SPEED_STAT];
        d[DF_STAMINA] = rd[db + F_STAMINA];
        d[DF_RAWST] = rd[db + F_RAWST];
        d[DF_GUTS] = rd[db + F_GUTS];
        d[DF_SPOTP] = rd[db + F_SPOTP];
        d[DF_DISTAPT] = (double)(int)ri[rb + R_DISTAPT];
        for (int k = DF_CMPM; k <= DF_SECCL; k++) {
            d[k] = 0.0;
        }
    }
    c->rp[CLS_RP_COMPLEN] = 0u;
    c->rp[CLS_RP_PHRCOUNT] = 0u;
    c->su[U_CAP] = 0u;
    c->su[U_STAT_KEEP] = 0u;
    c->su[U_STAT_CMP] = 0u;
    c->su[U_STAT_SEC] = 0u;
    c->su[U_STAT_GATE] = 0u;
    c->su[U_STAT_SYB] = 0u;
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
    // The constructor skill pass (race_solver.rs:1008) runs once per runner
    // in roster order after all derived constants exist. The host applied
    // the self stat bumps already; everything else lands through the fire.
    c->ri = ri;
    cc_sk_init(c);
    for (int r = 0; r < c->n; r++) {
        cc_sk_scan(c, r, 1);
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

// The kakari skillEvents entry the deactivate hook writes (lib.rs:692-698),
// tagged 1 in the unified stream: runner, activation clock, measured
// duration. Kakari and numeric-skill activations share one engine stream.
static void cc_emit_kakari(Cc* c, const unsigned int r,
    const double start_t, const double end_t) {
    const unsigned int len = c->rp[CLS_RP_SKLEN];
    if (len + 6u > (unsigned int)CLS_RP_SK_CAP) {
        c->rp_over = 1;
        return;
    }
    const ulong t = as_ulong(start_t);
    const ulong d = as_ulong(end_t - start_t);
    __global unsigned int* p = c->rp + CLS_RP_SK0 + len;
    p[0] = 1u;
    p[1] = r;
    p[2] = (unsigned int)(t & 0xffffffffu);
    p[3] = (unsigned int)(t >> 32u);
    p[4] = (unsigned int)(d & 0xffffffffu);
    p[5] = (unsigned int)(d >> 32u);
    c->rp[CLS_RP_SKLEN] = len + 6u;
}

// ---------------------------------------------------------------------------
// Classic skill runtime. The activation model mirrors the classic (non-retail)
// engine exactly: a pending-record scan (race_solver.rs:2131-2182), per-kind
// effect application (:2539-2641), end-of-tick delivery of cross effects
// (race_field.rs:486-492), and the lib.rs:632-721 event hooks over one
// shared stream. The host compiler replays the builder's wisdom lottery
// (race_solver_builder.rs:1006-1011) and compiles lost entries dead, so the
// kernel sees no draw. There is no additional activation and no kakari
// extend, so the running-skill machinery stays empty.

#define SKRD(rec) (c->su + CLS_SK_TABLE + SK_RECORDS + SK_RSTRIDE * (rec))
#define SKEF(e) (c->su + CLS_SK_TABLE + SK_EFFECTS + SK_ESTRIDE * (e))
#define SKARMF(a) (c->su + CLS_SK_TABLE + SK_ARMS + SK_ASTRIDE * (a))
#define SKREGF(g) (c->su + CLS_SK_TABLE + SK_REGIONS + SK_GSTRIDE * (g))

static double cc_sk_f64(const __global unsigned int* w) {
    return sk_read_f64(w);
}

static int cc_sk_nrec(Cc* c) {
    return (int)c->su[CLS_SK_TABLE + SK_NREC];
}

// The category an effect lands in, following the engine's expire walk order
// (race_solver.rs:1953-1996). Non-timer kinds never push a token.
static int cc_sk_cat(const int kind) {
    if (kind == 27) return SKC_TGT;
    if (kind == 21 || kind == 22) return SKC_CUR;
    if (kind == 31) return SKC_ACC;
    if (kind == 28) return SKC_LAN;
    return SKC_AUX;
}

static int cc_sk_tokened(const int kind) {
    return kind == 27 || kind == 21 || kind == 22 || kind == 31 || kind == 28
        || kind == 8 || kind == 11 || kind == 12 || kind == 35 || kind == 38;
}

// The runner-local slot of a skill id, matching the sid slot assigned at
// init (the engine keys cooldowns and the pending map by skill id string).
static int cc_sk_slot_of_sid(Cc* c, const int r, const unsigned int sid) {
    const int nrec = cc_sk_nrec(c);
    for (int rec = 0; rec < nrec; rec++) {
        const __global unsigned int* rd = SKRD(rec);
        if ((int)rd[1] == r && rd[0] == sid) {
            return (int)SKW(rec)[SKW_SLOT];
        }
    }
    return -1;
}

// Init: the sid slot walk per runner, the merged window list per record, and
// the zeroed runtime state. Windows merge with the engine's region algebra
// (region.rs Union through sk_union, skills.cl:399-447) over the arms, whose
// compiled post lists already carry the pre-clipped chain.
static void cc_sk_init(Cc* c) {
    const int nrec = cc_sk_nrec(c);
    c->su[CLS_SKD_ORD] = 0u;
    c->su[CLS_SKD_QLEN] = 0u;
    for (int w = 0; w < SK_RMAX; w++) {
        c->su[CLS_SKD_CONS + w] = 0u;
        c->su[CLS_SKD_PRE + w] = 0u;
    }
    for (int w = 0; w < CLS_MAXR * SKC_COUNT * SK_CAT; w++) {
        c->sd[UDB_SKMR + w] = 0.0;
    }
    for (int w = 0; w < CLS_MAXR * SK_PEND_SLOT; w++) {
        c->sd[UDB_SKCOOL + w] = -1e300;
        c->sd[UDB_SKPEND + 2 * w] = 0.0;
        c->sd[UDB_SKPEND + 2 * w + 1] = 0.0;
        for (int f = 0; f < 3; f++) {
            c->su[CLS_SKD_PEND + w * 3 + f] = 0u;
        }
    }
    for (int w = 0; w < CLS_MAXR * SKC_COUNT; w++) {
        c->su[CLS_SKD_SLEN + w] = 0u;
    }
    for (int r = 0; r < c->n; r++) {
        int slot = -1;
        unsigned int last_sid = 0xFFFFFFFFu;
        int prev_rec = -1;
        for (int rec = 0; rec < nrec; rec++) {
            const __global unsigned int* rd = SKRD(rec);
            if ((int)rd[1] != r) {
                continue;
            }
            if (prev_rec >= 0 && slot >= SK_PEND_SLOT) {
                c->rp_over = 1;
            }
            (void)prev_rec;
            // The staminasyoubu record shares the sid 0 sentinel with the
            // asitame record but keys its own cooldown slot in the engine
            // (race_solver.rs:2437), so it always opens a fresh slot.
            if (rd[0] != last_sid || (rd[2] & SK_FLAG_SYOUBU) != 0u) {
                last_sid = rd[0];
                slot++;
            }
            if (slot >= SK_PEND_SLOT) {
                c->rp_over = 1;
            }
            double merged[2 * SK_MWINMAX];
            int nm = 0;
            const int arms_base = (int)rd[21];
            const int arms_len = (int)rd[20];
            for (int a = 0; a < arms_len && a < SK_ARMMAX; a++) {
                const __global unsigned int* ar = SKARMF(arms_base + a);
                if (ar[6] & SK_ARM_DEAD) {
                    continue;
                }
                double cur[2 * SK_WINMAX];
                int nc = 0;
                const int rl = (int)ar[4];
                const int rb = (int)ar[5];
                for (int g = 0; g < rl && g < SK_WINMAX; g++) {
                    cur[2 * nc] = cc_sk_f64(SKREGF(rb + g));
                    cur[2 * nc + 1] = cc_sk_f64(SKREGF(rb + g) + 2);
                    nc++;
                }
                if (nc == 0) {
                    continue;
                }
                double out[2 * SK_MWINMAX];
                const int mo = sk_union(out, merged, nm, cur, nc);
                for (int w = 0; w < mo * 2 && w < 2 * SK_MWINMAX; w++) {
                    merged[w] = out[w];
                }
                nm = mo;
            }
            __global double* wd = SKW(rec);
            for (int w = 0; w < SKW_STRIDE; w++) {
                wd[w] = 0.0;
            }
            wd[SKW_SLOT] = (double)slot;
            if (nm > SK_WINMAXP) {
                c->rp_over = 1;
                nm = SK_WINMAXP;
            }
            wd[SKW_NWIN] = (double)nm;
            for (int w = 0; w < nm * 2; w++) {
                wd[SKW_WIN0 + w] = merged[w];
            }
            // The pending trigger spans the merged list (skills.cl:590-597
            // pattern; the engine builds trigger from the merged windows).
            wd[SKW_TRIG0] = nm > 0 ? merged[0] : 1e300;
            wd[SKW_TRIG1] = nm > 0 ? merged[2 * (nm - 1) + 1] : 1e300;
        }
    }
}

// The activate-hook pending lookup lives here, but the remaining fold reads
// the live timers first (race_solver.rs:731-740, finite Self-side entries).
static double cc_sk_remaining(Cc* c, const int r, const int slot) {
    double rem = 0.0;
    for (int cat = 0; cat < SKC_COUNT; cat++) {
        const int len = (int)SKSLEN(r, cat);
        for (int k = 0; k < len; k++) {
            const unsigned int tok = SKSEQ(r, cat, k);
            if (SKT_GET_S(tok) != slot || (tok & SKT_PERSP) != 0u) {
                continue;
            }
            const double timer = SKMR(r, cat, k);
            if (timer == -INFINITY) {
                continue;
            }
            rem = fmax(rem, fmax(-timer, 0.0));
        }
    }
    return rem;
}

// The deactivate hook of the engine (lib.rs:685-721) after one removal: when
// the last Self-side instance of a numeric skill ends, its pending
// skillEvents entry finalizes with the activation-time modeled duration.
static void cc_sk_deactivate_hook(Cc* c, const int r, const int slot,
    const unsigned int tok) {
    if ((tok & SKT_PERSP) != 0u) {
        return;
    }
    if (cc_sk_remaining(c, r, slot) > 0.0) {
        return;
    }
    const unsigned int flags = SKPEND(r, slot)[1];
    if ((flags & 1u) == 0u) {
        return;
    }
    const unsigned int ordinal = SKPEND(r, slot)[0];
    const unsigned int sid = SKPEND(r, slot)[2];
    const double start_t = SKPEND_T(r, slot);
    const double dur = SKPEND_D(r, slot);
    SKPEND(r, slot)[1] = 0u;
    const unsigned int len = c->rp[CLS_RP_SKLEN];
    if (len + 9u > (unsigned int)CLS_RP_SK_CAP) {
        c->rp_over = 1;
        return;
    }
    const ulong tb = as_ulong(start_t);
    const ulong db = as_ulong(dur);
    __global unsigned int* p = c->rp + CLS_RP_SK0 + len;
    p[0] = 0u;
    p[1] = ordinal;
    p[2] = (unsigned int)r;
    p[3] = sid;
    p[4] = (flags & 2u) ? 1u : 0u;
    p[5] = (unsigned int)(tb & 0xffffffffu);
    p[6] = (unsigned int)(tb >> 32u);
    p[7] = (unsigned int)(db & 0xffffffffu);
    p[8] = (unsigned int)(db >> 32u);
    c->rp[CLS_RP_SKLEN] = len + 9u;
}

// Remove one expired entry like Vec::remove keeps order, then run the
// release math and the deactivate hook of its category.
static void cc_sk_expire_entry(Cc* c, const int r, const int cat, const int k,
    const int len) {
    const unsigned int tok = SKSEQ(r, cat, k);
    for (int w = k; w + 1 < len; w++) {
        SKSEQ(r, cat, w) = SKSEQ(r, cat, w + 1);
        SKMR(r, cat, w) = SKMR(r, cat, w + 1);
    }
    SKSLEN(r, cat) = (unsigned int)(len - 1);
    const int eff = SKT_GET_E(tok);
    const int kind = (int)SKEF(eff)[0];
    const double mod = cc_sk_f64(SKEF(eff) + 6);
    if (cat == SKC_TGT) {
        cc_add(&SR(DF_TSA), &SR(DF_TSE), -mod);
    } else if (cat == SKC_CUR) {
        cc_add(&SR(DF_CSA), &SR(DF_CSE), -mod);
        if ((tok & SKT_NAT) != 0u) {
            SR(DF_ONEFRAME) += mod;
        }
    } else if (cat == SKC_ACC) {
        cc_add(&SR(DF_ACA), &SR(DF_ACE), -mod);
    }
    cc_sk_deactivate_hook(c, r, SKT_GET_S(tok), tok);
}

// expire_skill_effects (race_solver.rs:1949-2000), classic branch: the
// history pass only touches the always-empty running list, then the five
// vectors expire back-to-front with per-entry hooks between removals.
static void cc_sk_expire(Cc* c, const int r) {
    for (int cat = 0; cat < SKC_COUNT; cat++) {
        int len = (int)SKSLEN(r, cat);
        for (int k = len - 1; k >= 0; k--) {
            if (SKMR(r, cat, k) >= 0.0) {
                cc_sk_expire_entry(c, r, cat, k, len);
                len--;
            }
        }
    }
}

// Aux presence and sums, recomputed at every read like the engine's live
// vec scans (race_solver.rs:1310-1323, :1356-1358, :1380-1381, :2454-2456).
// Source-side Noop placeholders never count.
static double cc_sk_vision(Cc* c, const int r) {
    double v = 0.0;
    const int len = (int)SKSLEN(r, SKC_AUX);
    for (int k = 0; k < len; k++) {
        const unsigned int tok = SKSEQ(r, SKC_AUX, k);
        if ((tok & SKT_PLACE) != 0u) {
            continue;
        }
        if ((int)SKEF(SKT_GET_E(tok))[0] == 8) {
            v += cc_sk_f64(SKEF(SKT_GET_E(tok)) + 6);
        }
    }
    return fmax(20.0 + v, 0.1);
}

static int cc_sk_lane_pref(Cc* c, const int r) {
    int outside = 0;
    int inside = 0;
    const int len = (int)SKSLEN(r, SKC_AUX);
    for (int k = 0; k < len; k++) {
        const unsigned int tok = SKSEQ(r, SKC_AUX, k);
        if ((tok & SKT_PLACE) != 0u) {
            continue;
        }
        const int kind = (int)SKEF(SKT_GET_E(tok))[0];
        if (kind == 11) outside = 1;
        if (kind == 12) inside = 1;
    }
    return outside | (inside << 1);
}

// The latest SetTargetLane entry wins (race_solver.rs:1356-1358 rev().find()).
static int cc_sk_setlane(Cc* c, const int r, double* out) {
    const int len = (int)SKSLEN(r, SKC_AUX);
    for (int k = len - 1; k >= 0; k--) {
        const unsigned int tok = SKSEQ(r, SKC_AUX, k);
        if ((tok & SKT_PLACE) != 0u) {
            continue;
        }
        const int eff = SKT_GET_E(tok);
        if ((int)SKEF(eff)[0] == 35) {
            *out = cc_sk_f64(SKEF(eff) + 6);
            return 1;
        }
    }
    return 0;
}

static double cc_sk_lane_speed_sum(Cc* c, const int r) {
    double s = 0.0;
    const int len = (int)SKSLEN(r, SKC_LAN);
    for (int k = 0; k < len; k++) {
        s += cc_sk_f64(SKEF(SKT_GET_E(SKSEQ(r, SKC_LAN, k))) + 6);
    }
    return s;
}

static int cc_sk_immune(Cc* c, const int r) {
    const int len = (int)SKSLEN(r, SKC_AUX);
    for (int k = 0; k < len; k++) {
        const unsigned int tok = SKSEQ(r, SKC_AUX, k);
        if ((tok & SKT_PLACE) != 0u) {
            continue;
        }
        if ((int)SKEF(SKT_GET_E(tok))[0] == 38) {
            return 1;
        }
    }
    return 0;
}

// skill_targets (race_field.rs:407-463). Ranks come from the tick-start
// order, matching the engine's borrow of field.order during step_forward.
// out gets the kept indices; returns the count. cap is at least CLS_MAXR.
static int cc_sk_targets(Cc* c, const int r, const int target,
    const int value, const int debuff, int* out) {
    int cand[CLS_MAXR];
    int n = 0;
    const int rank = cc_rank(c, r);
    if (target == 1) {
        cand[n++] = r;
    } else if (target == 4) {
        for (int k = 0; k < c->n; k++) {
            const int j = (int)c->su[U_ORDER + k];
            if (j == r) {
                continue;
            }
            const double gap = SO(DF_POS, j) - SR(DF_POS);
            const double vision = cc_sk_vision(c, r);
            if (gap >= 0.0 && gap <= vision &&
                fabs(SO(DF_LANE, j) - SR(DF_LANE)) <=
                    (gap / vision * 11.5 + 2.0) * CLS_HORSE_LANE / 2.0) {
                cand[n++] = j;
            }
        }
    } else if (target == 9) {
        const int limit = value;
        for (int k = rank - 2; k >= 0 && n < limit; k--) {
            const int j = (int)c->su[U_ORDER + k];
            if (j != r) {
                cand[n++] = j;
            }
        }
        n = n < limit ? n : limit;
    } else if (target == 10) {
        const int limit = value;
        for (int k = rank; k < c->n && n < limit; k++) {
            const int j = (int)c->su[U_ORDER + k];
            if (j != r) {
                cand[n++] = j;
            }
        }
        n = n < limit ? n : limit;
    }
    if (target == 4) {
        const int limit = value;
        if (n > limit) {
            n = limit;
        }
    }
    int m = 0;
    const unsigned int team_r =
        c->ri[CLS_RI_RUN0 + r * CLS_RI_RUN_STRIDE + R_TEAM];
    for (int k = 0; k < n; k++) {
        const int j = cand[k];
        if (debuff != 0) {
            if (cc_sk_immune(c, j) != 0) {
                continue;
            }
            const unsigned int team_j =
                c->ri[CLS_RI_RUN0 + j * CLS_RI_RUN_STRIDE + R_TEAM];
            if (j != r && team_r != 0u && team_j == team_r) {
                continue;
            }
        }
        out[m++] = j;
    }
    return m;
}

// push one instance; overflow flags the race (a 32-deep category is far
// past any compiled skill's footprint).
static void cc_sk_push(Cc* c, const int r, const int cat,
    const unsigned int tok, const double timer) {
    const int len = (int)SKSLEN(r, cat);
    if (len >= SK_CAT) {
        c->rp_over = 1;
        return;
    }
    SKSEQ(r, cat, len) = tok;
    SKMR(r, cat, len) = timer;
    SKSLEN(r, cat) = (unsigned int)(len + 1);
}

static double cc_sk_new_timer(const double duration) {
    return duration < 0.0 ? -INFINITY : -duration;
}

// Construction constants the refresh recomputes from live stats, mirroring
// race_solver.rs:43-88 and hp_policy.rs:20.
__constant double CC_SPEED_STRAT[6][3] = {
    {0.0, 0.0, 0.0},
    {1.0, 0.98, 0.962},
    {0.978, 0.991, 0.975},
    {0.938, 0.998, 0.994},
    {0.931, 1.0, 1.0},
    {1.063, 0.962, 0.95},
};
__constant double CC_SPEED_DIST[8] =
    {1.05, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1};
__constant double CC_ACCEL_STRAT[6][3] = {
    {0.0, 0.0, 0.0},
    {1.0, 1.0, 0.996},
    {0.985, 1.0, 0.996},
    {0.975, 1.0, 1.0},
    {0.945, 1.0, 0.997},
    {1.17, 0.94, 0.956},
};
__constant double CC_ACCEL_GROUND[8] =
    {1.05, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1};
__constant double CC_ACCEL_DIST[8] =
    {1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4};
__constant double CC_HP_STRAT[6] = {0.0, 0.95, 0.89, 1.0, 0.995, 0.86};

// Mid-race refresh_stat_effects (race_solver.rs:1330-1340). The classic
// readers of these constants are DF_BTS0..2, DF_LSS, DF_MINSPEED, DF_BACC0..5
// and the hp ceiling; the duel and spot constants are construction-time.
static void cc_sk_refresh(Cc* c, const int r) {
    const double guts = SR(DF_GUTS);
    const double power = SR(DF_POWER);
    const double speed = SR(DF_SPEEDST);
    const double stamina = SR(DF_STAMINA);
    const double wisdom = SR(DF_WISDOM);
    const int strategy = (int)SU(UR_STRATEGY);
    const int distapt = (int)SR(DF_DISTAPT);
    const int surfapt = (int)c->ri[CLS_RI_RUN0 + r * CLS_RI_RUN_STRIDE + R_SURFAPT];
    const double base = SR(DF_BASESPEED);
    const double dist = c->cd[D_DISTANCE];
    SR(DF_MINSPEED) = 0.85 * base + sqrt(200.0 * guts) * 0.001;
    const double spd_root = sqrt(500.0 * speed);
    for (int p = 0; p < 3; p++) {
        SR(DF_BTS0 + p) = base * CC_SPEED_STRAT[strategy][p] +
            (p == 2 ? 1.0 : 0.0) * spd_root * CC_SPEED_DIST[distapt] * 0.002;
    }
    SR(DF_LSS) = (SR(DF_BTS0 + 2) + 0.01 * base) * 1.05 +
        spd_root * CC_SPEED_DIST[distapt] * 0.002 +
        gs_pow(450.0 * guts, 0.597) * 0.0001;
    for (int slot = 0; slot < 6; slot++) {
        const double konst = slot > 2 ? 0.0004 : 0.0006;
        const int p = slot % 3;
        SR(DF_BACC0 + slot) = konst * sqrt(500.0 * power) *
            CC_ACCEL_STRAT[strategy][p] *
            CC_ACCEL_GROUND[surfapt] * CC_ACCEL_DIST[distapt];
    }
    const double new_max = 0.8 * CC_HP_STRAT[strategy] * stamina + dist;
    const double spent = SR(DF_MAXHP) - SR(DF_HP);
    SR(DF_MAXHP) = new_max;
    SR(DF_HP) = fmin(fmax(new_max - spent, 0.0), new_max);
    SR(DF_GUTSMOD) = 1.0 + 200.0 / sqrt(600.0 * guts);
    SR(DF_SUBPAR) = rint((15.0 + 0.05 * wisdom) * 1000.0);
}

// One applied effect on the activation or delivery path (the per-kind match
// at race_solver.rs:2539-2641). fieldless marks the constructor pass, where
// the host already consumed the stat bumps and the kakari chance bump
// (recovery included: the engine lands it before hp init, which then
// re-seeds full hp, so it is net zero).
static void cc_sk_apply_effect(Cc* c, const int r, const int eff,
    const int slot, const int fieldless, const int own, const double dur,
    const unsigned int rec_flags) {
    const __global unsigned int* ef = SKEF(eff);
    const int kind = (int)ef[0];
    double mod = cc_sk_f64(ef + 6);
    // The staminasyoubu wire modifier 0 is replaced at activation by the
    // live formula (race_solver.rs:2496-2498, race_solver_builder.rs:181-184):
    // the packed factor times sqrt(raw stamina - 1200) times 0.0085.
    if (kind == 27 && (rec_flags & SK_FLAG_SYOUBU) != 0u) {
        mod = (sqrt(fmax(SR(DF_RAWST) - 1200.0, 0.0)) * 0.0085) * mod;
    }
    const unsigned int tok = SKT_E(eff) | SKT_S(slot) | (own ? 0u : SKT_PERSP);
    if (fieldless) {
        if (kind == 1 || kind == 2 || kind == 3 || kind == 4 || kind == 5 ||
            kind == 9 || kind == 29) {
            return;
        }
    }
    if (kind == 21 || kind == 22) {
        cc_add(&SR(DF_CSA), &SR(DF_CSE), mod);
        cc_sk_push(c, r, SKC_CUR,
            tok | ((kind == 22) ? SKT_NAT : 0u), cc_sk_new_timer(dur));
    } else if (kind == 27) {
        cc_add(&SR(DF_TSA), &SR(DF_TSE), mod);
        cc_sk_push(c, r, SKC_TGT, tok, cc_sk_new_timer(dur));
    } else if (kind == 31) {
        cc_add(&SR(DF_ACA), &SR(DF_ACE), mod);
        cc_sk_push(c, r, SKC_ACC, tok, cc_sk_new_timer(dur));
    } else if (kind == 28) {
        cc_sk_push(c, r, SKC_LAN, tok, cc_sk_new_timer(dur));
    } else if (kind == 8 || kind == 11 || kind == 12 || kind == 35 ||
        kind == 38) {
        cc_sk_push(c, r, SKC_AUX, tok, cc_sk_new_timer(dur));
    } else if (kind == 10) {
        SR(DF_DELAY) *= mod;
    } else if (kind == 14) {
        SR(DF_DELAY) = mod;
    } else if (kind == 9) {
        SR(DF_HP) = fmax(fmin(SR(DF_MAXHP),
            SR(DF_HP) + SR(DF_MAXHP) * mod), 0.0);
        // An HP-recovery effect clears Stamina Keep on whoever applies it
        // (race_solver.rs:2632).
        SU(UR_KEEP) &= ~UR_KEEP_KEEP;
        if (mod > 0.0 && (int)SU(UR_PHASE) >= 2 && !SU(UR_SPURT)) {
            SR(DF_LSTTRANS) = -1.0;
            cc_update_last_spurt(c, r);
        }
    } else if (kind == 1 || kind == 2 || kind == 3 || kind == 4 || kind == 5) {
        if (kind == 1) {
            SR(DF_SPEEDST) = fmax(SR(DF_SPEEDST) + mod, 1.0);
        } else if (kind == 2) {
            SR(DF_STAMINA) = fmax(SR(DF_STAMINA) + mod, 1.0);
            // The engine moves the raw stat beside the adjusted one
            // (race_solver.rs:2558), feeding the staminasyoubu gate and value.
            SR(DF_RAWST) = fmax(SR(DF_RAWST) + mod, 1.0);
        } else if (kind == 3) {
            SR(DF_POWER) = fmax(SR(DF_POWER) + mod, 1.0);
        } else if (kind == 4) {
            SR(DF_GUTS) = fmax(SR(DF_GUTS) + mod, 1.0);
        } else {
            SR(DF_WISDOM) = fmax(SR(DF_WISDOM) + mod, 1.0);
        }
    }
}

// The activate-hook tail (lib.rs:632-680) after one direct activation of a
// numeric skill, plus the detail and target events the wrapper writes
// (race_solver.rs:2448-2451). The asitame record instead records the
// conserve release with no skillEvents entry at all (lib.rs:634-641).
static void cc_sk_hook_activate(Cc* c, const int r, const int slot,
    const unsigned int sid, const unsigned int flags, const int detail) {
    const int numeric = (flags & SK_FLAG_ASITAME) == 0u && sid != 0u;
    if ((flags & SK_FLAG_ASITAME) != 0u) {
        const unsigned int len = c->rp[CLS_RP_CNSLEN];
        if (len + 3u > (unsigned int)CLS_RP_CNS_CAP) {
            c->rp_over = 1;
            return;
        }
        const ulong tb = as_ulong(SR(DF_ACCUM));
        __global unsigned int* p = c->rp + CLS_RP_CNS0 + len;
        p[0] = (unsigned int)r;
        p[1] = (unsigned int)(tb & 0xffffffffu);
        p[2] = (unsigned int)(tb >> 32u);
        c->rp[CLS_RP_CNSLEN] = len + 3u;
    }
    if (!numeric) {
        return;
    }
    const double rem = cc_sk_remaining(c, r, slot);
    if (rem <= 0.0) {
        unsigned int ord = c->su[CLS_SKD_ORD];
        c->su[CLS_SKD_ORD] = ord + 1u;
        const unsigned int len = c->rp[CLS_RP_SKLEN];
        if (len + 9u > (unsigned int)CLS_RP_SK_CAP) {
            c->rp_over = 1;
            return;
        }
        const ulong tb = as_ulong(SR(DF_ACCUM));
        __global unsigned int* p = c->rp + CLS_RP_SK0 + len;
        p[0] = 0u;
        p[1] = ord;
        p[2] = (unsigned int)r;
        p[3] = sid;
        p[4] = (flags & SK_FLAG_UNIQUE) ? 1u : 0u;
        p[5] = (unsigned int)(tb & 0xffffffffu);
        p[6] = (unsigned int)(tb >> 32u);
        p[7] = 0u;
        p[8] = 0u;
        c->rp[CLS_RP_SKLEN] = len + 9u;
    } else {
        unsigned int ord = SKPEND(r, slot)[0];
        if ((SKPEND(r, slot)[1] & 1u) == 0u) {
            ord = c->su[CLS_SKD_ORD];
            c->su[CLS_SKD_ORD] = ord + 1u;
        }
        SKPEND(r, slot)[0] = ord;
        SKPEND(r, slot)[1] = 1u | ((flags & SK_FLAG_UNIQUE) ? 2u : 0u);
        SKPEND(r, slot)[2] = sid;
        SKPEND_T(r, slot) = SR(DF_ACCUM);
        SKPEND_D(r, slot) = rem;
    }
    // skillDetailEvents and skillTargetEvents (race_solver.rs:2448-2451;
    // the reply renders them per runner in push order, lib.rs:1078-1094).
    {
        const unsigned int len = c->rp[CLS_RP_DETLEN];
        if (len + 5u > (unsigned int)CLS_RP_DET_CAP) {
            c->rp_over = 1;
        } else {
            const ulong tb = as_ulong(SR(DF_ACCUM));
            __global unsigned int* p = c->rp + CLS_RP_DET0 + len;
            p[0] = (unsigned int)r;
            p[1] = sid;
            p[2] = (unsigned int)detail;
            p[3] = (unsigned int)(tb & 0xffffffffu);
            p[4] = (unsigned int)(tb >> 32u);
            c->rp[CLS_RP_DETLEN] = len + 5u;
        }
    }
}

// Fire one record: the per-effect loop of apply_activation (2458-2641) with
// the classic value chain compiled in, the cross queue of queue_skill_effect
// (race_field.rs:465-484), the source-side Noop placeholder, and the hooks.
// The duration_usage 3 HP tiers (race_solver.rs:283-290): the first bound the
// frozen ledger value is strictly below, else 4.0.
static double cc_sk_du3_coeff(const double hp) {
    if (hp < 2000.0) { return 1.; }
    if (hp < 2400.0) { return 1.5; }
    if (hp < 2600.0) { return 2.; }
    if (hp < 2800.0) { return 2.200000047683716; }
    if (hp < 3000.0) { return 2.5; }
    if (hp < 3200.0) { return 3.; }
    if (hp < 3500.0) { return 3.5; }
    return 4.0;
}

static void cc_sk_fire(Cc* c, const int r, const int rec, const int fieldless) {
    const __global unsigned int* rd = SKRD(rec);
    const unsigned int sid = rd[0];
    const unsigned int flags = rd[2];
    const int detail = (int)rd[26];
    const int eff_base = (int)rd[6];
    const int n_eff = (int)rd[5];
    const int slot = (int)SKW(rec)[SKW_SLOT];
    // duration_coefficient reads the ledger value the engine freezes before
    // the effect loop (race_solver.rs:2466).
    const double act_hp = SR(DF_HP);
    // The wrapper only records DET and TGT events for numeric skill ids
    // (race_solver.rs:2448-2451).
    const int numeric = (flags & SK_FLAG_ASITAME) == 0u && sid != 0u;
    if (numeric && n_eff <= 31) {
        const unsigned int len = c->rp[CLS_RP_TGTLEN];
        if (len + 6u > (unsigned int)CLS_RP_TGT_CAP) {
            c->rp_over = 1;
        } else {
            const ulong tb = as_ulong(SR(DF_ACCUM));
            __global unsigned int* p = c->rp + CLS_RP_TGT0 + len;
            p[0] = (unsigned int)r;
            p[1] = sid;
            p[2] = (unsigned int)detail;
            p[3] = (unsigned int)(tb & 0xffffffffu);
            p[4] = (unsigned int)(tb >> 32u);
            p[5] = (unsigned int)n_eff;
            c->rp[CLS_RP_TGTLEN] = len + 6u;
        }
    }
    if (numeric && n_eff > 31) {
        c->rp_over = 1;
    }
    for (int e = 0; e < n_eff; e++) {
        const __global unsigned int* ef = SKEF(eff_base + e);
        const int kind = (int)ef[0];
        const int target = (int)ef[1];
        const int debuff = (int)ef[6];
        // The classic branch scales base duration by the tier coefficient and
        // the course factor in f64 (race_solver.rs:2508), left to right.
        double dur = cc_sk_f64(ef + 8);
        if (ef[3] == 3u) {
            dur *= cc_sk_du3_coeff(act_hp);
        }
        dur *= c->cd[D_DISTANCE] / 1000.0;
        int sel[CLS_MAXR];
        int ns = 0;
        // Selection precedes delivery on every caster activation and covers
        // the self target too (race_solver.rs:2479-2484). The constructor
        // pass has no field, so every list stays null there.
        if (!fieldless && kind != 0 && kind != 6) {
            if (target == 1) {
                ns = debuff != 0 && cc_sk_immune(c, r) != 0 ? 0 : 1;
                if (ns == 1) {
                    sel[0] = r;
                }
            } else {
                ns = cc_sk_targets(c, r, target, (int)ef[4], debuff, sel);
            }
        }
        if (numeric && n_eff <= 31) {
            const unsigned int len = c->rp[CLS_RP_TGTLEN];
            const int nullish = fieldless || kind == 0 || kind == 6;
            if (len + 1u + (nullish ? 0u : (unsigned int)ns) >
                (unsigned int)CLS_RP_TGT_CAP) {
                c->rp_over = 1;
            } else {
                c->rp[CLS_RP_TGT0 + len] =
                    nullish ? 0xFFFFFFFFu : (unsigned int)ns;
                for (int k = 0; k < (nullish ? 0 : ns); k++) {
                    c->rp[CLS_RP_TGT0 + len + 1 + k] = (unsigned int)sel[k];
                }
                c->rp[CLS_RP_TGTLEN] =
                    len + 1u + (nullish ? 0u : (unsigned int)ns);
            }
        }
        if (target == 1 && debuff != 0 && cc_sk_immune(c, r) != 0) {
            continue;
        }
        if (target != 1) {
            const int q = (int)c->su[CLS_SKD_QLEN];
            for (int k = 0; k < ns; k++) {
                if (q + k >= SK_QMAX) {
                    c->rp_over = 1;
                    break;
                }
                c->su[CLS_SKD_QMETA + q + k] =
                    (unsigned int)r |
                    ((unsigned int)sel[k] << 7) |
                    ((unsigned int)e << 14) |
                    ((unsigned int)rec << 18);
                c->sd[UDB_SKQDUR + q + k] = dur;
            }
            if (ns > 0) {
                c->su[CLS_SKD_QLEN] = (unsigned int)(q + ns);
            }
            if (dur > 0.0) {
                cc_sk_push(c, r, SKC_AUX,
                    SKT_E(eff_base + e) | SKT_S(slot) | SKT_PLACE,
                    cc_sk_new_timer(dur));
            }
            continue;
        }
        cc_sk_apply_effect(c, r, eff_base + e, slot, fieldless, 1, dur, flags);
    }
    // The hook runs with the post-apply state (race_solver.rs:2673 fires
    // after the whole effect list), and asitame never carries skill events.
    cc_sk_hook_activate(c, r, slot, sid, flags, detail);
}

// The live-condition gate. The engine folds the condition tree's dynamic
// checks into one Or closure over per-arm checks
// (activation_conditions.rs:1454-1464): an arm passes when the position sits
// in its own filtered chain and all of its comparisons hold. The guard at
// activation_conditions.rs:303 rejects every field-read comparison in the
// fieldless constructor pass, so those readers answer NaN there and the NaN
// check fails every comparison, `!=` included.

// The engine's is_visible cone (retail_lane.rs:484-499), the same one the
// retail lane machine ports.
static int cc_is_visible(const float gap, const float lane_gap, const float vision) {
    if (vision <= 0. || !(gap >= 0. && gap <= vision)) {
        return 0;
    }
    // Keep the width expression double until the final f32 store, or an
    // intermediate rounding admits a runner just outside the cone.
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

static double cc_feat_nan(void) {
    return (double)as_float(0x7fc00000u);
}

// The scan gate's live predicate values, one FEAT_* code per dynamic leaf.
// hp_per answers the classic ratio branch and accumulatetime the f32-floored
// clock; both read solver state only, so the fieldless pass evaluates them.
// The blocked-side timer (activation_conditions.rs:1294) is solver state too.
static double cc_sk_feat_value(const uint code, Cc* c, const int r, const int fieldless) {
    const uint feat = code >> 3;
    switch (feat) {
    case FEAT_CLOCK:
        return floor((double)convert_float_rte((float)SR(DF_ACCUM)));
    case FEAT_HP_PER:
        return fmax(SR(DF_HP) / SR(DF_MAXHP), 0.0);
    case FEAT_BSIDE:
        return trunc(SR(DF_BSIDE));
    case FEAT_ORDER:
    case FEAT_ORDER_RATE:
        if (fieldless != 0) {
            return cc_feat_nan();
        }
        return (double)cc_rank(c, r);
    case FEAT_NEAR: {
        if (fieldless != 0) {
            return cc_feat_nan();
        }
        const double pos = SR(DF_POS);
        const double lane = SR(DF_LANE);
        const double lane_lim = (double)0.16666f;
        int cnt = 0;
        for (int j = 0; j < c->n; j++) {
            if (j == r) {
                continue;
            }
            if (fabs(SO(DF_POS, j) - pos) <= 3.0 &&
                fabs(SO(DF_LANE, j) - lane) <= lane_lim) {
                cnt++;
            }
        }
        return (double)cnt;
    }
    case FEAT_NEARIF: {
        if (fieldless != 0) {
            return cc_feat_nan();
        }
        const double pos = SR(DF_POS);
        const double lane = SR(DF_LANE);
        const double lane_lim = (double)0.16666f;
        int cnt = 0;
        for (int j = 0; j < c->n; j++) {
            if (j == r) {
                continue;
            }
            const double gap = SO(DF_POS, j) - pos;
            if (gap > 0.0 && gap <= 3.0 && fabs(SO(DF_LANE, j) - lane) <= lane_lim) {
                cnt++;
            }
        }
        return (double)cnt;
    }
    case FEAT_SURR: {
        if (fieldless != 0) {
            return cc_feat_nan();
        }
        const double pos = SR(DF_POS);
        const double lane = SR(DF_LANE);
        int out = 0;
        int front = 0;
        int behind = 0;
        for (int j = 0; j < c->n; j++) {
            if (j == r) {
                continue;
            }
            const double dg = (double)convert_float_rte(SO(DF_POS, j) - pos);
            const double lg = (double)convert_float_rte(SO(DF_LANE, j) - lane);
            if (fabs(dg) <= 1.5 && lg >= 0.0 && lg <= (double)0.16666f) out = 1;
            if (dg > 0.0 && dg <= 3.0 && fabs(lg) <= (double)0.08333f) front = 1;
            if (dg >= -3.0 && dg < 0.0 && fabs(lg) <= (double)0.08333f) behind = 1;
            if (out && front && behind) break;
        }
        return out && front && behind ? 1.0 : 0.0;
    }
    case FEAT_VISN: {
        if (fieldless != 0) {
            return cc_feat_nan();
        }
        const double pos = SR(DF_POS);
        const double lane = SR(DF_LANE);
        const float vision = (float)cc_sk_vision(c, r);
        int cnt = 0;
        for (int j = 0; j < c->n; j++) {
            if (j == r) {
                continue;
            }
            if (cc_is_visible(convert_float_rte(SO(DF_POS, j) - pos),
                convert_float_rte(SO(DF_LANE, j) - lane), vision)) {
                cnt++;
            }
        }
        return (double)cnt;
    }
    default:
        return 0.0;
    }
}

// The engine's comparisons. order_rate rounds the field-size threshold
// half away from zero; the classic hp_per branch keeps the ratio and divides
// the percent argument (activation_conditions.rs:612-624).
static int cc_sk_feat_cmp(const uint code, const double v, const double arg, const int n) {
    if (isnan(v)) {
        return 0;
    }
    double b = arg;
    const uint feat = code >> 3;
    if (feat == FEAT_ORDER_RATE) {
        b = round((double)n * (arg / 100.0));
    } else if (feat == FEAT_HP_PER) {
        b = arg / 100.0;
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

// One arm's condition program: a conjunction of leaf checks, since the '@'
// structure lives in the arms table and not in the programs.
static int cc_sk_prog(Cc* c, const int r, const __global const unsigned int* prog,
    const int len, const int fieldless) {
    for (int i = 0; i < len; i++) {
        const __global const unsigned int* nd = prog + SK_PSTRIDE * i;
        if (nd[0] == OP_TRUE) {
            if (nd[1] == 0u) {
                return 0;
            }
        } else if (nd[0] == OP_FEAT_I) {
            const double arg = cc_sk_f64(nd + 2);
            const double v = cc_sk_feat_value(nd[1], c, r, fieldless);
            if (!cc_sk_feat_cmp(nd[1], v, arg, c->n)) {
                return 0;
            }
        } else if (nd[0] != OP_AND) {
            return 0;
        }
    }
    return 1;
}

// The arm fold of the engine's Or closure: one arm passes when the position
// sits in its own runtime chain and its program holds.
static int cc_sk_live_ok(Cc* c, const int r, const __global unsigned int* rd,
    const double pos, const int fieldless) {
    const int arms_base = (int)rd[21];
    const int arms_len = (int)rd[20];
    for (int a = 0; a < arms_len; a++) {
        const __global unsigned int* ar = SKARMF(arms_base + a);
        if (ar[6] & SK_ARM_DEAD) {
            continue;
        }
        const int rl = (int)ar[4];
        const int rb = (int)ar[5];
        int member = 0;
        for (int g = 0; g < rl; g++) {
            if (cc_sk_f64(SKREGF(rb + g)) <= pos &&
                pos < cc_sk_f64(SKREGF(rb + g) + 2)) {
                member = 1;
                break;
            }
        }
        if (member == 0) {
            continue;
        }
        if (cc_sk_prog(c, r, c->su + CLS_SK_TABLE + SK_PROGRAMS + SK_PSTRIDE * (int)ar[1],
            (int)ar[0], fieldless)) {
            return 1;
        }
    }
    return 0;
}

// The engine's per-tick prerequisite watch (race_solver.rs:2089-2098), the
// same fold the retail lane's sk_prereq_check runs: membership in the
// compiled precondition regions and the precondition program.
static int cc_sk_prereq(Cc* c, const int r, const __global unsigned int* rd,
    const double pos, const int fieldless) {
    const int npre = (int)rd[22];
    for (int g = 0; g < npre; g++) {
        const __global const unsigned int* rg = SKREGF((int)rd[23] + g);
        if (cc_sk_f64(rg) <= pos && pos < cc_sk_f64(rg + 2)) {
            return cc_sk_prog(c, r, c->su + CLS_SK_TABLE + SK_PROGRAMS +
                SK_PSTRIDE * (int)rd[25], (int)rd[24], fieldless);
        }
    }
    return 0;
}

// process_pending_armed (race_solver.rs:2131-2182) for one runner: the armed
// queue runs in descending sequence order (race_solver.rs:2142-2145), so the
// staminasyoubu record, created last, is gated before any wire record of the
// same tick, then the wire records in wire (creation) order, the order the
// full scan nets by walking the reversed pending list from the back
// (race_solver.rs:2200). trigger-end consumption, the fire gate, and the
// cooldown parking. fieldless is the constructor pass, where only
// all-self-target records fire.
static void cc_sk_scan(Cc* c, const int r, const int fieldless) {
    const int nrec = cc_sk_nrec(c);
    const double pos = fieldless ? 0.0 : SR(DF_POS);
    for (int pass = 0; pass < 2; pass++) {
    for (int rec = 0; rec < nrec; rec++) {
        const __global unsigned int* rd = SKRD(rec);
        if ((int)rd[1] != r || (rd[2] & SK_FLAG_DEAD) != 0u) {
            continue;
        }
        if (((rd[2] & SK_FLAG_SYOUBU) != 0u) != (pass == 0)) {
            continue;
        }
        // The watch runs ahead of the trigger gate, exactly like the engine's
        // prerequisite loop, and latches once for the record's lifetime.
        if (rd[22] != 0u && c->su[CLS_SKD_CONS + rec] == 0u &&
            c->su[CLS_SKD_PRE + rec] == 0u &&
            cc_sk_prereq(c, r, rd, pos, fieldless) != 0) {
            c->su[CLS_SKD_PRE + rec] = 1u;
        }
        __global double* wd = SKW(rec);
        const double trig0 = wd[SKW_TRIG0];
        const double trig1 = wd[SKW_TRIG1];
        if (trig0 > pos) {
            continue;
        }
        if (c->su[CLS_SKD_CONS + rec] != 0u) {
            continue;
        }
        if (pos >= trig1) {
            c->su[CLS_SKD_CONS + rec] = 1u;
            continue;
        }
        if (rd[22] != 0u && c->su[CLS_SKD_PRE + rec] == 0u) {
            continue;
        }
        const int slot = (int)wd[SKW_SLOT];
        int all_self = 1;
        const int eff_base = (int)rd[6];
        const int n_eff = (int)rd[5];
        for (int e = 0; e < n_eff; e++) {
            if ((int)SKEF(eff_base + e)[1] != 1) {
                all_self = 0;
                break;
            }
        }
        if (fieldless && all_self == 0) {
            continue;
        }
        if (SKCOOL(r, slot) > SR(DF_ACCUM)) {
            continue;
        }
        int win = 0;
        const int nwin = (int)wd[SKW_NWIN];
        for (int k = 0; k < nwin; k++) {
            if (wd[SKW_WIN0 + 2 * k] <= pos &&
                pos < wd[SKW_WIN0 + 2 * k + 1]) {
                win = 1;
                break;
            }
        }
        if (win == 0) {
            continue;
        }
        if ((rd[2] & SK_FLAG_DYN) != 0u &&
            cc_sk_live_ok(c, r, rd, pos, fieldless) == 0) {
            continue;
        }
        if ((rd[2] & SK_FLAG_ASITAME) != 0u && !SU(UR_SPURT)) {
            continue;
        }
        // The staminasyoubu extra condition (race_solver_builder.rs:833):
        // raw stamina past 1200 and the current speed at or over the last
        // spurt speed, both read live at the scan.
        if ((rd[2] & SK_FLAG_SYOUBU) != 0u &&
            (SR(DF_RAWST) <= 1200.0 || SR(DF_SPEED) < SR(DF_LSS))) {
            continue;
        }
        cc_sk_fire(c, r, rec, fieldless);
        if ((rd[2] & SK_FLAG_SYOUBU) != 0u) {
            c->su[U_STAT_SYB] += 1u;
        }
        const double cd = cc_sk_f64(rd + 13);
        if (cd <= 0.0) {
            c->su[CLS_SKD_CONS + rec] = 1u;
            SKCOOL(r, slot) = INFINITY;
        } else {
            SKCOOL(r, slot) = SR(DF_ACCUM) + cd * c->cd[D_DISTANCE] / 1000.0;
        }
    }
    }
}

// The per-tick skill pass, process_skill_activations (race_solver.rs:1943)
// with field: expire first, then the armed scan.
static void cc_sk_pass(Cc* c, const int r) {
    cc_sk_expire(c, r);
    cc_sk_scan(c, r, 0);
}

// flush_skill_effects (race_field.rs:486-492): the FIFO drains in push
// order; every delivery is a follow_up application on the receiver, so no
// hooks and no detail events, with the stat refresh of race_solver.rs:2660-2663.
static void cc_sk_flush(Cc* c) {
    int q = (int)c->su[CLS_SKD_QLEN];
    c->su[CLS_SKD_QLEN] = 0u;
    int head = 0;
    while (head < q) {
        const unsigned int meta = c->su[CLS_SKD_QMETA + head];
        head++;
        const int src = (int)(meta & 0x7fu);
        const int dst = (int)((meta >> 7) & 0x7fu);
        const int eff_local = (int)((meta >> 14) & 0xfu);
        const int rec = (int)(meta >> 18);
        const int eff = (int)SKRD(rec)[6] + eff_local;
        // A receiver that does not own the skill still applies the effect,
        // like the engine's unchecked solver delivery (race_field.rs:489).
        // Other-perspective instances never reach the deactivate hook, so the
        // slot only matters for self deliveries, which always own it.
        int slot = cc_sk_slot_of_sid(c, dst, SKRD(rec)[0]);
        if (slot < 0) {
            if (src == dst) {
                c->rp_over = 1;
                continue;
            }
            slot = 0;
        }
        const int kind = (int)SKEF(eff)[0];
        const double queued_dur = c->sd[UDB_SKQDUR + head - 1];
        cc_sk_apply_effect(c, dst, eff, slot, 0, src == dst ? 1 : 0, queued_dur, 0u);
        if (kind == 1 || kind == 2 || kind == 3 || kind == 4 || kind == 5) {
            cc_sk_refresh(c, dst);
        }
    }
    (void)q;
}

// The race-end finalizer (lib.rs:1005-1017): still-open pending entries
// append their skillEvents entries, sorted by the activation ordinal.
static void cc_sk_finalize_pending(Cc* c) {
    unsigned int ord[CLS_MAXR * SK_PEND_SLOT];
    unsigned int sid[CLS_MAXR * SK_PEND_SLOT];
    unsigned int run[CLS_MAXR * SK_PEND_SLOT];
    unsigned int uni[CLS_MAXR * SK_PEND_SLOT];
    double t[CLS_MAXR * SK_PEND_SLOT];
    double d[CLS_MAXR * SK_PEND_SLOT];
    int m = 0;
    for (int r = 0; r < c->n; r++) {
        for (int s = 0; s < SK_PEND_SLOT; s++) {
            if ((SKPEND(r, s)[1] & 1u) == 0u) {
                continue;
            }
            ord[m] = SKPEND(r, s)[0];
            sid[m] = SKPEND(r, s)[2];
            run[m] = (unsigned int)r;
            uni[m] = (SKPEND(r, s)[1] & 2u) ? 1u : 0u;
            t[m] = SKPEND_T(r, s);
            d[m] = SKPEND_D(r, s);
            SKPEND(r, s)[1] = 0u;
            m++;
        }
    }
    for (int i = 1; i < m; i++) {
        for (int j = i; j > 0 && ord[j] < ord[j - 1]; j--) {
            unsigned int x;
            double y;
            x = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = x;
            x = sid[j]; sid[j] = sid[j - 1]; sid[j - 1] = x;
            x = run[j]; run[j] = run[j - 1]; run[j - 1] = x;
            x = uni[j]; uni[j] = uni[j - 1]; uni[j - 1] = x;
            y = t[j]; t[j] = t[j - 1]; t[j - 1] = y;
            y = d[j]; d[j] = d[j - 1]; d[j - 1] = y;
        }
    }
    for (int i = 0; i < m; i++) {
        const unsigned int len = c->rp[CLS_RP_SKLEN];
        if (len + 9u > (unsigned int)CLS_RP_SK_CAP) {
            c->rp_over = 1;
            return;
        }
        const ulong tb = as_ulong(t[i]);
        const ulong db = as_ulong(d[i]);
        __global unsigned int* p = c->rp + CLS_RP_SK0 + len;
        p[0] = 0u;
        p[1] = ord[i];
        p[2] = run[i];
        p[3] = sid[i];
        p[4] = uni[i];
        p[5] = (unsigned int)(tb & 0xffffffffu);
        p[6] = (unsigned int)(tb >> 32u);
        p[7] = (unsigned int)(db & 0xffffffffu);
        p[8] = (unsigned int)(db >> 32u);
        c->rp[CLS_RP_SKLEN] = len + 9u;
    }
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
        // flush_skill_effects (race_field.rs:291-323), the tick-tail FIFO
        // that delivers queued cross effects on the receivers.
        cc_sk_flush(c);
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
    // The race-end pending finalizer (lib.rs:1005-1017) appends the events
    // of still-open activations, in activation order.
    cc_sk_finalize_pending(c);
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
        if (r == 0) {
            st[rb + S_STAT0] = c->su[U_STAT_KEEP];
            st[rb + S_STAT1] = c->su[U_STAT_CMP];
            st[rb + S_STAT2] = c->su[U_STAT_SEC];
            st[rb + S_STAT3] = c->su[U_STAT_GATE];
            st[rb + S_STAT4] = c->su[U_STAT_SYB];
        }
    }
    st[S_TICKS] = (unsigned int)ticks;
    cc_bits(st, S_END, t);
    if (c->rp_over && !st[S_FALLBACK]) {
        st[S_FALLBACK] = 4u;
    }
}
