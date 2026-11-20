// Initialization kernel (handoff step 6, initialization slice). One work item
// runs one complete input-to-output chain, line for line against the Rust
// reference; the rng_core.cl helpers are prepended by the host (gpu/src/gpu.rs).
//
// Exactness per docs/GPU_HANDOFF.md: both FP_CONTRACT OFF pragmas are in
// rng_core.cl, the program builds with -cl-fp32-correctly-rounded-divide-sqrt,
// no fast math, and every f64-to-f32 store is a single-rounding
// convert_float_rte matching the Rust `as f32`.

// Runners per initialization case (the race kernel's roster cap, mirrored by
// the host's MAX_RUNNERS), hills per course, section count.
#define MAXR 144
#define MAXH 8
#define SEC 24

#define IM_STATS 0
#define IM_R30 1
#define IM_INIT 2
#define IM_HP 3

// Input u32 slots, stride IIU (sized to the IM_INIT per-runner block:
// US_R0 + IIS_STRIDE * MAXR).
#define IIU 448
#define US_MODE 0
#define US_A 1
#define US_B 2
#define US_C 3
#define US_D 4
#define US_E 5
#define US_F 6
#define US_CSS0 7
#define US_CSS1 8
#define US_CSS2 9
#define US_CSS3 10
// IM_STATS: the singleMode flag (build_adjusted_stats_for_mode's +400).
#define US_SINGLE 11
// IM_INIT per-runner block, slot 16 + IIS_STRIDE * r.
#define US_R0 16
#define IIS_STRIDE 3
#define US_STYLE 0
#define US_WISB 1
#define US_BONUS 2

// Input f64 slots (raw bits), stride IIL (IL_PARAMS + 10 parameters).
#define IIL 168
// IM_STATS: 0..4 = raw speed, stamina, power, guts, wisdom.
// IM_R30: 0 = first slope start, 1 = first slope value, 2 = adjusted wisdom,
//         3 = course distance.
// IM_INIT: 0 = course distance, 8 + r = per-runner base_wisdom,
//          params from IL_PARAMS: force1..4, force_span, start_delay_max,
//          minus_val1, minus_val2, plus_val1, temptation_start_per_val1.
#define IL_DISTANCE 0
#define IL_RUNNER0 8
#define IL_PARAMS (IL_RUNNER0 + MAXR)  // directly after the per-runner block
#define IL_FORCE1 (IL_PARAMS + 0)
#define IL_FORCE2 (IL_PARAMS + 1)
#define IL_FORCE3 (IL_PARAMS + 2)
#define IL_FORCE4 (IL_PARAMS + 3)
#define IL_FORCE_SPAN (IL_PARAMS + 4)
#define IL_DELAY_MAX (IL_PARAMS + 5)
#define IL_MINUS1 (IL_PARAMS + 6)
#define IL_MINUS2 (IL_PARAMS + 7)
#define IL_PLUS1 (IL_PARAMS + 8)
#define IL_TEMPT1 (IL_PARAMS + 9)
// IM_HP: 0 stamina, 1 guts, 2 wisdom, 3 distance, 4 hp_initial,
//        5..9 hp_style[0..4], 10..12 course speed c0, c1, c2,
//        13..14 hp_drain[3] (HpGutsCoefSqrt), hp_drain[4] (HpGutsCoef).

// Output u32 slots, stride OOU (the per-runner block, its continuation
// bank, and tail margin). 0..59 is the RetailRandom bank, 60..119 the
// per-runner RUSH bank (paired mode). Slot 120 is the fallback flag:
// 0 ok, 1 a capped uniform() ran out, 2 an item input exceeded a layout bound.
#define OOU 4608
#define OS_BANK 0
#define OS_RUSHBANK 60
#define OS_FLAG 120
// IM_INIT draw-stage counters.
#define OS_STAGES 121
// IM_INIT per-runner block, slot 128 + 30 * r, 30 slots per runner. IM_R30
// scalars reuse 128 up.
#define OS_RUN0 128
#define IRS_STRIDE 30
#define OR_FORCE 0
#define OR_CHANCE 1
#define OR_DELAY 2
#define OR_RCHANCE 3
#define OR_RSECTION 4
#define OR_RPRESENT 5
#define OR_SECTIONS 6
// IM_R30 scalar slots, from 128.
#define O3_D1 128
#define O3_D2 129
#define O3_HP_SEED 130
#define O3_HP_HI 131
#define O3_HP_LO 132
#define O3_GATE_ROLL 133
#define O3_RANDOM_LOT 134
#define O3_KAKARI_U 135
#define O3_KAKARI_IDX 136
#define O3_HILL0_ROLL 137
#define O3_DOWNHILL 138
#define O3_HILLS 139      // 2 * MAXH seed words, lo first per pair
#define O3_HILL0_HI 155   // hill_rng[0] state after the gate roll
#define O3_HILL0_LO 156
#define O3_CAPFLAG 157    // 1 when one of the three uniform() caps was hit
#define O3_SOLVER_HI 158  // solver_rng state after every draw of this chain
#define O3_SOLVER_LO 159
// IM_INIT continuation draws (state equality without exposing internals),
// directly after the per-runner block. The bank state the runtime rng clones
// at the after_delays point follows with a gap kept for growth.
#define OS_CONT_I (OS_RUN0 + IRS_STRIDE * MAXR)
#define OS_BANK_FOR_RACE (OS_CONT_I + 16)
// IM_HP outputs, from 120.
#define OH_GUTS 170
#define OH_WISDOM 171

// Output f64 slots (raw bits), stride OOL (MAXR continuation plus margin).
#define OOL 168
// IM_HP: out_l 0 ground_modifier, 1 base_speed, 2 max_hp (all raw f64).
// IM_STATS: 0..4 base stats, 5 raw_stamina, 6 raw_power, 7..11 adjusted.
// IM_R30: 0 start-delay random, 1 kakari check random, 2..4 r1..r3,
//         5..28 the 24 section randoms, 29 the gate hill-roll bits.
// IM_INIT: 0..MAXR-1 per-runner kakari_start, OL_CONT..OL_CONT+7 continuation
//          next_double bits.
#define OL_CONT MAXR

static double clamp_1_2000(const double v) {
    // Rust clamp passes NaN through, which both comparisons failing does.
    if (v < 1.0) return 1.0;
    if (v > 2000.0) return 2000.0;
    return v;
}

static float clamp_1_2000_f32(const float v) {
    if (v < 1.0f) return 1.0f;
    if (v > 2000.0f) return 2000.0f;
    return v;
}

static double adjust_overcap(const double stat) {
    if (stat > 1200.0) return 1200.0 + (stat - 1200.0) * 0.5;
    return stat;
}

static float team_rank_bonus(const int rank) {
    if (rank <= 0) return 0.0f;
    if (rank < 6) return 50.0f;
    if (rank < 11) return 30.0f;
    if (rank < 16) return 20.0f;
    if (rank < 21) return 10.0f;
    return 0.0f;
}

// initial_base_wisdom from src/retail_initialization.rs.
static float initial_base_wisdom(const double raw, const int mood) {
    const int moodc = (mood < -2) ? -2 : ((mood > 2) ? 2 : mood);
    const float coefficient = (float)(10000 + 200 * moodc) * 0.0001f;
    const float v = convert_float_rte(raw) * coefficient;
    if (v < 1.0f) return 1.0f;
    if (v > 2000.0f) return 2000.0f;
    return v;
}

// ranked_base_stat from src/race_solver_builder.rs, the one base-stat path the
// retail chain uses (team bonus added in f32 before the mood coefficient).
static float ranked_base_stat(const double raw, const int mood, const int rank) {
    const float base = convert_float_rte(adjust_overcap(raw)) + team_rank_bonus(rank);
    return initial_base_wisdom((double)base, mood);
}

// log10 ported from the msun/libm algorithm the wasm (musl) and native Linux
// reference builds use (libm 0.2.16 math/log10.rs, FreeBSD msun e_log10.c).
// The f32 casts of every engine log10 result absorb the 1-ulp raw deltas that
// Windows-native std produces; gpu/README.md carries the exhaustive check.
static double engine_log10(double x) {
    const double x1p54 = 0x1.0p54;

    ulong ui = as_ulong(x);
    double hfsq;
    double f;
    double s;
    double z;
    double r;
    double w;
    double t1;
    double t2;
    double dk;
    double y;
    double hi;
    double lo;
    double val_hi;
    double val_lo;
    uint hx;
    int k;

    hx = (uint)(ui >> 32);
    k = 0;
    if (hx < 0x00100000u || (hx >> 31) > 0u) {
        if (ui << 1 == 0UL) {
            return -1.0 / (x * x); /* log(+-0)=-inf */
        }
        if ((hx >> 31) > 0u) {
            return (x - x) / 0.0; /* log(-#) = NaN */
        }
        /* subnormal number, scale x up */
        k -= 54;
        x *= x1p54;
        ui = as_ulong(x);
        hx = (uint)(ui >> 32);
    } else if (hx >= 0x7ff00000u) {
        return x;
    } else if (hx == 0x3ff00000u && ui << 32 == 0UL) {
        return 0.0;
    }

    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    hx += 0x3ff00000u - 0x3fe6a09eu;
    k += (int)(hx >> 20) - 0x3ff;
    hx = (hx & 0x000fffffu) + 0x3fe6a09eu;
    ui = ((ulong)hx << 32) | (ui & 0xffffffffUL);
    x = as_double(ui);

    f = x - 1.0;
    hfsq = 0.5 * f * f;
    s = f / (2.0 + f);
    z = s * s;
    w = z * z;
    t1 = w * (0x1.999999997fa04p-2 + w * (0x1.c71c51d8e78afp-3 + w * 0x1.39a09d078c69fp-3));
    t2 = z * (0x1.5555555555593p-1 + w * (0x1.2492494229359p-2 + w * (0x1.7466496cb03dep-3 + w * 0x1.2f112df3e5244p-3)));
    r = t2 + t1;

    /* hi+lo = f - hfsq + s*(hfsq+R) ~ log(1+f) */
    hi = f - hfsq;
    ui = as_ulong(hi);
    ui &= 0xffffffff00000000UL;
    hi = as_double(ui);
    lo = f - hi - hfsq + s * (hfsq + r);

    /* val_hi+val_lo ~ log10(1+f) + k*log10(2) */
    val_hi = hi * 0x1.bcb7b152p-2;
    dk = (double)k;
    y = dk * 0x1.34413509f6p-2;
    val_lo = dk * 0x1.9fef311f12b36p-42 + (lo + hi) * 0x1.b9438ca9aadd5p-36 + lo * 0x1.bcb7b152p-2;

    w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;

    return val_lo + val_hi;
}

// The four mode bodies are static functions so the race kernel can run the
// same chains over its scratch blocks. The standalone kernel dispatches, and
// the zeroing of the output slots belongs to it; each function writes only
// the slots its mode documents.

// build_base_stats_with_team_rank then build_adjusted_stats_for_mode with
// the US_SINGLE flag, the retail path in race_solver_builder.rs.
static void im_stats(__global const uint* ib, __global const ulong* ilb,
    __global uint* ob, __global ulong* olb) {
        const double single_bonus = ib[US_SINGLE] != 0u ? 400.0 : 0.0;
        const int mood = (int)ib[US_A];
        const int rank = (int)ib[US_B];
        const int surface = (int)ib[US_C];
        const int strategy_apt = (int)ib[US_D];
        const int ground = (int)ib[US_E];
        const int css_len = (int)ib[US_F];
        const double raw_speed = as_double(ilb[0]);
        const double raw_stamina = as_double(ilb[1]);
        const double raw_power = as_double(ilb[2]);
        const double raw_guts = as_double(ilb[3]);
        const double raw_wisdom = as_double(ilb[4]);
        if (surface < 1 || surface > 2 || ground < 1 || ground > 4
            || strategy_apt < 0 || strategy_apt > 7 || css_len < 0 || css_len > 4) {
            ob[OS_FLAG] = 2u;
            return;
        }
        const double motiv = 1.0 + 0.02 * (double)mood; // motiv uses the raw mood
        const uint css[4] = {ib[US_CSS0], ib[US_CSS1], ib[US_CSS2], ib[US_CSS3]};

        double values[6];
        values[0] = 0.0;
        values[1] = (double)ranked_base_stat(raw_speed, mood, rank);
        values[2] = (double)ranked_base_stat(raw_stamina, mood, rank);
        values[3] = (double)ranked_base_stat(raw_power, mood, rank);
        values[4] = (double)ranked_base_stat(raw_guts, mood, rank);
        values[5] = (double)ranked_base_stat(raw_wisdom, mood, rank);
        olb[0] = as_ulong(values[1]);
        olb[1] = as_ulong(values[2]);
        olb[2] = as_ulong(values[3]);
        olb[3] = as_ulong(values[4]);
        olb[4] = as_ulong(values[5]);
        olb[5] = as_ulong(raw_stamina * motiv);
        olb[6] = as_ulong(values[3] + fmax(raw_power - 1200.0, 0.0) * 0.5);

        // course_speed_modifier from course_data.rs, a f32 accumulator.
        int count = 0;
        for (int i = 0; i < css_len; i++) {
            if (css[i] != 0u) count += 1;
        }
        double race_course_modifier = 1.0;
        if (count > 0) {
            float coefficient = 1.0f;
            for (int i = 0; i < css_len; i++) {
                const int stat = (int)css[i];
                if (stat == 0) continue;
                if (stat < 1 || stat > 5) {
                    // Rust indexes values[stat] here and panics past 5.
                    ob[OS_FLAG] = 2u;
                    continue;
                }
                const float value = convert_float_rte(values[stat]);
                const float term = (value <= 300.0f) ? 0.05f
                    : ((value <= 600.0f) ? 0.1f
                    : ((value <= 900.0f) ? 0.15f
                    : ((value <= 2000.0f) ? 0.2f : 0.0f)));
                coefficient += term / (float)count;
            }
            race_course_modifier = (double)coefficient;
        }
        // GROUND_SPEED_MODIFIER and GROUND_POWER_MODIFIER[surface][ground].
        const double gspd[3][5] = {
            {0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, -50.0},
            {0.0, 0.0, 0.0, 0.0, -50.0},
        };
        const double gpow[3][5] = {
            {0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, -50.0, -50.0, -50.0},
            {0.0, -100.0, -50.0, -100.0, -100.0},
        };
        const double strategy_t[8] = {11000., 10000., 8500., 7500., 6000., 4000., 2000., 1000.};
        // Each combination rounds once, in f32, only where Rust stores a float.
        const float csf = convert_float_rte(race_course_modifier);
        const double adj_speed = clamp_1_2000(convert_float_rte(
            (double)convert_float_rte(values[1]) * (double)csf + gspd[surface][ground] + single_bonus));
        const double adj_stamina = clamp_1_2000(values[2] + single_bonus);
        const double adj_power = clamp_1_2000(values[3] + gpow[surface][ground] + single_bonus);
        const double adj_guts = clamp_1_2000(values[4] + single_bonus);
        const float wisdom_coef = (float)strategy_t[strategy_apt] * 0.0001f;
        const double adj_wisdom = clamp_1_2000(convert_float_rte(
            convert_float_rte(values[5]) * (double)wisdom_coef + single_bonus));
        olb[7] = as_ulong(adj_speed);
        olb[8] = as_ulong(adj_stamina);
        olb[9] = as_ulong(adj_power);
        olb[10] = as_ulong(adj_guts);
        olb[11] = as_ulong(adj_wisdom);
}

// The per-solver Rule30 seeding chain, race_solver_builder.rs prepare()
// and build() plus the rng-consuming part of RaceSolver::new through
// init_hills(), the kakari draws and the 24 section draws.
static void im_r30(__global const uint* ib, __global const ulong* ilb,
    __global uint* ob, __global ulong* olb) {
        const uint seed_lo_bits = ib[US_A];
        const uint seed_hi_bits = ib[US_B];
        const uint runner = ib[US_C];
        const int n_slopes = (int)ib[US_D];
        const double first_start = as_double(ilb[0]);
        const double first_per = as_double(ilb[1]);
        const double wisdom = as_double(ilb[2]);
        const double distance = as_double(ilb[3]);
        if (n_slopes < 0 || n_slopes > MAXH) {
            ob[OS_FLAG] = 2u;
            return;
        }
        const long seed = (long)(((ulong)seed_hi_bits << 32) | (ulong)seed_lo_bits);
        const uint lo = (uint)(seed + (long)runner * 7919L);
        // wisdom_seed(lo, 0x52414345), the race_seed_hi finalizer.
        uint x = lo ^ 0x52414345u ^ 0x57495344u;
        x ^= x >> 16;
        x = x * 0x85ebca6bu;
        x ^= x >> 13;
        x = x * 0xc2b2ae35u;
        x ^= x >> 16;
        const uint hi = (x == 0u) ? 1u : x;

        uint b_hi = hi;
        uint b_lo = lo;
        const uint d1 = r30_int32(&b_hi, &b_lo);
        const uint d2 = r30_int32(&b_hi, &b_lo);
        ob[O3_D1] = d1;
        ob[O3_D2] = d2;

        // Rule30CARng::new(seed_lo, seed_hi) keeps hi at seed_hi, so the
        // (d1, 0) streams start with hi = 0.
        // The continuing stream: build() draws the hp seed from solver_rng
        // first (GameHpState::new argument), then RaceSolver::new draws on.
        uint s_hi = 0u;
        uint s_lo = d1;
        const uint hp_seed = r30_int32(&s_hi, &s_lo);
        ob[O3_HP_SEED] = hp_seed;
        // hp_rng = new(hp_seed, 0) is stored but never drawn in retail, the
        // retail runtime replaces it. Its start state is pinned for parity.
        ob[O3_HP_HI] = 0u;
        ob[O3_HP_LO] = hp_seed;
        int found = 0;
        const int gate_roll = r30_uniform(&s_hi, &s_lo, 12252240.0, &found);
        uint cap_flag = (uint)(found ? 0 : 1);
        ob[O3_GATE_ROLL] = (uint)gate_roll;
        const int random_lot = r30_uniform(&s_hi, &s_lo, 100.0, &found);
        cap_flag |= (uint)(found ? 0 : 1);
        ob[O3_RANDOM_LOT] = (uint)random_lot;
        olb[0] = as_ulong(0.1 * r30_random(&s_hi, &s_lo));

        for (int hill = 0; hill < n_slopes; hill++) {
            const uint h_lo = r30_int32(&s_hi, &s_lo);
            const uint h_hi = r30_int32(&s_hi, &s_lo);
            ob[O3_HILLS + 2 * hill] = h_lo;
            ob[O3_HILLS + 2 * hill + 1] = h_hi;
        }
        if (n_slopes > 0 && first_start == 0.0) {
            uint h0_hi = ob[O3_HILLS + 1];
            uint h0_lo = ob[O3_HILLS];
            const double roll = r30_random(&h0_hi, &h0_lo);
            olb[29] = as_ulong(roll);
            ob[O3_HILL0_HI] = h0_hi;
            ob[O3_HILL0_LO] = h0_lo;
            ob[O3_DOWNHILL] = (first_per < 0.0 && roll < wisdom * 0.0004) ? 1u : 0u;
            ob[O3_HILL0_ROLL] = 1u;
        } else {
            ob[O3_HILL0_ROLL] = 0u;
            ob[O3_DOWNHILL] = 0u;
        }

        const int kakari_u = r30_uniform(&s_hi, &s_lo, 7.0, &found);
        cap_flag |= (uint)(found ? 0 : 1);
        ob[O3_KAKARI_U] = (uint)kakari_u;
        olb[1] = as_ulong(r30_random(&s_hi, &s_lo));
        const double r1 = r30_random(&s_hi, &s_lo);
        const double r2 = r30_random(&s_hi, &s_lo);
        const double r3 = r30_random(&s_hi, &s_lo);
        olb[2] = as_ulong(r1);
        olb[3] = as_ulong(r2);
        olb[4] = as_ulong(r3);
        // seq = [0, r1, r2, r3, 1], first index whose value > 0.45 (always found).
        int idx = 4;
        if (0.0 > 0.45) idx = 0;
        else if (r1 > 0.45) idx = 1;
        else if (r2 > 0.45) idx = 2;
        else if (r3 > 0.45) idx = 3;
        ob[O3_KAKARI_IDX] = (uint)idx;
        for (int s = 0; s < SEC; s++) {
            olb[5 + s] = as_ulong(r30_random(&s_hi, &s_lo));
        }
        ob[O3_CAPFLAG] = cap_flag;
        if (cap_flag) ob[OS_FLAG] = 1u;
        // The continuing solver_rng state, which the tick loop inherits.
        ob[O3_SOLVER_HI] = s_hi;
        ob[O3_SOLVER_LO] = s_lo;
}

/// Replay the enabled wisdom-gated records' self WisdomUp bumps onto the
/// per-runner init inputs, mirroring the engine's constructor fire ordering.
static void sk_init_wis_bump(__global const uint* sk, __global const uint* sks,
    __global uint* ib) {
    const int nrec = (int)sk[SK_NREC];
    for (int rec = 0; rec < nrec; rec++) {
        __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
        if ((rd[2] & SK_FLAG_INIT_BUMP) == 0) continue;
        if (!sk_gate_enabled(sk, sks, rec)) continue;
        __global uint* rb = ib + US_R0 + IIS_STRIDE * (int)rd[1];
        float value = convert_float_rte(as_float(rb[US_WISB]));
        for (int e = 0; e < (int)rd[5]; e++) {
            const __global const uint* ef = sk + SK_EFFECTS + SK_ESTRIDE * ((int)rd[6] + e);
            if ((int)ef[0] != 5) continue;
            value += convert_float_rte(as_double((ulong)ef[6] | ((ulong)ef[7] << 32)));
        }
        rb[US_WISB] = as_uint(clamp_1_2000_f32(value));
    }
}

static void im_init(__global uint* ib, __global const ulong* ilb,
    __global uint* ob, __global ulong* olb, __global const uint* sk,
    __global uint* sks, __global double* skd, __global uint* sub) {
        const int seed = (int)ib[US_A];
        const int n = (int)ib[US_B];
        const int paired = (int)ib[US_C];
        const int paired_seed = (int)ib[US_D];
        const int lot_min = (int)ib[US_E];
        const int lot_max = (int)ib[US_F];
        if (n < 1 || n > MAXR) {
            ob[OS_FLAG] = 2u;
            return;
        }
        __global uint* bank = ob + OS_BANK;
        __global uint* rush_bank = ob + OS_RUSHBANK;
        const double distance = as_double(ilb[IL_DISTANCE]);
        const float dist_f = convert_float_rte(distance);
        const int sk_on = (sk != 0) && (int)sk[SK_NREC] > 0;
        double sk_mult[MAXR];

        r128_new(bank, seed);
        if (sk_on) sk_init_triggers(sk, sks, skd, bank, sub, paired, seed, n, distance);
        ob[OS_STAGES + 0] = bank[BANK_DC]; // after_triggers

        // force-in-move pass
        const float force_style[4] = {
            convert_float_rte(as_double(ilb[IL_FORCE1])),
            convert_float_rte(as_double(ilb[IL_FORCE2])),
            convert_float_rte(as_double(ilb[IL_FORCE3])),
            convert_float_rte(as_double(ilb[IL_FORCE4])),
        };
        const float span = convert_float_rte(as_double(ilb[IL_FORCE_SPAN]));
        for (int r = 0; r < n; r++) {
            __global const uint* rb = ib + US_R0 + IIS_STRIDE * r;
            const int style = (int)rb[0];
            const float coefficient = (style >= 1 && style <= 4) ? force_style[style - 1] : 0.0f;
            float offset;
            if (coefficient > 0.0f) offset = r128_real(bank, 0.0f, span);
            else if (coefficient < 0.0f) offset = r128_real(bank, -span, 0.0f);
            else offset = 0.0f;
            ob[OS_RUN0 + IRS_STRIDE * r + OR_FORCE] = as_uint(coefficient + offset);
        }
        ob[OS_STAGES + 1] = bank[BANK_DC]; // after_ai

        // wisdom chance, then the skill lottery on the shared stream when the
        // race carries a skill table
        if (sk_on) sk_init_wisdom(sk, sks, bank, sub, ilb, sk_mult, paired, seed, n,
            IL_RUNNER0);
        else for (int r = 0; r < n; r++) sk_mult[r] = 1.0;
        // The engine's constructor fires the enabled wisdom-gated records
        // before the section and rushing draws (race_solver.rs:1008 before
        // lib.rs:760), so their wisdom bumps land here, behind both.
        if (sk_on) sk_init_wis_bump(sk, sks, ib);
        for (int r = 0; r < n; r++) {
            const float base_wisdom = convert_float_rte(as_double(ilb[IL_RUNNER0 + r]));
            float chance = 100.0f - 9000.0f / base_wisdom;
            if (chance < 20.0f) chance = 20.0f;
            ob[OS_RUN0 + IRS_STRIDE * r + OR_CHANCE] = as_uint(chance);
        }
        ob[OS_STAGES + 2] = bank[BANK_DC]; // after_wisdom

        // start-delay pass, scaled by the enabled always-alternative factors
        const float delay_max = convert_float_rte(as_double(ilb[IL_DELAY_MAX]));
        for (int r = 0; r < n; r++) {
            ob[OS_RUN0 + IRS_STRIDE * r + OR_DELAY] = as_uint(
                r128_real(bank, 0.0f, delay_max) * convert_float_rte(sk_mult[r]));
        }
        ob[OS_STAGES + 3] = bank[BANK_DC]; // after_delays

        // initialize_sections_and_rushing: base, ranges, 24 draws per runner
        const float base = 20.0f - (dist_f - 2000.0f) / 1000.0f;
        const float minus1 = convert_float_rte(as_double(ilb[IL_MINUS1]));
        const float minus2 = convert_float_rte(as_double(ilb[IL_MINUS2]));
        const float plus1 = convert_float_rte(as_double(ilb[IL_PLUS1]));
        for (int r = 0; r < n; r++) {
            __global const uint* rb = ib + US_R0 + IIS_STRIDE * r;
            const float wisdom = as_float(rb[1]);
            const float logarithm = convert_float_rte(engine_log10((double)(wisdom * 0.1f)));
            const float lower = convert_float_rte(
                ((double)wisdom / (double)minus2) * (double)logarithm + (double)minus1);
            const float upper = convert_float_rte(((double)wisdom / (double)plus1) * (double)logarithm);
            for (int s = 0; s < SEC; s++) {
                const float sample = convert_float_rte(r128_next_double(bank));
                const float width = fabs(upper - lower);
                float random = convert_float_rte((double)sample * (double)width + (double)lower);
                if (random < lower) random = lower;
                else if (random > upper) random = upper;
                ob[OS_RUN0 + IRS_STRIDE * r + OR_SECTIONS + s] = as_uint(
                    convert_float_rte(((double)random * (double)0.01f) * (double)base));
            }
        }
        ob[OS_STAGES + 4] = bank[BANK_DC]; // after_sections

        const float tempt1 = convert_float_rte(as_double(ilb[IL_TEMPT1]));
        for (int r = 0; r < n; r++) {
            __global const uint* rb = ib + US_R0 + IIS_STRIDE * r;
            const float wisdom = as_float(rb[1]);
            const float bonus = as_float(rb[2]);
            const float term = wisdom * 0.1f + 1.0f;
            const float logarithm = convert_float_rte(engine_log10((double)(term * term)));
            const float fraction = tempt1 / logarithm;
            const float rushing_chance = fraction * fraction + bonus;
            ob[OS_RUN0 + IRS_STRIDE * r + OR_RCHANCE] = as_uint(rushing_chance);
            __global uint* draw_bank = bank;
            if (paired) {
                const ulong hash = r128_substream_hash4(paired_seed, (uint)r);
                r128_new(rush_bank, (int)(uint)(hash >> 33));
                draw_bank = rush_bank;
            }
            int section = 0;
            uint present = 0u;
            if (rushing_chance > (float)r128_integer(draw_bank, 0, 100)) {
                section = r128_integer(draw_bank, lot_min, lot_max + 1);
                present = 1u;
            }
            ob[OS_RUN0 + IRS_STRIDE * r + OR_RSECTION] = (uint)section;
            ob[OS_RUN0 + IRS_STRIDE * r + OR_RPRESENT] = present;
            // enable_retail_runtime overwrites the f64 feed with this f32 form
            // (race_solver_retail.rs: (s-1) f32 * (distance f32 / 24.)).
            olb[r] = present
                ? as_ulong((double)((float)(section - 1) * (dist_f / 24.0f)))
                : as_ulong((double)(dist_f + 9999.0f));
        }
        ob[OS_STAGES + 5] = bank[BANK_DC]; // after_rushing (shared stream)

        // enable_retail_runtime clones the manager rng after sections and the
        // rushing lottery; anything drawn later stays out of the race stream.
        for (int i = 0; i < BANK_WORDS; i++) ob[OS_BANK_FOR_RACE + i] = bank[i];

        for (int i = 0; i < 8; i++) ob[OS_CONT_I + i] = r128_next_int(bank);
        for (int i = 0; i < 8; i++) olb[OL_CONT + i] = as_ulong(r128_next_double(bank));
}

// GameHpState::with_retail + init() from race_solver_retail.rs,
// hp_policy.rs and retail_physics.rs (the retail branch).
static void im_hp(__global const uint* ib, __global const ulong* ilb,
    __global uint* ob, __global ulong* olb) {
        const int strategy = (int)ib[US_A];
        const int surface = (int)ib[US_B];
        const int ground = (int)ib[US_C];
        if (strategy < 1 || strategy > 5 || surface < 1 || surface > 2 || ground < 1 || ground > 4) {
            ob[OS_FLAG] = 2u;
            return;
        }
        const float stamina = convert_float_rte(as_double(ilb[0]));
        const float guts = convert_float_rte(as_double(ilb[1]));
        const float wisdom = convert_float_rte(as_double(ilb[2]));
        const float distance = convert_float_rte(as_double(ilb[3]));
        const float hp_initial = convert_float_rte(as_double(ilb[4]));
        float hp_style[5];
        for (int i = 0; i < 5; i++) hp_style[i] = convert_float_rte(as_double(ilb[5 + i]));
        const float course0 = convert_float_rte(as_double(ilb[10]));
        const float course1 = convert_float_rte(as_double(ilb[11]));
        const float course2 = convert_float_rte(as_double(ilb[12]));
        const float drain3 = convert_float_rte(as_double(ilb[13]));
        const float drain4 = convert_float_rte(as_double(ilb[14]));

        // initial_hp: bounded_stat, double chain, one rounding on the store.
        const float bs = clamp_1_2000_f32(stamina);
        const float max_hp = convert_float_rte(
            ((double)bs * (double)hp_initial) * (double)hp_style[strategy - 1] + (double)distance);
        // course_speed, kept as the f64 the GameHpState field holds.
        const double base_speed = (double)(course0 - (distance - course1) / course2);
        // hp_guts_factor: root() is a correctly-rounded f64 sqrt, then f32.
        const float root = convert_float_rte(sqrt((double)(clamp_1_2000_f32(guts) * drain3)));
        const float guts_factor = convert_float_rte(1.0 + (double)drain4 / (double)root);
        // HP_CONSUMPTION_GROUND_MODIFIER[surface][ground], an f64 field.
        const double hp_ground[3][5] = {
            {0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 1.0, 1.02, 1.02},
            {0.0, 1.0, 1.0, 1.01, 1.02},
        };
        ob[OH_GUTS] = as_uint(guts_factor);
        ob[OH_WISDOM] = as_uint(wisdom);
        olb[0] = as_ulong(hp_ground[surface][ground]);
        olb[1] = as_ulong(base_speed);
        olb[2] = as_ulong((double)max_hp);
}

__kernel void initialization(__global const uint* in_u, __global const ulong* in_l,
    __global uint* out_u, __global ulong* out_l) {
    const int id = get_global_id(0);
    __global const uint* ib = in_u + id * IIU;
    __global const ulong* ilb = in_l + id * IIL;
    __global uint* ob = out_u + id * OOU;
    __global ulong* olb = out_l + id * OOL;
    for (int i = 0; i < OOL; i++) olb[i] = 0UL;
    for (int i = 0; i < 120; i++) ob[i] = 0u;
    ob[OS_FLAG] = 0u;
    const uint mode = ib[US_MODE];
    if (mode == IM_STATS) {
        im_stats(ib, ilb, ob, olb);
    } else if (mode == IM_R30) {
        im_r30(ib, ilb, ob, olb);
    } else if (mode == IM_INIT) {
        im_init((__global uint*)ib, ilb, ob, olb, 0, 0, 0, 0);
    } else {
        im_hp(ib, ilb, ob, olb);
    }
}

