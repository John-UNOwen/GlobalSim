// Race-loop kernel (handoff step 6 items 3-5 minus skills): one work item
// runs one complete no-skill race, single runner, line for line against the
// Rust reference. The rng_core.cl, initialization.cl, geometry.cl and
// retail_core.cl sources are concatenated ahead of this file by the host
// (gpu/src/gpu.rs).
//
// Domain: retail runtime, no skills, no startup mode. Paired mode gives each
// runner its own RUN substream bank, fresh from the race seed. Anything else
// sets ST_FALLBACK with the domain code in ST_ARG (1 bad n, 3 startup,
// 4 host flag, 5 initialization input, 6 geometry, 7 event log overflow).
// The field logic (ranks, pacemaker, lane blocking and movement, duel and
// spot competition) runs in the work item in roster order. Skills keep their
// own streams on the CPU, so a request carrying any skill still falls back
// at the host.

#define FB_N 1u
#define FB_STARTUP 3u
#define FB_HOST 4u
#define FB_INIT 5u
#define FB_GEOMETRY 6u

static void race_fb(__global uint* st, const uint code) {
    st[ST_FALLBACK] = 1u;
    st[ST_ARG] = code;
}

// One TickFrame row, the lib.rs capture layout.
static void race_frame(__global double* frames_all, const int id, const int tick,
    const int n, const int r, const double pos, const double speed, const double hp,
    const double lane, const double mode, const double blocker) {
    __global double* row = frames_all
        + ((long)id * RACE_MAX_TICKS + tick) * n * 6 + r * 6;
    row[0] = pos;
    row[1] = speed;
    row[2] = hp;
    row[3] = lane;
    row[4] = mode;
    row[5] = blocker;
}

// Race scalars in the f64 scratch tail (RD_T): four phase boundaries, then
// the ordinary-nige field count and the pace controller's end section.
#define RD_B0 0
#define RD_NIGE 4
#define RD_PACE_END 5

// The InitProper{Ground,Distance} basis points, one f32 round each.
static float race_apt_ground(const int apt) {
    const uint table[8] = {10500u, 10000u, 9000u, 8000u, 7000u, 5000u, 3000u, 1000u};
    return (float)table[apt] * 0.0001f;
}

static float race_apt_distance(const int apt) {
    const uint table[8] = {10000u, 10000u, 10000u, 10000u, 10000u, 6000u, 5000u, 4000u};
    return (float)table[apt] * 0.0001f;
}

static float race_apt_spurt(const int apt) {
    const uint table[8] = {10500u, 10000u, 9000u, 8000u, 6000u, 4000u, 2000u, 1000u};
    return (float)table[apt] * 0.0001f;
}

// step_forward_retail_with_roster, the no-skill path. The live solver
// columns are the roster, so the dirty-refresh re-snapshot has nothing to
// rebuild and the start-delay rest path returns 0.0 like the engine.
// Returns the movement distance, or -1.0 for the gated rest tick.
static float race_step(__global uint* si, __global uint* scr, __global float* sf,
    __global double* sd,
    __global double* shl, __global uint* lists, __global float* cached,
    __global uint* bank, __global float* c_start, __global float* c_speed,
    __global float* c_time, __global const double* pd, __global const double* rd,
    const int r, const int n, const int empty_move, const float dt,
    const double course_distance, __global const uint* skt,
    __global uint* ev, __global uint* st) {
    const int strategy = (int)si[SI_STRATEGY + r];
    const float dist_f = convert_float_rte(course_distance);
    const int finished = (int)si[SI_FINISHED + r];
    const float power = sf[SF_ACCEL_BASE + r];
    const int phase = (int)si[SI_PHASE + r];
    __global double* shl_r = shl + r * SH_STRIDE;

    // minimum: finished 5, start-dash 3, else the guts-keyed base cache.
    const int blocker = (int)si[SI_BLOCKER + r];
    float minimum;
    if (finished) {
        minimum = 5.f;
    } else if ((int)si[SI_DASH + r]) {
        minimum = 3.f;
    } else {
        minimum = sf[SF_BMIN_VAL + r];
    }
    if (!finished && !(int)si[SI_DASH + r] && blocker >= 0) {
        minimum = fmin(minimum, sf[SF_MIN + blocker]);
    }
    sf[SF_MIN + r] = minimum;
    sd[SD_MS + r] = (double)minimum;

    // Start-delay gate next: a gated runner rests this whole tick.
    float remaining = sf[SF_DELAY + r];
    double gate_fraction = 1.0;
    const int gated = advance_start_delay(&remaining, dt, &gate_fraction);
    sf[SF_DELAY + r] = remaining;
    if (!gated) {
        return -1.f;
    }

    // update_retail_pace. The collect loop keeps lane_max current, and the
    // duel proximity pass feeds only the duel, which takes no pace input.
    if (!finished) {
        const int kakari = (int)si[SI_KAKARI + r];
        if ((int)si[SI_PS_MODE + r] == 7 && !kakari && !(int)si[SI_PS_RUSHING + r]
            && convert_float_rte(sd[SD_POS + r])
                >= floor(dist_f / 24.0f) * convert_float_rte(rd[RD_PACE_END])
            && sd[SD_POS + r] >= (double)(long)(course_distance / 24.0)
                * rd[RD_PACE_END]) {
            // An idle controller past both windows only mirrors its style.
            si[SI_PS_STYLE + r] = (uint)(strategy == 5 ? 1 : strategy);
            si[SI_PACEDOWN + r] = 0u;
            sd[SD_PKCOEF + r] = (double)sf[SF_PS_COEF + r];
        } else {
            const __global uint* rs = scr + SCR_RACE;
            const int rank = (int)si[SI_RANKCOL + r];
            const int pm = (int)rs[RS_PM] - 1;
            int has_behind = 0;
            float behind_gap = 0.f;
            if (rank < n) {
                has_behind = 1;
                behind_gap = convert_float_rte(fmax(
                    sd[SD_POS + r] - sd[SD_POS + (int)rs[RS_ORDER + rank]], 0.0));
            }
            pace_update(si, sf, r, dt, pd, bank, dist_f,
                convert_float_rte(sd[SD_POS + r]), sf[SF_HPW + r], strategy,
                kakari, (int)si[SI_RMODE + r], phase, rank,
                (int)si[SI_RANKCOL + pm], (int)si[SI_STRATEGY + pm],
                convert_float_rte(sd[SD_POS + pm] - sd[SD_POS + r]), pm == r,
                (int)rs[RS_STYLE + r] > field_style(strategy),
                (int)rs[RS_COUNTS + 1], has_behind, behind_gap,
                sk_has_speed(skt, scr, si, r, pd));
            si[SI_PACEDOWN + r] = ((int)si[SI_PS_MODE + r] == 3) ? 1u : 0u;
            sd[SD_PKCOEF + r] = (double)sf[SF_PS_COEF + r];
        }
        // The v8 legacy conserve gauge runs in the position-keep window,
        // with the truncated-section divisor rather than the rounded one.
        const double cp_sec = (double)(long)(course_distance / 24.0);
        if (cp_sec * (convert_float_rte(pd[PK_START_SECTION]) - 1.0)
                <= sd[SD_POS + r]
            && sd[SD_POS + r] < cp_sec * convert_float_rte(pd[PK_END_SECTION])) {
            const int cp_mode = (int)si[SI_PS_MODE + r];
            const double rate = (cp_mode == 7) ? 6.7 : ((cp_mode == 3) ? 4.2 : 0.);
            sd[SD_CP_ACC + r] += (double)dt * rate
                * ((strategy == 2) ? 0.75 : ((strategy == 4) ? 0.7 : 1.));
            uint cpfl = si[SI_CPFLAGS + r];
            if ((int)si[SI_SPOT_ACTIVE + r]) {
                cpfl |= CP_STRUGGLE;
            }
            if (kakari) {
                cpfl |= CP_KAKARI;
            }
            si[SI_CPFLAGS + r] = cpfl;
        }
    }

    // update_targets, then the empty_in_move-gated target search (it owns
    // the outward lottery draw).
    const float position_f = convert_float_rte(sd[SD_POS + r]);
    const float lane_f = convert_float_rte(sd[SD_LANE + r]);
    lane_update_targets(si, sf, lists, cached, sd, sf, r, n, pd, dt,
        position_f, lane_f, sf[SF_OBSERVED + r], convert_float_rte(sd[SD_TSWSK + r]));
    if (empty_move && sf[SF_AFTER_GOAL + r] < 1.f) {
        const float target = lane_target(si, sf, lists, cached, sd, sf, pd, bank,
            r, n, position_f, lane_f, sf[SF_OBSERVED + r],
            convert_float_rte(sd[SD_TSWSK + r]), phase, (int)(shl_r[SH_HP] > 0.),
            (int)si[SI_PACEDOWN + r]);
        sd[SD_LANET + r] = (double)target;
    }

    // The move engine reads the freshly written lane target.
    sf[SF_MV_TARGET + r] = convert_float_rte(sd[SD_LANET + r]);

    // Rushing transition, skipped once finished. The gates mirror step().
    // A start bumps temptation_count. An end is where the deactivate hook
    // emits the kakari skillEvents entry.
    if (!finished) {
        const int rush = rushing_step(si, sf, r, position_f, dist_f, dt, pd, bank);
        if (rush == 1) {
            si[SI_TEMPT + r] += 1u;
        }
        if (rush == -1) {
            race_event(ev, st, r, EV_KAKARI);
        }
        si[SI_KAKARI + r] = si[SI_RS_ACTIVE + r];
        si[SI_RMODE + r] = si[SI_RS_MODE + r];
    }

    // The clean conserve ledger, skipped once the last spurt has started.
    if (!finished && !(int)si[SI_SP_ACTIVE + r]) {
        const int cp_mode = (int)si[SI_PS_MODE + r];
        cp_update(si, sf, sd, r, course_distance, strategy, dt,
            cp_mode != 7 && cp_mode != 3, (int)si[SI_PACEDOWN + r],
            (int)si[SI_KAKARI + r],
            (int)si[SI_SPOT_ACTIVE + r] && sd[SD_SPOT_TIMER + r] < 0.0);
    }

    // update_retail_spurt. The plan closure only runs when the state needs a
    // new plan, which is where the search's lotteries live.
    const int was_spurting = (int)si[SI_SP_ACTIVE + r];
    if (phase >= 2) {
        float out_start = 0.f, out_speed = 0.f;
        int out_result = 0;
        if ((int)si[SI_SP_COUNT + r] == 0 || (int)si[SI_SP_PENDING + r]) {
            spurt_choose(pd, shl_r, sf[SF_HPG + r], sf[SF_HPW + r], dist_f,
                position_f, convert_float_rte(shl_r[SH_HP]), sf[SF_SPURT_MAX + r],
                convert_float_rte(sd[SD_BTS + 2 * RACE_MAXR + r]), bank, c_start, c_speed,
                c_time, &out_start, &out_speed, &out_result);
        }
        si[SI_SP_ACTIVE + r] = (uint)spurt_state_update(si, sf, r, position_f,
            out_start, out_speed, out_result);
        sd[SD_LSS + r] = (double)sf[SF_SP_SPEED + r];
        // The Conserve Power release rides the last-spurt transition.
        if (!was_spurting && (int)si[SI_SP_ACTIVE + r]) {
            asitame_release(si, sf, sd, r, pd, course_distance, strategy);
        }
    }

    // Downhill mode from the slope region, drawing its lottery.
    const double slope_per = (double)sf[SF_SL_VALUE + r];
    downhill_update(si, sf, r, slope_per < 0., sf[SF_HPW + r], dt, pd, bank);
    si[SI_DOWNHILL + r] = si[SI_DS_ACTIVE + r];

    // The expired spot receiver is removed before the target chain, after
    // the pace context already read its token.
    if ((int)si[SI_SPOT_ACTIVE + r] && sd[SD_SPOT_TIMER + r] >= 0.0) {
        si[SI_SPOT_TOKEN + r] = 0u;
    }
    const float spot_token = (int)si[SI_SPOT_TOKEN + r]
        ? convert_float_rte(pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_SPOT_M]) : 0.f;
    const float duel_t_token = (int)si[SI_DUEL_TOKEN + r]
        ? convert_float_rte(pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_DUEL_T]) : 0.f;

    // Target speed. The SectionIndex walk equals the quotient port.
    const int section_index = phys_section(position_f, dist_f);
    const int target_phase = phys_phase_of_section(section_index);
    const int last_spurt = (int)si[SI_SP_ACTIVE + r];
    const float base = last_spurt ? convert_float_rte(sd[SD_LSS + r])
        : convert_float_rte(sd[SD_BTS + target_phase * RACE_MAXR + r]);
    const float section = last_spurt ? 0.f
        : convert_float_rte(sd[SD_SECMOD + section_index * RACE_MAXR + r]);
    if (!finished) {
        const float slope = convert_float_rte(slope_per / 10000.);
        __global float* tmods = (__global float*)(scr + SK_STAGED);
        const int ntm = sk_mods(skt, scr, r, SK_EFF_TARGET, tmods);
        sd[SD_TS + r] = (double)phys_target_speed(pd, base, section,
            convert_float_rte(sd[SD_PKCOEF + r]),
            (int)si[SI_L_FORCEIN + r] ? sf[SF_FORCE_IN + r] : 0.f,
            power, slope, (int)si[SI_DOWNHILL + r], minimum,
            (int)(shl_r[SH_HP] > 0.), tmods, ntm,
            spot_token, duel_t_token);
        sk_column_one(skt, scr, sd, r);
    }
    float cap = 30.f;
    if (blocker >= 0) {
        cap = phys_blocking_speed_cap(pd, minimum, sf[SF_OBSERVED + blocker],
            sf[SF_BLOCKER_GAP + r]);
    }

    // Movement step over the collect side room.
    lane_move_step(si, sf, r, pd,
        finished && sf[SF_AFTER_GOAL + r] >= 1.f, position_f, power, phase,
        (int)si[SI_RANKCOL + r], convert_float_rte(pd[PC_FIRST_MOVE_POINT]),
        sf[SF_LANEMAX + r], (int)si[SI_L_CANIN + r], (int)si[SI_L_CANOUT + r], dt);
    sd[SD_LANE + r] = (double)sf[SF_MV_LANE + r];
    sd[SD_LANET + r] = (double)sf[SF_MV_TARGET + r];
    sd[SD_LCS + r] = (double)sf[SF_MV_SPEED + r];

    // Longitudinal speed chain.
    const float current = convert_float_rte(sd[SD_CS + r]);
    const float target = convert_float_rte(sd[SD_TS + r]);
    float acceleration;
    if (finished) {
        acceleration = (current > target) ? -4.f : ((current < target) ? 1.f : 0.f);
    } else if (!(shl_r[SH_HP] > 0.)) {
        acceleration = -1.2f;
    } else if (current > target) {
        acceleration = (int)si[SI_PACEDOWN + r] ? -0.5f
            : convert_float_rte((phase == 0) ? -1.2
                : ((phase == 1) ? -0.8 : -1.0));
    } else if (current == target && current > 0.f) {
        // A positive finite dt always reaches the target and clamps there.
        acceleration = 0.f;
    } else {
        const float base_accel = phys_base_acceleration(pd, power,
            race_apt_ground((int)si[SI_SUAPT + r]),
            race_apt_distance((int)si[SI_DISTAPT + r]),
            strategy, phase, slope_per > 0.);
        __global float* amods = (__global float*)(scr + SK_STAGED);
        const int nam = sk_mods(skt, scr, r, SK_EFF_ACCEL, amods);
        acceleration = phys_acceleration_from_base(pd, base_accel,
            (int)si[SI_DASH + r], amods, nam,
            convert_float_rte(pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_DUEL_A]),
            (int)si[SI_DUEL_TOKEN + r],
            sf[SF_AS_VALUE + r], (int)si[SI_CPFLAGS + r] & CP_ASIT);
    }
    float speed = current + acceleration * dt;
    speed = (current <= target) ? fmin(speed, target) : fmax(speed, target);
    const float dash_limit = phys_start_dash_speed(pd, dist_f);
    if ((int)si[SI_DASH + r]) {
        speed = fmin(speed, dash_limit);
    }
    speed = fmax(fmin(speed, cap), minimum);
    sd[SD_CS + r] = (double)speed;
    // retail_current_speed: the natural sum (empty here, no such effect is
    // ported) plus the ordinary current-skill sum, over f32 speed.
    const float observed = fmin(fmax(sk_current_fold(skt, scr, r) + speed,
        minimum), cap);
    sf[SF_OBSERVED + r] = observed;
    float advance_distance = observed * dt;
    if (gate_fraction != 1.0) {
        advance_distance = convert_float_rte((double)advance_distance * gate_fraction);
    }

    // HP tick, the retail branch. Competing is the live spot struggle with
    // an unexpired timer, shared by the dash and normal paths.
    const int runaway = (strategy == 5) ? 1 : 0;
    const int competing = (int)si[SI_SPOT_ACTIVE + r] && sd[SD_SPOT_TIMER + r] < 0.0;
    if (!finished && (int)si[SI_DASH + r]) {
        hp_tick_start_dash(pd, shl_r, (int)si[SI_KAKARI + r], competing, runaway,
            (int)si[SI_PACEDOWN + r], (int)si[SI_DOWNHILL + r], dt);
    } else if (!finished) {
        hp_tick(pd, shl_r, sf[SF_HPG + r], (int)si[SI_KAKARI + r], competing, runaway,
            (int)si[SI_PACEDOWN + r], (int)si[SI_DOWNHILL + r], phase,
            sd[SD_CS + r], (double)dt);
    }
    if (!finished && !(shl_r[SH_HP] > 0.)) {
        si[SI_HPEMPTY + r] = 1u;
    }

    // Phase recomputation on the pre-advance position.
    int new_phase = 0;
    for (int b = 0; b < 4; b++) {
        if (sd[SD_POS + r] >= rd[RD_B0 + b]) {
            new_phase += 1;
        }
    }
    si[SI_RPHASE + r] = (uint)new_phase;
    si[SI_PHASE + r] = (uint)(new_phase < 2 ? new_phase : 2);
    return advance_distance;
}

__kernel void race(__global const uint* ri_all, __global const double* rd_all,
    __global uint* scr_u_all, __global double* scr_d_all,
    __global double* frames_all, __global uint* status_all,
    __global const uint* cs, __global const double* sg_x,
    __global const double* sg_xd, __global const double* sg_z,
    __global const double* sg_zd, __global const float* sg_q,
    __global const float* sg_qd, __global const float* lm_b,
    __global const float* lm_l, __global const float* ce_b, __global const float* ce_e,
    __global const uint* ce_i, __global const uint* ce_f,
    __global const uint* sk_all, __global uint* events_all) {
    const int id = get_global_id(0);
    __global const uint* ri = ri_all + id * RI_STRIDE;
    __global const double* pd = rd_all + id * RIL_STRIDE;
    __global uint* scr = scr_u_all + (long)id * SCR_U_WORDS;
    __global double* scr_d = scr_d_all + (long)id * SRL_WORDS;
    __global uint* st = status_all + id * ST_STRIDE;
    __global uint* ev = events_all + id * EV_STRIDE;
    st[ST_FALLBACK] = 0u;
    st[ST_ARG] = 0u;
    st[ST_TICKS] = 0u;
    ev[0] = 0u;

    // The skill table block for this race, zero-filled (NREC 0) when the
    // host compiled no skills. Views over it stay live for every race.
    __global const uint* skt = sk_all + (long)id * SK_STRIDE;
    const int sk_on = (int)skt[SK_NREC] > 0;
    __global uint* sks = scr + SCR_SK;
    __global double* skd = scr_d + SRL_SKD;
    __global uint* sub = scr + SK_SUB;

    const int n = (int)ri[RI_N];
    if (n < 1 || n > RACE_MAXR) {
        race_fb(st, FB_N);
        return;
    }
    if ((int)ri[RI_STARTUP]) {
        race_fb(st, FB_STARTUP);
        return;
    }
    if (ri[RI_HOST_FLAG] != 0u) {
        race_fb(st, FB_HOST);
        return;
    }

    // Scratch views.
    __global uint* si = scr + SCR_STATE;
    __global float* sf = (__global float*)(scr + SCR_SF);
    __global uint* lists = scr + SCR_LISTS;
    __global float* cached = (__global float*)(scr + SCR_CACHED);
    __global float* c_start = (__global float*)(scr + SCR_CAND);
    __global float* c_speed = (__global float*)(scr + SCR_CAND + CAND_MAX);
    __global float* c_time = (__global float*)(scr + SCR_CAND + 2 * CAND_MAX);
    __global uint* bank = scr + SCR_INIT_OUT + OS_BANK;
    __global double* sd = scr_d + SRL_RUNNER;
    __global double* shl = scr_d + SRL_HP;
    __global double* rds = scr_d + RD_T;
    const int geom_on = (int)ri[RI_GEOM_ON];
    __global const uint* cb = cs + (int)ri[RI_COURSE] * CS_STRIDE;
    const int empty_move = (int)ri[RI_EMPTY_MOVE];
    const double course_distance = pd[PI_DISTANCE];
    const float dist_f = convert_float_rte(course_distance);
    const int surface = (int)ri[RI_SURFACE];
    const int ground = (int)ri[RI_GROUND];
    const float dt = 0.0666f;

    // Per-runner adjusted stats through the initialization chain.
    for (int r = 0; r < n; r++) {
        __global const uint* rb = ri + RI_RUN0 + RI_RUN_STRIDE * r;
        __global uint* ib = scr + SCR_STATS_IN + r * IIU;
        ib[US_MODE] = (uint)IM_STATS;
        ib[US_A] = rb[RR_MOOD];
        ib[US_B] = rb[RR_RANK];
        ib[US_C] = (uint)surface;
        ib[US_D] = rb[RR_STAPT];
        ib[US_E] = (uint)ground;
        ib[US_F] = rb[RR_CSSN];
        ib[US_CSS0] = rb[RR_CSS0];
        ib[US_CSS1] = rb[RR_CSS1];
        ib[US_CSS2] = rb[RR_CSS2];
        ib[US_CSS3] = rb[RR_CSS3];
        for (int s = 0; s < 121; s++) {
            scr[SCR_STATS_OUT + r * 160 + s] = 0u;
        }
        __global ulong* ilb =
            (__global ulong*)(scr_d + SRL_STATS_IN_L + r * IIL);
        for (int s = 0; s < 5; s++) {
            ilb[s] = as_ulong(pd[PI_RUNNER0 + 6 * r + s]);
        }
        __global uint* ob = scr + SCR_STATS_OUT + r * 160;
        __global ulong* olb = (__global ulong*)(ob + 128);
        im_stats(ib, ilb, ob, olb);
        if (ob[OS_FLAG] != 0u) {
            race_fb(st, FB_INIT);
            return;
        }
    }

    // One shared initialization block drives force offsets, delays, section
    // modifiers and the rushing lottery from the run's RetailRandom stream.
    // Paired mode sends each runner's rushing lottery to its own RUSH
    // substream (lib.rs:760), same as the engine's init.
    const int paired = (int)ri[RI_PAIRED];
    {
        __global uint* ib = scr + SCR_INIT_IN;
        ib[US_MODE] = (uint)IM_INIT;
        ib[US_A] = ri[RI_SEED];
        ib[US_B] = (uint)n;
        ib[US_C] = (uint)paired;
        ib[US_D] = ri[RI_SEED];
        ib[US_E] = ri[RI_LOT_MIN];
        ib[US_F] = ri[RI_LOT_MAX];
        for (int r = 0; r < n; r++) {
            __global const uint* rb = ri + RI_RUN0 + RI_RUN_STRIDE * r;
            __global const ulong* scol =
                (__global const ulong*)(scr + SCR_STATS_OUT + r * 160 + 128);
            __global uint* ib_run = ib + US_R0 + IIS_STRIDE * r;
            ib_run[US_STYLE] = rb[RR_STYLE];
            ib_run[US_WISB] = as_uint(convert_float_rte(as_double(scol[11])));
            ib_run[US_BONUS] = 0u;
        }
        __global ulong* ilb = (__global ulong*)(scr_d + SRL_INIT_IN_L);
        ilb[IL_DISTANCE] = as_ulong(course_distance);
        for (int r = 0; r < n; r++) {
            const __global ulong* scol =
                (__global const ulong*)(scr + SCR_STATS_OUT + r * 160 + 128);
            ilb[IL_RUNNER0 + r] = scol[4];
        }
        ilb[IL_FORCE1] = as_ulong(pd[PI_FORCE1]);
        ilb[IL_FORCE2] = as_ulong(pd[PI_FORCE2]);
        ilb[IL_FORCE3] = as_ulong(pd[PI_FORCE3]);
        ilb[IL_FORCE4] = as_ulong(pd[PI_FORCE4]);
        ilb[IL_FORCE_SPAN] = as_ulong(pd[PI_FORCE_SPAN]);
        ilb[IL_DELAY_MAX] = as_ulong(pd[PI_DELAY_MAX]);
        ilb[IL_MINUS1] = as_ulong(pd[PI_MINUS1]);
        ilb[IL_MINUS2] = as_ulong(pd[PI_MINUS2]);
        ilb[IL_PLUS1] = as_ulong(pd[PI_PLUS1]);
        ilb[IL_TEMPT1] = as_ulong(pd[PI_TEMPT1]);
        __global uint* ob = scr + SCR_INIT_OUT;
        for (int s = 0; s < 121; s++) ob[s] = 0u;
        __global ulong* olb = (__global ulong*)(scr_d + SRL_INIT_OUT_L);
        im_init(ib, ilb, ob, olb, skt, sk_on ? sks : 0, sk_on ? skd : 0, sub);
        if (ob[OS_FLAG] != 0u) {
            race_fb(st, FB_INIT);
            return;
        }
        // Unpaired: the race rng is the clone the runtime takes at the
        // after_delays point, so later initialization draws never touch the
        // race stream. Paired: lib.rs:794 replaces that clone per runner with
        // a fresh RUN substream, so each runner ticks on its own bank.
        if (paired) {
            for (int r = 0; r < n; r++) {
                const ulong hash = r128_substream_hash_run((int)ri[RI_SEED], (uint)r);
                r128_new(scr + SCR_BANKS + r * BANK_WORDS, (int)(uint)(hash >> 33));
            }
        } else {
            for (int i = 0; i < BANK_WORDS; i++) bank[i] = ob[OS_BANK_FOR_RACE + i];
        }
    }

    int ordinary_nige = 0;
    for (int r = 0; r < n; r++) {
        if ((int)ri[RI_RUN0 + RI_RUN_STRIDE * r + RR_STRATEGY] == 1) {
            ordinary_nige += 1;
        }
    }
    rds[RD_NIGE] = (double)ordinary_nige;
    rds[RD_PACE_END] = pd[PK_END_SECTION];
    // phase_boundaries, keeping the rounded section length product form.
    {
        const float section = dist_f / 24.0f;
        rds[RD_B0 + 0] = (double)(section * 4.0f);
        rds[RD_B0 + 1] = (double)(section * 16.0f);
        rds[RD_B0 + 2] = (double)(section * 20.0f);
        rds[RD_B0 + 3] = (double)dist_f;
    }

    // HP state through the initialization kernel's im_hp chain.
    for (int r = 0; r < n; r++) {
        __global const uint* rb = ri + RI_RUN0 + RI_RUN_STRIDE * r;
        const __global ulong* scol =
            (__global const ulong*)(scr + SCR_STATS_OUT + r * 160 + 128);
        __global uint* ib = scr + SCR_HP_IN + r * IIU;
        ib[US_MODE] = (uint)IM_HP;
        ib[US_A] = rb[RR_STRATEGY];
        ib[US_B] = (uint)surface;
        ib[US_C] = (uint)ground;
        __global ulong* ilb = (__global ulong*)(scr_d + SRL_HP_IN_L + r * IIL);
        ilb[0] = scol[8];
        ilb[1] = scol[10];
        ilb[2] = scol[11];
        ilb[3] = as_ulong(course_distance);
        ilb[4] = as_ulong(pd[PT_HP_INITIAL]);
        for (int s = 0; s < 5; s++) ilb[5 + s] = as_ulong(pd[PT_HP_STYLE + s]);
        ilb[10] = as_ulong(pd[PT_COURSE0]);
        ilb[11] = as_ulong(pd[PT_COURSE1]);
        ilb[12] = as_ulong(pd[PT_COURSE2]);
        ilb[13] = as_ulong(pd[PT_HP_DRAIN + 3]);
        ilb[14] = as_ulong(pd[PT_HP_DRAIN + 4]);
        for (int s = 0; s < 178; s++) {
            scr[SCR_HP_OUT + r * 180 + s] = 0u;
        }
        __global uint* ob = scr + SCR_HP_OUT + r * 180;
        __global ulong* olb = (__global ulong*)(ob + 172);
        im_hp(ib, ilb, ob, olb);
        if (ob[OS_FLAG] != 0u) {
            race_fb(st, FB_INIT);
            return;
        }
        shl[r * SH_STRIDE + SH_GROUNDMOD] = as_double(olb[0]);
        shl[r * SH_STRIDE + SH_BASESPEED] = as_double(olb[1]);
        shl[r * SH_STRIDE + SH_MAX] = as_double(olb[2]);
        shl[r * SH_STRIDE + SH_HP] = as_double(olb[2]);
        sf[SF_HPG + r] = as_float(ob[OH_GUTS]);
        sf[SF_HPW + r] = as_float(ob[OH_WISDOM]);
    }

    // Solver state: the enable_retail_runtime snapshot plus the RaceSolver
    // constructor defaults (current_speed 3, lane_target 0, rest zero).
    for (int r = 0; r < n; r++) {
        __global const uint* rb = ri + RI_RUN0 + RI_RUN_STRIDE * r;
        const __global ulong* scol =
            (__global const ulong*)(scr + SCR_STATS_OUT + r * 160 + 128);
        const int strategy = (int)rb[RR_STRATEGY];
        si[SI_STRATEGY + r] = (uint)strategy;
        si[SI_DISTAPT + r] = rb[RR_DISTAPT];
        si[SI_SUAPT + r] = rb[RR_SUAPT];
        si[SI_RANK + r] = rb[RR_RANK];
        si[SI_PHASE + r] = 0u;
        si[SI_RPHASE + r] = 0u;
        si[SI_RMODE + r] = 0u;
        si[SI_KAKARI + r] = 0u;
        si[SI_PACEDOWN + r] = 0u;
        si[SI_DOWNHILL + r] = 0u;
        si[SI_DASH + r] = 1u;
        si[SI_EVENTS_READY + r] = 0u;
        si[SI_FINISHED + r] = 0u;
        si[SI_BLOCKER + r] = 0xFFFFFFFFu;
        si[SI_HPEMPTY + r] = 0u;
        si[SI_TEMPT + r] = 0u;
        si[SI_SECT + r] = 0u;
        si[SI_RS_ACTIVE + r] = 0u;
        si[SI_RS_USED + r] = 0u;
        si[SI_RS_MODE + r] = 0u;
        si[SI_DS_ACTIVE + r] = 0u;
        si[SI_SL_NEXT + r] = 0u;
        si[SI_SL_REGION + r] = 0xFFFFFFFFu;
        si[SI_PS_RUSHING + r] = 0u;
        si[SI_PS_STYLE + r] = 0u;
        si[SI_PS_MODE + r] = 7u;
        si[SI_L_FORCE + r] = 0u;
        si[SI_L_XOP + r] = 0u;
        si[SI_SP_PENDING + r] = 0u;
        si[SI_SP_COUNT + r] = 0u;
        si[SI_SP_RESULT + r] = 8u;  // SpurtState::default
        si[SI_MV_DIR + r] = 0u;
        si[SI_L_INNER_IDX + r] = 0xFFFFFFFFu;
        si[SI_L_TGTN + r] = 0u;
        si[SI_L_VISN + r] = 0u;
        si[SI_L_CANIN + r] = 1u;
        si[SI_L_CANOUT + r] = 1u;
        si[SI_L_FORCEIN + r] = 0u;
        si[SI_RANKCOL + r] = 0u;
        si[SI_SP_ACTIVE + r] = 0u;
        si[SI_CORNER + r] = 0u;
        si[SI_FINALE + r] = 0u;
        si[SI_LS_ENTERED + r] = 0u;
        si[SI_DUEL_ACTIVE + r] = 0u;
        si[SI_DUEL_FIGHT + r] = 0u;
        si[SI_DUEL_GROUP + r] = 0xFFFFFFFFu;
        si[SI_DUEL_PRE_FIN + r] = 0u;
        si[SI_DUEL_PRE_STR + r] = 0u;
        si[SI_GOAL_SNAP + r] = 0u;
        si[SI_SPOT_ACTIVE + r] = 0u;
        si[SI_SPOT_STARTED + r] = 0u;
        si[SI_SPOT_EXITED + r] = 0u;
        si[SI_SPOT_LATCH + r] = 0u;
        si[SI_SPOT_CLOSED + r] = 0u;
        si[SI_SPOT_TOKEN + r] = 0u;
        si[SI_DUEL_TOKEN + r] = 0u;
        si[SI_CPFLAGS + r] = 0u;
        sd[SD_SPOT_TIMER + r] = 0.0;
        sd[SD_CP_GAUGE + r] = 0.0;
        sd[SD_CP_PREVRATE + r] = 0.0;
        sd[SD_CP_INITPW + r] = 0.0;
        sd[SD_CP_ACC + r] = 0.0;
        sd[SD_AS_TIMER + r] = 0.0;
        sf[SF_AS_VALUE + r] = 0.f;
        sf[SF_CP_GELAP + r] = 0.f;
        sf[SF_CP_GTHRESH + r] = 0.f;
        sf[SF_DUEL_PRE + r] = 3.f;
        for (int j = 0; j < RACE_MAXR; j++) {
            sd[SD_DUEL + j * RACE_MAXR + r] = -1.0;
        }

        const uint runb = SCR_INIT_OUT + OS_RUN0 + IRS_STRIDE * r;
        si[SI_SECT + r] = (uint)(int)(scr[runb + OR_RPRESENT]
            ? (int)scr[runb + OR_RSECTION] - 1 : -1);

        sf[SF_FORCE_IN + r] = as_float(scr[runb + OR_FORCE]);
        sf[SF_ACCEL_BASE + r] = convert_float_rte(as_double(scol[9]));
        sd[SD_RAWPW + r] = as_double(scol[2]);
        sf[SF_PENDING_HP + r] = 0.f;
        sf[SF_AFTER_GOAL + r] = 0.f;
        sf[SF_DELAY + r] = as_float(scr[runb + OR_DELAY]);
        st[ST_DELAY0 + r] = scr[runb + OR_DELAY];
        sf[SF_RS_CHECK + r] = 0.f;
        sf[SF_RS_MAXREM + r] = 0.f;
        sf[SF_DS_REM + r] = 0.f;
        sf[SF_SL_END + r] = 0.f;
        sf[SF_SL_VALUE + r] = 0.f;
        sf[SF_PS_COOL + r] = 0.f;
        sf[SF_PS_INTVL + r] = 0.f;
        sf[SF_PS_COEF + r] = 1.f;
        sf[SF_PS_ESTART + r] = 0.f;
        sf[SF_PS_EEND + r] = 0.f;
        sf[SF_PS_EXIT + r] = 0.f;
        sf[SF_L_COOLDOWN + r] = 0.f;
        sf[SF_L_XO + r] = 0.f;
        sf[SF_L_SIDE_INNER + r] = 0.f;
        sf[SF_L_SIDE_INS + r] = 0.f;
        sf[SF_L_SIDE_OUTS + r] = 0.f;
        sf[SF_SP_START + r] = -1.f;
        sf[SF_SP_SPEED + r] = 0.f;
        sf[SF_MV_SPEED + r] = 0.f;
        sf[SF_MV_TIME + r] = 0.f;
        sf[SF_BLOCKER_GAP + r] = 0.f;
        sf[SF_RSCORE + r] = 0.f;
        sf[SF_RREM + r] = 1.f;
        sf[SF_OBSERVED + r] = 3.f;
        sf[SF_MIN + r] = 3.f;

        const float speed = convert_float_rte(as_double(scol[7]));
        const float guts = convert_float_rte(as_double(scol[10]));
        const float lane_f = convert_float_rte(pd[PI_RUNNER0 + 6 * r + 5]);
        sf[SF_MV_LANE + r] = lane_f;
        sf[SF_MV_TARGET + r] = 0.f;
        sf[SF_LANEMAX + r] = geom_on ? geo_lane_max_at(cb, lm_b, lm_l, 0.f)
            : convert_float_rte(pd[PC_MAX_LANE]);
        if (geom_on) {
            float wx, wz;
            if (geo_position(cb, 0.f, lane_f, sg_x, sg_xd, sg_z, sg_zd,
                sg_q, sg_qd, &wx, &wz)) {
                race_fb(st, FB_GEOMETRY);
                return;
            }
            sf[SF_WORLD_X + r] = wx;
            sf[SF_WORLD_Y + r] = wz;
        } else {
            sf[SF_WORLD_X + r] = 0.f;
            sf[SF_WORLD_Y + r] = 0.f;
        }

        const float aptitude = race_apt_spurt((int)rb[RR_DISTAPT]);
        for (int ph = 0; ph < 3; ph++) {
            sd[SD_BTS + ph * RACE_MAXR + r] = (double)phys_base_target_speed(pd, speed,
                strategy, ph, aptitude, dist_f);
        }
        sd[SD_POS + r] = 0.0;
        sd[SD_LANE + r] = (double)lane_f;
        sd[SD_LANET + r] = 0.0;
        sd[SD_LCS + r] = 0.0;
        sd[SD_LCT + r] = 0.0;
        sd[SD_CS + r] = 3.0;
        sd[SD_TS + r] = (double)convert_float_rte(
            convert_float_rte(sd[SD_BTS + r])
            + convert_float_rte(as_float(scr[runb + OR_SECTIONS + 0])));
        sd[SD_TSWSK + r] = sd[SD_TS + r];
        const float minimum_speed = phys_minimum_speed(pd, guts, dist_f);
        sd[SD_MS + r] = (double)minimum_speed;
        sf[SF_BMIN_VAL + r] = minimum_speed;
        si[SI_BMIN_BITS + r] = as_uint(guts);
        sd[SD_LSS + r] = (double)phys_maximum_spurt_speed(pd, speed, strategy,
            aptitude, dist_f, pd[PD_RUN0 + r * PD_RUN_STRIDE + PRD_POWF]);
        sf[SF_SPURT_MAX + r] = convert_float_rte(sd[SD_LSS + r]);
        for (int s = 0; s < 24; s++) {
            sd[SD_SECMOD + s * RACE_MAXR + r] =
                (double)as_float(scr[runb + OR_SECTIONS + s]);
        }
        sd[SD_SECMOD + 24 * RACE_MAXR + r] = 0.0;
        sd[SD_PKCOEF + r] = 1.0;
    }

    // RaceField::new runs update() once: at zero positions that publishes
    // rank i+1, order 0..n, prev rank 0 and the min-style leader as section
    // pacemaker, which the first pace context then reads.
    __global uint* rs = scr + SCR_RACE;
    rs[RS_PM] = 0u;
    rs[RS_PM_SEC] = 0u;
    rs[RS_PM_COUNT] = 0u;
    rs[RS_PM_HANDED] = 0u;
    rs[RS_PM_HAS] = 0u;
    const double section_length = course_distance / 24.0;
    const float pace_end_f = convert_float_rte(rds[RD_PACE_END]);
    const float pm_range = convert_float_rte(pd[PD_PM_RANGE]);
    const int pm_notfwd = (int)pd[PD_PM_NOTFWD];
    const int has_corner_events = (int)ri[RI_NSTR] && (int)ri[RI_NCOR];
    field_update(scr, si, sd, n, dist_f, pace_end_f, course_distance,
        pm_range, pm_notfwd);

    // Frame 0 is the pre-loop capture (lib.rs), the initial snapshot.
    for (int r = 0; r < n; r++) {
        race_frame(frames_all, id, 0, n, r, sd[SD_POS + r],
            (double)sf[SF_OBSERVED + r], fmax(shl[r * SH_STRIDE + SH_HP], 0.0),
            sd[SD_LANE + r], (double)(int)si[SI_RMODE + r], -1.0);
    }

    double t = 0.0;
    double prev_pos[RACE_MAXR];
    int finish_state[RACE_MAXR];
    for (int r = 0; r < n; r++) {
        prev_pos[r] = sd[SD_POS + r];
        finish_state[r] = 0;
    }
    int tick = 0;
    int any_open = 1;
    while (any_open && t < 400.0) {
        // Rows over the pre-movement state: the last-straight flags feed the
        // duel collect and the duel entry test.
        uint any_straight = 0u;
        for (int r = 0; r < n; r++) {
            uint flag = 0u;
            if (has_corner_events
                && sd[SD_POS + r] >= pd[PC_LAST_STRAIGHT_START]
                && sd[SD_POS + r] < pd[PC_LAST_STRAIGHT_END]) {
                flag = 1u;
            }
            rs[RS_STRAIGHT + r] = flag;
            any_straight |= flag;
        }
        // Collect: duel proximity timers, then lane caches with refreshed
        // lane_max. All read the same pre-movement columns.
        for (int r = 0; r < n; r++) {
            duel_collect(si, sf, sd, r, n, pd, dt, course_distance,
                (int)any_straight, rs + RS_STRAIGHT);
            if (geom_on) {
                sf[SF_LANEMAX + r] = geo_lane_max_at(cb, lm_b, lm_l,
                    convert_float_rte(sd[SD_POS + r]));
            }
            lane_collect(si, sf, lists, cached, sd, sf, r, n, pd, empty_move,
                (int)si[SI_PHASE + r], sf[SF_LANEMAX + r],
                convert_float_rte(sd[SD_LANE + r]),
                convert_float_rte(sd[SD_POS + r]));
        }
        // prepare_retail_tick: after-goal, the spot timer step, the corner
        // state read and the slope events.
        for (int r = 0; r < n; r++) {
            if ((int)si[SI_FINISHED + r]) {
                sf[SF_AFTER_GOAL + r] += dt;
            }
            sd[SD_SPOT_TIMER + r] =
                (double)(convert_float_rte((float)sd[SD_SPOT_TIMER + r]) + dt);
            // The asitame effect timer steps and expires with the same pass
            // the engine advances timers and runs expire_skill_effects.
            if ((int)si[SI_CPFLAGS + r] & CP_ASIT) {
                const double at =
                    (double)(convert_float_rte((float)sd[SD_AS_TIMER + r]) + dt);
                sd[SD_AS_TIMER + r] = at;
                if (at >= 0.0) {
                    si[SI_CPFLAGS + r] &= ~CP_ASIT;
                }
            }
            if (geom_on) {
                geo_corner_state(cb, ce_b, ce_e, ce_i, ce_f,
                    convert_float_rte(sd[SD_POS + r]),
                    (int)si[SI_EVENTS_READY + r],
                    &si[SI_CORNER + r], &si[SI_FINALE + r]);
            }
            if ((int)si[SI_EVENTS_READY + r]) {
                slope_update(si, sf, r, (int)ri[RI_NSLOPES], pd,
                    convert_float_rte(sd[SD_POS + r]));
            }
            if ((int)si[SI_EVENTS_READY + r] && (int)si[SI_FINALE + r]
                && !(int)si[SI_LS_ENTERED + r] && (int)ri[RI_NSTR]
                && sd[SD_POS + r] >= pd[PC_LAST_STRAIGHT_START]) {
                si[SI_LS_TICK + r] = 1u;
                si[SI_LS_ENTERED + r] = 1u;
            } else {
                si[SI_LS_TICK + r] = 0u;
            }
            si[SI_EVENTS_READY + r] = 1u;
        }
        // The skill runtime sits where prepare_retail_tick ends: timer step,
        // expiry and cooldown start, then the priority passes with their
        // fire gate, then the live Horse rows the sample below reads.
        if (sk_on) {
            skills_advance(skt, sks, skd, scr, dt, n, dist_f);
            skills_scan(skt, sks, skd, scr, si, sd, shl, rs, 1, n, dist_f, ev, st);
            skills_scan(skt, sks, skd, scr, si, sd, shl, rs, 0, n, dist_f, ev, st);
            sk_columns(skt, scr, sd, n);
        }
        // blocking_sample over the post-skill rows, which skills left alone.
        for (int r = 0; r < n; r++) {
            int front = -1;
            float front_gap = 0.f;
            int side = 0;
            lane_blocking_sample(sd, si, r, n, pd, &front, &front_gap, &side);
            si[SI_BLOCKER + r] = (uint)front;
            sf[SF_BLOCKER_GAP + r] = front_gap;
        }
        // Movement in roster order; the live columns are the roster. Paired
        // runners draw race dice from their own bank, the others share one.
        for (int r = 0; r < n; r++) {
            __global uint* runner_bank =
                paired ? scr + SCR_BANKS + r * BANK_WORDS : bank;
            const float advance_distance = race_step(si, scr, sf, sd, shl, lists,
                cached, runner_bank, c_start, c_speed, c_time, pd, rds, r, n,
                empty_move, dt, course_distance, skt, ev, st);
            // stadium_score::RushingScore.update(is_kakari, dt), after the
            // step and before the gated continue, as the engine loop does.
            if ((int)si[SI_KAKARI + r]) {
                float rem = sf[SF_RREM + r] - dt;
                if (rem <= 0.f) {
                    rem = 1.f;
                    sf[SF_RSCORE + r] += 1.f;
                }
                sf[SF_RREM + r] = rem;
            }
            rs[RS_ADV + r] = 0u;
            if (advance_distance < 0.f) {
                continue;  // gated rest: no movement, dash or goal test
            }
            rs[RS_ADV + r] = 1u;
            // Geometry advance with the world cursor.
            float next;
            if (geom_on) {
                float wx = sf[SF_WORLD_X + r];
                float wz = sf[SF_WORLD_Y + r];
                if (geo_advance(cb, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd,
                    convert_float_rte(sd[SD_POS + r]),
                    convert_float_rte(sd[SD_LANE + r]), advance_distance,
                    wx, wz, &next, &wx, &wz)) {
                    race_fb(st, FB_GEOMETRY);
                    return;
                }
                sf[SF_WORLD_X + r] = wx;
                sf[SF_WORLD_Y + r] = wz;
            } else {
                next = convert_float_rte(sd[SD_POS + r]) + advance_distance;
            }
            sd[SD_POS + r] = (double)next;
            if ((int)si[SI_FINISHED + r] == 0 && sd[SD_POS + r] > course_distance) {
                si[SI_FINISHED + r] = 1u;
                si[SI_GOAL_SNAP + r] = 1u;
            }
            if ((int)si[SI_DASH + r]
                && sf[SF_OBSERVED + r] >= phys_start_dash_speed(pd, dist_f)) {
                si[SI_DASH + r] = 0u;
            }
        }
        // Overlap resolution over the post-movement rows.
        lane_resolve_overlaps(si, sd, n, rs + RS_LKEYS, rs + RS_LORDER);
        // The duel start pass first: a fresh participant keeps its partner
        // created in the same tick before any exit evaluation runs.
        for (int r = 0; r < n; r++) {
            if (rs[RS_ADV + r] && !(int)si[SI_DUEL_ACTIVE + r]
                && (int)si[SI_DUEL_FIGHT + r] == 0) {
                duel_update(si, sf, sd, shl, r, n, course_distance,
                    has_corner_events);
            }
        }
        for (int r = 0; r < n; r++) {
            if (rs[RS_ADV + r] && (int)si[SI_SPOT_ACTIVE + r]
                && sd[SD_SPOT_TIMER + r] < 0.0
                && sd[SD_POS + r] >= section_length * 8.0) {
                si[SI_SPOT_LATCH + r] = 1u;
            }
        }
        // step_field_conditions: spot struggle then duel per advanced row.
        for (int r = 0; r < n; r++) {
            if (!rs[RS_ADV + r]) {
                continue;
            }
            spot_update(scr, si, sd, r, n, pd, section_length);
            duel_update(si, sf, sd, shl, r, n, course_distance,
                has_corner_events);
        }
        // The duel and spot receivers catch up with the flag columns after
        // the competition passes, in the engine's registration order.
        sk_token_sync(scr, si, n);
        // update(): ranks, order and the pacemaker, read next tick.
        field_update(scr, si, sd, n, dist_f, pace_end_f, course_distance,
            pm_range, pm_notfwd);
        // lib.rs finish detection, then the f32 time sum, then the capture.
        const float end_f = convert_float_rte(t) + dt;
        for (int r = 0; r < n; r++) {
            if (finish_state[r] == 0 && sd[SD_POS + r] > course_distance) {
                finish_state[r] = 1;
                const double step64 = (double)end_f;
                const double raw64 = (double)phys_finish_time(end_f, dt,
                    convert_float_rte(prev_pos[r]),
                    convert_float_rte(sd[SD_POS + r]),
                    convert_float_rte(course_distance));
                const ulong step_bits = as_ulong(step64);
                const ulong raw_bits = as_ulong(raw64);
                st[ST_FINISH0 + 4 * r + 0] = (uint)(step_bits & 0xFFFFFFFFu);
                st[ST_FINISH0 + 4 * r + 1] = (uint)(step_bits >> 32);
                st[ST_FINISH0 + 4 * r + 2] = (uint)(raw_bits & 0xFFFFFFFFu);
                st[ST_FINISH0 + 4 * r + 3] = (uint)(raw_bits >> 32);
            }
        }
        t = (double)convert_float_rte(convert_float_rte(t) + dt);
        for (int r = 0; r < n; r++) {
            race_frame(frames_all, id, tick + 1, n, r, sd[SD_POS + r],
                (double)sf[SF_OBSERVED + r], fmax(shl[r * SH_STRIDE + SH_HP], 0.0),
                sd[SD_LANE + r], (double)(int)si[SI_RMODE + r],
                (double)(int)si[SI_BLOCKER + r]);
            prev_pos[r] = sd[SD_POS + r];
        }
        any_open = 0;
        for (int r = 0; r < n; r++) {
            if (finish_state[r] == 0) {
                any_open = 1;
            }
        }
        tick += 1;
        if (tick > RACE_MAX_TICKS - 2) {
            break;
        }
    }
    st[ST_TICKS] = (uint)tick;
    // Runners never past the line finish at the loop's own end time, which
    // the host recomputes from the tick count with the same f32 sum.
    for (int r = 0; r < n; r++) {
        st[ST_TEMPT0 + r] = si[SI_TEMPT + r];
        st[ST_RSCORE0 + r] = convert_uint(sf[SF_RSCORE + r]);
        st[ST_HPEMPTY0 + r] = si[SI_HPEMPTY + r];
        st[ST_LSS0 + r] = as_uint(sf[SF_SP_START + r]);
    }
}


