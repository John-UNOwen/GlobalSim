// Skill table constants and initialization samplers (handoff step 7). Linked
// between rng_core.cl and initialization.cl: im_init runs the trigger
// samplers and the wisdom lottery inside the engine's exact stream order, and
// race.cl reads the same SK_* layout through the sk_all block. The encoding
// and the condition-support matrix are documented in gpu/SKILLS_DESIGN.md.
//
// Exactness follows rng_core.cl: both FP_CONTRACT OFF pragmas and
// -cl-fp32-correctly-rounded-divide-sqrt; every Rust `as f32` mirrors
// convert_float_rte and no constant expression may contract to an FMA.

#define SK_STRIDE 54912

#define SK_NREC 0
#define SK_NPROG 1
#define SK_NREG 2
#define SK_NEFF 3
#define SK_NTAG 4
#define SK_NSTR 5
#define SK_NCOR 6
#define SK_NSLOPE 7
#define SK_NTERM 8

#define SK_STRAIGHTS 16
#define SK_CORNERS 40
#define SK_SLOPES 64
#define SK_RECORDS 112
#define SK_EFFECTS 25072
#define SK_PROGRAMS 26992
#define SK_ARMS 39280
#define SK_REGIONS 41328
#define SK_TERMS 49520
#define SK_TAGS 50800
// The runner row remain_distance_viewer_id reads (playerHorseIndex).
#define SK_VIDX 9

#define SK_RSTRIDE 26
#define SK_ESTRIDE 10
#define SK_PSTRIDE 4
#define SK_GSTRIDE 4
#define SK_MSTRIDE 8
#define SK_ASTRIDE 8

// Records per race: the grid-mode cap of 120 runners with a full skill kit
// plus unique skill (8 per runner). sk_targets caps targets at the roster,
// so raising this past 120 changes no result, only the table budget.
#define SK_RMAX 960
#define SK_STRMAX 12
#define SK_CORMAX 12
#define SK_SLOPEMAX 16
#define SK_WINMAX 16
#define SK_MWINMAX 64
#define SK_ARMMAX 4

#define SK_FLAG_RETAIN 1
#define SK_FLAG_WISDOM 2
#define SK_FLAG_SAMPLED 4
#define SK_FLAG_DEAD 8
#define SK_FLAG_ANYACT 16
#define SK_FLAG_WISDOM_FOLLOW 32
#define SK_FLAG_INIT_DEFER 64
#define SK_FLAG_GREEN 128
#define SK_FLAG_PRIMARY 256
#define SK_FLAG_RSELIG 512
#define SK_FLAG_RANGOLD 1024
#define SK_FLAG_RANWHITE 2048
#define SK_FLAG_GENERAL 4096
#define SK_FLAG_BUMPED 8192
#define SK_FLAG_INIT_BUMP 16384

#define SK_TERM_ALL_CORNER 0
#define SK_TERM_STRAIGHT 1
#define SK_TERM_DOWN_SLOPE 2
#define SK_TERM_UP_SLOPE 3
#define SK_TERM_PHASE_RANDOM 4
#define SK_TERM_PHASE_FIRSTHALF 5
#define SK_TERM_PHASE_LATERHALF 6
#define SK_TERM_FC_RANDOM 7
#define SK_TERM_DIST_RATE_AFTER 8
#define SK_TERM_CORNER_RANDOM 9

// Arm entry flags. Bit 1 marks an arm whose static chain collapsed: its
// random term still draws, and the arm never passes the membership gate.
#define SK_ARM_SAMPLED 1
#define SK_ARM_DEAD 2

// Condition program node kinds, and the dynamic leaves with the comparison
// code packed after them (feat * 8 + cmp, cmp in engine token order).
#define OP_TRUE 0
#define OP_AND 1
#define OP_OR 2
#define OP_FEAT_I 4
#define FEAT_ORDER 1
#define FEAT_ORDER_RATE 2
#define FEAT_HP_PER 3
#define FEAT_IS_OVERTAKE 4
#define FEAT_IS_TEMPTATION 5
#define FEAT_LS_ENTRY 6
#define FEAT_PHASE 7
#define FEAT_CORNER 8
#define FEAT_FINALE 9
#define FEAT_FC_HALF 10
#define FEAT_DIST_RATE 11
#define FEAT_REMAIN 12
#define FEAT_CLOCK 13
#define FEAT_TEMPT 14
#define FEAT_ACT0 15
#define FEAT_ACT1 16
#define FEAT_BSIDE 17
#define FEAT_IFNEAR 18
#define FEAT_BHNEAR 19
#define FEAT_ORDER_DELTA 20
// The order_out_filter band cell, the percentage coming from the comparison
// argument word (20/40/50/70), always compared ==1 like the engine's filter.
#define FEAT_OUTBAND 21
#define FEAT_NEAR 22
#define FEAT_BASH_BEHIND 23
#define FEAT_BASH_INFRONT 24
#define FEAT_LASTSPURT 25
#define FEAT_BHNEAR1 26
#define FEAT_SLOPE 27
// The per-visit changed flag of the engine's observe_anyskill.
#define FEAT_ANYACT 28
#define FEAT_OTIME 29
#define FEAT_DDTOP 30
#define FEAT_HEAL 31
#define FEAT_CHGUP_END 32
#define FEAT_LANETYPE 33
#define FEAT_FIGHT 34
#define FEAT_BHIN 35
#define FEAT_MOVELANE 36
#define FEAT_BFRONT 37
// Wave-3 leaves: the in-band order cells read the band rate from the argument
// word (20/40/50/80), FEAT_USED the queried skill id. The band and used
// leaves only compare ==1, like the engine's filters.
#define FEAT_INBAND 38
#define FEAT_NEARIF 39
#define FEAT_VISN 40
#define FEAT_USED 41
#define FEAT_ACT2 42
#define FEAT_ACTLH 43
#define FEAT_ACTALL 44
#define FEAT_NOTOP 45
#define FEAT_NOTOD 46
#define FEAT_CHGUP_MID 47
#define FEAT_CHGUP_FC 48
#define FEAT_BALL 49
#define FEAT_BFRONTB 50
#define FEAT_SURR 51
#define FEAT_TEMPT_B 52
#define FEAT_TEMPT_F 53
#define FEAT_DDRATE 55
#define FEAT_DDTOPF 56
#define FEAT_RPHASE 57
// running_style_temptation_count_{nige,senko,sashi,oikomi} take 58 to 61, the
// group in the low two id bits. The _opponent_count_ variants take 62 to 65
// and leave the querying runner out, like the engine's opponents_only arm.
#define FEAT_RTEMP 58
#define FEAT_RTEMP_OPP 62
#define FEAT_RTEMP_END 65
// remain_distance_viewer_id: the engine's course-distance minus floor over the
// viewer row from SK_VIDX, the same form as FEAT_REMAIN for that runner.
#define FEAT_VREMAIN 66

// The (skill, effect) dispatch kinds the port covers.
#define SK_EFF_SPEEDUP 1
#define SK_EFF_STAMINUP 2
#define SK_EFF_GUTSUP 4
#define SK_EFF_WISDOMUP 5
#define SK_EFF_RUNAWAY 6
#define SK_EFF_VISION 8
#define SK_EFF_NOOP 10
#define SK_EFF_GOAL_OUT 11     // PreferOutsideLane
#define SK_EFF_GOAL_IN 12      // PreferInsideLane
#define SK_EFF_EXTKAKARI 13    // ExtendKakari
#define SK_EFF_SETDELAY 14
#define SK_EFF_CURRENT 21
#define SK_EFF_CURRENT_NAT 22
#define SK_EFF_TARGET 27
#define SK_EFF_LANESPEED 28
#define SK_EFF_KAKARI 29
#define SK_EFF_ACCEL 31
#define SK_EFF_SETLANE 35      // SetTargetLane
#define SK_EFF_IMMUNE 38       // DebuffImmunity
#define SK_EFF_RECOVERY 9

// Runtime record state, per record in the per-race scratch state block.
#define SK_S_READY 0
#define SK_S_WAITING 1
#define SK_S_COOLDOWN 2
#define SK_S_DEAD 3
#define SK_S_DISABLED 4

// Word layout of one record's state block (SKR_STRIDE words).
#define SKR_STRIDE 24
#define SKR_STATE 0
#define SKR_NWIN 1
#define SKR_SINGLE 2
#define SKR_LATCH 3
#define SKR_AWIN0 4
#define SKR_AWIN1 5
#define SKR_AWIN2 6
#define SKR_AWIN3 7
#define SKR_USED 8         // the record owner has activated this skill id
#define SKR_AA_RUN 9       // a delayed running activation exists for this record
#define SKR_AA_UNAPPLIED 10 // activations queued for this record's next turn
#define SKR_GREEN 11       // this (runner, skill id) counted in the green set
#define SKR_RESERVED 12    // the random-skill reservation picked this record
#define SKR_AA_COUNT 13    // per-effect-slot delivered counts, SKR_AA_COUNT + e
#define SKR_WISFAIL 21     // the ordinary lottery failed; only reservations fire

// Double layout of one record's dynamic block (SKD_STRIDE doubles). The
// effect timers are indexed by effect slot; an effect fires at most once per
// (skill, detail) cycle, so one timer chain per slot mirrors the engine's
// per-receiver duration timers.
#define SKD_STRIDE 276
#define SKD_EFF_T 268    // per-effect duration timers, negative while active
#define SKD_ANY_LAST 4    // observe_anyskill: last activation count seen
#define SKD_ANY_TRUES 5   // observe_anyskill: activations already counted
#define SKD_ANY_RESULT 6  // observe_anyskill: changed flag, read by FEAT_ANYACT
#define SKD_INSTANCE 8    // random_active_details (skill, detail) instance timer
#define SKD_COOLDOWN 9    // retail cooldown timer
#define SKD_RUN_T 0       // running_skills timer, extended on an overtake
#define SKD_AEXT 3        // overtake extensions of that instance, three at most
#define SKD_TRIG0 10      // trigger start, first merged window start
#define SKD_TRIG1 11      // trigger end, last merged window end
#define SKD_WINDOWS 12    // merged arm chains, up to 64 (start, end) pairs
#define SKD_ARMS 140      // per-arm runtime chains, 16 pairs per arm slot

// f32::MAX, the phase sampler's tail bound.
#define SK_FLT_MAX 0x1.fffffep+127f

static void sk_substream_bank(__global uint* bank, const int seed, const uint runner,
    __global const uchar* tag, const int len) {
    const ulong hash = r128_substream_hash(seed, runner, tag, len);
    r128_new(bank, (int)(hash >> 33));
}

// distance_window in retail_initialization.rs: real_distance over
// (start, max(start, end-10)), and the window end is the value plus ten.
static void sk_distance_window(__global uint* bank, const float start, const float end,
    float* lo, float* hi) {
    const float v = r128_real_distance(bank, start, fmax(start, end - 10.0f));
    *lo = v;
    *hi = v + 10.0f;
}

/// sample_trigger in retail_initialization.rs for the ported samplers,
/// drawing on the caller's bank. all_corner_random returns its windows sorted
/// by start, the others are single windows. Returns the window count.
static int sk_sample_trigger(const uint kind, const int arg, __global uint* bank,
    __global const uint* sk, const float distance, const int corner_idx,
    float* wlo, float* whi) {
    int n = 0;
    if (kind == SK_TERM_ALL_CORNER) {
        const int nc = (int)sk[SK_NCOR];
        float cs[SK_CORMAX];
        float ce[SK_CORMAX];
        int m = 0;
        for (int i = 0; i < nc; i++) {
            cs[i] = as_float(sk[SK_CORNERS + 2 * i]);
            ce[i] = as_float(sk[SK_CORNERS + 2 * i + 1]);
            m++;
        }
        for (int round = 0; round < 4 && m > 0 && n < 4; round++) {
            const uint index = (uint)r128_integer(bank, 0, m);
            float start = cs[index];
            const float end = ce[index];
            float lo, hi;
            sk_distance_window(bank, start, end, &lo, &hi);
            wlo[n] = lo;
            whi[n] = hi;
            n++;
            // The engine drops the picked candidate when its run is almost
            // covered, and otherwise raises its start to the window end. The
            // retain keeps candidates at or after that start, which in the
            // drop branch is the candidate's original start.
            if (end - hi < 10.0f) {
                for (int i = (int)index; i + 1 < m; i++) {
                    cs[i] = cs[i + 1];
                    ce[i] = ce[i + 1];
                }
                m--;
            } else {
                cs[index] = hi;
                start = hi;
            }
            int k = 0;
            for (int i = 0; i < m; i++) {
                if (cs[i] >= start) {
                    cs[k] = cs[i];
                    ce[k] = ce[i];
                    k++;
                }
            }
            m = k;
        }
        // windows.sort_by(start) from the engine, stable insertion sort.
        for (int i = 1; i < n; i++) {
            for (int j = i; j > 0 && wlo[j] < wlo[j - 1]; j--) {
                const float a = wlo[j];
                wlo[j] = wlo[j - 1];
                wlo[j - 1] = a;
                const float b = whi[j];
                whi[j] = whi[j - 1];
                whi[j - 1] = b;
            }
        }
        return n;
    }
    if (kind == SK_TERM_STRAIGHT) {
        const int ns = (int)sk[SK_NSTR];
        if (ns > 0) {
            const uint index = (uint)r128_integer(bank, 0, ns);
            sk_distance_window(bank, as_float(sk[SK_STRAIGHTS + 2 * index]),
                as_float(sk[SK_STRAIGHTS + 2 * index + 1]), wlo, whi);
            return 1;
        }
        return 0;
    }
    if (kind >= SK_TERM_PHASE_RANDOM && kind <= SK_TERM_PHASE_LATERHALF) {
        const float section = distance / 6.0f;
        const float phases[6] = {0.0f, section, section * 4.0f, section * 5.0f,
            distance, SK_FLT_MAX};
        float start = phases[arg];
        float end = phases[arg + 1];
        const float middle = (end - start) * 0.5f + start;
        if (kind == SK_TERM_PHASE_FIRSTHALF) end = middle;
        if (kind == SK_TERM_PHASE_LATERHALF) start = middle;
        sk_distance_window(bank, start, end, wlo, whi);
        return 1;
    }
    if (kind == SK_TERM_FC_RANDOM || kind == SK_TERM_CORNER_RANDOM) {
        // corner_random resolves its request argument to one course corner row
        // at compile time, so both kinds draw the same window there.
        if (corner_idx >= 0 && corner_idx < (int)sk[SK_NCOR]) {
            sk_distance_window(bank, as_float(sk[SK_CORNERS + 2 * corner_idx]),
                as_float(sk[SK_CORNERS + 2 * corner_idx + 1]), wlo, whi);
            return 1;
        }
        return 0;
    }
    if (kind == SK_TERM_DIST_RATE_AFTER) {
        // distance_rate_after_random: one distance_window opened at the rate
        // point of the race distance, matching the retail trigger arm.
        sk_distance_window(bank, distance * (float)arg / 100.0f, distance, wlo, whi);
        return 1;
    }
    // up_slope_random / down_slope_random: one pick among the sign-matching
    // slopes in course order, windowed over the slope run.
    const int uphill = kind == SK_TERM_UP_SLOPE ? 1 : 0;
    float ss[SK_SLOPEMAX];
    float se[SK_SLOPEMAX];
    int m = 0;
    const int ns = (int)sk[SK_NSLOPE];
    for (int i = 0; i < ns; i++) {
        const float v = as_float(sk[SK_SLOPES + 3 * i + 2]);
        if (v != 0.0f && ((v > 0.0f) == uphill)) {
            ss[m] = as_float(sk[SK_SLOPES + 3 * i]);
            se[m] = as_float(sk[SK_SLOPES + 3 * i + 1]);
            m++;
        }
    }
    if (m > 0) {
        const uint index = (uint)r128_integer(bank, 0, m);
        sk_distance_window(bank, ss[index], se[index], wlo, whi);
        return 1;
    }
    return 0;
}

static double sk_read_f64(__global const uint* w) {
    return as_double((ulong)w[0] | ((ulong)w[1] << 32));
}

/// The engine's Region chain step for the runtime window list: every pair of
/// surviving intervals against every list entry, empties dropped, exactly
/// like rmap_multi's intersect with its start filter.
static int sk_chain(double* out, const double* in, int m,
    const double* lst, int k) {
    int c = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            const double s = fmax(in[2 * i], lst[2 * j]);
            const double e = fmin(in[2 * i + 1], lst[2 * j + 1]);
            if (e > s && c < SK_WINMAX) {
                out[2 * c] = s;
                out[2 * c + 1] = e;
                c++;
            }
        }
    }
    return c;
}

/// region.rs union: append, stable sort by start, then fold. Only strict
/// overlaps merge, adjacent regions stay separate.
static int sk_union(double* out, const double* a, const int na, const double* b,
    const int nb) {
    double s[2 * SK_MWINMAX];
    int m = 0;
    for (int i = 0; i < na; i++) {
        s[2 * m] = a[2 * i];
        s[2 * m + 1] = a[2 * i + 1];
        m++;
    }
    for (int i = 0; i < nb; i++) {
        s[2 * m] = b[2 * i];
        s[2 * m + 1] = b[2 * i + 1];
        m++;
    }
    for (int i = 1; i < m; i++) {
        for (int j = i; j > 0 && s[2 * j] < s[2 * (j - 1)]; j--) {
            const double x = s[2 * j];
            s[2 * j] = s[2 * (j - 1)];
            s[2 * (j - 1)] = x;
            const double y = s[2 * j + 1];
            s[2 * j + 1] = s[2 * (j - 1) + 1];
            s[2 * (j - 1) + 1] = y;
        }
    }
    if (m == 0) return 0;
    double a0 = s[0];
    double a1 = s[1];
    int c = 0;
    for (int i = 1; i < m; i++) {
        const double b0 = s[2 * i];
        const double b1 = s[2 * i + 1];
        if (a0 <= b0 && a1 >= b1) {
            continue;
        } else if (a0 <= b0 && b0 < a1) {
            a1 = b1;
        } else if (a0 < b1 && b1 <= a1) {
            a0 = b0;
        } else {
            out[2 * c] = a0;
            out[2 * c + 1] = a1;
            c++;
            a0 = b0;
            a1 = b1;
        }
    }
    out[2 * c] = a0;
    out[2 * c + 1] = a1;
    return c + 1;
}

/// The initialization skill stage, runners then wire-order records: the
/// trigger draws and the per-arm window chains. im_init calls this right
/// after r128_new, before the force-in pass, so the shared stream advances
/// over the trigger draws exactly where the engine's after_triggers stamp
/// falls. Each arm chains its compiled pre-list through the sampled windows
/// into its compiled post-list; a chain that is already fully folded at
/// compile time is copied. The merged list is the arms folded right through
/// the engine's union, matching the Or closure's region algebra.
static void sk_init_triggers(__global const uint* sk, __global uint* sks,
    __global double* skd, __global uint* bank, __global uint* sub, const int paired,
    const int seed, const int n, const double distance) {
    const int nrec = (int)sk[SK_NREC];
    for (int r = 0; r < n; r++) {
        for (int rec = 0; rec < nrec; rec++) {
            const __global uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            if ((int)rd[1] != r) continue;
            __global double* wd = skd + SKD_STRIDE * rec;
            for (int w = 0; w < SKD_STRIDE; w++) wd[w] = 0.0;
            sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_READY;
            sks[SKR_STRIDE * rec + SKR_LATCH] = 0u;
            sks[SKR_STRIDE * rec + SKR_USED] = 0u;
            for (int w = SKR_AWIN0; w < SKR_AWIN0 + SK_ARMMAX; w++) {
                sks[SKR_STRIDE * rec + w] = 0u;
            }
            sks[SKR_STRIDE * rec + SKR_AA_RUN] = 0u;
            sks[SKR_STRIDE * rec + SKR_AA_UNAPPLIED] = 0u;
            sks[SKR_STRIDE * rec + SKR_GREEN] = 0u;
            sks[SKR_STRIDE * rec + SKR_RESERVED] = 0u;
            for (int w = SKR_AA_COUNT; w < SKR_STRIDE; w++) {
                sks[SKR_STRIDE * rec + w] = 0u;
            }
            const uint dead = (rd[2] & SK_FLAG_DEAD) ? 1u : 0u;
            int mcount = 0;
            if (dead) {
                // Trigger sampling happens in the engine before any region is
                // evaluated, so a record whose precondition empties it still
                // spends its random draws. Take them so the shared stream
                // stays aligned, then leave the record inert.
                for (int a = 0; a < (int)rd[20]; a++) {
                    const __global uint* ar =
                        sk + SK_ARMS + SK_ASTRIDE * ((int)rd[21] + a);
                    if (!(ar[6] & SK_ARM_SAMPLED)) continue;
                    const __global uint* tm =
                        sk + SK_TERMS + SK_MSTRIDE * (int)ar[7];
                    float wlo[4];
                    float whi[4];
                    if (paired) {
                        sk_substream_bank(sub, seed, (uint)r,
                            (__global const uchar*)(sk + SK_TAGS) + tm[2],
                            (int)tm[3]);
                        (void)sk_sample_trigger(tm[0], (int)tm[1], sub, sk,
                            convert_float_rte(distance), (int)tm[4], wlo, whi);
                    } else {
                        (void)sk_sample_trigger(tm[0], (int)tm[1], bank, sk,
                            convert_float_rte(distance), (int)tm[4], wlo, whi);
                    }
                }
            } else {
                const int narms = (int)rd[20];
                double cur[2 * SK_WINMAX];
                double nxt[2 * SK_WINMAX];
                double merged[2 * SK_MWINMAX];
                for (int a = 0; a < narms; a++) {
                    const __global uint* ar =
                        sk + SK_ARMS + SK_ASTRIDE * ((int)rd[21] + a);
                    __global double* ad = wd + SKD_ARMS + 2 * SK_WINMAX * a;
                    int nwin = 0;
                    const uint sampled = (ar[6] & SK_ARM_SAMPLED) ? 1u : 0u;
                    double samp[2 * 4];
                    int nsamp = 0;
                    if (sampled) {
                        const __global uint* tm =
                            sk + SK_TERMS + SK_MSTRIDE * (int)ar[7];
                        float wlo[4];
                        float whi[4];
                        if (paired) {
                            // The keyed stream is seeded from (seed, runner,
                            // tag) and never touches the shared bank.
                            sk_substream_bank(sub, seed, (uint)r,
                                (__global const uchar*)(sk + SK_TAGS) + tm[2],
                                (int)tm[3]);
                            nsamp = sk_sample_trigger(tm[0], (int)tm[1], sub, sk,
                                convert_float_rte(distance), (int)tm[4], wlo, whi);
                        } else {
                            nsamp = sk_sample_trigger(tm[0], (int)tm[1], bank, sk,
                                convert_float_rte(distance), (int)tm[4], wlo, whi);
                        }
                        for (int i = 0; i < nsamp; i++) {
                            samp[2 * i] = (double)wlo[i];
                            samp[2 * i + 1] = (double)whi[i];
                        }
                    }
                    if (!(ar[6] & SK_ARM_DEAD)) {
                        if (sampled) {
                            nwin = (int)ar[2];
                            const __global uint* ra =
                                sk + SK_REGIONS + SK_GSTRIDE * (int)ar[3];
                            for (int i = 0; i < nwin; i++) {
                                cur[2 * i] = sk_read_f64(ra + SK_GSTRIDE * i);
                                cur[2 * i + 1] = sk_read_f64(ra + SK_GSTRIDE * i + 2);
                            }
                            nwin = sk_chain(nxt, cur, nwin, samp, nsamp);
                            for (int i = 0; i < 2 * nwin; i++) cur[i] = nxt[i];
                            if (ar[4] > 0 && nwin > 0) {
                                const __global uint* rb =
                                    sk + SK_REGIONS + SK_GSTRIDE * (int)ar[5];
                                double post[2 * SK_WINMAX];
                                const int k = (int)ar[4];
                                for (int i = 0; i < k; i++) {
                                    post[2 * i] = sk_read_f64(rb + SK_GSTRIDE * i);
                                    post[2 * i + 1] =
                                        sk_read_f64(rb + SK_GSTRIDE * i + 2);
                                }
                                nwin = sk_chain(nxt, cur, nwin, post, k);
                                for (int i = 0; i < 2 * nwin; i++) cur[i] = nxt[i];
                            }
                        } else {
                            nwin = (int)ar[4];
                            const __global uint* rb =
                                sk + SK_REGIONS + SK_GSTRIDE * (int)ar[5];
                            for (int i = 0; i < nwin; i++) {
                                cur[2 * i] = sk_read_f64(rb + SK_GSTRIDE * i);
                                cur[2 * i + 1] = sk_read_f64(rb + SK_GSTRIDE * i + 2);
                            }
                        }
                        for (int i = 0; i < 2 * nwin; i++) ad[i] = cur[i];
                    }
                    sks[SKR_STRIDE * rec + SKR_AWIN0 + a] = (uint)nwin;
                    if (a == 0) {
                        for (int i = 0; i < 2 * nwin; i++) merged[i] = cur[i];
                        mcount = nwin;
                    } else {
                        double um[2 * SK_MWINMAX];
                        const int u = sk_union(um, merged, mcount, cur, nwin);
                        for (int i = 0; i < 2 * u; i++) merged[i] = um[i];
                        mcount = u;
                    }
                }
                for (int i = 0; i < 2 * mcount; i++) wd[SKD_WINDOWS + i] = merged[i];
            }
            sks[SKR_STRIDE * rec + SKR_NWIN] = (uint)mcount;
            double t0 = 0.0;
            double t1 = 0.0;
            if (mcount > 0) {
                t0 = wd[SKD_WINDOWS];
                t1 = wd[SKD_WINDOWS + 2 * (mcount - 1) + 1];
            }
            wd[SKD_TRIG0] = t0;
            wd[SKD_TRIG1] = t1;
            // w.single means the one merged window spanning the trigger extent.
            uint single = (mcount == 1) ? 1u : 0u;
            sks[SKR_STRIDE * rec + SKR_SINGLE] = single;
            // No surviving window means the condition rejected the trigger
            // (engine: an empty window list never fires).
            if (mcount == 0) sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DEAD;
        }
    }
}

/// The wisdom lottery: one roll per wisdomCheck record in the engine's order
/// (runners, then wire-order skills), chance from base_wisdom in f64. A
/// failed roll disables the record, and the enabled records' delay
/// multipliers are the product per runner the delay draw then needs. Rolls
/// run even for records whose window list is already empty, since the engine
/// rolls before it ever looks at windows.
static void sk_init_wisdom(__global const uint* sk, __global uint* sks,
    __global uint* bank, __global uint* sub, __global const ulong* ilb,
    double* mult_out, const int paired, const int seed, const int n,
    const int wis_slot) {
    const int nrec = (int)sk[SK_NREC];
    for (int r = 0; r < n; r++) {
        const double base_wisdom = as_double(ilb[wis_slot + r]);
        const double chance = fmax(100.0 - 9000.0 / base_wisdom, 20.0);
        float mult = 1.0f;
        // The engine's retail init rolls the skill lottery once per skill id,
        // so one record of each wisdom group rolls and followers inherit it.
        uint group_sid = 0xffffffffu;
        uint group_enabled = 1u;
        for (int rec = 0; rec < nrec; rec++) {
            __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
            if ((int)rd[1] != r) continue;
            uint enabled = 1u;
            if (rd[2] & SK_FLAG_WISDOM) {
                float roll;
                if (paired) {
                    sk_substream_bank(sub, seed, (uint)r,
                        (__global const uchar*)(sk + SK_TAGS) + rd[15], (int)rd[16]);
                    roll = r128_real(sub, 0.0f, 100.0f);
                } else {
                    roll = r128_real(bank, 0.0f, 100.0f);
                }
                enabled = (double)chance > (double)roll ? 1u : 0u;
                group_sid = rd[0];
                group_enabled = enabled;
            } else if (rd[2] & SK_FLAG_WISDOM_FOLLOW) {
                enabled = rd[0] == group_sid ? group_enabled : 1u;
            }
            if (!enabled && sks[SKR_STRIDE * rec + SKR_STATE] == (uint)SK_S_READY) {
                sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DISABLED;
                // A retained primary stays pending for a reservation fire;
                // the extra condition of the ordinary path never holds again
                // (race_solver_builder.rs:1002-1035).
                sks[SKR_STRIDE * rec + SKR_WISFAIL] = 1u;
            }
            if (enabled) mult *= as_float(rd[17]);
        }
        mult_out[r] = (double)mult;
    }
}

/// The lottery outcome of a skill's (runner, skill id) group, consulted by
/// records whose static constructor fire the kernel bakes after the roll.
static uint sk_gate_enabled(__global const uint* sk, __global const uint* sks,
    int rec) {
    const int nrec = (int)sk[SK_NREC];
    __global const uint* rd = sk + SK_RECORDS + SK_RSTRIDE * rec;
    for (int q = 0; q < nrec; q++) {
        __global const uint* qd = sk + SK_RECORDS + SK_RSTRIDE * q;
        if (qd[0] == rd[0] && qd[1] == rd[1] && (qd[2] & SK_FLAG_WISDOM)) {
            return sks[SKR_STRIDE * q + SKR_STATE] != (uint)SK_S_DISABLED;
        }
    }
    return 1u;
}
