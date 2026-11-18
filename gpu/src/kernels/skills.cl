// Skill table constants and initialization samplers (handoff step 7). Linked
// between rng_core.cl and initialization.cl: im_init runs the trigger
// samplers and the wisdom lottery inside the engine's exact stream order, and
// race.cl reads the same SK_* layout through the sk_all block. The encoding
// and the condition-support matrix are documented in gpu/SKILLS_DESIGN.md.
//
// Exactness follows rng_core.cl: both FP_CONTRACT OFF pragmas and
// -cl-fp32-correctly-rounded-divide-sqrt; every Rust `as f32` mirrors
// convert_float_rte and no constant expression may contract to an FMA.

#define SK_STRIDE 4480

#define SK_NREC 0
#define SK_NPROG 1
#define SK_NREG 2
#define SK_NEFF 3
#define SK_NTAG 4
#define SK_NSTR 5
#define SK_NCOR 6
#define SK_NSLOPE 7

#define SK_STRAIGHTS 16
#define SK_CORNERS 40
#define SK_SLOPES 64
#define SK_RECORDS 112
#define SK_EFFECTS 752
#define SK_PROGRAMS 1232
#define SK_REGIONS 2512
#define SK_TAGS 3536
#define SK_TERMS 4048

#define SK_RSTRIDE 20
#define SK_ESTRIDE 10
#define SK_PSTRIDE 4
#define SK_GSTRIDE 4
#define SK_MSTRIDE 8

#define SK_RMAX 32
#define SK_STRMAX 12
#define SK_CORMAX 12
#define SK_SLOPEMAX 16
#define SK_WINMAX 16

#define SK_FLAG_RETAIN 1
#define SK_FLAG_WISDOM 2
#define SK_FLAG_SAMPLED 4
#define SK_FLAG_DEAD 8

#define SK_TERM_ALL_CORNER 0
#define SK_TERM_STRAIGHT 1
#define SK_TERM_DOWN_SLOPE 2
#define SK_TERM_UP_SLOPE 3

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

// The (skill, effect) dispatch kinds the port covers.
#define SK_EFF_TARGET 27
#define SK_EFF_ACCEL 31
#define SK_EFF_CURRENT 21

// Runtime record state, per record in the per-race scratch state block.
#define SK_S_READY 0
#define SK_S_WAITING 1
#define SK_S_COOLDOWN 2
#define SK_S_DEAD 3
#define SK_S_DISABLED 4

// Word layout of one record's state block (SKR_STRIDE words).
#define SKR_STRIDE 8
#define SKR_STATE 0
#define SKR_NWIN 1
#define SKR_SINGLE 2

// Double layout of one record's dynamic block (SKD_STRIDE doubles).
#define SKD_STRIDE 44
#define SKD_T_TIMER 0     // target effect duration timer, negative while active
#define SKD_A_TIMER 1     // accel effect duration timer
#define SKD_C_TIMER 2     // current-speed effect duration timer
#define SKD_INSTANCE 8    // (skill, detail) instance timer
#define SKD_COOLDOWN 9    // retail cooldown timer
#define SKD_TRIG0 10      // trigger start, first window start
#define SKD_TRIG1 11      // trigger end, last window end
#define SKD_WINDOWS 12    // up to 16 (start, end) pairs

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

/// sample_trigger in retail_initialization.rs for the four ported samplers,
/// drawing on the caller's bank. all_corner_random returns its windows sorted
/// by start, the others are single windows. Returns the window count.
static int sk_sample_trigger(const uint kind, const int arg, __global uint* bank,
    __global const uint* sk, float* wlo, float* whi) {
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

/// The initialization skill stage, runners then wire-order records: the
/// trigger draws and the final window list. im_init calls this right after
/// r128_new, before the force-in pass, so the shared stream advances over the
/// trigger draws exactly where the engine's after_triggers stamp falls. The
/// final list is the engine's region chain: the compiled pre-list intersected
/// with the sampled windows in draw order, then the compiled post-list. A
/// non-sampled record stores its whole folded chain as the post-list, so the
/// pre-list slot stays at zero.
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
            const uint dead = (rd[2] & SK_FLAG_DEAD) ? 1u : 0u;
            int nwin = 0;
            if (!dead) {
                double cur[2 * SK_WINMAX];
                double nxt[2 * SK_WINMAX];
                const uint sampled = (rd[2] & SK_FLAG_SAMPLED) ? 1u : 0u;
                if (sampled) {
                    nwin = (int)rd[11];
                    const __global uint* ra =
                        sk + SK_REGIONS + SK_GSTRIDE * (int)rd[12];
                    for (int i = 0; i < nwin; i++) {
                        cur[2 * i] = sk_read_f64(ra + SK_GSTRIDE * i);
                        cur[2 * i + 1] = sk_read_f64(ra + SK_GSTRIDE * i + 2);
                    }
                    const __global uint* tm =
                        sk + SK_TERMS + SK_MSTRIDE * (int)rd[10];
                    float wlo[4];
                    float whi[4];
                    int nsamp = 0;
                    if (paired) {
                        // The keyed stream is seeded from (seed, runner, tag)
                        // and never touches the shared bank.
                        sk_substream_bank(sub, seed, (uint)r,
                            (__global const uchar*)(sk + SK_TAGS) + tm[2], (int)tm[3]);
                        nsamp = sk_sample_trigger(tm[0], (int)tm[1], sub, sk, wlo, whi);
                    } else {
                        nsamp = sk_sample_trigger(tm[0], (int)tm[1], bank, sk, wlo, whi);
                    }
                    double samp[2 * 4];
                    for (int i = 0; i < nsamp; i++) {
                        samp[2 * i] = (double)wlo[i];
                        samp[2 * i + 1] = (double)whi[i];
                    }
                    nwin = sk_chain(nxt, cur, nwin, samp, nsamp);
                    for (int i = 0; i < 2 * nwin; i++) cur[i] = nxt[i];
                    if (rd[18] > 0 && nwin > 0) {
                        const __global uint* rb =
                            sk + SK_REGIONS + SK_GSTRIDE * (int)rd[19];
                        double post[2 * SK_WINMAX];
                        const int k = (int)rd[18];
                        for (int i = 0; i < k; i++) {
                            post[2 * i] = sk_read_f64(rb + SK_GSTRIDE * i);
                            post[2 * i + 1] = sk_read_f64(rb + SK_GSTRIDE * i + 2);
                        }
                        nwin = sk_chain(nxt, cur, nwin, post, k);
                    }
                } else {
                    nwin = (int)rd[18];
                    const __global uint* rb =
                        sk + SK_REGIONS + SK_GSTRIDE * (int)rd[19];
                    for (int i = 0; i < nwin; i++) {
                        cur[2 * i] = sk_read_f64(rb + SK_GSTRIDE * i);
                        cur[2 * i + 1] = sk_read_f64(rb + SK_GSTRIDE * i + 2);
                    }
                }
                (void)distance;
                for (int i = 0; i < 2 * nwin; i++) wd[SKD_WINDOWS + i] = cur[i];
            }
            sks[SKR_STRIDE * rec + SKR_NWIN] = (uint)nwin;
            double t0 = 0.0;
            double t1 = 0.0;
            if (nwin > 0) {
                t0 = wd[SKD_WINDOWS];
                t1 = wd[SKD_WINDOWS + 2 * (nwin - 1) + 1];
            }
            wd[SKD_TRIG0] = t0;
            wd[SKD_TRIG1] = t1;
            // w.single means the one window spanning the trigger extent.
            uint single = (nwin == 1) ? 1u : 0u;
            sks[SKR_STRIDE * rec + SKR_SINGLE] = single;
            // No surviving window means the condition rejected the trigger
            // (engine: an empty window list never fires).
            if (nwin == 0) sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DEAD;
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
            }
            if (!enabled && sks[SKR_STRIDE * rec + SKR_STATE] == (uint)SK_S_READY) {
                sks[SKR_STRIDE * rec + SKR_STATE] = (uint)SK_S_DISABLED;
            }
            if (enabled) mult *= as_float(rd[17]);
        }
        mult_out[r] = (double)mult;
    }
}
