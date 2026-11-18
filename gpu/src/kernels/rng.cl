// Engine RNGs, ported line for line from src/random_gen.rs (Rule30CARng) and
// src/retail_random.rs (RetailRandom legacy System.Random plus substream
// keying). The Rust reference wrappers in gpu/src/rng.rs are the authority.
//
// Exactness per docs/GPU_HANDOFF.md: no contract FMA formation, no
// -cl-fast-relaxed-math family options, build with
// -cl-fp32-correctly-rounded-divide-sqrt. Every RNG operation is u32/i32/f64 or
// f32, and each width is kept exactly as the Rust source.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// The handoff's rule against FMA formation, in both the C and OpenCL spellings.
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF

// A work item runs one chain. Buffers are flat arrays of OUT_STRIDE words per
// item, so each item keeps a private bank and the helpers can hold the 56-word
// RetailRandom state there instead of in a 56-entry local array.
#define IN_STRIDE 16
#define OUT_STRIDE 80
#define TAG_CAP 64

#define MODE_RULE30 0
#define MODE_RETAIL 1
#define MODE_SUBSTREAM 2

// One retail item runs one op against one fresh RNG.
#define SERIES_NEXTINT 0
#define SERIES_NEXTDOUBLE 1
#define SERIES_INTEGER 2
#define SERIES_REAL 3
#define SERIES_REALDIST 4

// Fixed chain lengths, mirrored in rng.rs and tests/rng_gate.rs.
// rule30 step() count before the final hi/lo snapshot.
#define R30_N 64
// rule30 int32() draws and random() draws after the snapshot.
#define R30_K 8
// retail per-series draw count. 12 matches the fixture integers array.
#define R_ARR 12
// substream derived-RNG next_int and next_double draws.
#define SUB_CHAIN 4
#define SUB_DBL 2
// Hard cap on the uniform() rejection loop.
#define UNIFORM_CAP 4096
// Written when uniform() hits UNIFORM_CAP, with the fallback flag set.
#define SENTINEL 2147483647

// Input slots in in_u, per item.
#define IS_MODE 0
#define IS_SERIES 1
#define IS_SEED_LO 2
#define IS_SEED_HI 3
#define IS_SEED_I32 4
#define IS_RUNNER 5
#define IS_REAL_LO 6
#define IS_REAL_HI 7
#define IS_INT_LO 8
#define IS_INT_HI 9
#define IS_TAG_LEN 10

// Output slots in out_u, per item. Slots 64 and up carry results.
#define OS_R_STATE 0      // 56 uints of RetailRandom state
#define OS_R_A 56
#define OS_R_B 57
#define OS_R_DC 58        // draw_count
#define OS_SUB_SEED 60    // substream derived seed, i32 bits
#define OS_FLAG 63        // 1 when uniform() hit UNIFORM_CAP
#define OS_RES 64         // retail series results. rule30 hi and substream next_int
                          // reuse this base: one mode runs per item.
#define OS_LO 65
#define OS_UNI 66
#define OS_I32 67         // rule30 int32 array, OS_I32 to OS_I32+R30_K-1

// Output slots in out_l, per item.
#define OL_RES 0          // retail next_double bits, rule30 random() bits
#define OL_SUB_DBL 8      // substream derived next_double bits

// next_double's multiplier is Rust's `1.0/i32::MAX as f64` (i32::MAX = 2147483647),
// the correctly-rounded double 1/2147483647, raw bits 0x3e00000000200000. The hex
// literal is that exact constant, and rng_gate.rs ties it to the Rust value.
#define RD_SCALE 0x1.0000000200000p-31

// step() from src/random_gen.rs. All u32. The Rust fixed-width shifts already
// truncate to 32 bits, so the Python masks from the source have no equivalent here.
static void r30_step(uint* hi_p, uint* lo_p) {
    const uint hi = *hi_p;
    const uint lo = *lo_p;
    const uint rot = hi >> 31;
    const uint rolhi = (hi << 1) | (lo >> 31);
    const uint rollo = (lo << 1) | rot;
    const uint rot2 = hi << 31;
    const uint rorhi = (hi >> 1) | (lo << 31);
    const uint rorlo = (lo >> 1) | rot2;
    *hi_p = rorhi ^ (hi | rolhi);
    *lo_p = rorlo ^ (lo | rollo);
}

static void r30_pair(uint* hi, uint* lo, uint* x, uint* y) {
    uint xx = 0u;
    uint yy = 0u;
    for (int j = 0; j < 16; j++) {
        xx = (xx << 2) | ((*hi & 0x10000u) >> 15) | (*hi & 1u);
        yy = (yy << 2) | ((*hi & 0x1000000u) >> 23) | ((*hi & 0x100u) >> 8);
        r30_step(hi, lo);
    }
    *x = xx;
    *y = yy;
}

static uint r30_int32(uint* hi, uint* lo) {
    uint x;
    uint y;
    r30_pair(hi, lo, &x, &y);
    return x;
}

static double r30_random(uint* hi, uint* lo) {
    uint x;
    uint y;
    r30_pair(hi, lo, &x, &y);
    // (hi & MASK_HI) * EXP + (lo & MASK_LO), exact in u64 since the result is
    // below 2^53, then one correctly-rounded f64 divide by 2^53.
    const ulong v = ((ulong)(x & 0x03FFFFFFu) << 27) + (ulong)(y & 0x07FFFFFFu);
    return (double)v / 9007199254740992.0;
}

// uniform(), with a hard cap. found=0 means the cap was hit, so the caller
// stores SENTINEL and sets the fallback flag instead of looping forever.
static int r30_uniform(uint* hi, uint* lo, const double upper, int* found) {
    const long v = (long)(upper - 1.0);
    // v | 1 is always odd, so clz stays in 0 to 31 and the mask shift is defined.
    const uint ored = (uint)(v | 1);
    const uint mask = 0xFFFFFFFFu >> clz(ored);
    for (int i = 0; i < UNIFORM_CAP; i++) {
        const uint n_u = r30_int32(hi, lo) & mask;
        // Cast through int to sign-extend, matching the Rust two's-complement branch.
        const long n = (long)((int)n_u);
        if ((double)n < upper) {
            *found = 1;
            return (int)n;
        }
    }
    *found = 0;
    return SENTINEL;
}

// RetailRandom from src/retail_random.rs. The state lives in out_u scratch.
// i32 values are stored as u32 bit patterns, + and - are the u32 wraps Rust's
// wrapping_* ops perform, and every signed test is an explicit (int) cast, so
// the kernel never runs a signed add that could overflow. The
// `current.wrapping_add(i32::MAX)` and `value.wrapping_add(i32::MAX)` steps are
// `+ 0x7FFFFFFF`, and the `value == i32::MAX` case is `== 0x7FFFFFFF`.

static void r128_new(__global uint* out, const int id, const int seed) {
    __global uint* st = out + id * OUT_STRIDE + OS_R_STATE;
    const uint su = (uint)seed;
    uint mag;
    if (su == 0x80000000u) {
        // seed == i32::MIN takes magnitude i32::MAX, since i32::MIN.abs() overflows.
        mag = 0x7FFFFFFFu;
    } else {
        mag = ((int)su < 0) ? (0u - su) : su;
    }
    uint previous = 161803398u - mag;
    st[55] = previous;
    uint current = 1u;
    for (int n = 1; n < 55; n++) {
        const int slot = (21 * n) % 55;
        st[slot] = current;
        current = previous - current;
        if ((int)current < 0) current = current + 0x7FFFFFFFu;
        previous = st[slot];
    }
    for (int r = 0; r < 4; r++) {
        for (int slot = 1; slot < 56; slot++) {
            const uint value = st[slot] - st[1 + ((slot + 30) % 55)];
            st[slot] = ((int)value < 0) ? (value + 0x7FFFFFFFu) : value;
        }
    }
    out[id * OUT_STRIDE + OS_R_A] = 0u;
    out[id * OUT_STRIDE + OS_R_B] = 21u;
    out[id * OUT_STRIDE + OS_R_DC] = 0u;
}

static uint r128_next_int(__global uint* out, const int id) {
    __global uint* st = out + id * OUT_STRIDE + OS_R_STATE;
    uint a = out[id * OUT_STRIDE + OS_R_A];
    uint b = out[id * OUT_STRIDE + OS_R_B];
    a = (a < 55u) ? (a + 1u) : 1u;
    b = (b < 55u) ? (b + 1u) : 1u;
    uint value = st[a] - st[b];
    if (value == 0x7FFFFFFFu) value = value - 1u;
    if ((int)value < 0) value = value + 0x7FFFFFFFu;
    st[a] = value;
    out[id * OUT_STRIDE + OS_R_A] = a;
    out[id * OUT_STRIDE + OS_R_B] = b;
    out[id * OUT_STRIDE + OS_R_DC] += 1u;
    return value;
}

static double r128_next_double(__global uint* out, const int id) {
    const uint value = r128_next_int(out, id);
    return (double)((int)value) * RD_SCALE;
}

// integer(lower, upper). width above i32::MAX takes the wide branch. The caller
// picks per-item bounds so both widths run, with MIN..MAX forcing the wide one.
static int r128_integer(__global uint* out, const int id, const int lower, const int upper) {
    const long width = (long)upper - (long)lower;
    if (width <= 2147483647L) {
        const double nd = r128_next_double(out, id);
        const long v = (long)(nd * (double)width);
        return (int)(v + (long)lower);
    }
    const uint v0 = r128_next_int(out, id);
    long value = (long)((int)v0);
    const int ni = (int)r128_next_int(out, id);
    // C's % keeps the sign of the dividend, matching Rust, so -1 % 2 == -1.
    if ((ni % 2) == 0) value = -value;
    const double sample = (double)(value + 2147483646L) / 4294967293.0;
    const long t = (long)(sample * (double)width);
    return (int)(t + (long)lower);
}

static float r128_real_distance(__global uint* out, const int id, const float lower, const float upper) {
    const double nd = r128_next_double(out, id);
    const float sample = convert_float_rte(nd);
    const float width = fabs(upper - lower);
    const float value = convert_float_rte((double)sample * (double)width + (double)lower);
    if (value < lower) return lower;
    else if (value > upper) return upper;
    return value;
}

static float r128_real(__global uint* out, const int id, const float lower, const float upper) {
    const double nd = r128_next_double(out, id);
    const float sample = convert_float_rte(nd);
    const float value = sample * fabs(upper - lower) + lower;
    if (value < lower) return lower;
    else if (value > upper) return upper;
    return value;
}

static ulong r128_substream_hash(const int seed, const uint runner,
    __global const uchar* tag, const int len) {
    ulong hash = 0xcbf29ce484222325UL ^ (ulong)(uint)seed;
    hash = (hash * 0x100000001b3UL) ^ (ulong)runner;
    for (int i = 0; i < len; i++) {
        hash = (hash ^ (ulong)tag[i]) * 0x100000001b3UL;
    }
    hash ^= hash >> 33;
    hash = hash * 0xff51afd7ed558ccdUL;
    hash ^= hash >> 33;
    hash = hash * 0xc4ceb9fe1a85ec53UL;
    hash ^= hash >> 33;
    return hash;
}

__kernel void rng(__global const uint* in_u, __global const double* in_upper,
    __global const uchar* in_tag, __global uint* out_u, __global ulong* out_l) {
    const int id = get_global_id(0);
    __global const uint* ib = in_u + id * IN_STRIDE;
    __global uint* ob = out_u + id * OUT_STRIDE;
    __global ulong* olb = out_l + id * OUT_STRIDE;
    const uint mode = ib[IS_MODE];
    const int series = (int)ib[IS_SERIES];

    if (mode == MODE_RULE30) {
        // new(seed_lo, seed_hi) stores hi = seed_hi and lo = seed_lo.
        uint hi = ib[IS_SEED_HI];
        uint lo = ib[IS_SEED_LO];
        for (int s = 0; s < R30_N; s++) r30_step(&hi, &lo);
        ob[OS_RES] = hi;
        ob[OS_LO] = lo;
        for (int k = 0; k < R30_K; k++) {
            ob[OS_I32 + k] = r30_int32(&hi, &lo);
        }
        for (int k = 0; k < R30_K; k++) {
            const double rv = r30_random(&hi, &lo);
            olb[OL_RES + k] = as_ulong(rv);
        }
        int found = 0;
        const int uni = r30_uniform(&hi, &lo, in_upper[id], &found);
        ob[OS_UNI] = (uint)uni;
        ob[OS_FLAG] = found ? 0u : 1u;
    } else if (mode == MODE_RETAIL) {
        const int seed = (int)ib[IS_SEED_I32];
        r128_new(out_u, id, seed);
        if (series == SERIES_NEXTINT) {
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = r128_next_int(out_u, id);
        } else if (series == SERIES_NEXTDOUBLE) {
            for (int i = 0; i < R_ARR; i++) olb[OL_RES + i] = as_ulong(r128_next_double(out_u, id));
        } else if (series == SERIES_INTEGER) {
            const int ilo = (int)ib[IS_INT_LO];
            const int ihi = (int)ib[IS_INT_HI];
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = (uint)r128_integer(out_u, id, ilo, ihi);
        } else if (series == SERIES_REAL) {
            const float rlo = as_float(ib[IS_REAL_LO]);
            const float rhi = as_float(ib[IS_REAL_HI]);
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = as_uint(r128_real(out_u, id, rlo, rhi));
        } else {
            const float rlo = as_float(ib[IS_REAL_LO]);
            const float rhi = as_float(ib[IS_REAL_HI]);
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = as_uint(r128_real_distance(out_u, id, rlo, rhi));
        }
        ob[OS_FLAG] = 0u;
    } else {
        const int seed = (int)ib[IS_SEED_I32];
        const uint runner = ib[IS_RUNNER];
        const int len = (int)ib[IS_TAG_LEN];
        const ulong hash = r128_substream_hash(seed, runner, in_tag + id * TAG_CAP, len);
        olb[OL_RES] = hash;
        const int derived = (int)(uint)(hash >> 33);
        ob[OS_SUB_SEED] = (uint)derived;
        r128_new(out_u, id, derived);
        for (int i = 0; i < SUB_CHAIN; i++) ob[OS_RES + i] = r128_next_int(out_u, id);
        for (int i = 0; i < SUB_DBL; i++) olb[OL_SUB_DBL + i] = as_ulong(r128_next_double(out_u, id));
        ob[OS_FLAG] = 0u;
    }
}
