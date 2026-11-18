// The rng kernel (handoff step 6.1). The ported engine RNG helpers live in
// rng_core.cl, which the host prepends to this file when it builds the program
// (see gpu/src/gpu.rs), so the initialization kernel shares the same port.
//
// A work item runs one chain. Buffers are flat arrays of OUT_STRIDE words per
// item, so each item keeps a private bank and the helpers hold the 56-word
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

// Output slots in out_u, per item. Slots 64 and up carry results. Bank 0..59
// is the RetailRandom bank (rng_core.cl BANK_*); the old slot names stay so
// rng.rs and tests/rng_gate.rs keep their layout.
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

__kernel void rng(__global const uint* in_u, __global const double* in_upper,
    __global const uchar* in_tag, __global uint* out_u, __global ulong* out_l) {
    const int id = get_global_id(0);
    __global const uint* ib = in_u + id * IN_STRIDE;
    __global uint* ob = out_u + id * OUT_STRIDE;
    __global ulong* olb = out_l + id * OUT_STRIDE;
    __global uint* bank = ob;
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
        r128_new(bank, seed);
        if (series == SERIES_NEXTINT) {
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = r128_next_int(bank);
        } else if (series == SERIES_NEXTDOUBLE) {
            for (int i = 0; i < R_ARR; i++) olb[OL_RES + i] = as_ulong(r128_next_double(bank));
        } else if (series == SERIES_INTEGER) {
            const int ilo = (int)ib[IS_INT_LO];
            const int ihi = (int)ib[IS_INT_HI];
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = (uint)r128_integer(bank, ilo, ihi);
        } else if (series == SERIES_REAL) {
            const float rlo = as_float(ib[IS_REAL_LO]);
            const float rhi = as_float(ib[IS_REAL_HI]);
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = as_uint(r128_real(bank, rlo, rhi));
        } else {
            const float rlo = as_float(ib[IS_REAL_LO]);
            const float rhi = as_float(ib[IS_REAL_HI]);
            for (int i = 0; i < R_ARR; i++) ob[OS_RES + i] = as_uint(r128_real_distance(bank, rlo, rhi));
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
        r128_new(bank, derived);
        for (int i = 0; i < SUB_CHAIN; i++) ob[OS_RES + i] = r128_next_int(bank);
        for (int i = 0; i < SUB_DBL; i++) olb[OL_SUB_DBL + i] = as_ulong(r128_next_double(bank));
        ob[OS_FLAG] = 0u;
    }
}
