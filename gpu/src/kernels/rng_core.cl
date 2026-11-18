// Engine RNGs, ported line for line from src/random_gen.rs (Rule30CARng) and
// src/retail_random.rs (RetailRandom legacy System.Random plus substream
// keying). The Rust reference wrappers in gpu/src/rng.rs are the authority.
//
// Shared between the rng kernel (rng.cl) and the initialization kernel
// (initialization.cl). The host builds each program by prepending this file,
// so the two kernels share one ported source (see gpu/src/gpu.rs).
//
// Exactness per docs/GPU_HANDOFF.md: no contract FMA formation, no
// -cl-fast-relaxed-math family options, build with
// -cl-fp32-correctly-rounded-divide-sqrt. Every RNG operation is u32/i32/f64 or
// f32, and each width is kept exactly as the Rust source.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// The handoff's rule against FMA formation, in both the C and OpenCL spellings.
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF

// Hard cap on the uniform() rejection loop.
#define UNIFORM_CAP 4096
// Written when uniform() hits UNIFORM_CAP, with the fallback flag set.
#define SENTINEL 2147483647

// A RetailRandom lives in one bank of words: state[0..56), a at 56, b at 57,
// draw_count at 58. The state stays in a buffer because OpenCL C cannot pass a
// private array to a helper.
#define BANK_WORDS 60
#define BANK_A 56
#define BANK_B 57
#define BANK_DC 58

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

// RetailRandom from src/retail_random.rs, over one bank. i32 values are stored
// as u32 bit patterns, + and - are the u32 wraps Rust's wrapping_* ops perform,
// and every signed test is an explicit (int) cast, so the kernel never runs a
// signed add that could overflow. The `current.wrapping_add(i32::MAX)` and
// `value.wrapping_add(i32::MAX)` steps are `+ 0x7FFFFFFF`, and the
// `value == i32::MAX` case is `== 0x7FFFFFFF`.

static void r128_new(__global uint* bank, const int seed) {
    const uint su = (uint)seed;
    uint mag;
    if (su == 0x80000000u) {
        // seed == i32::MIN takes magnitude i32::MAX, since i32::MIN.abs() overflows.
        mag = 0x7FFFFFFFu;
    } else {
        mag = ((int)su < 0) ? (0u - su) : su;
    }
    uint previous = 161803398u - mag;
    bank[55] = previous;
    uint current = 1u;
    for (int n = 1; n < 55; n++) {
        const int slot = (21 * n) % 55;
        bank[slot] = current;
        current = previous - current;
        if ((int)current < 0) current = current + 0x7FFFFFFFu;
        previous = bank[slot];
    }
    for (int r = 0; r < 4; r++) {
        for (int slot = 1; slot < 56; slot++) {
            const uint value = bank[slot] - bank[1 + ((slot + 30) % 55)];
            bank[slot] = ((int)value < 0) ? (value + 0x7FFFFFFFu) : value;
        }
    }
    bank[BANK_A] = 0u;
    bank[BANK_B] = 21u;
    bank[BANK_DC] = 0u;
}

static uint r128_next_int(__global uint* bank) {
    uint a = bank[BANK_A];
    uint b = bank[BANK_B];
    a = (a < 55u) ? (a + 1u) : 1u;
    b = (b < 55u) ? (b + 1u) : 1u;
    uint value = bank[a] - bank[b];
    if (value == 0x7FFFFFFFu) value = value - 1u;
    if ((int)value < 0) value = value + 0x7FFFFFFFu;
    bank[a] = value;
    bank[BANK_A] = a;
    bank[BANK_B] = b;
    bank[BANK_DC] += 1u;
    return value;
}

static double r128_next_double(__global uint* bank) {
    const uint value = r128_next_int(bank);
    return (double)((int)value) * RD_SCALE;
}

// integer(lower, upper). width above i32::MAX takes the wide branch. The caller
// picks per-item bounds so both widths run, with MIN..MAX forcing the wide one.
static int r128_integer(__global uint* bank, const int lower, const int upper) {
    const long width = (long)upper - (long)lower;
    if (width <= 2147483647L) {
        const double nd = r128_next_double(bank);
        const long v = (long)(nd * (double)width);
        return (int)(v + (long)lower);
    }
    const uint v0 = r128_next_int(bank);
    long value = (long)((int)v0);
    const int ni = (int)r128_next_int(bank);
    // C's % keeps the sign of the dividend, matching Rust, so -1 % 2 == -1.
    if ((ni % 2) == 0) value = -value;
    const double sample = (double)(value + 2147483646L) / 4294967293.0;
    const long t = (long)(sample * (double)width);
    return (int)(t + (long)lower);
}

static float r128_real_distance(__global uint* bank, const float lower, const float upper) {
    const double nd = r128_next_double(bank);
    const float sample = convert_float_rte(nd);
    const float width = fabs(upper - lower);
    const float value = convert_float_rte((double)sample * (double)width + (double)lower);
    if (value < lower) return lower;
    else if (value > upper) return upper;
    return value;
}

static float r128_real(__global uint* bank, const float lower, const float upper) {
    const double nd = r128_next_double(bank);
    const float sample = convert_float_rte(nd);
    const float value = sample * fabs(upper - lower) + lower;
    if (value < lower) return lower;
    else if (value > upper) return upper;
    return value;
}

static ulong r128_substream_hash(const int seed, const uint runner,
    const __global const uchar* tag, const int len) {
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

// The four-byte key "RUSH", so the initialization kernel needs no tag buffer.
static ulong r128_substream_hash4(const int seed, const uint runner) {
    ulong hash = 0xcbf29ce484222325UL ^ (ulong)(uint)seed;
    hash = (hash * 0x100000001b3UL) ^ (ulong)runner;
    const uchar bytes[4] = {'R', 'U', 'S', 'H'};
    for (int i = 0; i < 4; i++) {
        hash = (hash ^ (ulong)bytes[i]) * 0x100000001b3UL;
    }
    hash ^= hash >> 33;
    hash = hash * 0xff51afd7ed558ccdUL;
    hash ^= hash >> 33;
    hash = hash * 0xc4ceb9fe1a85ec53UL;
    hash ^= hash >> 33;
    return hash;
}

// The three-byte key "RUN", the fresh stream lib.rs:794 hands each runner's
// retail runtime in paired mode.
static ulong r128_substream_hash_run(const int seed, const uint runner) {
    ulong hash = 0xcbf29ce484222325UL ^ (ulong)(uint)seed;
    hash = (hash * 0x100000001b3UL) ^ (ulong)runner;
    const uchar bytes[3] = {'R', 'U', 'N'};
    for (int i = 0; i < 3; i++) {
        hash = (hash ^ (ulong)bytes[i]) * 0x100000001b3UL;
    }
    hash ^= hash >> 33;
    hash = hash * 0xff51afd7ed558ccdUL;
    hash ^= hash >> 33;
    hash = hash * 0xc4ceb9fe1a85ec53UL;
    hash ^= hash >> 33;
    return hash;
}
