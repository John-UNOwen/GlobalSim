// Restricted flat no-skills loop, ported line for line from
// src/retail_physics.rs plus the restricted oracle's tick chain. The Rust
// reference in gpu/src/reference.rs is the authority on every value.
//
// Exactness per docs/GPU_HANDOFF.md: no contract FMA formation, no
// -cl-fast-relaxed-math family options, build with
// -cl-fp32-correctly-rounded-divide-sqrt. Rust evaluates float literals
// round-to-nearest and rounds `as` casts the same way, so decimal literals
// and convert_*_rte here match the Rust text.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// The handoff's rule against FMA formation, in both the C and OpenCL spellings.
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF

#ifndef ROWS
#define ROWS 8
#endif

// Mirrors reference.rs KernelParams (repr(C) f32-only, no padding).
typedef struct {
    float course[3];
    float phase_speed[5][4];
    float phase_accel[5][4];
    float minimum[3];
    float end_speed_sqrt;
    float add_speed;
    float slope[3];
    float target_min;
    float accel[4];
    float hp_drain[5];
} Params;

// Mirrors reference.rs KernelCase.
typedef struct {
    float time, pos, speed, hp, delay, finish;
    int phase, dash, finished;
    float speed_in, guts, power;
    int strategy;
    float aptitude, ground, distance, dt;
    float section, pace, force_in, slope;
    int moving_lane_skill, downhill, uphill;
    float spot_struggle, bonus;
} Case;

// Mirrors reference.rs KernelRow: the state snapshot emitted after each tick.
typedef struct {
    float time, pos, speed, hp, delay, finish;
    int phase, dash, finished;
} Row;

typedef struct {
    float time, pos, speed, hp, delay, finish;
    int phase, dash, finished;
} State;

static float bounded_stat(float value) {
    // Rust clamp(1., 2000.); all inputs are finite.
    return fmin(fmax(value, 1.0f), 2000.0f);
}

// Rust root(value) is (value as f64).sqrt() as f32: widen, correctly-rounded
// f64 sqrt, one f32 narrowing store.
static float root(float value) {
    return convert_float_rte(sqrt((double)value));
}

// All f32, like course_speed. The build flag makes the divide correctly
// rounded, matching Rust's f32 divide.
static float course_speed(__constant const Params* p, float distance) {
    return p->course[0] - (distance - p->course[1]) / p->course[2];
}

static float start_dash_speed(__constant const Params* p, float distance) {
    return course_speed(p, distance) * p->minimum[0];
}

static float base_target_speed(__constant const Params* p, float speed, int strategy, int phase,
    float aptitude, float distance) {
    // v19 x87-style: f64 intermediates (0.01 is an f32 literal widened, and
    // course_speed's rounded f32 return widened), one narrowing at the store.
    double base = ((double)p->phase_speed[strategy - 1][phase] * (double)0.01f)
        * (double)course_speed(p, distance);
    if (phase >= 2) {
        // end_add rounds through an f32 store before widening back into base.
        float end_add = convert_float_rte(((double)root(bounded_stat(speed) * p->end_speed_sqrt)
            * (double)aptitude) * (double)p->add_speed);
        base += (double)end_add;
    }
    return convert_float_rte(base);
}

static float minimum_speed(__constant const Params* p, float guts, float distance) {
    // Wider product until the single field store, like the Rust helper.
    return convert_float_rte((double)start_dash_speed(p, distance)
        + (double)root(bounded_stat(guts) * p->minimum[1]) * (double)p->minimum[2]);
}

// target_speed_with_late_terms_iter; the oracle's empty modifier slice makes
// apply_additive_modifiers the identity, so there is no modifier loop here.
static float target_speed(__constant const Params* p, float base, const Case* c, float power,
    float minimum, int hp_remaining) {
    float speed = (base + c->section) * c->pace + c->force_in;
    if (c->moving_lane_skill) {
        speed += root(bounded_stat(power) * 0.0002f);
    }
    // v69: the slope correction is evaluated in f64, rounded once to an f32
    // local, then added in f32. The else arm's 0.0 is an f64 zero, like Rust.
    float correction = convert_float_rte(c->slope > 0.0f
        ? -((double)c->slope * ((double)p->slope[0] / (double)bounded_stat(power)))
        : (c->slope < 0.0f && c->downhill)
            ? ((double)p->slope[1] + (double)fabs(c->slope) / (double)p->slope[2])
            : 0.0);
    speed += correction;
    speed += c->spot_struggle;
    speed += c->bonus;
    if (!hp_remaining) {
        speed = minimum;
    }
    return fmax(fmin(speed, 30.0f), p->target_min);
}

// Acceleration chain; v20 comment: the f32 sqrt is a rounded call result and
// the product chain runs in f64 with one store. No clamp before the (empty)
// modifier application, so a negative base survives into the dash term.
static float acceleration(__constant const Params* p, float power, int strategy, int phase,
    float ground, float distance_field, int uphill, int start_dash) {
    double wide = (double)root(bounded_stat(power) * p->accel[0]);
    wide = ((wide * (double)p->accel[uphill ? 2 : 1] * (double)ground)
        * (double)distance_field) * (double)p->phase_accel[strategy - 1][phase];
    float value = convert_float_rte(wide);
    if (start_dash) {
        value += p->accel[3];
    }
    return fmax(value, 0.0f);
}

static float hp_guts_factor(__constant const Params* p, float guts) {
    // Keep the f32 sqrt result, then round quotient plus one together.
    return convert_float_rte(1.0 + (double)p->hp_drain[4]
        / (double)root(bounded_stat(guts) * p->hp_drain[3]));
}

static float hp_per_second(__constant const Params* p, float speed, float base_speed,
    float ground, float guts, int phase, float rate) {
    float gap = (speed - base_speed) + p->hp_drain[0];
    // gap*gap/hp_drain[1] stays f32; the coefficient product runs in f64 until one store.
    float value = convert_float_rte(
        (((double)((gap * gap) / p->hp_drain[1])) * (double)p->hp_drain[2] * (double)rate)
            * (double)ground);
    if (phase >= 2) {
        value *= hp_guts_factor(p, guts);
    }
    return value;
}

// advance_start_delay; fraction stays f64 (the official crossing ratio is
// wider than the f32 countdown), so the caller passes a double out-parameter.
static int advance_start_delay(float* delay, float dt, double* fraction) {
    if (dt == 0.0f) {
        return 0;
    }
    float remaining = *delay;
    if (remaining > 0.0f) {
        remaining -= dt;
        if (remaining > 0.0f) {
            *delay = remaining;
            return 0;
        }
    }
    *delay = 0.0f;
    *fraction = remaining < 0.0f ? -(double)remaining / (double)dt : 1.0;
    return 1;
}

// finish_time; `crossing` is the post-move position, `course` the race distance.
static float finish_time(float frame_time, float dt, float previous, float crossing,
    float course) {
    float span = crossing - previous;
    float fraction = fmin(fmax((crossing - course) / span, 0.0f), 1.0f);
    return convert_float_rte((double)frame_time - (double)fraction * (double)dt);
}

static void tick(__constant const Params* p, State* s, const Case* c) {
    // Field scheduling mask: position == goal is no longer stepped.
    if (s->finished || s->pos >= c->distance) {
        return;
    }
    const float dt = c->dt;
    double fraction = 1.0;
    if (!advance_start_delay(&s->delay, dt, &fraction)) {
        s->time += dt;
        return;
    }
    // target_speed_indices section quotient: f32 section length store, f64
    // quotient truncated toward zero like Rust's `as i32`, then clamped.
    int section = (int)((double)s->pos / (double)(c->distance / 24.0f));
    section = min(max(section, 0), 23);
    const int target_phase = section < 4 ? 0 : (section < 16 ? 1 : 2);
    const float base = base_target_speed(p, c->speed_in, c->strategy, target_phase,
        c->aptitude, c->distance);
    const float ordinary = minimum_speed(p, c->guts, c->distance);
    const float minimum = s->dash ? 3.0f : ordinary;
    const float target = target_speed(p, base, c, c->power, minimum, s->hp > 0.0f);
    float accel;
    if (s->hp <= 0.0f) {
        accel = -1.2f;
    } else if (s->speed > target) {
        const float decelerations[3] = {-1.2f, -0.8f, -1.0f};
        accel = decelerations[s->phase];
    } else {
        // The oracle passes aptitude (not the race distance) as the
        // Acceleration distance field; keep that quirk exactly.
        accel = acceleration(p, c->power, c->strategy, s->phase, c->ground, c->aptitude,
            c->uphill, s->dash);
    }
    float next = s->speed + accel * dt;
    next = s->speed <= target ? fmin(next, target) : fmax(next, target);
    const float dash_limit = start_dash_speed(p, c->distance);
    if (s->dash) {
        next = fmin(next, dash_limit);
    }
    next = fmax(next, minimum);
    s->speed = next;
    const float drain = s->dash
        ? 20.0f * c->ground
        : hp_per_second(p, next, course_speed(p, c->distance), c->ground, c->guts, s->phase, 1.0f);
    s->hp = fmax(s->hp - drain * dt, 0.0f);
    // phase_boundaries, all f32: section = distance/24. then 4x/16x/20x and
    // the distance itself; count crossed boundaries, cap at 2.
    const float section_length = c->distance / 24.0f;
    int count = 0;
    count += s->pos >= section_length * 4.0f;
    count += s->pos >= section_length * 16.0f;
    count += s->pos >= section_length * 20.0f;
    count += s->pos >= c->distance;
    s->phase = min(count, 2);
    // The speed-time product rounds through f32 first; the crossing ratio
    // (f64) applies to the rounded product, one final f32 store.
    const float moved = convert_float_rte((double)(next * dt) * fraction);
    const float previous = s->pos;
    s->pos += moved;
    s->time += dt;
    if (previous < c->distance && s->pos >= c->distance) {
        s->finished = 1;
        s->finish = finish_time(s->time, dt, previous, s->pos, c->distance);
    }
    if (s->dash && next >= dash_limit) {
        s->dash = 0;
    }
}

__kernel void restricted_tick(__constant const Params* params, __global const Case* cases,
    __global Row* rows) {
    const int id = get_global_id(0);
    const Case c = cases[id];
    State s;
    s.time = c.time;
    s.pos = c.pos;
    s.speed = c.speed;
    s.hp = c.hp;
    s.delay = c.delay;
    s.finish = c.finish;
    s.phase = c.phase;
    s.dash = c.dash;
    s.finished = c.finished;
    for (int row = 0; row < ROWS; row++) {
        tick(params, &s, &c);
        Row out;
        out.time = s.time;
        out.pos = s.pos;
        out.speed = s.speed;
        out.hp = s.hp;
        out.delay = s.delay;
        out.finish = s.finish;
        out.phase = s.phase;
        out.dash = s.dash;
        out.finished = s.finished;
        rows[id * ROWS + row] = out;
    }
}

// Throughput probes for the f64/f32 cost question, not race code. Four
// independent chains keep the pipe busy instead of measuring FMA latency.
__kernel void fma_probe_f64(__global const double* source, __global double* sink, int iterations) {
    const int id = get_global_id(0);
    const double a = source[2 * id];
    const double b = source[2 * id + 1];
    double x0 = a, x1 = a + 1.0, x2 = a - 1.0, x3 = b;
    for (int i = 0; i < iterations; i++) {
        x0 = fma(x0, b, a);
        x1 = fma(x1, b, a);
        x2 = fma(x2, b, a);
        x3 = fma(x3, b, a);
    }
    sink[id] = x0 + x1 + x2 + x3;
}

__kernel void fma_probe_f32(__global const float* source, __global float* sink, int iterations) {
    const int id = get_global_id(0);
    const float a = source[2 * id];
    const float b = source[2 * id + 1];
    float x0 = a, x1 = a + 1.0f, x2 = a - 1.0f, x3 = b;
    for (int i = 0; i < iterations; i++) {
        x0 = fma(x0, b, a);
        x1 = fma(x1, b, a);
        x2 = fma(x2, b, a);
        x3 = fma(x3, b, a);
    }
    sink[id] = x0 + x1 + x2 + x3;
}
