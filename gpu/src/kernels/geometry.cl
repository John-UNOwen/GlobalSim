// Course geometry runtime queries, ported line for line from
// src/retail_geometry.rs: AnimationSegment::interpolate,
// CourseGeometry::corner_state_at, lane_max_at, position and advance.
// The Rust reference side is the same file, included in the gpu crate.
//
// Exactness per docs/GPU_HANDOFF.md: no contract FMA formation, no
// -cl-fast-relaxed-math family options, build with
// -cl-fp32-correctly-rounded-divide-sqrt. Widths follow the Rust source:
// segment x/z interpolation is double with one float store each, quaternion
// lerp and normalization are float with the float dot widened once for the
// sqrt, the lane-offset chain keeps the float num2..num11 products with one
// float store per component, and advance squares/sums/compares in double with
// single float stores.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// The handoff's rule against FMA formation, in both the C and OpenCL spellings.
#pragma FP_CONTRACT OFF
#pragma OPENCL FP_CONTRACT OFF

#define NSEG 1001
#define CS_STRIDE 16
#define IN_STRIDE 8
#define OUT_STRIDE 8

#define MODE_CORNER 0
#define MODE_LANE_MAX 1
#define MODE_POSITION 2
#define MODE_ADVANCE 3

// Course table, per course. Floats are stored as their raw bits.
#define CS_DISTANCE 0
#define CS_D2W 1        // distance_to_world
#define CS_LANE0 2      // initial_lane_max
#define CS_OVR_PRESENT 3
#define CS_OVR_DISTANCE 4
#define CS_LM_BASE 5    // lane_max_changes block in lm_*
#define CS_LM_COUNT 6
#define CS_CE_BASE 7    // corner_events block in ce_*
#define CS_CE_COUNT 8
#define CS_RF0 9        // rotation_fix, four float bits
#define CS_RF1 10
#define CS_RF2 11
#define CS_RF3 12
#define CS_OVR_BASE 13  // overrun segment block start in sg_*
#define CS_SEG_BASE 14  // main segment block start in sg_*

// Per-item input. Floats are stored as their raw bits.
#define IS_COURSE 0
#define IS_MODE 1
#define IS_READY 2
#define IS_DISTANCE 3
#define IS_LANE 4
#define IS_MOVEMENT 5
#define IS_PREVX 6
#define IS_PREVZ 7

// Per-item output. Corner writes GO_ID and GO_FINAL, lane_max GO_A,
// position GO_A/GO_B (world x/z), advance GO_A/GO_B/GO_C (next distance,
// world x/z). GO_FLAG is the fallback flag: 1 where the engine returns an
// error and the outputs carry the sentinel NaN instead.
#define GO_FLAG 0
#define GO_ID 1
#define GO_FINAL 2
#define GO_A 3
#define GO_B 4
#define GO_C 5

// Rust's f32::NAN bit pattern, written to result slots on the error paths.
#define SENTINEL_BITS 0x7fc00000u

// AnimationSegment::interpolate. Coordinates are double with one float store.
// The endpoint (the last sample) keeps its raw rotation, without normalization.
static void interpolate(const int seg, const float t, const int endpoint,
    __global const double* sg_x, __global const double* sg_xd,
    __global const double* sg_z, __global const double* sg_zd,
    __global const float* sg_q, __global const float* sg_qd,
    float* x, float* z, float* q) {
    *x = convert_float_rte(sg_x[seg] + sg_xd[seg] * (double)t);
    *z = convert_float_rte(sg_z[seg] + sg_zd[seg] * (double)t);
    const int qb = 4 * seg;
    q[0] = sg_q[qb];
    q[1] = sg_q[qb + 1];
    q[2] = sg_q[qb + 2];
    q[3] = sg_q[qb + 3];
    if (endpoint) {
        return;
    }
    for (int k = 0; k < 4; k++) {
        q[k] = q[k] + sg_qd[qb + k] * t;
    }
    // norm = (dot(q, q) as f64).sqrt() as f32, the dot in float, then float divides.
    const float dqq = ((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2]) + q[3] * q[3];
    const float norm = convert_float_rte(sqrt((double)dqq));
    for (int k = 0; k < 4; k++) {
        q[k] = q[k] / norm;
    }
}

// quaternion_multiply, the Rust parenthesization carried term by term.
static void quat_mul(const float* a, const float* b, float* o) {
    const float ax = a[0], ay = a[1], az = a[2], aw = a[3];
    const float bx = b[0], by = b[1], bz = b[2], bw = b[3];
    o[0] = ((aw * bx + ax * bw) + ay * bz) - az * by;
    o[1] = ((aw * by + ay * bw) + az * bx) - ax * bz;
    o[2] = ((aw * bz + az * bw) + ax * by) - ay * bx;
    o[3] = ((aw * bw - ax * bx) - ay * by) - az * bz;
}

// CourseGeometry::position. Returns 1 where the engine returns Err.
static int geo_position(__global const uint* cb, const float distance, const float lane,
    __global const double* sg_x, __global const double* sg_xd,
    __global const double* sg_z, __global const double* sg_zd,
    __global const float* sg_q, __global const float* sg_qd,
    float* wx, float* wz) {
    if (!isfinite(distance) || distance < 0.f || !isfinite(lane)) {
        return 1;
    }
    const float course_distance = as_float(cb[CS_DISTANCE]);
    float d = distance;
    float length = course_distance;
    int seg = (int)cb[CS_SEG_BASE];
    if (distance > course_distance) {
        if (cb[CS_OVR_PRESENT] == 0u) {
            return 1;
        }
        // Rust's f32 % is fmod, and the remainder of the overrun wrap is exact
        // for both operators, so fmod is the same value.
        const float ovr = as_float(cb[CS_OVR_DISTANCE]);
        d = fmod(distance - course_distance, ovr);
        length = ovr;
        seg = (int)cb[CS_OVR_BASE];
    }
    const float sample = convert_float_rte(((double)d / (double)length) * 1000.0);
    // Rust casts the finite nonnegative sample (<= 1000 by the wrap above) to
    // usize and clamps to 1000.
    const int i = min(convert_int_sat_rtz(sample), 1000);
    const float t = sample - (float)i;
    float x, z, q[4];
    interpolate(seg + i, t, i == 1000, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd, &x, &z, q);
    const float rf[4] = {as_float(cb[CS_RF0]), as_float(cb[CS_RF1]), as_float(cb[CS_RF2]), as_float(cb[CS_RF3])};
    float qq[4];
    quat_mul(q, rf, qq);
    const float q_x = qq[0], q_y = qq[1], q_z = qq[2], q_w = qq[3];
    // The v12 lane-offset chain: float products with float stores, the
    // component expressions evaluated once at double, rounded on the store.
    const float num2 = q_y * 2.f;
    const float num3 = q_z * 2.f;
    const float num5 = q_y * num2;
    const float num6 = q_z * num3;
    const float num8 = q_x * num3;
    const float num11 = q_w * num2;
    const double offset = (double)(lane * 11.25f);
    const float dx = convert_float_rte((1.0 - ((double)num5 + (double)num6)) * offset);
    const float dz = convert_float_rte(((double)num8 - (double)num11) * offset);
    *wx = x + dx;
    *wz = z + dz;
    return 0;
}

// CourseGeometry::corner_state_at, the reverse scan of the MODE_CORNER body.
// The first event at or before the distance from the end is the active corner;
// an event ending exactly at the distance is active unless its closing flag is
// set. ready=0 (the engine's events-not-yet-ready state) keeps (0, false).
static void geo_corner_state(__global const uint* cb, __global const float* ce_b,
    __global const float* ce_e, __global const uint* ce_i, __global const uint* ce_f,
    const float distance, const int ready, __global uint* id, __global uint* is_final) {
    *id = 0u;
    *is_final = 0u;
    if (!ready) {
        return;
    }
    const int base = (int)cb[CS_CE_BASE];
    const int end = base + (int)cb[CS_CE_COUNT];
    for (int k = end - 1; k >= base; k--) {
        if (distance >= ce_b[k]) {
            const uint fl = ce_f[k];
            const int active = (distance < ce_e[k]) || (distance == ce_e[k] && (fl & 2u) == 0u);
            *id = active ? ce_i[k] : 0u;
            *is_final = fl & 1u;
            break;
        }
    }
}

// CourseGeometry::lane_max_at.
static float geo_lane_max_at(__global const uint* cb, __global const float* lm_b,
    __global const float* lm_l, const float distance) {
    float value = as_float(cb[CS_LANE0]);
    const int base = (int)cb[CS_LM_BASE];
    const int end = base + (int)cb[CS_LM_COUNT];
    for (int k = base; k < end; k++) {
        if (distance < lm_b[k]) {
            break;
        }
        value = lm_l[k];
    }
    return value;
}

// CourseGeometry::advance, the MODE_ADVANCE body. Returns 1 where the engine
// returns Err. The movement check runs before position, as in Rust.
static int geo_advance(__global const uint* cb,
    __global const double* sg_x, __global const double* sg_xd,
    __global const double* sg_z, __global const double* sg_zd,
    __global const float* sg_q, __global const float* sg_qd,
    const float distance, const float lane, float movement,
    const float px, const float pz, float* next, float* wx, float* wz) {
    int err = !isfinite(movement) || movement < 0.f || !isfinite(px) || !isfinite(pz);
    float x, z;
    if (!err) {
        err = geo_position(cb, distance, lane, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd, &x, &z);
    }
    if (err) {
        return 1;
    }
    const float ddx = x - px;
    const float ddz = z - pz;
    const double squared = (double)ddx * (double)ddx + (double)ddz * (double)ddz;
    float allowed = movement * as_float(cb[CS_D2W]);
    if (!(allowed >= 0.f && squared <= (double)allowed * (double)allowed)) {
        const float world_distance = convert_float_rte(sqrt(squared));
        if (world_distance > allowed) {
            movement = convert_float_rte((double)movement * ((double)allowed / (double)world_distance));
        }
    }
    *next = distance + movement;
    *wx = x;
    *wz = z;
    return 0;
}

// One work item runs one query for one course. The course table, the segment,
// lane-max and corner arrays, the item and the output are flat per-index
// buffers. Each mode writes the slots named in the GO_* comments.
__kernel void geometry(__global const uint* cs,
    __global const double* sg_x, __global const double* sg_xd,
    __global const double* sg_z, __global const double* sg_zd,
    __global const float* sg_q, __global const float* sg_qd,
    __global const float* lm_b, __global const float* lm_l,
    __global const float* ce_b, __global const float* ce_e,
    __global const uint* ce_i, __global const uint* ce_f,
    __global const uint* it, __global uint* out) {
    const int id = get_global_id(0);
    __global const uint* ib = it + (size_t)id * IN_STRIDE;
    __global uint* ob = out + (size_t)id * OUT_STRIDE;
    __global const uint* cb = cs + (size_t)ib[IS_COURSE] * CS_STRIDE;
    const int mode = (int)ib[IS_MODE];
    const float distance = as_float(ib[IS_DISTANCE]);
    ob[GO_FLAG] = 0u;
    ob[GO_ID] = 0u;
    ob[GO_FINAL] = 0u;
    ob[GO_A] = 0u;
    ob[GO_B] = 0u;
    ob[GO_C] = 0u;

    if (mode == MODE_CORNER) {
        geo_corner_state(cb, ce_b, ce_e, ce_i, ce_f, distance, (int)ib[IS_READY],
            &ob[GO_ID], &ob[GO_FINAL]);
    } else if (mode == MODE_LANE_MAX) {
        ob[GO_A] = as_uint(geo_lane_max_at(cb, lm_b, lm_l, distance));
    } else if (mode == MODE_POSITION) {
        float x, z;
        if (geo_position(cb, distance, as_float(ib[IS_LANE]), sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd, &x, &z)) {
            ob[GO_FLAG] = 1u;
            ob[GO_A] = SENTINEL_BITS;
            ob[GO_B] = SENTINEL_BITS;
        } else {
            ob[GO_A] = as_uint(x);
            ob[GO_B] = as_uint(z);
        }
    } else {
        float next, x, z;
        if (geo_advance(cb, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd, distance, as_float(ib[IS_LANE]),
                as_float(ib[IS_MOVEMENT]), as_float(ib[IS_PREVX]), as_float(ib[IS_PREVZ]), &next, &x, &z)) {
            ob[GO_FLAG] = 1u;
            ob[GO_A] = SENTINEL_BITS;
            ob[GO_B] = SENTINEL_BITS;
            ob[GO_C] = SENTINEL_BITS;
        } else {
            ob[GO_A] = as_uint(next);
            ob[GO_B] = as_uint(x);
            ob[GO_C] = as_uint(z);
        }
    }
}

