//! Hosts the engine's course-geometry module in place and adds the
//! flattening accessors the kernel buffers are built from.
//!
//! The engine file arrives through `build.rs` as a doc-stripped copy, included
//! at module scope, so the accessors below sit inside the engine module and can
//! read the private fields of `CourseGeometry`, `AnimationSegment` and
//! `OverrunAnimation` (field privacy is module-scoped). `lib.rs` gates this
//! module with `#[cfg(all(feature = "geometry", not(test)))]` because the
//! engine module's tests read `env!("CARGO_MANIFEST_DIR")/data/...`, which
//! would resolve inside this crate's unit-test target (see gpu/README.md).

include!(concat!(env!("OUT_DIR"), "/retail_geometry_engine.rs"));

pub const NSEG: usize = 1001;

/// One course as flat read-only kernel arrays. The segment arrays hold the
/// main animation block first, then the overrun block (all zeros when the
/// course has no overrun).
#[derive(Clone, Debug)]
pub struct FlatCourse {
    pub course_id: i32,
    pub distance: f32,
    pub distance_to_world: f32,
    pub initial_lane_max: f32,
    pub rotation_fix: [f32; 4],
    pub overrun_present: bool,
    pub overrun_distance: f32,
    pub seg_x: Vec<f64>,
    pub seg_xd: Vec<f64>,
    pub seg_z: Vec<f64>,
    pub seg_zd: Vec<f64>,
    pub seg_q: Vec<f32>,
    pub seg_qd: Vec<f32>,
    pub lm_boundary: Vec<f32>,
    pub lm_limit: Vec<f32>,
    pub ce_start: Vec<f32>,
    pub ce_end: Vec<f32>,
    pub ce_id: Vec<i32>,
    pub ce_flags: Vec<u32>,
    pub final_corner_half: Option<(f32, f32)>,
}

/// Clears the course's overrun animation, mirroring the engine test that
/// strips it from a clone to exercise the "course geometry lacks an overrun
/// animation" error path. Captured courses all carry an overrun, so a gate
/// can only reach that domain this way.
pub fn drop_overrun(geometry: &mut CourseGeometry) {
    geometry.overrun = None;
}

fn push_segments(
    segments: &[AnimationSegment],
    x: &mut Vec<f64>,
    xd: &mut Vec<f64>,
    z: &mut Vec<f64>,
    zd: &mut Vec<f64>,
    q: &mut Vec<f32>,
    qd: &mut Vec<f32>,
) {
    for s in segments {
        x.push(s.x);
        xd.push(s.x_delta);
        z.push(s.z);
        zd.push(s.z_delta);
        q.extend_from_slice(&s.rotation);
        qd.extend_from_slice(&s.rotation_delta);
    }
}

/// Copies the runtime state `CourseGeometry::from_value` built, field for
/// field, into kernel arrays. The kernel then reproduces the runtime queries
/// against the same tables the engine answers from.
pub fn flatten_course(geometry: &CourseGeometry) -> FlatCourse {
    let mut course = FlatCourse {
        course_id: geometry.course_id,
        distance: geometry.distance,
        distance_to_world: geometry.distance_to_world,
        initial_lane_max: geometry.initial_lane_max,
        rotation_fix: geometry.rotation_fix,
        overrun_present: geometry.overrun.is_some(),
        overrun_distance: geometry.overrun.as_ref().map_or(0., |a| a.distance),
        seg_x: Vec::with_capacity(2 * NSEG),
        seg_xd: Vec::with_capacity(2 * NSEG),
        seg_z: Vec::with_capacity(2 * NSEG),
        seg_zd: Vec::with_capacity(2 * NSEG),
        seg_q: Vec::with_capacity(8 * NSEG),
        seg_qd: Vec::with_capacity(8 * NSEG),
        lm_boundary: Vec::new(),
        lm_limit: Vec::new(),
        ce_start: Vec::new(),
        ce_end: Vec::new(),
        ce_id: Vec::new(),
        ce_flags: Vec::new(),
        final_corner_half: geometry.final_corner_half,
    };
    push_segments(
        &geometry.segments,
        &mut course.seg_x,
        &mut course.seg_xd,
        &mut course.seg_z,
        &mut course.seg_zd,
        &mut course.seg_q,
        &mut course.seg_qd,
    );
    match &geometry.overrun {
        Some(animation) => push_segments(
            &animation.segments,
            &mut course.seg_x,
            &mut course.seg_xd,
            &mut course.seg_z,
            &mut course.seg_zd,
            &mut course.seg_q,
            &mut course.seg_qd,
        ),
        None => {
            // Zero-filled placeholder block. The kernel never reads it while
            // `overrun_present` is false.
            course.seg_x.resize(2 * NSEG, 0.);
            course.seg_xd.resize(2 * NSEG, 0.);
            course.seg_z.resize(2 * NSEG, 0.);
            course.seg_zd.resize(2 * NSEG, 0.);
            course.seg_q.resize(8 * NSEG, 0.);
            course.seg_qd.resize(8 * NSEG, 0.);
        }
    }
    for (boundary, limit) in &geometry.lane_max_changes {
        course.lm_boundary.push(*boundary);
        course.lm_limit.push(*limit);
    }
    for (start, end, id, final_corner, straight_clears) in &geometry.corner_events {
        course.ce_start.push(*start);
        course.ce_end.push(*end);
        course.ce_id.push(*id);
        course.ce_flags.push((*final_corner as u32) | ((*straight_clears as u32) << 1));
    }
    course
}
