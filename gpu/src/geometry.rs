//! Host side of the geometry port (handoff step 6, geometry slice): the
//! flat-buffer layout shared with kernels/geometry.cl, the run through the
//! gated device, and the loader for the engine's captured-geometry fixture.
//!
//! The Rust reference side is the engine's own runtime queries, available in
//! this crate through the `retail_geometry` module (`position`, `advance`,
//! `lane_max_at`, `corner_state_at`). `flatten_course` there copies the
//! loaded `CourseGeometry` state into `FlatCourse`, the input of `run`.

use crate::retail_geometry::FlatCourse;

pub const NSEG: usize = 1001;
pub const CS_STRIDE: usize = 16;
pub const IN_STRIDE: usize = 8;
pub const OUT_STRIDE: usize = 8;

pub const MODE_CORNER: u32 = 0;
pub const MODE_LANE_MAX: u32 = 1;
pub const MODE_POSITION: u32 = 2;
pub const MODE_ADVANCE: u32 = 3;

pub fn mode_name(mode: u32) -> &'static str {
    match mode {
        MODE_CORNER => "corner",
        MODE_LANE_MAX => "lane_max",
        MODE_POSITION => "position",
        _ => "advance",
    }
}

pub const CS_DISTANCE: usize = 0;
pub const CS_D2W: usize = 1;
pub const CS_LANE0: usize = 2;
pub const CS_OVR_PRESENT: usize = 3;
pub const CS_OVR_DISTANCE: usize = 4;
pub const CS_LM_BASE: usize = 5;
pub const CS_LM_COUNT: usize = 6;
pub const CS_CE_BASE: usize = 7;
pub const CS_CE_COUNT: usize = 8;
pub const CS_RF0: usize = 9;
pub const CS_OVR_BASE: usize = 13;
pub const CS_SEG_BASE: usize = 14;

pub const IS_COURSE: usize = 0;
pub const IS_MODE: usize = 1;
pub const IS_READY: usize = 2;
pub const IS_DISTANCE: usize = 3;
pub const IS_LANE: usize = 4;
pub const IS_MOVEMENT: usize = 5;
pub const IS_PREVX: usize = 6;
pub const IS_PREVZ: usize = 7;

pub const OS_FLAG: usize = 0;
pub const OS_ID: usize = 1;
pub const OS_FINAL: usize = 2;
pub const OS_A: usize = 3;
pub const OS_B: usize = 4;
pub const OS_C: usize = 5;

// Rust's f32::NAN bits, the sentinel the kernel writes where the engine
// returns Err (the handoff step 5 fallback domain).
pub const SENTINEL_BITS: u32 = 0x7FC0_0000;

#[derive(Clone, Copy)]
pub struct GeometryItem {
    pub course: usize,
    pub mode: u32,
    pub ready: bool,
    pub distance: f32,
    pub lane: f32,
    pub movement: f32,
    pub prev: [f32; 2],
}

impl GeometryItem {
    fn new(course: usize, mode: u32) -> GeometryItem {
        GeometryItem { course, mode, ready: false, distance: 0., lane: 0., movement: 0., prev: [0.; 2] }
    }

    pub fn corner(course: usize, distance: f32, ready: bool) -> GeometryItem {
        GeometryItem { ready, distance, ..GeometryItem::new(course, MODE_CORNER) }
    }

    pub fn lane_max(course: usize, distance: f32) -> GeometryItem {
        GeometryItem { distance, ..GeometryItem::new(course, MODE_LANE_MAX) }
    }

    pub fn position(course: usize, distance: f32, lane: f32) -> GeometryItem {
        GeometryItem { distance, lane, ..GeometryItem::new(course, MODE_POSITION) }
    }

    pub fn advance(course: usize, distance: f32, lane: f32, movement: f32, prev: [f32; 2]) -> GeometryItem {
        GeometryItem { distance, lane, movement, prev, ..GeometryItem::new(course, MODE_ADVANCE) }
    }

    fn words(&self) -> [u32; IN_STRIDE] {
        [
            self.course as u32,
            self.mode,
            self.ready as u32,
            self.distance.to_bits(),
            self.lane.to_bits(),
            self.movement.to_bits(),
            self.prev[0].to_bits(),
            self.prev[1].to_bits(),
        ]
    }
}

pub fn out_word(out: &[u32], item: usize, slot: usize) -> u32 {
    out[item * OUT_STRIDE + slot]
}

/// An engine fixture row: the `advance` inputs and the four expected bit
/// patterns (next distance, world x, world z, lane_max_at) the engine's own
/// `snapshot_vectors_match_bits_for_all_captured_courses` test replays.
pub struct FixtureRow {
    pub position: f32,
    pub lane: f32,
    pub movement: f32,
    pub previous: [f32; 2],
    pub expected_bits: [u32; 4],
}

pub struct FixtureCourse {
    pub course_id: i32,
    pub distance: f32,
    pub rows: Vec<FixtureRow>,
}

fn fixture_f32(value: &crate::json::Value) -> f32 {
    // Mirrors the engine test's `finite`/`number` helpers: the JSON f64 narrows
    // to f32 once, and the fixture only stores finite values.
    value.as_f64().unwrap() as f32
}

pub fn load_port_fixture() -> Vec<FixtureCourse> {
    // The engine's independent captured-game reference fixture, relative to
    // the engine package root (see gpu/README.md).
    let text = include_str!("../../tests/fixtures/retail_geometry_port.json");
    let data = crate::json::parse(text).unwrap();
    let mut courses = Vec::new();
    for course in data.get_array("courses").unwrap() {
        let mut rows = Vec::new();
        for row in course.get_array("rows").unwrap() {
            let previous_world = row.get_array("previous_world").unwrap();
            let expected_bits = row.get_array("expected_bits").unwrap();
            rows.push(FixtureRow {
                position: fixture_f32(row.get("position").unwrap()),
                lane: fixture_f32(row.get("lane").unwrap()),
                movement: fixture_f32(row.get("movement").unwrap()),
                previous: [fixture_f32(&previous_world[0]), fixture_f32(&previous_world[1])],
                expected_bits: expected_bits.iter().map(|e| e.as_i64().unwrap() as u32).collect::<Vec<_>>().try_into().unwrap(),
            });
        }
        courses.push(FixtureCourse {
            course_id: course.get_i64("course_id").unwrap() as i32,
            distance: course.get_f64("distance").unwrap() as f32,
            rows,
        });
    }
    courses
}

#[cfg(feature = "gpu")]
pub struct GeometryOutput {
    pub out: Vec<u32>,
    pub device: crate::gpu::DeviceReport,
    pub build_ms: f64,
}

#[cfg(feature = "gpu")]
fn pad(values: &mut Vec<f32>) {
    // OpenCL rejects zero-length buffers. A padded element can never be read:
    // the course table counts never cover it.
    if values.is_empty() {
        values.push(0.);
    }
}

#[cfg(feature = "gpu")]
fn pad_u(values: &mut Vec<u32>) {
    if values.is_empty() {
        values.push(0);
    }
}

#[cfg(feature = "gpu")]
fn buffer<T: ocl::OclPrm>(queue: &ocl::Queue, data: &[T]) -> Result<ocl::Buffer<T>, String> {
    let buffer = ocl::Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(data.len())
        .build()
        .map_err(|e| e.to_string())?;
    buffer.cmd().write(data).enq().map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// The flat kernel tables for a set of courses, exactly the buffer contents
/// the geometry kernel's cs/sg/lm/ce arguments are built from.
#[cfg(feature = "gpu")]
pub struct CourseTables {
    pub cs: Vec<u32>,
    pub lm_b: Vec<f32>,
    pub lm_l: Vec<f32>,
    pub ce_b: Vec<f32>,
    pub ce_e: Vec<f32>,
    pub ce_i: Vec<u32>,
    pub ce_f: Vec<u32>,
    pub sg_x: Vec<f64>,
    pub sg_xd: Vec<f64>,
    pub sg_z: Vec<f64>,
    pub sg_zd: Vec<f64>,
    pub sg_q: Vec<f32>,
    pub sg_qd: Vec<f32>,
}

/// Flattens courses into kernel tables. Segment buffers are one main block
/// per course followed by one overrun block per course. Lane-max and corner
/// arrays are concatenated spans addressed through the course table.
#[cfg(feature = "gpu")]
pub fn course_tables<'a>(courses: impl IntoIterator<Item = &'a FlatCourse>) -> CourseTables {
    let courses: Vec<&FlatCourse> = courses.into_iter().collect();
    let n_courses = courses.len();
    let mut cs: Vec<u32> = vec![0; n_courses * CS_STRIDE];
    let mut lm_b: Vec<f32> = Vec::new();
    let mut lm_l: Vec<f32> = Vec::new();
    let mut ce_b: Vec<f32> = Vec::new();
    let mut ce_e: Vec<f32> = Vec::new();
    let mut ce_i: Vec<u32> = Vec::new();
    let mut ce_f: Vec<u32> = Vec::new();
    for (ci, co) in courses.iter().enumerate() {
        let row = ci * CS_STRIDE;
        cs[row + CS_DISTANCE] = co.distance.to_bits();
        cs[row + CS_D2W] = co.distance_to_world.to_bits();
        cs[row + CS_LANE0] = co.initial_lane_max.to_bits();
        cs[row + CS_OVR_PRESENT] = co.overrun_present as u32;
        cs[row + CS_OVR_DISTANCE] = co.overrun_distance.to_bits();
        cs[row + CS_LM_BASE] = lm_b.len() as u32;
        cs[row + CS_LM_COUNT] = co.lm_boundary.len() as u32;
        cs[row + CS_CE_BASE] = ce_b.len() as u32;
        cs[row + CS_CE_COUNT] = co.ce_start.len() as u32;
        for k in 0..4 {
            cs[row + CS_RF0 + k] = co.rotation_fix[k].to_bits();
        }
        cs[row + CS_OVR_BASE] = ((n_courses + ci) * NSEG) as u32;
        cs[row + CS_SEG_BASE] = (ci * NSEG) as u32;
        lm_b.extend_from_slice(&co.lm_boundary);
        lm_l.extend_from_slice(&co.lm_limit);
        ce_b.extend_from_slice(&co.ce_start);
        ce_e.extend_from_slice(&co.ce_end);
        ce_i.extend(co.ce_id.iter().map(|id| *id as u32));
        ce_f.extend_from_slice(&co.ce_flags);
    }
    pad(&mut lm_b);
    pad(&mut lm_l);
    pad_u(&mut ce_i);
    pad_u(&mut ce_f);
    let mut sg_x: Vec<f64> = Vec::with_capacity(2 * n_courses * NSEG);
    let mut sg_xd = Vec::with_capacity(2 * n_courses * NSEG);
    let mut sg_z = Vec::with_capacity(2 * n_courses * NSEG);
    let mut sg_zd = Vec::with_capacity(2 * n_courses * NSEG);
    let mut sg_q = Vec::<f32>::with_capacity(8 * n_courses * NSEG);
    let mut sg_qd = Vec::<f32>::with_capacity(8 * n_courses * NSEG);
    for co in &courses {
        sg_x.extend_from_slice(&co.seg_x[..NSEG]);
        sg_xd.extend_from_slice(&co.seg_xd[..NSEG]);
        sg_z.extend_from_slice(&co.seg_z[..NSEG]);
        sg_zd.extend_from_slice(&co.seg_zd[..NSEG]);
        sg_q.extend_from_slice(&co.seg_q[..4 * NSEG]);
        sg_qd.extend_from_slice(&co.seg_qd[..4 * NSEG]);
    }
    for co in courses {
        sg_x.extend_from_slice(&co.seg_x[NSEG..2 * NSEG]);
        sg_xd.extend_from_slice(&co.seg_xd[NSEG..2 * NSEG]);
        sg_z.extend_from_slice(&co.seg_z[NSEG..2 * NSEG]);
        sg_zd.extend_from_slice(&co.seg_zd[NSEG..2 * NSEG]);
        sg_q.extend_from_slice(&co.seg_q[4 * NSEG..8 * NSEG]);
        sg_qd.extend_from_slice(&co.seg_qd[4 * NSEG..8 * NSEG]);
    }
    CourseTables { cs, lm_b, lm_l, ce_b, ce_e, ce_i, ce_f, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd }
}

/// Runs every item against every course's flattened tables on the gated
/// device.
#[cfg(feature = "gpu")]
pub fn run(courses: &[FlatCourse], items: &[GeometryItem]) -> Result<GeometryOutput, String> {
    let spike = crate::gpu::Spike::open()?;
    let program = spike.build_program(crate::gpu::GEOMETRY_KERNEL_SOURCE)?;
    let queue = spike.queue();
    let t = course_tables(courses);
    let CourseTables { cs, lm_b, lm_l, ce_b, ce_e, ce_i, ce_f, sg_x, sg_xd, sg_z, sg_zd, sg_q, sg_qd } = t;
    let it: Vec<u32> = items.iter().flat_map(|item| item.words()).collect();

    let b_cs = buffer(&queue, &cs)?;
    let b_sg_x = buffer(&queue, &sg_x)?;
    let b_sg_xd = buffer(&queue, &sg_xd)?;
    let b_sg_z = buffer(&queue, &sg_z)?;
    let b_sg_zd = buffer(&queue, &sg_zd)?;
    let b_sg_q = buffer(&queue, &sg_q)?;
    let b_sg_qd = buffer(&queue, &sg_qd)?;
    let b_lm_b = buffer(&queue, &lm_b)?;
    let b_lm_l = buffer(&queue, &lm_l)?;
    let b_ce_b = buffer(&queue, &ce_b)?;
    let b_ce_e = buffer(&queue, &ce_e)?;
    let b_ce_i = buffer(&queue, &ce_i)?;
    let b_ce_f = buffer(&queue, &ce_f)?;
    let b_it = buffer(&queue, &it)?;
    let b_out = ocl::Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(items.len() * OUT_STRIDE)
        .build()
        .map_err(|e| e.to_string())?;

    let kernel = ocl::Kernel::builder()
        .program(&program)
        .name("geometry")
        .queue(queue.clone())
        .global_work_size(items.len())
        .arg(&b_cs)
        .arg(&b_sg_x)
        .arg(&b_sg_xd)
        .arg(&b_sg_z)
        .arg(&b_sg_zd)
        .arg(&b_sg_q)
        .arg(&b_sg_qd)
        .arg(&b_lm_b)
        .arg(&b_lm_l)
        .arg(&b_ce_b)
        .arg(&b_ce_e)
        .arg(&b_ce_i)
        .arg(&b_ce_f)
        .arg(&b_it)
        .arg(&b_out)
        .build()
        .map_err(|e| e.to_string())?;
    unsafe { kernel.enq().map_err(|e| e.to_string())? };
    queue.finish().map_err(|e| e.to_string())?;

    let mut out = vec![0u32; items.len() * OUT_STRIDE];
    b_out.cmd().read(&mut out).enq().map_err(|e| e.to_string())?;
    Ok(GeometryOutput { out, device: spike.device, build_ms: spike.build_ms })
}
