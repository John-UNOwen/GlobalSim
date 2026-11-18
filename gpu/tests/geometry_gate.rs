//! Handoff step 6 (geometry slice) gate: the OpenCL `geometry` kernel
//! reproduces the engine's course-geometry runtime queries bit for bit on
//! the gated device. Every float comparison goes through `to_bits()`.
//!
//! Three layers, in the order the gate reports them:
//! 1. the independent captured-game fixture `tests/fixtures/
//!    retail_geometry_port.json` (engine root), replayed row for row through
//!    the kernel, same role as the retail RNG fixture pin,
//! 2. an engine parity sweep over 10504, 10604, 10613 and 11102: dense
//!    distance grids, ULP probes around every corner-event and lane-width
//!    boundary, per-distance lane sets including `lane_max_at(d)` itself,
//!    ready on and off, the overrun wrap on 10604, and `advance` with
//!    `previous_world` taken from `position(d - 0.1, lane)` across movements
//!    that clip and do not clip,
//! 3. the fallback domain: engine errors become the sentinel NaN plus the
//!    fallback flag, and no valid sweep sample carries either.
#![cfg(feature = "gpu")]
#![cfg(feature = "geometry")]

use globalsim_gpu::{geometry, retail_geometry};
use std::path::Path;

fn geometry_dir() -> std::path::PathBuf {
    // The engine's data dir, reached from this crate's manifest dir.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/course_geometry")
}

fn report_device(out: &geometry::GeometryOutput) {
    println!(
        "device: {} ({} CUs, single_fp_config {:#x}, double_fp_config {:#x}, build {:.1} ms)",
        out.device.name, out.device.compute_units, out.device.single_fp_config, out.device.double_fp_config, out.build_ms
    );
}

fn ulp_trio(x: f32) -> [f32; 3] {
    [f32::from_bits(x.to_bits() - 1), x, f32::from_bits(x.to_bits() + 1)]
}

/// One sweep item plus enough context to name the first diverging sample.
struct Checked {
    item: geometry::GeometryItem,
    course_id: i32,
}

impl Checked {
    fn tag(&self) -> String {
        let i = &self.item;
        format!(
            "mode {} course {} distance {:#010x} lane {:#010x} movement {:#010x} prev [{:#010x},{:#010x}] ready {}",
            geometry::mode_name(i.mode),
            self.course_id,
            i.distance.to_bits(),
            i.lane.to_bits(),
            i.movement.to_bits(),
            i.prev[0].to_bits(),
            i.prev[1].to_bits(),
            i.ready,
        )
    }
}

fn f32_eq(got: u32, want: u32, field: &str, tag: &str) {
    if got != want {
        panic!("{tag}: kernel {field} gpu {got:#010x} rust {want:#010x}");
    }
}

#[test]
fn kernel_reproduces_the_captured_geometry_fixture() {
    // The engine's own snapshot test replays this fixture through
    // advance/lane_max_at. Here the same rows run through the kernel, pinning
    // the device to the independent captured-game reference, not to Rust.
    let fixture = geometry::load_port_fixture();
    let dir = geometry_dir();
    let mut courses: Vec<retail_geometry::FlatCourse> = Vec::new();
    let mut items: Vec<geometry::GeometryItem> = Vec::new();
    let mut expected: Vec<(i32, usize, [u32; 4])> = Vec::new();
    for course in &fixture {
        let course_index = courses.len();
        let loaded = retail_geometry::CourseGeometry::load(&dir, course.course_id, course.distance)
            .unwrap_or_else(|e| panic!("course {}: {e}", course.course_id));
        courses.push(retail_geometry::flatten_course(&loaded));
        for (row_index, row) in course.rows.iter().enumerate() {
            items.push(geometry::GeometryItem::advance(
                course_index,
                row.position,
                row.lane,
                row.movement,
                row.previous,
            ));
            items.push(geometry::GeometryItem::lane_max(course_index, row.position));
            expected.push((course.course_id, row_index, row.expected_bits));
        }
    }
    let row_count: usize = fixture.iter().map(|c| c.rows.len()).sum();
    let out = geometry::run(&courses, &items).expect("fixture kernel run");
    report_device(&out);
    assert_eq!(items.len(), 2 * row_count);

    let mut checked = 0;
    for (ri, (course_id, row_index, bits)) in expected.iter().enumerate() {
        let (adv, lm) = (ri * 2, ri * 2 + 1);
        let tag = || format!("fixture course {course_id} row {row_index}");
        assert_eq!(geometry::out_word(&out.out, adv, geometry::OS_FLAG), 0, "{}: fallback flag", tag());
        assert_eq!(geometry::out_word(&out.out, lm, geometry::OS_FLAG), 0, "{}: lane_max fallback flag", tag());
        f32_eq(geometry::out_word(&out.out, adv, geometry::OS_A), bits[0], "next", &tag());
        f32_eq(geometry::out_word(&out.out, adv, geometry::OS_B), bits[1], "world x", &tag());
        f32_eq(geometry::out_word(&out.out, adv, geometry::OS_C), bits[2], "world z", &tag());
        f32_eq(geometry::out_word(&out.out, lm, geometry::OS_A), bits[3], "lane_max", &tag());
        for item in [adv, lm] {
            for slot in [geometry::OS_A, geometry::OS_B, geometry::OS_C] {
                assert_ne!(geometry::out_word(&out.out, item, slot), geometry::SENTINEL_BITS, "{}: sentinel in valid row", tag());
            }
        }
        checked += 4;
    }
    println!(
        "kernel pinned to the captured fixture: {checked} values across {} courses and {row_count} rows",
        fixture.len()
    );
}

fn compare_rows(
    out: &[u32],
    rows: &[(usize, geometry::GeometryItem, i32)],
    courses: &[retail_geometry::FlatCourse],
    engines: &[retail_geometry::CourseGeometry],
) {
    for (index, (course_index, item, course_id)) in rows.iter().enumerate() {
        let engine = &engines[*course_index];
        let tag = Checked { item: *item, course_id: *course_id }.tag();
        let flag = geometry::out_word(out, index, geometry::OS_FLAG);
        match item.mode {
            geometry::MODE_CORNER => {
                let (id, final_corner) = engine.corner_state_at(item.distance, item.ready);
                assert_eq!(flag, 0, "{tag}: fallback flag set on a valid corner query");
                assert_eq!(
                    geometry::out_word(out, index, geometry::OS_ID) as i32,
                    id,
                    "{tag}: corner id"
                );
                assert_eq!(
                    geometry::out_word(out, index, geometry::OS_FINAL),
                    final_corner as u32,
                    "{tag}: final corner flag"
                );
            }
            geometry::MODE_LANE_MAX => {
                assert_eq!(flag, 0, "{tag}: fallback flag set on a valid lane_max query");
                f32_eq(
                    geometry::out_word(out, index, geometry::OS_A),
                    engine.lane_max_at(item.distance).to_bits(),
                    "lane_max",
                    &tag,
                );
            }
            geometry::MODE_POSITION => {
                let want = engine.position(item.distance, item.lane).unwrap_or_else(|e| panic!("{tag}: engine errored ({e}) on a valid sweep sample"));
                assert_eq!(flag, 0, "{tag}: fallback flag set on a valid position query");
                f32_eq(geometry::out_word(out, index, geometry::OS_A), want[0].to_bits(), "world x", &tag);
                f32_eq(geometry::out_word(out, index, geometry::OS_B), want[1].to_bits(), "world z", &tag);
            }
            _ => {
                let (next, world) = engine
                    .advance(item.distance, item.lane, item.movement, item.prev)
                    .unwrap_or_else(|e| panic!("{tag}: engine errored ({e}) on a valid sweep sample"));
                assert_eq!(flag, 0, "{tag}: fallback flag set on a valid advance query");
                f32_eq(geometry::out_word(out, index, geometry::OS_A), next.to_bits(), "next", &tag);
                f32_eq(geometry::out_word(out, index, geometry::OS_B), world[0].to_bits(), "world x", &tag);
                f32_eq(geometry::out_word(out, index, geometry::OS_C), world[1].to_bits(), "world z", &tag);
            }
        }
        for slot in [geometry::OS_A, geometry::OS_B, geometry::OS_C] {
            assert_ne!(
                geometry::out_word(out, index, slot),
                geometry::SENTINEL_BITS,
                "{tag}: sentinel bit pattern on a valid sample (course {})",
                courses[*course_index].course_id
            );
        }
    }
}

#[test]
fn kernel_matches_the_engine_on_a_dense_sweep() {
    let dir = geometry_dir();
    let engines = vec![
        retail_geometry::CourseGeometry::load(&dir, 10504, 2000.).expect("load 10504"),
        retail_geometry::CourseGeometry::load(&dir, 10604, 2000.).expect("load 10604"),
        retail_geometry::CourseGeometry::bundled(10613, 2400.).expect("bundle 10613"),
        retail_geometry::CourseGeometry::bundled(11102, 1800.).expect("bundle 11102"),
    ];
    let courses: Vec<retail_geometry::FlatCourse> = engines.iter().map(retail_geometry::flatten_course).collect();
    let ids: Vec<i32> = courses.iter().map(|c| c.course_id).collect();
    let flat = |c: &retail_geometry::FlatCourse| {
        assert_eq!(c.seg_x.len(), 2 * geometry::NSEG);
        assert_eq!(c.seg_q.len(), 8 * geometry::NSEG);
    };
    courses.iter().for_each(flat);

    let mut rows: Vec<(usize, geometry::GeometryItem, i32)> = Vec::new();
    let tiny = f32::from_bits(1);
    for (ci, (course, id)) in courses.iter().zip(&ids).enumerate() {
        let step = course.distance / 4096.;
        for k in 0..=4096 {
            let d = k as f32 * step;
            let lane_max_here = engines[ci].lane_max_at(d);
            let lanes = [0., 1. / 18., -1. / 18., 6. / 18., lane_max_here, 1.0e6];
            for ready in [true, false] {
                rows.push((ci, geometry::GeometryItem::corner(ci, d, ready), *id));
            }
            rows.push((ci, geometry::GeometryItem::lane_max(ci, d), *id));
            for lane in lanes {
                rows.push((ci, geometry::GeometryItem::position(ci, d, lane), *id));
            }
            if k % 4 == 0 {
                // previous_world from position(d - 0.1, lane), clamped at the
                // start line so every sample stays in the engine's domain.
                let pd = if d >= 0.1 { d - 0.1 } else { 0. };
                for lane in [0., 1. / 18., 6. / 18., 1.0e4] {
                    let prev = engines[ci].position(pd, lane).expect("valid previous position");
                    let cur = engines[ci].position(d, lane).expect("valid position");
                    let dx = cur[0] - prev[0];
                    let dz = cur[1] - prev[1];
                    let wd = (dx as f64 * dx as f64 + dz as f64 * dz as f64).sqrt() as f32;
                    let movements = [0., tiny, wd * 0.5, wd, wd * 10., 1000.];
                    for m in movements {
                        rows.push((ci, geometry::GeometryItem::advance(ci, d, lane, m, prev), *id));
                    }
                }
            }
        }
        // ULP probes: exact corner and lane-width boundary distances, plus the
        // adjacent f32 on each side, through all three query modes.
        let mut boundaries: Vec<f32> = Vec::new();
        for i in 0..course.ce_start.len() {
            boundaries.push(course.ce_start[i]);
            boundaries.push(course.ce_end[i]);
        }
        boundaries.extend_from_slice(&course.lm_boundary);
        for b in boundaries {
            for d in ulp_trio(b) {
                rows.push((ci, geometry::GeometryItem::corner(ci, d, true), *id));
                rows.push((ci, geometry::GeometryItem::corner(ci, d, false), *id));
                rows.push((ci, geometry::GeometryItem::lane_max(ci, d), *id));
                rows.push((ci, geometry::GeometryItem::position(ci, d, 6. / 18.), *id));
            }
        }
    }

    // The 10604 overrun: distances past the course end, and distances past
    // the end by whole multiples of the overrun distance (the fmod wrap).
    let ovr_ci = 1;
    let ovr = courses[ovr_ci].overrun_distance;
    assert!(courses[ovr_ci].overrun_present && ovr > 0., "10604 must carry an overrun");
    for k in 1..=200 {
        let d = 2000. + k as f32;
        rows.push((ovr_ci, geometry::GeometryItem::corner(ovr_ci, d, true), ids[ovr_ci]));
        rows.push((ovr_ci, geometry::GeometryItem::lane_max(ovr_ci, d), ids[ovr_ci]));
        rows.push((ovr_ci, geometry::GeometryItem::position(ovr_ci, d, 6. / 18.), ids[ovr_ci]));
        let prev = engines[ovr_ci].position(d - 0.1, 0.).expect("valid overrun previous position");
        rows.push((ovr_ci, geometry::GeometryItem::advance(ovr_ci, d, 0., 0.5, prev), ids[ovr_ci]));
    }
    for d in [2025. + ovr, 2050. + 2. * ovr, 2199. + 5. * ovr] {
        rows.push((ovr_ci, geometry::GeometryItem::position(ovr_ci, d, -1. / 18.), ids[ovr_ci]));
    }

    let out = geometry::run(&courses, &rows.iter().map(|r| r.1).collect::<Vec<_>>()).expect("sweep kernel run");
    report_device(&out);
    compare_rows(&out.out, &rows, &courses, &engines);
    println!("{} sweep items bit-identical across courses {:?}", rows.len(), ids);
}

#[test]
fn engine_errors_become_the_sentinel_and_the_fallback_flag() {
    let dir = geometry_dir();
    // Every captured course carries an overrun animation, so the engine's
    // own "lacks an overrun animation" error path (its clone-and-strip test)
    // is the only way into it. The kernel reaches it through the same state.
    let mut no_overrun = retail_geometry::CourseGeometry::load(&dir, 10504, 2000.).expect("load 10504");
    retail_geometry::drop_overrun(&mut no_overrun);
    let engines = vec![
        retail_geometry::CourseGeometry::load(&dir, 10504, 2000.).expect("load 10504"),
        retail_geometry::CourseGeometry::load(&dir, 10604, 2000.).expect("load 10604"),
        no_overrun,
    ];
    let courses: Vec<retail_geometry::FlatCourse> = engines.iter().map(retail_geometry::flatten_course).collect();
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let mut rows: Vec<(usize, geometry::GeometryItem, i32)> = Vec::new();
    let push = |rows: &mut Vec<(usize, geometry::GeometryItem, i32)>, ci, item| {
        rows.push((ci, item, courses[ci].course_id));
    };
    // position: non-finite or negative distance, non-finite lane.
    for d in [nan, -1., -0.0001] {
        push(&mut rows, 0, geometry::GeometryItem::position(0, d, 0.));
    }
    for lane in [inf, -inf, nan] {
        push(&mut rows, 0, geometry::GeometryItem::position(0, 100., lane));
    }
    // Past the course end, without and with an overrun (engine errors only
    // without: 10504 and 10604 both carry overruns in their captured data).
    push(&mut rows, 2, geometry::GeometryItem::position(2, 2001., 0.));
    push(&mut rows, 1, geometry::GeometryItem::position(1, 2001., 0.));
    // advance: bad movement or previous_world, and the no-overrun past-end.
    for m in [nan, -1., -inf] {
        push(&mut rows, 1, geometry::GeometryItem::advance(1, 100., 0., m, [0., 0.]));
    }
    for p in [[inf, 0.], [nan, nan], [0., -inf]] {
        push(&mut rows, 1, geometry::GeometryItem::advance(1, 100., 0., 0.5, p));
    }
    push(&mut rows, 2, geometry::GeometryItem::advance(2, 2001., 0., 0.5, [0., 0.]));
    push(&mut rows, 2, geometry::GeometryItem::advance(2, nan, 0., 0.5, [0., 0.]));
    // Valid samples in the same batch: the flag is per item, and the overrun
    // domain stays error-free through advance's clip path (prev is far away).
    push(&mut rows, 1, geometry::GeometryItem::advance(1, 2001., 0., 0.5, [0., 0.]));
    push(&mut rows, 0, geometry::GeometryItem::position(0, 2001., 6. / 18.));

    let out = geometry::run(&courses, &rows.iter().map(|r| r.1).collect::<Vec<_>>()).expect("error kernel run");
    report_device(&out);
    let mut error_items = 0;
    for (index, (ci, item, course_id)) in rows.iter().enumerate() {
        let tag = Checked { item: *item, course_id: *course_id }.tag();
        let flag = geometry::out_word(&out.out, index, geometry::OS_FLAG);
        // The engine's own verdict decides which items are the error domain.
        let engine_verdict = match item.mode {
            geometry::MODE_POSITION => engines[*ci].position(item.distance, item.lane).is_err(),
            _ => engines[*ci]
                .advance(item.distance, item.lane, item.movement, item.prev)
                .is_err(),
        };
        assert_eq!(flag as usize, engine_verdict as usize, "{tag}: fallback flag versus engine Err");
        if engine_verdict {
            error_items += 1;
            let slots = match item.mode {
                geometry::MODE_POSITION => [geometry::OS_A, geometry::OS_B].as_slice(),
                _ => [geometry::OS_A, geometry::OS_B, geometry::OS_C].as_slice(),
            };
            for slot in slots {
                assert_eq!(
                    geometry::out_word(&out.out, index, *slot),
                    geometry::SENTINEL_BITS,
                    "{tag}: sentinel expected in slot {slot}"
                );
            }
        } else {
            for slot in [geometry::OS_A, geometry::OS_B, geometry::OS_C] {
                assert_ne!(
                    geometry::out_word(&out.out, index, slot),
                    geometry::SENTINEL_BITS,
                    "{tag}: sentinel on a valid sample"
                );
            }
            match item.mode {
                geometry::MODE_POSITION => {
                    let want = engines[*ci].position(item.distance, item.lane).expect("valid");
                    f32_eq(geometry::out_word(&out.out, index, geometry::OS_A), want[0].to_bits(), "world x", &tag);
                    f32_eq(geometry::out_word(&out.out, index, geometry::OS_B), want[1].to_bits(), "world z", &tag);
                }
                _ => {
                    let (next, world) = engines[*ci]
                        .advance(item.distance, item.lane, item.movement, item.prev)
                        .expect("valid");
                    f32_eq(geometry::out_word(&out.out, index, geometry::OS_A), next.to_bits(), "next", &tag);
                    f32_eq(geometry::out_word(&out.out, index, geometry::OS_B), world[0].to_bits(), "world x", &tag);
                    f32_eq(geometry::out_word(&out.out, index, geometry::OS_C), world[1].to_bits(), "world z", &tag);
                }
            }
        }
    }
    println!("{error_items} error-domain items carry the sentinel and the fallback flag, {} valid items in the same batch do not", rows.len() - error_items);
}

#[test]
fn kernel_structures_match_the_opencl_layout() {
    // The kernel indexes these flat layouts. A host-side mismatch is UB.
    assert_eq!(geometry::CS_STRIDE * std::mem::size_of::<u32>(), 64);
    assert_eq!(geometry::IN_STRIDE * std::mem::size_of::<u32>(), 32);
    assert_eq!(geometry::OUT_STRIDE * std::mem::size_of::<u32>(), 32);
    assert_eq!(geometry::SENTINEL_BITS, f32::NAN.to_bits());
}
