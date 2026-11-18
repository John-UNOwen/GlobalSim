//! The spike's core claim: the OpenCL restricted_tick kernel reproduces the
//! Rust reference bit for bit on the B580, for the six fixture cases x eight
//! ticks. Fails with case/row/field and both bit patterns on first mismatch.
#![cfg(feature = "gpu")]

use globalsim_gpu::{gpu, reference};

const FIELDS: [(&str, fn(&reference::KernelRow) -> u32, fn(&reference::S) -> u32); 6] = [
    ("time", |r| r.time.to_bits(), |s| s.time.to_bits()),
    ("pos", |r| r.pos.to_bits(), |s| s.pos.to_bits()),
    ("speed", |r| r.speed.to_bits(), |s| s.speed.to_bits()),
    ("hp", |r| r.hp.to_bits(), |s| s.hp.to_bits()),
    ("delay", |r| r.delay.to_bits(), |s| s.delay.to_bits()),
    ("finish", |r| r.finish.to_bits(), |s| s.finish.to_bits()),
];

#[test]
fn gpu_kernel_matches_the_rust_reference_bit_for_bit() {
    let parameters = reference::load_parameters();
    let cases = reference::cases(&parameters);
    let kernel_params = reference::kernel_params();
    let spike = gpu::Spike::open().expect("an OpenCL device passing the exactness gate");
    println!(
        "device: {} ({} CUs, single_fp_config {:#x}, double_fp_config {:#x}, build {:.1} ms, options {:?})",
        spike.device.name,
        spike.device.compute_units,
        spike.device.single_fp_config,
        spike.device.double_fp_config,
        spike.build_ms,
        gpu::BUILD_OPTIONS,
    );

    let job = spike.prepare(&kernel_params, cases.len()).expect("job buffers");
    let upload_ms = job.upload(&cases.iter().map(reference::kernel_case).collect::<Vec<_>>()).expect("upload");
    let run_ms = job.run().expect("run");
    let (rows, read_ms) = job.read().expect("read");
    assert_eq!(rows.len(), cases.len() * reference::ROWS);

    for (case_index, case) in cases.iter().enumerate() {
        let mut state = case.state;
        for row_index in 0..reference::ROWS {
            state = reference::tick(
                &parameters,
                state,
                case.speed,
                case.guts,
                case.power,
                case.strategy,
                case.aptitude,
                case.ground,
                case.distance,
                reference::DT,
            );
            let row = &rows[case_index * reference::ROWS + row_index];
            for (field, gpu_bits, rust_bits) in FIELDS {
                let (got, want) = (gpu_bits(row), rust_bits(&state));
                if got != want {
                    panic!(
                        "case {case_index} ({}) row {row_index} field {field}: rust {want:#010x} gpu {got:#010x} \
                         (inputs speed {} guts {} power {} strategy {} aptitude {} ground {} distance {} dt {})",
                        case.name,
                        case.speed,
                        case.guts,
                        case.power,
                        case.strategy,
                        case.aptitude,
                        case.ground,
                        case.distance,
                        reference::DT,
                    );
                }
            }
            assert_eq!(row.phase as usize, state.phase, "case {case_index} ({}) row {row_index} field phase", case.name);
            assert_eq!(row.dash != 0, state.dash, "case {case_index} ({}) row {row_index} field dash", case.name);
            assert_eq!(row.finished != 0, state.finished, "case {case_index} ({}) row {row_index} field finished", case.name);
        }
    }
    println!(
        "{} cases x {} ticks bit-identical (upload {upload_ms:.3} ms, run {run_ms:.3} ms, read {read_ms:.3} ms)",
        cases.len(),
        reference::ROWS,
    );
}

#[test]
fn kernel_struct_sizes_match_the_opencl_layout() {
    // The kernel reads these exact byte layouts; a mismatch would be UB.
    assert_eq!(std::mem::size_of::<reference::KernelCase>(), 104);
    assert_eq!(std::mem::size_of::<reference::KernelRow>(), 36);
    assert_eq!(std::mem::size_of::<reference::KernelParams>(), 244);
}
