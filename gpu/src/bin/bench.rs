//! Timing bench for the restricted spike: the same eight-tick chain run as
//! (a) native Rust on N worker threads, (b) the OpenCL kernel, and (c) the
//! f64/f32 fma microkernels that quantify B580 double throughput.
//!
//! Fixed iteration counts, stated here and in every table header:
//! - ticks per race: 8 (the kernel's ROWS, same as the correctness gate)
//! - GPU batch runs per N: 8 warmup + 256 measured, average reported
//! - Rust: every thread runs its chunk of the N cases 256 times
//! - fma probes: 4096 fma per chain, 4 independent chains per work item,
//!   one warmup launch + 20 measured launches, average reported
#![cfg(feature = "gpu")]

use globalsim_gpu::{gpu, reference};
use globalsim_gpu::reference::Case;
use globalsim_gpu::retail_physics::Parameters;
use std::time::Instant;

const REPEATS: usize = 256;
const WARMUP: usize = 8;
const FMA_REPEATS: usize = 20;
const FMA_ITERATIONS: i32 = 4096;
const SIZES: [usize; 5] = [1, 64, 256, 4096, 65536];

fn main() -> Result<(), String> {
    let parameters = reference::load_parameters();
    let kernel_params = reference::kernel_params();
    let spike = gpu::Spike::open()?;
    println!(
        "device: {} ({} CUs, single_fp_config {:#x}, double_fp_config {:#x})",
        spike.device.name,
        spike.device.compute_units,
        spike.device.single_fp_config,
        spike.device.double_fp_config,
    );
    println!("program build (once): {:.1} ms, options {:?}", spike.build_ms, gpu::BUILD_OPTIONS);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(20);
    println!("rust baseline: {threads} native threads, kernel: one work item per case\n");
    println!(
        "{:>6} | {:>8} {:>9} {:>8} {:>9} | {:>13} | {:>12} {:>13} | {:>7}",
        "N", "upload", "run(avg)", "read", "total", "GPU races/s", "Rust ms", "Rust races/s", "GPU/Rust"
    );
    for n in SIZES {
        let cases = reference::bench_cases(&parameters, n);
        let kernel_cases: Vec<_> = cases.iter().map(reference::kernel_case).collect();
        let job = spike.prepare(&kernel_params, n)?;
        let upload_ms = job.upload(&kernel_cases)?;
        for _ in 0..WARMUP {
            job.run()?;
        }
        let run_ms = (0..REPEATS).map(|_| job.run()).sum::<Result<f64, String>>()? / REPEATS as f64;
        let (rows, read_ms) = job.read()?;
        if n == 4096 {
            // The timing runs must execute the same chain the gate proved.
            check_rows(&parameters, &cases, &rows);
            println!("bit-exact spot check at N={n}: {} rows identical to the Rust reference", rows.len());
        }
        let rust_ms = bench_rust(&parameters, &cases, threads);
        // run_ms is the average milliseconds per launch; each launch runs N races.
        let gpu_rps = n as f64 / (run_ms / 1000.);
        let rust_rps = (n * REPEATS) as f64 / (rust_ms / 1000.);
        println!(
            "{n:>6} | {:>8.3} {:>9.3} {:>8.3} {:>9.3} | {:>13.0} | {:>12.3} {:>13.0} | {:>7.2}",
            upload_ms,
            run_ms,
            read_ms,
            upload_ms + run_ms + read_ms,
            gpu_rps,
            rust_ms,
            rust_rps,
            gpu_rps / rust_rps,
        );
    }

    println!(
        "\nfma probes: {FMA_ITERATIONS} fma per chain, 4 chains per work item, \
         {FMA_REPEATS} measured launches, avg reported"
    );
    println!("{:>6} | {:>14} {:>9} | {:>14} {:>9} | {:>11}", "N", "f64 ms", "f64 GF/s", "f32 ms", "f32 GF/s", "f64/f32");
    for n in SIZES {
        let f64_ms = spike.fma_probe(true, n, FMA_ITERATIONS, FMA_REPEATS)?;
        let f32_ms = spike.fma_probe(false, n, FMA_ITERATIONS, FMA_REPEATS)?;
        let gflops = |ms: f64| 2. * 4. * FMA_ITERATIONS as f64 * n as f64 / (ms / 1000.) / 1e9;
        println!(
            "{n:>6} | {:>14.3} {:>9.1} | {:>14.3} {:>9.1} | {:>11.2}",
            f64_ms,
            gflops(f64_ms),
            f32_ms,
            gflops(f32_ms),
            f64_ms / f32_ms,
        );
    }
    if spike.kernel_build_failures() != 0 {
        return Err("kernel build reported failures".to_string());
    }
    Ok(())
}

fn bench_rust(p: &Parameters, cases: &[Case], threads: usize) -> f64 {
    let started = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(cases.len().div_ceil(threads))
            .map(|chunk| {
                scope.spawn(move || {
                    let mut sink: u32 = 0;
                    for _ in 0..REPEATS {
                        for case in chunk {
                            let mut state = case.state;
                            for _ in 0..reference::ROWS {
                                state = reference::tick(
                                    p,
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
                            }
                            sink ^= state.time.to_bits();
                        }
                    }
                    sink
                })
            })
            .collect();
        for handle in handles {
            // Keep every thread's sink alive so the ticks cannot be optimized out.
            std::hint::black_box(handle.join().expect("rust baseline thread"));
        }
    });
    started.elapsed().as_secs_f64() * 1000.
}

fn check_rows(p: &Parameters, cases: &[Case], rows: &[reference::KernelRow]) {
    for (case_index, case) in cases.iter().enumerate() {
        let mut state = case.state;
        for row_index in 0..reference::ROWS {
            state = reference::tick(
                p,
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
            for (field, gpu_bits, rust_bits) in [
                ("time", row.time.to_bits(), state.time.to_bits()),
                ("pos", row.pos.to_bits(), state.pos.to_bits()),
                ("speed", row.speed.to_bits(), state.speed.to_bits()),
                ("hp", row.hp.to_bits(), state.hp.to_bits()),
                ("delay", row.delay.to_bits(), state.delay.to_bits()),
                ("finish", row.finish.to_bits(), state.finish.to_bits()),
            ] {
                assert_eq!(gpu_bits, rust_bits, "bench case {case_index} row {row_index} field {field}");
            }
            assert_eq!(row.phase as usize, state.phase, "bench case {case_index} row {row_index} phase");
            assert_eq!(row.dash != 0, state.dash, "bench case {case_index} row {row_index} dash");
            assert_eq!(row.finished != 0, state.finished, "bench case {case_index} row {row_index} finished");
        }
    }
}
