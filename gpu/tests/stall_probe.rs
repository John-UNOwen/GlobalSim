//! Multi-consumer stall probe: one process, several OpenCL contexts, the same
//! v77 line-1 batches the app drives. Prints one `PROBE {json}` line per run
//! so a script can compare concurrent wall time against solo runs.
//!
//! Modes (argv 1, then K threads, reps, seeds per batch):
//!   solo K R N       one context, R sequential batches (baseline)
//!   threads K R N    K threads, own context each, batch released by a barrier
//!   shared K R N     K threads, one context behind a Mutex (fix candidate)
//!   stagger K R N    like threads, thread i sleeps i*400 ms before entering
//!
//! Run: cargo test --features gpu,initialization,geometry,race --release \
//!        --test stall_probe -- --ignored --nocapture
//! One mode per invocation via PROBE_MODE (plus PROBE_K PROBE_R PROBE_N).

use globalsim_gpu::race_exec::{self, Prepared, RaceContext};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// Same justification as the app's gpu_lane::Slot: the queue and programs are
/// touched by one thread at a time (own thread in threads/stagger, under the
/// Mutex in shared), never concurrently across threads.
struct SendCtx(RaceContext);
unsafe impl Send for SendCtx {}

fn v77_line1() -> String {
    std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .expect("examples/v77-race.jsonl")
        .lines()
        .next()
        .expect("empty request file")
        .to_string()
}

/// One batch on `ctx` exactly like facade::run_repeated: prepare once, one
/// with_seed clone per seed, one launch. Returns (wall_ms, stage timings).
fn run_batch(ctx: &mut RaceContext, request: &str, base_seed: i32, n: i32) -> (f64, race_exec::StageTimings) {
    let t = Instant::now();
    let template: Prepared = ctx.prepare(request, base_seed as i64).expect("prepare");
    let prepare_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut prepared: Vec<Prepared> = (0..n).map(|i| template.with_seed(base_seed + i)).collect();
    let clone_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let (run, _) = ctx.run_prepared(&mut prepared, None).expect("launch");
    assert!(run.races.iter().all(|r| r.fallback.is_none()), "probe race fell back");
    let run_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("PROBE-GS {{\"prepare_ms\":{prepare_ms:.1},\"clone_ms\":{clone_ms:.1},\"run_prepared_ms\":{run_ms:.1},\"stage_total_ms\":{:.1}}}", run.stage_ms.total_ms);
    (prepare_ms + clone_ms + run_ms, run.stage_ms)
}

fn report(mode: &str, tid: usize, rep: usize, open_ms: f64, since: Duration, wall_ms: f64, s: &race_exec::StageTimings) {
    println!("PROBE {{\"mode\":\"{mode}\",\"tid\":{tid},\"rep\":{rep},\"open_ms\":{:.1},\"t_ms\":{:.1},\"wall_ms\":{wall_ms:.1},\"launch_ms\":{:.1},\"upload_ms\":{:.1},\"readback_ms\":{:.1},\"decode_ms\":{:.1},\"prepass_ms\":{:.1}}}",
        open_ms, since.as_secs_f64() * 1e3, s.launch_ms, s.upload_ms, s.readback_ms, s.decode_ms, s.prepass_ms);
}

#[test]
#[ignore = "diagnostic: needs the B580 and prints a timing timeline"]
fn stall_probe() {
    let mode = std::env::var("PROBE_MODE").unwrap_or_else(|_| "threads".into());
    let k: usize = std::env::var("PROBE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let reps: usize = std::env::var("PROBE_R").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let n: i32 = std::env::var("PROBE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let request = v77_line1();

    match mode.as_str() {
        "solo" => {
            let t = Instant::now();
            let mut ctx = RaceContext::open().expect("open");
            let open_ms = t.elapsed().as_secs_f64() * 1e3;
            for rep in 0..reps {
                let (wall, s) = run_batch(&mut ctx, &request, 20261115, n);
                report("solo", 0, rep, open_ms, Duration::ZERO, wall, &s);
            }
        }
        "threads" | "stagger" => {
            let barrier = Arc::new(Barrier::new(k));
            let stagger = mode == "stagger";
            let mode_label: &'static str = if stagger { "stagger" } else { "threads" };
            std::thread::scope(|scope| {
                for tid in 0..k {
                    let barrier = Arc::clone(&barrier);
                    let request = request.clone();
                    scope.spawn(move || {
                        let t = Instant::now();
                        let mut ctx = RaceContext::open().expect("open");
                        let open_ms = t.elapsed().as_secs_f64() * 1e3;
                        if stagger && tid > 0 {
                            std::thread::sleep(Duration::from_millis(400 * tid as u64));
                        }
                        for rep in 0..reps {
                            barrier.wait();
                            let start = Instant::now();
                            let (wall, s) = run_batch(&mut ctx, &request, 20261115 + 1000 * tid as i32, n);
                            report(mode_label, tid, rep, open_ms, start.elapsed(), wall, &s);
                        }
                    });
                }
            });
        }
        "shared" => {
            let t = Instant::now();
            let ctx = Arc::new(Mutex::new(SendCtx(RaceContext::open().expect("open"))));
            let open_ms = t.elapsed().as_secs_f64() * 1e3;
            let barrier = Arc::new(Barrier::new(k));
            std::thread::scope(|scope| {
                for tid in 0..k {
                    let barrier = Arc::clone(&barrier);
                    let ctx = Arc::clone(&ctx);
                    let request = request.clone();
                    scope.spawn(move || {
                        for rep in 0..reps {
                            barrier.wait();
                            let start = Instant::now();
                            let (wall, s) = {
                                let mut guard = ctx.lock().expect("shared ctx");
                                run_batch(&mut guard.0, &request, 20261115 + 1000 * tid as i32, n)
                            };
                            report("shared", tid, rep, open_ms, start.elapsed(), wall, &s);
                        }
                    });
                }
            });
        }
        other => panic!("unknown PROBE_MODE {other}"),
    }
}
