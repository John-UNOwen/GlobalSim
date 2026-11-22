//! Shared launches for the one-call entry points (`race_exec::run`,
//! `race_exec::run_full`, `classic_exec::run`).
//!
//! A launch costs about as much for one race as for a few hundred (one race
//! per work item, the device mostly idle), and kernels from separate contexts
//! or processes run one after another, not side by side. So when several
//! threads call in at once, the calls share one context and one launch: the
//! first caller becomes the leader and launches every job queued so far, the
//! others wait, and whoever queued while that launch ran goes out in the next
//! one. A lone caller launches at once, with no added wait.
//!
//! Every race is still one work item with its own inputs, so a race's answer
//! does not depend on what else is in the launch. `GSIM_GPU_NO_COALESCE=1`
//! turns it off (one context and one launch per call, the old behaviour).

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Condvar, Mutex, MutexGuard};

pub(crate) fn enabled() -> bool {
    std::env::var_os("GSIM_GPU_NO_COALESCE").is_none_or(|v| v.is_empty() || v == "0")
}

/// The context only ever runs under `Coalescer::ctx`'s lock, one thread at a
/// time, the same rule the app's gpu_lane::Slot relies on.
struct SendCtx<C>(C);
unsafe impl<C> Send for SendCtx<C> {}

struct State<J, R> {
    pending: Vec<(u64, J)>,
    done: BTreeMap<u64, Result<R, String>>,
    next: u64,
    leading: bool,
}

pub(crate) struct Coalescer<C, J, R> {
    state: Mutex<State<J, R>>,
    wake: Condvar,
    ctx: Mutex<Option<SendCtx<C>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl<C, J, R> Coalescer<C, J, R> {
    pub(crate) const fn new() -> Self {
        Coalescer {
            state: Mutex::new(State { pending: Vec::new(), done: BTreeMap::new(), next: 0, leading: false }),
            wake: Condvar::new(),
            ctx: Mutex::new(None),
        }
    }

    /// Queue `job` and return its answer. `batch` gets every queued job in
    /// queue order and must return one answer per job, in the same order.
    pub(crate) fn submit(
        &self,
        job: J,
        open: impl Fn() -> Result<C, String>,
        batch: impl Fn(&mut C, Vec<J>) -> Vec<Result<R, String>>,
    ) -> Result<R, String> {
        let mut state = lock(&self.state);
        let id = state.next;
        state.next += 1;
        state.pending.push((id, job));
        loop {
            if let Some(answer) = state.done.remove(&id) {
                return answer;
            }
            if state.leading {
                state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            state.leading = true;
            let taken = std::mem::take(&mut state.pending);
            drop(state);

            let ids: Vec<u64> = taken.iter().map(|(id, _)| *id).collect();
            let jobs: Vec<J> = taken.into_iter().map(|(_, job)| job).collect();
            let answers = {
                let mut ctx = lock(&self.ctx);
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    if ctx.is_none() {
                        *ctx = Some(SendCtx(open()?));
                    }
                    let answers = batch(&mut ctx.as_mut().expect("opened above").0, jobs);
                    if answers.len() != ids.len() {
                        return Err(format!("a shared launch answered {} of {} jobs", answers.len(), ids.len()));
                    }
                    Ok(answers)
                }));
                match outcome {
                    Ok(Ok(answers)) => answers,
                    Ok(Err(reason)) => ids.iter().map(|_| Err(reason.clone())).collect(),
                    Err(_) => {
                        // A panic mid-launch may leave the context half
                        // used. The next leader opens a fresh one.
                        *ctx = None;
                        ids.iter().map(|_| Err("the shared launch panicked".to_string())).collect()
                    }
                }
            };

            state = lock(&self.state);
            for (id, answer) in ids.into_iter().zip(answers) {
                state.done.insert(id, answer);
            }
            state.leading = false;
            self.wake.notify_all();
        }
    }
}
