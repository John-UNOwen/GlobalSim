//! Tick-by-tick, runner-by-runner, field-by-field comparison of the CPU
//! engine's replay frames against a GPU race outcome (handoff step 4).

use crate::dep_cpu::TickFrame;
use crate::race::{FallbackReason, RaceOutcome};

/// The six replay values in row order, as `docs/BINARY-REPLAY.md` numbers them.
pub const FIELDS: [&str; 6] = ["pos", "speed", "hp", "lane", "rushing", "blocker"];

#[derive(Clone, Debug, PartialEq)]
pub struct Divergence {
    pub tick: usize,
    pub runner: usize,
    pub field: &'static str,
    pub cpu_bits: u64,
    pub gpu_bits: u64,
}

#[derive(Debug, PartialEq)]
pub enum Mismatch {
    /// The frames agree up to the count, and only one side keeps going.
    TickCount { cpu: usize, gpu: usize },
    RunnerCount { tick: usize, cpu: usize, gpu: usize },
    Field(Divergence),
    /// Every compared field agreed, but the runner flagged the race, so
    /// agreement is not parity.
    Flagged(FallbackReason),
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "tick {} runner {} field {}: cpu bits {:#018x} gpu bits {:#018x}",
            self.tick, self.runner, self.field, self.cpu_bits, self.gpu_bits
        )
    }
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::TickCount { cpu, gpu } => write!(f, "tick count differs: cpu {cpu} gpu {gpu}"),
            Self::RunnerCount { tick, cpu, gpu } => {
                write!(f, "tick {tick} runner count differs: cpu {cpu} gpu {gpu}")
            }
            Self::Field(divergence) => divergence.fmt(f),
            Self::Flagged(reason) => write!(f, "frames agree but the race is flagged: {reason:?}"),
        }
    }
}

/// Compares every tick of the CPU race against the GPU outcome. The first
/// mismatch wins: field differences report both raw bit patterns, and a tick
/// or runner present on only one side reports the two counts. `Ok` needs an
/// equal race from a runner that set no flag.
pub fn compare(cpu: &[TickFrame], gpu: &RaceOutcome) -> Result<(), Mismatch> {
    let shared_ticks = cpu.len().min(gpu.frames.len());
    for tick in 0..shared_ticks {
        let (cpu_runners, gpu_runners) = (&cpu[tick], &gpu.frames[tick]);
        for runner in 0..cpu_runners.len().min(gpu_runners.len()) {
            for (field, (c, g)) in FIELDS
                .iter()
                .zip(cpu_runners[runner].iter().zip(gpu_runners[runner].iter()))
            {
                let (cpu_bits, gpu_bits) = (c.to_bits(), g.to_bits());
                if cpu_bits != gpu_bits {
                    return Err(Mismatch::Field(Divergence {
                        tick,
                        runner,
                        field,
                        cpu_bits,
                        gpu_bits,
                    }));
                }
            }
        }
        if cpu_runners.len() != gpu_runners.len() {
            return Err(Mismatch::RunnerCount {
                tick,
                cpu: cpu_runners.len(),
                gpu: gpu_runners.len(),
            });
        }
    }
    if cpu.len() != gpu.frames.len() {
        return Err(Mismatch::TickCount { cpu: cpu.len(), gpu: gpu.frames.len() });
    }
    if let Some(reason) = &gpu.fallback {
        return Err(Mismatch::Flagged(reason.clone()));
    }
    Ok(())
}
