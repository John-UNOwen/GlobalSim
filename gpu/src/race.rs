//! The GPU race-runner seam and its fallback flag (handoff step 5). The
//! scheduler re-runs on the CPU every race whose outcome carries a flag.
//! A whole-race kernel runner sits behind this trait and plugs into the
//! comparator and the tool with no other change.

// The kernel-backed helpers exist only with the gpu feature.
#[cfg(feature = "gpu")]
use crate::{dep_cpu, reference};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// The race crosses a domain the kernel cannot guarantee bit-exactly.
    UnsupportedDomain(String),
    /// The kernel could not run: no device passes the exactness gate, the
    /// crate was built without the gpu feature, or OpenCL failed.
    DeviceUnavailable(String),
}

/// One race as a GPU backend produced it. `frames[tick]` holds one row of
/// the six replay values per runner, in the order of `docs/BINARY-REPLAY.md`.
/// Tick timestamps are not part of the outcome (see README).
pub struct RaceOutcome {
    pub frames: Vec<Vec<[f64; 6]>>,
    /// A set flag is the scheduler's rerun-on-CPU queue entry.
    pub fallback: Option<FallbackReason>,
}

pub trait RaceRunner {
    fn run_race(&mut self, request: &str, seed: i64) -> RaceOutcome;
}

/// Today's whole-race GPU side: the `restricted_tick` kernel run once per
/// race for the player lane, no field, no geometry, 8 ticks. The kernel
/// chain is device-proven, but field/RNG/geometry are not ported, so every
/// race comes back flagged from the first tick. The emitted ticks still get
/// compared: their first divergence is the progress measure (handoff step 4).
pub struct RestrictedStub;

const NOT_PORTED: &str = "field/rng/geometry not ported";

#[cfg(feature = "gpu")]
fn stub_strategy(name: &str) -> Result<usize, String> {
    match name.to_ascii_uppercase().as_str() {
        "NIGE" => Ok(1),
        "SENKOU" => Ok(2),
        "SASI" | "SASHI" => Ok(3),
        "OIKOMI" => Ok(4),
        "OONIGE" => Ok(5),
        _ => Err(format!("restricted stub cannot map strategy {name:?}")),
    }
}

/// The player lane as one restricted-kernel case. The request's stats ride
/// straight through: the engine's stat adjustments, initialization lottery,
/// start delay and terrain are exactly the domains the fallback flag marks.
#[cfg(feature = "gpu")]
fn stub_case(request: &str) -> Result<reference::KernelCase, String> {
    let root = crate::json::parse(request)?;
    let distance = dep_cpu::course_distance(root.get_i64("courseId")? as i32)?;
    let player = root.get("playerHorseIndex").and_then(crate::json::Value::as_i64).unwrap_or(0);
    let player = usize::try_from(player).unwrap_or(0);
    let horse = root.get_array("horses")?.get(player).ok_or("playerHorseIndex outside the roster")?;
    let strategy = stub_strategy(horse.get_str("strategy")?)?;
    let text = dep_cpu::engine_data("global_parameters.json")?;
    let parameters = crate::json::parse(&text)?
        .get("parameters")
        .and_then(|value| crate::retail_physics::Parameters::from_value(value).ok())
        .ok_or("global_parameters.json has no usable parameters")?;
    Ok(reference::KernelCase {
        time: 0.,
        pos: 0.,
        speed: 0.,
        hp: parameters.initial_hp(horse.get_f64("stamina")? as f32, strategy, distance),
        // The start-delay lottery comes from the unported RNG, so the stub starts unheld.
        delay: 0.,
        finish: f32::NAN,
        phase: 0,
        dash: 1,
        finished: 0,
        speed_in: horse.get_f64("speed")? as f32,
        guts: horse.get_f64("guts")? as f32,
        power: horse.get_f64("power")? as f32,
        strategy: strategy as i32,
        aptitude: 1.,
        ground: 1.,
        distance,
        dt: reference::DT,
        section: 0.,
        pace: 1.,
        force_in: 0.,
        slope: 0.,
        moving_lane_skill: 0,
        downhill: 0,
        uphill: 0,
        spot_struggle: 0.,
        bonus: 0.,
    })
}

/// Runs the kernel for the one case and reads back `ROWS` ticks x one runner.
#[cfg(feature = "gpu")]
fn stub_frames(case: &reference::KernelCase) -> Result<Vec<Vec<[f64; 6]>>, String> {
    let spike = crate::gpu::Spike::open()?;
    let parameters = reference::kernel_params_from_text(&dep_cpu::engine_data("global_parameters.json")?);
    let job = spike.prepare(&parameters, 1)?;
    job.upload(std::slice::from_ref(case))?;
    job.run()?;
    let (rows, _) = job.read()?;
    // One case, so the kernel's ROWS rows are the eight ticks of one lane.
    Ok(rows
        .iter()
        .map(|row| {
            // The restricted chain has no field: no lane movement, no
            // rushing, no front blocker, so those three stay zero.
            vec![[row.pos as f64, row.speed as f64, row.hp as f64, 0., 0., -1.]]
        })
        .collect())
}

#[cfg(feature = "gpu")]
impl RaceRunner for RestrictedStub {
    /// The seed stays unused: the restricted chain contains no RNG, so the
    /// unported RNG is part of the flag this stub always sets.
    fn run_race(&mut self, request: &str, _seed: i64) -> RaceOutcome {
        let case = match stub_case(request) {
            Ok(case) => case,
            Err(reason) => {
                return RaceOutcome {
                    frames: Vec::new(),
                    fallback: Some(FallbackReason::UnsupportedDomain(reason)),
                }
            }
        };
        let frames = match stub_frames(&case) {
            Ok(frames) => frames,
            Err(reason) => {
                return RaceOutcome {
                    frames: Vec::new(),
                    fallback: Some(FallbackReason::DeviceUnavailable(reason)),
                }
            }
        };
        RaceOutcome { frames, fallback: Some(FallbackReason::UnsupportedDomain(NOT_PORTED.into())) }
    }
}

#[cfg(not(feature = "gpu"))]
impl RaceRunner for RestrictedStub {
    fn run_race(&mut self, _request: &str, _seed: i64) -> RaceOutcome {
        RaceOutcome {
            frames: Vec::new(),
            fallback: Some(FallbackReason::DeviceUnavailable(
                "crate built without the gpu feature".into(),
            )),
        }
    }
}
