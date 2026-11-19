//! Batch facade: what the app's run_compact lane calls. One `GpuRaces`
//! instance per app, built once, then `run_batch` (or `run_repeated` for one
//! request over many seeds) per launch. The module is not Sync: keep the
//! instance on one thread.
//!
//! The contract is the compact meta plus the GSIMRP01 packet. A race the
//! kernel cannot answer bit-exactly comes back as `fallback` instead, meta
//! and packet both None, and the batch's other races are untouched. Full
//! replies and CPU replays stay on the CPU lanes: the kernel does not export
//! skillDetailEvents, competitionEvents, positioningPhases or the per-tick
//! frame times inside the meta, and the compact contract does not ask for
//! them.

use crate::json::Value;
use crate::{race_exec, replay_packet};

/// The vendored engine json module, re-exported so callers can build and
/// parse meta with the same `Value` this facade uses.
pub use crate::json;

/// One request's answer. Exactly one arm is set: the compact meta and packet
/// together, or the reason the race stays on the CPU lane.
pub struct BatchOutcome {
    pub meta: Option<String>,
    pub packet: Option<Vec<u8>>,
    pub fallback: Option<String>,
}

pub struct GpuRaces {
    ctx: race_exec::RaceContext,
}

impl GpuRaces {
    /// Runs the exactness gate, loads the engine data and builds the kernel
    /// once. The instance then stays alive for the app's whole session.
    pub fn new() -> Result<GpuRaces, String> {
        Ok(GpuRaces { ctx: race_exec::RaceContext::open()? })
    }

    /// The gated device, for the app to report which GPU took the lane.
    pub fn device(&self) -> String {
        self.ctx.device().name.clone()
    }

    /// Register the app's course JSON (the text it feeds the CPU lane's
    /// `register_course_geometry`) as this instance's kernel geometry for the
    /// course, so both lanes flatten one copy. Injecting replaces any earlier
    /// copy, file or injected, from the next prepare. Courses the app never
    /// registers keep the `data/course_geometry` file fallback, which is what
    /// standalone tests and tools use.
    pub fn register_geometry(&mut self, course_id: i32, json_text: &str) -> Result<(), String> {
        self.ctx.register_geometry(course_id, json_text)
    }

    /// Runs one batch in a single launch. The launchable races must share one
    /// runner count, the same constraint the kernel grid carries, and a mixed
    /// batch is an error for the caller to split. A request the domain gates
    /// reject comes back as `fallback`, and a request that fails to parse
    /// comes back as `fallback` too.
    pub fn run_batch(
        &mut self,
        requests: &[String],
        seeds: &[i32],
    ) -> Result<Vec<BatchOutcome>, String> {
        if requests.len() != seeds.len() {
            return Err(format!(
                "run_batch needs one seed per request ({} requests, {} seeds)",
                requests.len(),
                seeds.len()
            ));
        }
        let mut prepared: Vec<_> = Vec::with_capacity(requests.len());
        for (request, seed) in requests.iter().zip(seeds) {
            prepared.push(
                self.ctx
                    .prepare(request, *seed as i64)
                    .unwrap_or_else(|reason| race_exec::Prepared::stub(reason)),
            );
        }
        let (run, _) = self.ctx.run_prepared(&mut prepared, None)?;
        Ok(build_outcomes(&run, requests.iter().map(|request| player_horse_index(request))))
    }

    /// Runs one request against many seeds in a single launch, preparing it
    /// once. Chosen over a `run_batch` prepare cache keyed by request hash:
    /// a cache would keep every request alive indefinitely and change what
    /// `run_batch` costs depending on call history, while this entry says
    /// explicitly which batches share their prepare.
    ///
    /// The seed enters the prepared inputs only through the `RI_SEED` word
    /// (see `race_exec::Prepared`), so the entries past the first are the
    /// template clone with that one word rewritten. Outputs are byte-identical
    /// to `run_batch` on the same request repeated per seed, fallbacks
    /// included.
    pub fn run_repeated(
        &mut self,
        request: &str,
        seeds: &[i32],
    ) -> Result<Vec<BatchOutcome>, String> {
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        let mut prepared: Vec<_> = match self.ctx.prepare(request, seeds[0] as i64) {
            Ok(template) => seeds.iter().map(|&seed| template.with_seed(seed)).collect(),
            // One parse failure flags every seed, same as run_batch does for
            // one bad request among many.
            Err(reason) => seeds
                .iter()
                .map(|_| race_exec::Prepared::stub(reason.clone()))
                .collect(),
        };
        let (run, _) = self.ctx.run_prepared(&mut prepared, None)?;
        // The one request parses once here, not once per seed: its reply
        // fields are the only request input the meta path reads.
        let player = player_horse_index(request);
        Ok(build_outcomes(&run, std::iter::repeat(player).take(seeds.len())))
    }
}

/// The playerHorseIndex of a request, defaulting to 0 like the engine lane.
fn player_horse_index(request: &str) -> f64 {
    json::parse(request)
        .ok()
        .and_then(|root| root.get("playerHorseIndex").and_then(Value::as_f64))
        .unwrap_or(0.)
}

/// One launch's answers, in input order: flagged races carry the reason, the
/// rest get the compact meta and the GSIMRP01 packet.
fn build_outcomes(
    run: &race_exec::RaceRun,
    players: impl Iterator<Item = f64>,
) -> Vec<BatchOutcome> {
    run.races
        .iter()
        .zip(players)
        .map(|(race, player)| match &race.fallback {
            Some(reason) => BatchOutcome {
                meta: None,
                packet: None,
                fallback: Some(reason.clone()),
            },
            None => {
                let meta = compact_meta(race, player);
                let (meta_text, packet) = replay_packet::build(&meta, &race.frames);
                BatchOutcome {
                    meta: Some(meta_text),
                    packet: Some(packet),
                    fallback: None,
                }
            }
        })
        .collect()
}

/// The reply object compact_record reads: playerHorseIndex, results in roster
/// order, and the skillEvents (horseIndex, skillId) pairs the kernel logged.
/// The keys carry the engine's names, spellings and JSON types.
fn compact_meta(race: &race_exec::RaceGpu, player: f64) -> Value {
    // finishOrder: the engine's stable sort of the roster by finish time,
    // unfinished runners at the loop's own end time.
    let n = race.finish_raw_all.len();
    let finished: Vec<f64> = (0..n)
        .map(|r| race.finish_raw_all[r].unwrap_or(race.loop_end))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| finished[a].total_cmp(&finished[b]));
    let mut order_index = vec![0usize; n];
    for (rank, &horse_i) in order.iter().enumerate() {
        order_index[horse_i] = rank;
    }

    let results = json::arr(
        (0..n)
            .map(|r| {
                json::obj(vec![
                    ("finishOrder", json::num(order_index[r] as f64)),
                    ("finishTimeRaw", json::num(finished[r])),
                    ("hpEmptyOnRace", Value::Bool(race.hp_empty_all[r])),
                    (
                        "lastSpurtStartDistance",
                        json::num(if race.lss_all[r] > 0. { race.lss_all[r] } else { -1. }),
                    ),
                    ("rushingPenaltyCount", json::num(race.rscore_all[r] as f64)),
                    ("rushingStartCount", json::num(race.tempt_all[r] as f64)),
                    ("startDelayTime", json::num(race.delay_all[r])),
                ])
            })
            .collect(),
    );
    let skill_events = json::arr(
        race.events
            .iter()
            .map(|&(runner, id)| {
                json::obj(vec![
                    ("horseIndex", json::num(runner as f64)),
                    (
                        "skillId",
                        if id == race_exec::EV_KAKARI {
                            Value::String("kakari".into())
                        } else {
                            json::num(id as f64)
                        },
                    ),
                ])
            })
            .collect(),
    );
    json::obj(vec![
        ("playerHorseIndex", json::num(player)),
        ("results", results),
        ("skillEvents", skill_events),
    ])
}

/// Device metadata stand-in for the pure gate: the extension string plus the
/// two fp-config words, as the OpenCL device reports them.
pub struct DeviceFacts {
    pub extensions: String,
    pub single_fp_config: u64,
    pub double_fp_config: u64,
}

impl DeviceFacts {
    /// The exactness-gate refusals for these facts, empty when accepted.
    pub fn rejections(&self) -> Vec<String> {
        crate::gpu::fp_gate_rejections(
            &self.extensions,
            self.single_fp_config,
            self.double_fp_config,
        )
    }
}

/// Why the batch would refuse a request, None when it would run. `device`
/// None models "no GPU passed the exactness gate", Some carries one device's
/// facts. The device checks come first, then the request checks, so a caller
/// can pre-filter requests against known facts without a device.
pub fn gate_reasons(request: &str, device: Option<&DeviceFacts>) -> Option<String> {
    let device = match device {
        None => return Some("no GPU passes the exactness gate".into()),
        Some(facts) => facts,
    };
    let reasons = device.rejections();
    if !reasons.is_empty() {
        return Some(reasons.join(", "));
    }
    let root = match json::parse(request) {
        Ok(root) => root,
        Err(error) => return Some(error),
    };
    if root.get("experimentalGrid").is_some() {
        return Some("grid mode".into());
    }
    if root.get("experimentalRetailRuntime").and_then(Value::as_bool) != Some(true) {
        return Some("non-retail runtime".into());
    }
    match root.get_array("horses") {
        Ok(horses) if (1..=race_exec::RACE_MAXR).contains(&horses.len()) => None,
        Ok(_) => Some("roster size outside the kernel's range".into()),
        Err(error) => Some(error),
    }
}
