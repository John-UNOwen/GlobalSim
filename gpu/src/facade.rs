//! Batch facade: what the app's run_compact lane calls. One `GpuRaces`
//! instance per app, built once, then `run_batch` (or `run_repeated` for one
//! request over many seeds) per launch. The module is not Sync: keep the
//! instance on one thread.
//!
//! The contract is the compact meta plus the GSIMRP01 packet, or the full
//! reply text for `run_batch_full`. A race the kernel cannot answer
//! bit-exactly comes back as `fallback`, everything else None, and the
//! batch's other races are untouched. CPU replays stay on the CPU lanes.
//!
//! Two kernel lanes share this facade. A request carrying any retail flag runs
//! the retail kernel, the rest run the classic kernel, which runs the
//! engine's classic skill lane through the shared skill compiler.

use crate::json::Value;
use crate::{classic_exec, race_exec, replay_packet};

/// The vendored engine json module, re-exported so callers can build and
/// parse meta with the same `Value` this facade uses.
pub use crate::json;

/// One request's answer. The compact lanes set `meta` and `packet` together,
/// the full lane sets `reply`. A race that stays on the CPU lane carries
/// `fallback` and nothing else.
pub struct BatchOutcome {
    pub meta: Option<String>,
    pub packet: Option<Vec<u8>>,
    pub fallback: Option<String>,
    pub reply: Option<String>,
}

pub struct GpuRaces {
    ctx: race_exec::RaceContext,
    classic: Option<classic_exec::ClassicContext>,
}

impl GpuRaces {
    /// Runs the exactness gate, loads the engine data and builds both kernels
    /// once. The instance then stays alive for the app's whole session.
    pub fn new() -> Result<GpuRaces, String> {
        let ctx = race_exec::RaceContext::open()?;
        // A classic program that will not build leaves the retail lane running
        // and sends classic requests back as fallbacks with the reason.
        let classic = classic_exec::ClassicContext::open().ok();
        Ok(GpuRaces { ctx, classic })
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

    /// Runs one batch, retail requests in one retail launch and classic
    /// requests in one classic launch. The retail launchable races must share
    /// one runner count, the same constraint the kernel grid carries, and a
    /// mixed batch is an error for the caller to split. The classic launch
    /// pads to the batch's largest field. A request the domain gates reject
    /// comes back as `fallback`, and a request that fails to parse comes back
    /// as `fallback` too.
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
        let mut outcomes: Vec<Option<BatchOutcome>> = (0..requests.len()).map(|_| None).collect();
        let mut retail: Vec<(usize, race_exec::Prepared)> = Vec::new();
        let mut classic: Vec<(usize, classic_exec::Prepared)> = Vec::new();
        for (index, (request, seed)) in requests.iter().zip(seeds).enumerate() {
            if takes_classic_lane(request) {
                match &mut self.classic {
                    Some(ctx) => classic.push((
                        index,
                        ctx.prepare(request, *seed as i64).unwrap_or_else(classic_exec::stub),
                    )),
                    None => outcomes[index] = Some(BatchOutcome {
                        meta: None,
                        packet: None,
                        fallback: Some("the classic lane's kernel did not build".into()),
                        reply: None,
                    }),
                }
            } else {
                retail.push((
                    index,
                    self.ctx
                        .prepare(request, *seed as i64)
                        .unwrap_or_else(|reason| race_exec::Prepared::stub(reason)),
                ));
            }
        }
        if !retail.is_empty() {
            let slots: Vec<usize> = retail.iter().map(|&(index, _)| index).collect();
            let players: Vec<f64> =
                slots.iter().map(|&index| player_horse_index(&requests[index])).collect();
            let mut prepared: Vec<race_exec::Prepared> =
                retail.drain(..).map(|(_, prepared)| prepared).collect();
            let (run, _) = self.ctx.run_prepared(&mut prepared, None)?;
            for (index, outcome) in slots.into_iter().zip(build_outcomes(&run, players.into_iter())) {
                outcomes[index] = Some(outcome);
            }
        }
        if !classic.is_empty() {
            let slots: Vec<usize> = classic.iter().map(|&(index, _)| index).collect();
            let players: Vec<f64> =
                slots.iter().map(|&index| player_horse_index(&requests[index])).collect();
            let mut prepared: Vec<classic_exec::Prepared> =
                classic.drain(..).map(|(_, prepared)| prepared).collect();
            let run = self
                .classic
                .as_ref()
                .expect("a classic batch means the classic lane opened")
                .run_prepared(&mut prepared)?;
            for (index, outcome) in
                slots.into_iter().zip(build_classic_outcomes(&run, players.into_iter()))
            {
                outcomes[index] = Some(outcome);
            }
        }
        Ok(outcomes.into_iter().map(|outcome| outcome.unwrap()).collect())
    }

    /// Runs one batch in a single launch and answers each launchable race
    /// with the engine's full (non-compact) reply text in `reply`, built
    /// from the kernel's reply channels (see `full_reply`). A race the
    /// kernel flags or the assembly refuses carries `fallback` instead, and
    /// the batch's other races are untouched.
    pub fn run_batch_full(
        &mut self,
        requests: &[String],
        seeds: &[i32],
    ) -> Result<Vec<BatchOutcome>, String> {
        if requests.len() != seeds.len() {
            return Err(format!(
                "run_batch_full needs one seed per request ({} requests, {} seeds)",
                requests.len(),
                seeds.len()
            ));
        }
        let mut outcomes: Vec<Option<BatchOutcome>> = (0..requests.len()).map(|_| None).collect();
        let mut retail: Vec<(usize, race_exec::Prepared)> = Vec::new();
        let mut classic: Vec<(usize, classic_exec::Prepared)> = Vec::new();
        for (index, (request, seed)) in requests.iter().zip(seeds).enumerate() {
            if takes_classic_lane(request) {
                match &mut self.classic {
                    Some(ctx) => classic.push((
                        index,
                        ctx.prepare(request, *seed as i64).unwrap_or_else(classic_exec::stub),
                    )),
                    None => outcomes[index] = Some(BatchOutcome {
                        meta: None,
                        packet: None,
                        fallback: Some("the classic lane's kernel did not build".into()),
                        reply: None,
                    }),
                }
            } else {
                match self.ctx.prepare(request, *seed as i64) {
                    Ok(mut prepared) => {
                        prepared.set_full();
                        retail.push((index, prepared));
                    }
                    Err(reason) => retail.push((index, race_exec::Prepared::stub(reason))),
                }
            }
        }
        if !retail.is_empty() {
            let slots: Vec<usize> = retail.iter().map(|&(index, _)| index).collect();
            let mut prepared: Vec<race_exec::Prepared> =
                retail.drain(..).map(|(_, prepared)| prepared).collect();
            let (run, _) = self.ctx.run_prepared(&mut prepared, None)?;
            for (index, outcome) in slots.into_iter().zip(run.races.iter()).map(|(index, race)| {
                let (request, seed) = (requests[index].as_str(), seeds[index] as i64);
                (index, full_outcome(race.fallback.as_ref(), || crate::full_reply::build(request, seed, race)))
            }) {
                outcomes[index] = Some(outcome);
            }
        }
        if !classic.is_empty() {
            let slots: Vec<usize> = classic.iter().map(|&(index, _)| index).collect();
            let mut prepared: Vec<classic_exec::Prepared> =
                classic.drain(..).map(|(_, prepared)| prepared).collect();
            let run = self
                .classic
                .as_ref()
                .expect("a classic batch means the classic lane opened")
                .run_prepared(&mut prepared)?;
            for (index, outcome) in slots.into_iter().zip(run.races.iter()).map(|(index, race)| {
                let (request, seed) = (requests[index].as_str(), seeds[index] as i64);
                (index, full_outcome(race.fallback.as_ref(), || crate::classic_reply::build(request, seed, race)))
            }) {
                outcomes[index] = Some(outcome);
            }
        }
        Ok(outcomes.into_iter().map(|outcome| outcome.unwrap()).collect())
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
    /// included. The classic lane's seed also feeds its host-side
    /// construction, so a classic request prepares once per seed.
    pub fn run_repeated(
        &mut self,
        request: &str,
        seeds: &[i32],
    ) -> Result<Vec<BatchOutcome>, String> {
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        // The one request parses once here, not once per seed: its reply
        // fields are the only request input the meta path reads.
        let player = player_horse_index(request);
        if takes_classic_lane(request) {
            let ctx = match &mut self.classic {
                Some(ctx) => ctx,
                None => {
                    return Ok(seeds
                        .iter()
                        .map(|_| BatchOutcome {
                            meta: None,
                            packet: None,
                            fallback: Some("the classic lane's kernel did not build".into()),
                            reply: None,
                        })
                        .collect())
                }
            };
            let mut prepared: Vec<classic_exec::Prepared> = seeds
                .iter()
                .map(|&seed| ctx.prepare(request, seed as i64).unwrap_or_else(classic_exec::stub))
                .collect();
            let run = ctx.run_prepared(&mut prepared)?;
            return Ok(build_classic_outcomes(&run, std::iter::repeat(player).take(seeds.len())));
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
                reply: None,
            },
            None => {
                let meta = compact_meta(race, player);
                let (meta_text, packet) = replay_packet::build(&meta, &race.frames);
                BatchOutcome {
                    meta: Some(meta_text),
                    packet: Some(packet),
                    fallback: None,
                    reply: None,
                }
            }
        })
        .collect()
}

/// Which kernel lane a request belongs to. The engine builds its retail
/// config when the runtime, sections or initialization flag is set and runs
/// the classic solver otherwise (src/api.rs:74-77, src/lib.rs:1521-1523).
/// `experimentalRetailStartup` requires the runtime flag (src/api.rs:66), so
/// listing it keeps a startup-only request on the path that quotes the
/// engine's own refusal.
fn takes_classic_lane(request: &str) -> bool {
    match json::parse(request) {
        Ok(root) => root_takes_classic_lane(&root),
        Err(_) => false,
    }
}

fn root_takes_classic_lane(root: &Value) -> bool {
    let flagged = |key: &str| root.get(key).and_then(Value::as_bool) == Some(true);
    !(flagged("experimentalRetailRuntime")
        || flagged("experimentalRetailInitialization")
        || flagged("experimentalRetailSections")
        || flagged("experimentalRetailStartup"))
}

/// A full-reply answer: the assembled text, or the reason the race stayed
/// on the CPU lane.
fn full_outcome(
    fallback: Option<&String>,
    build: impl FnOnce() -> Result<String, String>,
) -> BatchOutcome {
    if let Some(reason) = fallback {
        return BatchOutcome { meta: None, packet: None, fallback: Some(reason.clone()), reply: None };
    }
    match build() {
        Ok(text) => BatchOutcome { meta: None, packet: None, fallback: None, reply: Some(text) },
        Err(reason) => BatchOutcome { meta: None, packet: None, fallback: Some(reason), reply: None },
    }
}

/// One classic launch's answers, in input order: flagged races carry the
/// reason, the rest get the compact meta and the GSIMRP01 packet built from
/// the six replay columns.
fn build_classic_outcomes(
    run: &classic_exec::ClassicRun,
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
                reply: None,
            },
            None => {
                let meta = classic_meta(race, player);
                let frames: Vec<Vec<[f64; 6]>> = race
                    .frames
                    .iter()
                    .map(|tick| tick.iter().map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]]).collect())
                    .collect();
                let (meta_text, packet) = replay_packet::build_classic(&meta, &frames);
                BatchOutcome {
                    meta: Some(meta_text),
                    packet: Some(packet),
                    fallback: None,
                    reply: None,
                }
            }
        })
        .collect()
}

/// The compact meta of a classic race, same keys and types as `compact_meta`.
/// The engine leaves hpEmptyOnRace null unless the retail config ran
/// (src/lib.rs:994). skillEvents entries mirror the engine's identity pair
/// per hook-emitted event: numeric activations carry the skill id, kakari
/// episodes carry the string id (src/lib.rs:661-720).
fn classic_meta(race: &classic_exec::ClassicGpu, player: f64) -> Value {
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
                    ("hpEmptyOnRace", Value::Null),
                    (
                        "lastSpurtStartDistance",
                        json::num(if race.lsttrans_all[r] > 0. { race.lsttrans_all[r] } else { -1. }),
                    ),
                    ("rushingPenaltyCount", json::num(race.rscore_all[r] as f64)),
                    ("rushingStartCount", json::num(race.tempt_all[r] as f64)),
                    ("startDelayTime", json::num(race.start_delay_all[r])),
                ])
            })
            .collect(),
    );
    let mut skill_events: Vec<Value> = Vec::new();
    {
        let stream = &race.skev;
        let mut i = 0;
        while i < stream.len() {
            match stream[i] {
                0 if i + 9 <= stream.len() => {
                    skill_events.push(json::obj(vec![
                        ("horseIndex", json::num(stream[i + 2] as f64)),
                        ("skillId", json::num(stream[i + 3] as f64)),
                    ]));
                    i += 9;
                }
                1 if i + 6 <= stream.len() => {
                    skill_events.push(json::obj(vec![
                        ("horseIndex", json::num(stream[i + 1] as f64)),
                        ("skillId", Value::String("kakari".into())),
                    ]));
                    i += 6;
                }
                _ => break,
            }
        }
    }
    let skill_events = json::arr(skill_events);
    let mut fields = vec![
        ("playerHorseIndex", json::num(player)),
        ("results", results),
        ("skillEvents", skill_events),
    ];
    // The engine tags grid replies on every lane (src/lib.rs:1135-1138).
    if race.grid {
        fields.push(("experimentalGridVersion", json::num(1.0)));
    }
    json::obj(fields)
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
    let mut fields = vec![
        ("playerHorseIndex", json::num(player)),
        ("results", results),
        ("skillEvents", skill_events),
    ];
    // The engine tags grid replies (src/lib.rs:1135-1138), nothing else.
    if race.grid {
        fields.push(("experimentalGridVersion", json::num(1.0)));
    }
    json::obj(fields)
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
/// can pre-filter requests against known facts without a device. The checks
/// follow the lane the request routes to. One classic gate needs the course
/// table and stays in `classic_exec::prepare`: a course with more than
/// CLS_MAX_SLOPES slopes. A runner above the 1200 adjusted raw power is not a
/// gate: `prepare` injects the asitame record like the engine does.
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
    // The grid mirror of prepare(): the engine validates the grid key before
    // anything else (src/lib.rs:294), so the grid refusal comes first and
    // quotes the engine's message.
    let horse_count = match root.get_array("horses") {
        Ok(horses) => horses.len(),
        Err(error) => return Some(error),
    };
    let grid_valid = root.get("experimentalGrid").is_some_and(|g| {
        g.as_object().is_some_and(|o| o.len() == 2)
            && g.get("columns").and_then(Value::as_f64) == Some(20.0)
            && g.get("rowSpacing").and_then(Value::as_f64) == Some(3.0)
            && (21..=120).contains(&horse_count)
    });
    if root.get("experimentalGrid").is_some() && !grid_valid {
        return Some(
            "experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0."
                .into(),
        );
    }
    // Parse-time shapes the engine rejects, and engine-supported modes the
    // kernel does not port. Same texts prepare() flags with.
    if root.get("experimentalLaterStaminaMechanics").is_some_and(|v| v.as_bool().is_none()) {
        return Some("experimentalLaterStaminaMechanics must be a boolean.".into());
    }
    if root.get("traceConservePower").is_some_and(|v| v.as_bool().is_none()) {
        return Some("traceConservePower must be a boolean".into());
    }
    if root.get("experimentalRetailStartup").is_some_and(|v| v.as_bool().is_none()) {
        return Some("experimentalRetailStartup must be a boolean".into());
    }
    if root_takes_classic_lane(&root) {
        // The classic lane's refusals, the same texts and order as
        // classic_exec::prepare's domain gates.
        if root.get("retailRuntime").and_then(Value::as_bool) == Some(true)
            || root.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true)
        {
            return Some("retail runtime belongs to the retail lane".into());
        }
        // The engine errors on classic before answering any grid or mode
        // (lib.rs:1504-1506), so quote it ahead of the mode fallbacks; the
        // invalid-grid shape is already quoted by the prefilter above.
        if root.get("traceConservePower").and_then(Value::as_bool) == Some(true) {
            return Some("traceConservePower requires experimentalRetailRuntime".into());
        }
        // The three flag-only systems are ported; only the >2100 m
        // staminasyoubu injection still falls back.
        if let Some(reason) = classic_exec::later_stamina_refusal(request) {
            return Some(reason);
        }
        let skilled = root
            .get("horses")
            .and_then(Value::as_array)
            .is_some_and(|horses| {
                horses.iter().any(|h| {
                    h.get("skills").and_then(Value::as_array).is_some_and(|s| !s.is_empty())
                })
            });
        if skilled {
            if let Some(reason) = classic_exec::classic_gate_reason(request) {
                return Some(reason);
            }
        }
        if (1..=classic_exec::CLS_MAXR).contains(&horse_count) {
            None
        } else {
            Some("roster size outside the kernel's range".into())
        }
    } else {
        if root.get("experimentalRetailRuntime").and_then(Value::as_bool) != Some(true) {
            return Some("non-retail runtime".into());
        }
        // The engine refuses geometry with grid starts (api.rs:68-71).
        if grid_valid && root.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true)
        {
            return Some(
                "World geometry requires the experimental runtime and standard starting grid".into(),
            );
        }
        if root.get("experimentalLaterStaminaMechanics").and_then(Value::as_bool) == Some(true) {
            return Some(
                "experimentalLaterStaminaMechanics draws from a live solver stream \
                 the retail kernel resolves host-side".into(),
            );
        }
        if (1..=race_exec::RACE_MAXR).contains(&horse_count) {
            None
        } else {
            Some("roster size outside the kernel's range".into())
        }
    }
}
