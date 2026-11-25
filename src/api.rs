//! Prepared, thread-confined races with fresh mutable state for every seed.
use crate::*;

fn initialize_tables() -> Result<(), String> {
    static READY: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    READY.get_or_init(|| {
        course_data::init_courses(load_course_table(COURSE_DATA_JSON)?);
        Ok(())
    }).clone()
}

fn conditions() -> &'static BTreeMap<String, activation_conditions::Condition> {
    static TABLE: std::sync::OnceLock<BTreeMap<String, activation_conditions::Condition>> = std::sync::OnceLock::new();
    TABLE.get_or_init(activation_conditions::build_conditions)
}

/// Immutable race preparation plus reusable roster facts. Create one on each
/// calling thread. The handle cannot be sent between threads or shared by them.
pub struct PreparedRace {
    spec: RaceSpec,
    skills: Rc<SkillTable>,
    initialization: Option<retail_initialization::PreparedInitialization>,
    retail_sections: bool,
    retail_runtime: bool,
    paired_streams: bool,
    world_geometry: Option<Rc<retail_geometry::CourseGeometry>>,
    startup_pass: bool,
    retain_all_frames: bool,
    gate_perm: Option<Vec<usize>>,
    signature: Option<Value>,
    roster: Option<PreparedRoster>,
}

/// One session's skill table: the parsed `skillData` value and its decoded
/// form. A worker session registers it once and every request that omits
/// skillData shares it, so the 357 KB table parses once per session.
pub(crate) struct SessionSkills {
    raw: Value,
    decoded: Rc<SkillTable>,
}

// Per thread because a PreparedRace is confined to the thread that created it,
// so a worker session (one thread) is exactly the sharing scope.
thread_local! {
    static SESSION_SKILLS: std::cell::RefCell<Option<Rc<SessionSkills>>> = const { std::cell::RefCell::new(None) };
}

/// Register the skill table this thread's preparations share. Requests that
/// omit `skillData` then prepare from it instead of resending the table text.
pub fn register_skill_table(json_text: &str) -> Result<(), String> {
    let raw = json::parse(json_text)?;
    let decoded = Rc::new(load_skill_value(&raw)?);
    SESSION_SKILLS.with(|cell| *cell.borrow_mut() = Some(Rc::new(SessionSkills { raw, decoded })));
    Ok(())
}

/// Drop this thread's session table. Hosts that mix sessions on one thread (a
/// test harness) call it before preparing requests that carry their own table.
pub fn clear_skill_table() {
    SESSION_SKILLS.with(|cell| *cell.borrow_mut() = None);
}

fn session_skills() -> Option<Rc<SessionSkills>> {
    SESSION_SKILLS.with(|cell| cell.borrow().clone())
}

impl PreparedRace {
    /// Prepare a conventional JSON request. Only the seed changes between runs.
    /// Preparation cost is separate from per-seed simulation cost.
    pub fn new(input: &str) -> Result<Self, String> {
        initialize_tables()?;
        let input = json::parse(input)?;
        let session = match input.get("skillData") {
            Some(_) => None,
            None => session_skills(),
        };
        Self::prepare_with(input, None, session)
    }

    /// Prepare a parsed request. `skills`, when given, must be the decoded
    /// table of the request's `skillData` (or of the bundled table).
    pub(crate) fn prepare(input: Value, skills: Option<Rc<SkillTable>>) -> Result<Self, String> {
        Self::prepare_with(input, skills, None)
    }

    /// As `prepare`, plus the session table that answers a request omitting
    /// `skillData`. The raw value feeds retail initialization, the decoded form
    /// the run, so neither is rebuilt per prepare.
    fn prepare_with(input: Value, skills: Option<Rc<SkillTable>>, session: Option<Rc<SessionSkills>>) -> Result<Self, String> {
        let (input, gate_perm) = gate_order(input)?;
        let session = match input.get("skillData") {
            Some(_) => None,
            None => session,
        };
        let raw_skills: &Value = match (&session, input.get("skillData")) {
            (_, Some(data)) => data,
            (Some(table), None) => &table.raw,
            (None, None) => bundled_skills()?,
        };
        let skills: Rc<SkillTable> = match skills {
            Some(skills) => skills,
            None => match &session {
                Some(table) => table.decoded.clone(),
                None => Rc::new(load_skill_value(raw_skills)?),
            },
        };
        let spec = parse_race_spec(&input)?;
        let course = bundled_courses()?.get(&spec.course_id.to_string())
            .ok_or("Missing retail course")?;
        let retail_runtime = input.get("experimentalRetailRuntime").and_then(Value::as_bool) == Some(true);
        if spec.trace_conserve_power && !retail_runtime {
            return Err("traceConservePower requires experimentalRetailRuntime".into());
        }
        let startup_pass = match input.get("experimentalRetailStartup") {
            None => false,
            Some(value) => value.as_bool().ok_or("experimentalRetailStartup must be a boolean")?,
        };
        if startup_pass && !retail_runtime {
            return Err("experimentalRetailStartup requires the experimental runtime".into());
        }
        let world_geometry = if input.get("experimentalRetailGeometry").and_then(Value::as_bool) == Some(true) {
            if !retail_runtime || spec.experimental_grid {
                return Err("World geometry requires the experimental runtime and standard starting grid".into());
            }
            Some(bundled_geometry(spec.course_id, course_data::get_course(spec.course_id).distance as f32)?)
        } else { None };
        let retail_sections = input.get("experimentalRetailSections").and_then(Value::as_bool) == Some(true);
        let paired_streams = input.get("pairedRandomStreams").and_then(Value::as_bool) == Some(true);
        let initialization = if input.get("experimentalRetailInitialization").and_then(Value::as_bool) == Some(true)
            || retail_sections || retail_runtime {
            let horses = spec.horses.iter().map(|horse| {
                Ok(json::obj(vec![
                    ("skills", Value::Array(horse.skills.iter().map(|(sid, level, _)| json::obj(vec![
                        ("skill_id", json::num(sid.parse::<f64>().unwrap_or(0.))),
                        ("level", json::num(*level as f64)),
                    ])).collect())),
                    ("base_wisdom", json::num(race_solver_builder::ranked_base_stat(
                        horse.wisdom, horse.mood, horse.team_rank) as f64)),
                    ("running_style", json::num(Strategy::parse_str(&horse.strategy_str)?.base_running_style() as f64)),
                ]))
            }).collect::<Result<Vec<_>, String>>()?;
            Some(retail_initialization::PreparedInitialization::new(&horses, course, raw_skills,
                bundled_parameters()?.get("parameters").ok_or("Missing retail parameters")?, paired_streams)?)
        } else { None };
        Ok(Self { spec, skills, initialization, retail_sections, retail_runtime, paired_streams,
            world_geometry, startup_pass, retain_all_frames: input.get("retainAllFrames").and_then(Value::as_bool) == Some(true),
            gate_perm, signature: input.get("dataSignature").cloned(), roster: None })
    }

    /// Run one independent seed with every mutable solver, RNG, timer and
    /// activation observer rebuilt. The result owns its complete selected replay.
    pub fn run(&mut self, seed: i64) -> Result<Simulation, String> {
        self.run_value(seed, true).map(|value| Simulation { value })
    }

    /// `run` as the worker's reply value. Without `capture_frames` the reply
    /// carries no replay frames or frame metadata; every other key is unchanged.
    pub(crate) fn run_value(&mut self, seed: i64, capture_frames: bool) -> Result<Value, String> {
        self.spec.seed = seed;
        let initialization = self.initialization.as_ref().map(|plan| {
            let seed = i32::try_from(seed).map_err(|_| "Retail seed must be a signed 32-bit integer".to_string())?;
            plan.initialize(seed)
        }).transpose()?;
        let mut value = race_solver_builder::with_skills(self.skills.clone(), || run_race(
            &self.spec, conditions(), initialization.as_ref(), self.retail_sections, self.retail_runtime,
            self.paired_streams, self.world_geometry.clone(), self.startup_pass, self.retain_all_frames, capture_frames, None,
            &mut self.roster))?;
        if let Value::Object(map) = &mut value {
            map.insert("experimentalRetailInitialization".into(), Value::Bool(initialization.is_some()));
            // Only paired replies carry the key, so default-mode replies stay
            // byte-identical to the pinned engine.
            if self.paired_streams { map.insert("pairedRandomStreams".into(), Value::Bool(true)); }
            if let Some(signature) = &self.signature { map.insert("dataSignature".into(), signature.clone()); }
        }
        if let Some(perm) = &self.gate_perm { restore_request_order(&mut value, perm); }
        Ok(value)
    }
}

/// Owned output. Frame views borrow this result and remain valid until it drops.
pub struct Simulation { value: Value }

impl Simulation {
    pub fn frames(&self) -> &crate::ReplayFrames {
        match self.value.get("frames") {
            Some(Value::ReplayFrames(frames)) => frames,
            _ => unreachable!("Prepared runs always retain numeric replay storage"),
        }
    }

    pub fn write_json(&self, destination: &mut String) {
        json::to_string_into(&self.value, destination);
    }

    /// Consume the result, replacing both caller-owned buffers with its binary
    /// replay packet and JSON metadata. Existing capacity can be reused.
    pub fn into_binary(mut self, metadata: &mut String, packet: &mut Vec<u8>) {
        binary_replay::encode(&mut self.value, metadata, packet);
    }
}

#[cfg(test)]
mod session_table_tests {
    use super::*;

    fn reply(race: &mut PreparedRace, seed: i64) -> String {
        let mut out = String::new();
        race.run(seed).expect("run").write_json(&mut out);
        out
    }

    #[test]
    fn session_skill_table_answers_sparse_requests_identically() {
        let line = include_str!("../examples/race.jsonl").trim();
        let sparse = json::parse(line).expect("example request parses");
        // The example omits skillData, so it answers from the bundled table: make
        // that table the session's and compare the two prep paths on it.
        let table = bundled_skills().expect("bundled table").clone();
        register_skill_table(&json::to_string(&table)).expect("table registers");
        let mut full = sparse.clone();
        if let Value::Object(map) = &mut full {
            map.insert("skillData".into(), table);
        }
        let mut direct = PreparedRace::new(&json::to_string(&full)).expect("direct prepare");
        let mut session = PreparedRace::new(&json::to_string(&sparse)).expect("session prepare");
        for seed in [1i64, 20261115] {
            assert_eq!(reply(&mut direct, seed), reply(&mut session, seed), "seed {seed} differs");
        }
        clear_skill_table();
    }
}

