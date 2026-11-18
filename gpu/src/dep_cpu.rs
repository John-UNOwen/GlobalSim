//! The CPU side of the whole-race comparison tool: run the full engine
//! (path dependency, public API only) and decode its dense replay frames.

use crate::json::{self, Value};

/// One tick of replay: a row of the six replay values per runner, in the
/// order of `docs/BINARY-REPLAY.md`: pos, speed, hp, lane, rushing, blocker.
pub type TickFrame = Vec<[f64; 6]>;

pub struct CpuRace {
    pub frames: Vec<TickFrame>,
    pub runners: usize,
    /// Runners whose last-tick position has reached the course distance.
    pub finishers: usize,
    /// Each runner's `finishTimeRaw` in roster order, exactly as the result
    /// JSON reports it (the loop end time for a runner that never finished).
    pub finish_times: Vec<f64>,
    /// Skill activations as the reply reports them: (horse, skillId,
    /// detailIndex, t). t is the accumulatetime at the fire pass.
    pub detail_events: Vec<(usize, i64, i64, f64)>,
}

pub fn course_id(request: &str) -> Result<i32, String> {
    Ok(json::parse(request)?.get_i64("courseId")? as i32)
}

pub fn request_seed(request: &str) -> Result<i64, String> {
    json::parse(request)?.get_i64("seed")
}

/// Reads a file from the engine repo's `data/` directory. The tool points at
/// the checkout it is built inside, the same data the engine embeds.
pub fn engine_data(file: &str) -> Result<String, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data").join(file);
    std::fs::read_to_string(&path).map_err(|e| format!("reading {} failed: {e}", path.display()))
}

pub fn course_distance(course_id: i32) -> Result<f32, String> {
    let root = json::parse(&engine_data("course_data.json")?)?;
    root.get(&course_id.to_string())
        .and_then(|course| course.get_f64("distance").ok())
        .map(|distance| distance as f32)
        .ok_or_else(|| format!("course_data.json has no distance for {course_id}"))
}

/// The request with `retainAllFrames: true` forced on.
pub fn with_retain_all_frames(request: &str) -> Result<String, String> {
    let mut value = json::parse(request)?;
    match &mut value {
        Value::Object(map) => {
            map.insert("retainAllFrames".into(), Value::Bool(true));
        }
        _ => return Err("race request must be a JSON object".into()),
    }
    let mut out = String::new();
    json::to_string_into(&value, &mut out);
    Ok(out)
}

/// Runs the engine on (request, seed) with every frame retained and decodes
/// the runner rows. Tick timestamps stay out of the rows: the replay values
/// are what the parity contract compares.
pub fn run_cpu_race(request: &str, seed: i64) -> Result<CpuRace, String> {
    let distance = course_distance(course_id(request)?)? as f64;
    let mut prepared = globalsim::PreparedRace::new(&with_retain_all_frames(request)?)?;
    let simulation = prepared.run(seed)?;
    let mut reply = String::new();
    simulation.write_json(&mut reply);
    let finish_times = json::parse(&reply)?
        .get_array("results")?
        .iter()
        .map(|row| row.get_f64("finishTimeRaw"))
        .collect::<Result<Vec<_>, _>>()?;
    let detail_events = json::parse(&reply)?
        .get_array("skillDetailEvents")?
        .iter()
        .map(|row| {
            Ok::<_, String>((
                row.get_f64("horseIndex")? as usize,
                row.get_f64("skillId")? as i64,
                row.get_f64("detailIndex")? as i64,
                row.get_f64("t")?,
            ))
        })
        .collect::<Result<Vec<(usize, i64, i64, f64)>, _>>()?;
    let frames = simulation.frames();
    let runners = frames.horse_count();
    let decoded: Vec<TickFrame> =
        (0..frames.len()).map(|index| frames.frame(index).1.to_vec()).collect();
    let finishers = decoded
        .last()
        .map_or(0, |last| last.iter().filter(|row| row[0] >= distance).count());
    Ok(CpuRace { frames: decoded, runners, finishers, finish_times, detail_events })
}
