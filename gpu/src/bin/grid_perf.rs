//! Grid n=120 launch profile: the per-seed marginal cost at 120 runners for
//! single races and small batches, with the stage split, plus a check that a
//! batch crossing the frame-byte launch budget decodes identically to the
//! same races launched one at a time.

use globalsim_gpu::json::Value;
use globalsim_gpu::{json, race_exec};
use std::time::Instant;

const COURSE: i32 = 10504;
const GRID20: &str = r#"{"columns":20,"rowSpacing":3}"#;
const SIZES: [usize; 6] = [1, 2, 4, 8, 16, 32];

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn grid_request(base: &str, n: usize, seed: i64) -> Result<String, String> {
    let mut value = json::parse(base)?;
    let json::Value::Object(map) = &mut value else {
        return Err("request is not a JSON object".into());
    };
    let template = map
        .get("horses")
        .and_then(Value::as_array)
        .filter(|h| !h.is_empty())
        .ok_or("request has no horses array")?
        .first()
        .cloned()
        .expect("non-empty array has a first horse");
    let horses: Vec<Value> = (0..n)
        .map(|i| {
            let mut horse = template.clone();
            let json::Value::Object(hm) = &mut horse else { unreachable!() };
            hm.insert("gateIndex".into(), Value::Number(i as f64));
            horse
        })
        .collect();
    map.insert("courseId".into(), Value::Number(COURSE as f64));
    map.insert("seed".into(), Value::Number(seed as f64));
    map.insert("horses".into(), json::arr(horses));
    map.remove("pairedRandomStreams");
    map.insert("experimentalGrid".into(), json::parse(GRID20)?);
    // The engine refuses geometry with grid starts (api.rs:68-71).
    map.insert("experimentalRetailGeometry".into(), Value::Bool(false));
    let mut text = String::new();
    json::to_string_into(&value, &mut text);
    Ok(text)
}

fn main() -> Result<(), String> {
    let base = std::fs::read_to_string(repo_file("examples/v77-race.jsonl"))
        .map_err(|e| format!("request file: {e}"))?
        .lines()
        .next()
        .ok_or("request file empty")?
        .to_string();
    let mut ctx = race_exec::RaceContext::open()?;
    println!("device {}", ctx.device().name);

    let request = grid_request(&base, 120, 1)?;
    let t = Instant::now();
    let template = ctx.prepare(&request, 1)?;
    println!("prepare 120-horse grid request: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    println!("\nrun_repeated path, n 120 (ms)");
    println!(
        "  {:>4}  {:>8}  {:>7}  {:>8}  {:>9}  {:>10}  {:>8}  {:>11}",
        "m", "upload", "tables", "launch", "readback", "decode", "total", "ms per seed",
    );
    let mut batch_frames: Vec<Vec<Vec<[f64; 6]>>> = Vec::new();
    for &m in &SIZES {
        // One warm launch at m 1, then every size runs once on the same
        // warm context. A 120-horse race costs tens of seconds, so an
        // extra launch per size is not affordable.
        let seeds: Vec<i32> = (1..=m as i32).collect();
        let mut prepared: Vec<race_exec::Prepared> =
            seeds.iter().map(|&s| template.with_seed(s)).collect();
        let t = Instant::now();
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        let wall_ms = t.elapsed().as_secs_f64() * 1e3;
        for race in &run.races {
            if let Some(reason) = &race.fallback {
                return Err(format!("grid race flagged: {reason}"));
            }
        }
        let s = &run.stage_ms;
        println!(
            "  {:>4}  {:>8.1}  {:>7.1}  {:>9.1}  {:>10.1}  {:>8.1}  {:>11.1}  {:>11.2}",
            m, s.upload_ms, s.tables_ms, s.launch_ms, s.readback_ms,
            s.decode_ms, s.total_ms, wall_ms / m as f64,
        );
        if m == 16 {
            batch_frames = run.races.iter().map(|r| r.frames.clone()).collect();
        }
    }

    // A batch of 16 races crosses the 512 MiB frame budget (15 per launch at
    // n 120), so the same races run one at a time must decode identically.
    // Seed 1 is the first slot of launch 1 and seed 16 the last of launch 2.
    println!("\nchunked batch (m 16) versus single launches, to_bits equality");
    let mut mismatches = 0usize;
    for seed in [1i32, 16] {
        let mut prepared = vec![template.with_seed(seed)];
        let (run, _) = ctx.run_prepared(&mut prepared, None)?;
        let single = &run.races[0];
        if let Some(reason) = &single.fallback {
            return Err(format!("single launch flagged: {reason}"));
        }
        let batch = &batch_frames[(seed - 1) as usize];
        if batch.len() != single.frames.len() {
            println!("  seed {seed}: tick count {} vs {}", batch.len(), single.frames.len());
            mismatches += 1;
            continue;
        }
        for (tick, (b, s)) in batch.iter().zip(&single.frames).enumerate() {
            for (r, (brow, srow)) in b.iter().zip(s).enumerate() {
                for (c, (bv, sv)) in brow.iter().zip(srow).enumerate() {
                    if bv.to_bits() != sv.to_bits() {
                        println!("  seed {seed} tick {tick} runner {r} column {c} differs");
                        mismatches += 1;
                        break;
                    }
                }
            }
        }
    }
    if mismatches == 0 {
        println!("  all 16 races match frame for frame");
    }
    Ok(())
}
