//! Coverage audit (handoff E1): what the retail compiler still refuses.
//! For each request line the bin compiles the full field, then every skill
//! of the request's skillData alone on runner 0 (host-side only, no GPU),
//! and prints a histogram of refusal families plus the full-field table
//! usage against the SK_* caps. Usage:
//!
//!   cargo run --release --bin skill_coverage [REQUEST.jsonl ...]
//!
//! With no arguments it audits examples/cm-10701-race.jsonl and
//! examples/v77-race.jsonl.
use globalsim_gpu::json::{self, Value};
use globalsim_gpu::skills_compile::{self, ClassicExtras};
use std::collections::BTreeMap;
use std::path::Path;

fn repo_file(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// Drop the per-id prefix and the condition text so a refusal maps to its
/// family, the same normalization parity_audit uses.
fn refusal_family(id: &str, reason: &str) -> String {
    let mut family = reason.to_string();
    let prefix = format!("skill {id} ");
    if let Some(stripped) = family.strip_prefix(&prefix) {
        family = stripped.to_string();
    }
    if let Some((_, tail)) = family.rsplit_once("': ") {
        family = tail.to_string();
    }
    family
}

fn skill_id_in(reason: &str) -> Option<&str> {
    let mut words = reason.split_whitespace();
    while let Some(w) = words.next() {
        if w == "skill" {
            return words.next();
        }
    }
    None
}

/// Set runner 0's skills to the single (id, level 1), clear every other
/// runner, mirroring the coverage probe.
fn solo_field(root: &Value, id: &str) -> Value {
    let mut value = root.clone();
    let Value::Object(map) = &mut value else { unreachable!() };
    let Some(Value::Array(horses)) = map.get_mut("horses") else { unreachable!() };
    for (i, h) in horses.iter_mut().enumerate() {
        if let Value::Object(hm) = h {
            let skills = if i == 0 {
                let mut one = std::collections::BTreeMap::new();
                one.insert("skillId".to_string(), Value::String(id.to_string()));
                one.insert("level".to_string(), Value::Number(1.0));
                vec![Value::Object(one)]
            } else {
                Vec::new()
            };
            hm.insert("skills".to_string(), Value::Array(skills));
        }
    }
    value
}

/// Remove every copy of the named skill from every runner, as the strip loop
/// of the saturate bench does.
fn strip_skill(root: &mut Value, id: &str) -> usize {
    let mut removed = 0;
    let Value::Object(map) = root else { unreachable!() };
    let Some(Value::Array(horses)) = map.get_mut("horses") else { unreachable!() };
    for h in horses.iter_mut() {
        if let Value::Object(hm) = h {
            if let Some(Value::Array(skills)) = hm.get_mut("skills") {
                let before = skills.len();
                skills.retain(|s| s.get("skillId").and_then(Value::as_str) != Some(id));
                removed += before - skills.len();
            }
        }
    }
    removed
}

fn print_histogram(title: &str, families: &BTreeMap<String, usize>) {
    println!("{title}");
    let mut rows: Vec<(&String, &usize)> = families.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (family, count) in rows {
        println!("  {count:>4}  {family}");
    }
}

fn main() -> Result<(), String> {
    let files: Vec<String> = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => vec![
            "examples/cm-10701-race.jsonl".to_string(),
            "examples/v77-race.jsonl".to_string(),
        ],
        args => args.to_vec(),
    };
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").map_err(|e| e.to_string())?;
    let extras = ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    for file in &files {
        let path = repo_file(file);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{file}: {e}"))?;
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let root: Value = json::parse(line).map_err(|e| format!("{file}:{line_no}: {e}"))?;
            let Some(course_id) = root.get("courseId").and_then(Value::as_i64) else {
                println!("{file} line {}: no courseId, skipped", line_no + 1);
                continue;
            };
            let Some(course) = courses.get(&course_id.to_string()) else {
                println!("{file} line {}: course {course_id} missing from course_data.json, skipped", line_no + 1);
                continue;
            };
            println!("== {file} line {}: course {course_id}", line_no + 1);

            // Full field: compile, table usage against the caps, then the
            // strip loop until it compiles.
            let mut field = root.clone();
            let mut strip_families: BTreeMap<String, usize> = BTreeMap::new();
            let plan = loop {
                let plan = skills_compile::plan_retail_with(&field, course, &extras);
                let Some(reason) = &plan.reason else { break plan };
                let Some(id) = skill_id_in(reason) else {
                    println!("  full field refused without a skill id: {reason}");
                    break plan;
                };
                let family = refusal_family(id, reason);
                *strip_families.entry(family).or_default() += 1;
                if strip_skill(&mut field, id) == 0 {
                    println!("  strip loop stuck on {id}: {reason}");
                    break plan;
                }
            };
            let skills_total: usize = root
                .get("horses")
                .and_then(Value::as_array)
                .map(|hs| hs.iter().filter_map(|h| h.get("skills").and_then(Value::as_array)).map(|s| s.len()).sum())
                .unwrap_or(0);
            let removed: usize = strip_families.values().sum();
            println!(
                "  full field: {} skills, {removed} stripped, bucket {}, records {}, effects {}, programs {}/{} (SK_PMAX), regions {}/{} (SK_GMAX), terms {}/{} (SK_MMAX), tags {}/{} (SK_TMAX), arms {}/{} (SK_AMAX)",
                skills_total,
                plan.emax,
                plan.words[globalsim_gpu::skills_compile::SK_NREC],
                plan.words[globalsim_gpu::skills_compile::SK_NEFF],
                plan.words[globalsim_gpu::skills_compile::SK_NPROG], skills_compile::SK_PMAX,
                plan.words[globalsim_gpu::skills_compile::SK_NREG], skills_compile::SK_GMAX,
                plan.words[globalsim_gpu::skills_compile::SK_NTERM], skills_compile::SK_MMAX,
                plan.words[globalsim_gpu::skills_compile::SK_NTAG], skills_compile::SK_TMAX,
                plan.arms, skills_compile::SK_AMAX,
            );
            if !strip_families.is_empty() {
                print_histogram("  full-field strip histogram:", &strip_families);
            }

            // Solo audit over the request's skillData.
            let ids: Vec<String> = match root.get("skillData") {
                Some(Value::Object(m)) => m.keys().cloned().collect(),
                _ => {
                    println!("  no skillData, solo audit skipped");
                    continue;
                }
            };
            let mut compiled = 0usize;
            let mut families: BTreeMap<String, usize> = BTreeMap::new();
            for id in &ids {
                let plan = skills_compile::plan_retail_with(&solo_field(&root, id), course, &extras);
                match &plan.reason {
                    None => compiled += 1,
                    Some(reason) => {
                        *families.entry(refusal_family(id, reason)).or_default() += 1;
                    }
                }
            }
            println!("  solo audit: {} of {} ids compile, {} refused", compiled, ids.len(), ids.len() - compiled);
            print_histogram("  solo refusal histogram:", &families);
        }
    }
    Ok(())
}
