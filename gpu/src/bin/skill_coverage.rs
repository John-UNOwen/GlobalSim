//! Coverage audit (handoff E1): what the retail compiler still refuses.
//! For each request line the bin compiles the full field, then every skill
//! of the request's skillData alone on runner 0 (host-side only, no GPU),
//! and prints a histogram of refusal families plus the full-field table
//! usage against the SK_* caps. Usage:
//!
//!   cargo run --release --features race --bin skill_coverage [OPTIONS] [REQUEST.jsonl ...]
//!
//! With no request files it audits examples/cm-10701-race.jsonl and
//! examples/v77-race.jsonl.
//!
//! The compiler folds conditions against the course and runner 0's strategy,
//! so a skill that folds away on one course or style can hide a refusal. The
//! options sweep both:
//!
//!   --skills FILE      solo-audit FILE (an engine skill table, such as the
//!                      app's globalsim/skills.merged.json) instead of the
//!                      request's skillData; the full field keeps its own
//!   --all-courses      audit every course in data/course_data.json, each
//!                      request line as the base
//!   --styles A,B,...   audit each runner-0 strategy (Nige, Senkou, Sasi,
//!                      Oikomi, Oonige); "all" is all five
//!   --names FILE       skill names (the app's skillnames.json) for the report
//!   --by-id            print each refused id with where it was refused
//!
//! A sweep (--all-courses or --styles) folds the per-combination output into
//! one summary per request line and always prints the per-id report. The
//! solo compiles run on every core.
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

const ALL_STYLES: [&str; 5] = ["Nige", "Senkou", "Sasi", "Oikomi", "Oonige"];

struct Options {
    files: Vec<String>,
    skills: Option<String>,
    all_courses: bool,
    styles: Vec<String>,
    names: Option<String>,
    by_id: bool,
}

fn parse_args() -> Result<Options, String> {
    let mut o = Options { files: Vec::new(), skills: None, all_courses: false, styles: Vec::new(), names: None, by_id: false };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--skills" => o.skills = Some(args.next().ok_or("--skills needs a file")?),
            "--names" => o.names = Some(args.next().ok_or("--names needs a file")?),
            "--all-courses" => o.all_courses = true,
            "--by-id" => o.by_id = true,
            "--styles" => {
                let v = args.next().ok_or("--styles needs a list")?;
                o.styles = if v == "all" {
                    ALL_STYLES.iter().map(|s| s.to_string()).collect()
                } else {
                    v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
                };
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => o.files.push(a),
        }
    }
    if o.files.is_empty() {
        o.files = vec!["examples/cm-10701-race.jsonl".to_string(), "examples/v77-race.jsonl".to_string()];
    }
    Ok(o)
}

/// A path as given if it exists, otherwise relative to the repo root.
fn resolve(path: &str) -> std::path::PathBuf {
    let p = Path::new(path);
    if p.exists() { p.to_path_buf() } else { repo_file(path) }
}

fn read_json(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(resolve(path)).map_err(|e| format!("{path}: {e}"))?;
    json::parse(&text).map_err(|e| format!("{path}: {e}"))
}

fn set_key(root: &mut Value, key: &str, v: Value) {
    if let Value::Object(m) = root {
        m.insert(key.to_string(), v);
    }
}

fn set_strategy(root: &mut Value, style: &str) {
    if let Value::Object(m) = root {
        if let Some(Value::Array(hs)) = m.get_mut("horses") {
            if let Some(Value::Object(h0)) = hs.get_mut(0) {
                h0.insert("strategy".to_string(), Value::String(style.to_string()));
            }
        }
    }
}

/// The request with skillData removed and every runner's skills cleared, so
/// a solo compile only adds one skillData entry and runner 0's one skill.
fn solo_base(root: &Value) -> Value {
    let mut value = root.clone();
    if let Value::Object(map) = &mut value {
        map.remove("skillData");
        if let Some(Value::Array(horses)) = map.get_mut("horses") {
            for h in horses.iter_mut() {
                if let Value::Object(hm) = h {
                    hm.insert("skills".to_string(), Value::Array(Vec::new()));
                }
            }
        }
    }
    value
}

/// Runner 0 alone with the one skill at level 1. skillData is cut to that
/// skill's entry: with the other runners empty the compiler reads no other
/// entry, and the construction hooks need none.
fn solo_request(base: &Value, id: &str, entry: &Value) -> Value {
    let mut value = base.clone();
    let mut data = BTreeMap::new();
    data.insert(id.to_string(), entry.clone());
    set_key(&mut value, "skillData", Value::Object(data));
    if let Value::Object(map) = &mut value {
        if let Some(Value::Array(horses)) = map.get_mut("horses") {
            if let Some(Value::Object(hm)) = horses.get_mut(0) {
                let mut one = BTreeMap::new();
                one.insert("skillId".to_string(), Value::String(id.to_string()));
                one.insert("level".to_string(), Value::Number(1.0));
                hm.insert("skills".to_string(), Value::Array(vec![Value::Object(one)]));
            }
        }
    }
    value
}

/// Solo-compile every id across all cores; the refusals come back in id order.
fn solo_audit(base: &Value, ids: &[(String, Value)], course: &Value, extras: &ClassicExtras) -> Vec<Option<String>> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(ids.len().max(1));
    let chunk = ids.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = ids
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|(id, entry)| skills_compile::plan_retail_with(&solo_request(base, id, entry), course, extras).reason)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("solo audit thread panicked")).collect()
    })
}

struct FieldStats {
    removed: usize,
    tags: u32,
    usage: String,
    strip_families: BTreeMap<String, usize>,
    note: Option<String>,
}

/// Full field: compile, then strip refused skills until it compiles.
fn full_field(root: &Value, course: &Value, extras: &ClassicExtras) -> FieldStats {
    use skills_compile::*;
    let mut field = root.clone();
    let mut strip_families: BTreeMap<String, usize> = BTreeMap::new();
    let mut note = None;
    let plan = loop {
        let plan = plan_retail_with(&field, course, extras);
        let Some(reason) = &plan.reason else { break plan };
        let Some(id) = skill_id_in(reason) else {
            note = Some(format!("full field refused without a skill id: {reason}"));
            break plan;
        };
        *strip_families.entry(refusal_family(id, reason)).or_default() += 1;
        if strip_skill(&mut field, id) == 0 {
            note = Some(format!("strip loop stuck on {id}: {reason}"));
            break plan;
        }
    };
    let skills: usize = root
        .get("horses")
        .and_then(Value::as_array)
        .map(|hs| hs.iter().filter_map(|h| h.get("skills").and_then(Value::as_array)).map(|s| s.len()).sum())
        .unwrap_or(0);
    let removed = strip_families.values().sum();
    let w = &plan.words;
    let usage = format!(
        "{skills} skills, {removed} stripped, bucket {}/{}, records {}, effects {}, programs {}/{} (SK_PMAX), regions {}/{} (SK_GMAX), terms {}/{} (SK_MMAX), tags {} bytes used (cap {}), arms {}/{} (SK_AMAX)",
        plan.emax,
        plan.tmax,
        w[SK_NREC],
        w[SK_NEFF],
        w[SK_NPROG],
        SK_PMAX,
        w[SK_NREG],
        SK_GMAX,
        w[SK_NTERM],
        SK_MMAX,
        plan.tags_used,
        plan.tmax,
        plan.arms,
        SK_AMAX,
    );
    FieldStats { removed, tags: w[SK_NTAG], usage, strip_families, note }
}

fn load_names(path: &str) -> Result<BTreeMap<String, String>, String> {
    let v = read_json(path)?;
    let Some(m) = v.as_object() else { return Ok(BTreeMap::new()) };
    Ok(m.iter()
        .filter_map(|(k, v)| {
            let n = v.as_str().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_str))?;
            Some((k.clone(), n.to_string()))
        })
        .collect())
}

fn main() -> Result<(), String> {
    let o = parse_args()?;
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").map_err(|e| e.to_string())?;
    let extras = ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    let skills_override = match &o.skills {
        Some(f) => Some(read_json(f)?),
        None => None,
    };
    let names = match &o.names {
        Some(f) => load_names(f)?,
        None => BTreeMap::new(),
    };
    let sweep = o.all_courses || !o.styles.is_empty();
    let styles: Vec<Option<&str>> =
        if o.styles.is_empty() { vec![None] } else { o.styles.iter().map(|s| Some(s.as_str())).collect() };

    for file in &o.files {
        let text = std::fs::read_to_string(resolve(file)).map_err(|e| format!("{file}: {e}"))?;
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let root: Value = json::parse(line).map_err(|e| format!("{file}:{line_no}: {e}"))?;
            let Some(own_course) = root.get("courseId").and_then(Value::as_i64) else {
                println!("{file} line {}: no courseId, skipped", line_no + 1);
                continue;
            };
            let course_ids: Vec<String> = if o.all_courses {
                courses.as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default()
            } else {
                vec![own_course.to_string()]
            };
            let audited = skills_override.as_ref().or_else(|| root.get("skillData"));
            let ids: Vec<(String, Value)> = match audited {
                Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                _ => Vec::new(),
            };
            if sweep {
                println!(
                    "== {file} line {}: sweep over {} course(s) x {} style(s), {} skills",
                    line_no + 1,
                    course_ids.len(),
                    styles.len(),
                    ids.len()
                );
            }

            // id -> family -> the combinations it was refused on.
            let mut refused: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
            let mut combos = 0usize;
            let mut peak: Option<(String, FieldStats)> = None;
            let mut strips: Vec<(String, FieldStats)> = Vec::new();
            for cid in &course_ids {
                let Some(course) = courses.get(cid) else {
                    println!("{file} line {}: course {cid} missing from course_data.json, skipped", line_no + 1);
                    continue;
                };
                let mut croot = root.clone();
                set_key(&mut croot, "courseId", Value::Number(cid.parse::<f64>().unwrap_or(0.0)));
                for style in &styles {
                    combos += 1;
                    let label = match style {
                        Some(s) => format!("{cid}/{s}"),
                        None => cid.clone(),
                    };
                    let mut sroot = croot.clone();
                    if let Some(s) = style {
                        set_strategy(&mut sroot, s);
                    }
                    let field = full_field(&sroot, course, &extras);
                    if !sweep {
                        println!("== {file} line {}: course {cid}", line_no + 1);
                        if let Some(n) = &field.note {
                            println!("  {n}");
                        }
                        println!("  full field: {}", field.usage);
                        if !field.strip_families.is_empty() {
                            print_histogram("  full-field strip histogram:", &field.strip_families);
                        }
                    }

                    let mut here = 0usize;
                    if !ids.is_empty() {
                        let base = solo_base(&sroot);
                        for ((id, _), reason) in ids.iter().zip(solo_audit(&base, &ids, course, &extras)) {
                            if let Some(r) = reason {
                                here += 1;
                                refused.entry(id.clone()).or_default().entry(refusal_family(id, &r)).or_default().push(label.clone());
                            }
                        }
                    }
                    if !sweep && !ids.is_empty() {
                        println!("  solo audit: {} of {} ids compile, {here} refused", ids.len() - here, ids.len());
                    }

                    if field.removed > 0 || field.note.is_some() {
                        strips.push((label.clone(), FieldStats { usage: String::new(), strip_families: field.strip_families.clone(), note: field.note.clone(), ..field }));
                    } else if peak.as_ref().map_or(true, |(_, p)| field.tags > p.tags) {
                        peak = Some((label, field));
                    }
                }
            }
            if ids.is_empty() {
                println!("  no skillData, solo audit skipped");
                continue;
            }

            let mut families: BTreeMap<String, usize> = BTreeMap::new();
            for f in refused.values() {
                for fam in f.keys() {
                    *families.entry(fam.clone()).or_default() += 1;
                }
            }
            if sweep {
                if let Some((label, f)) = &peak {
                    println!("  full field, peak tag use at {label}: {}", f.usage);
                }
                if strips.is_empty() {
                    println!("  full field compiled unstripped on all {combos} combination(s)");
                } else {
                    println!("  full field needed stripping on {} of {combos} combination(s):", strips.len());
                    for (label, f) in strips.iter().take(20) {
                        let fams: Vec<String> = f.strip_families.iter().map(|(k, v)| format!("{v}x {k}")).collect();
                        let note = f.note.as_ref().map(|n| format!(" ({n})")).unwrap_or_default();
                        println!("    {label}: {}{note}", fams.join("; "));
                    }
                }
                let everywhere = refused.values().filter(|f| f.values().any(|v| v.len() >= combos)).count();
                println!(
                    "  solo audit: {} of {} ids compile on every combination, {} refused somewhere ({everywhere} on all {combos})",
                    ids.len() - refused.len(),
                    ids.len(),
                    refused.len()
                );
                print_histogram("  solo refusal histogram (ids per family):", &families);
            } else {
                print_histogram("  solo refusal histogram:", &families);
            }
            if o.by_id || sweep {
                println!("  refused ids:");
                for (id, fams) in &refused {
                    let name = names.get(id).map(|n| format!(" {n}")).unwrap_or_default();
                    for (fam, at) in fams {
                        let at = if at.len() >= combos {
                            "everywhere".to_string()
                        } else if at.len() <= 6 {
                            format!("on {}", at.join(", "))
                        } else {
                            format!("on {} of {combos}, e.g. {}", at.len(), at[..4].join(", "))
                        };
                        println!("    {id}{name}: {fam} ({at})");
                    }
                }
            }
        }
    }
    Ok(())
}
