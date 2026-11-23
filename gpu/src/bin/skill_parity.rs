//! Skill parity sweep: does every skill the GPU compiles also run frame for
//! frame like the engine, in a race where it actually fires?
//!
//!   cargo run --release --features race --bin skill_parity -- [OPTIONS] FIELD.jsonl ...
//!
//! Each FIELD line is a background race (a real CM field). Per field:
//!
//! 1. Prepare: strip the skills the compiler refuses, then, while the field
//!    still overflows a skill table, drop the last skill of the runner that
//!    carries the most. What was dropped is printed.
//! 2. Bare check: the prepared field on the GPU against the engine, every
//!    frame bit-exact. A field that fails here is reported and not used.
//!
//! Then per skill, CPU only: add it to one runner of a field and run the
//! engine. It fired when the reply's skillDetailEvents name that runner and
//! skill. Fields and runners are tried round-robin until --per-skill firing
//! cases are found, on the field's own seed and then --extra-seeds more.
//! Every firing case then runs on the GPU and must match the engine bit for
//! bit with no fallback.
//!
//! Options:
//!   --skills FILE       skill table to test (default: the first field's
//!                       skillData); entries the fields lack are added to them
//!   --names FILE        skill names (the app's skillnames.json)
//!   --only ID,ID        test only these ids
//!   --per-skill N       firing cases to check per skill (default 3)
//!   --extra-seeds N     seeds tried past the field's own (default 2)
//!   --batch N           races per GPU call (default 192)
//!   --bisect            only for fields failing the bare check: rerun each with
//!                       one skill (and each runner's whole list) removed, and
//!                       print the removals that make it bit-exact
//!   --reduce            only for fields failing the bare check: shrink each to
//!                       the smallest skill set that still diverges
//!   --trace             every field, side by side: the first full-reply
//!                       channel whose content differs, then the engine's and
//!                       the kernel's detail and target event streams
//!
//! Each skill ends as one of: match, diverge (first divergent tick), fallback
//! (the GPU refused the race), refused (does not compile in any field), cap
//! (compiles alone but overflows the field's tables), never fired.
use globalsim_gpu::json::{self, Value};
use globalsim_gpu::skills_compile::{self, ClassicExtras};
use globalsim_gpu::{dep_cpu, full_reply, race_exec};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

fn repo_file(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn resolve(path: &str) -> std::path::PathBuf {
    let p = Path::new(path);
    if p.exists() { p.to_path_buf() } else { repo_file(path) }
}

fn read_json(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(resolve(path)).map_err(|e| format!("{path}: {e}"))?;
    json::parse(&text).map_err(|e| format!("{path}: {e}"))
}

fn text_of(v: &Value) -> String {
    let mut s = String::new();
    json::to_string_into(v, &mut s);
    s
}

/// Compact one-line form of a reply value, for the trace print.
fn json_of(v: &Value) -> String {
    let mut s = String::new();
    json::to_string_into(v, &mut s);
    if s.len() > 120 {
        s.truncate(117);
        s.push_str("...");
    }
    s
}

/// The union of the two replies' top-level keys, sorted.
fn channel_keys(engine: &Value, gpu: &Value) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for v in [engine, gpu] {
        if let Value::Object(map) = v {
            for k in map.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
    }
    keys.sort();
    keys
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

fn horses_mut(root: &mut Value) -> &mut Vec<Value> {
    let Value::Object(m) = root else { unreachable!() };
    let Some(Value::Array(h)) = m.get_mut("horses") else { unreachable!() };
    h
}

fn skills_mut(horse: &mut Value) -> &mut Vec<Value> {
    let Value::Object(hm) = horse else { unreachable!() };
    if !matches!(hm.get("skills"), Some(Value::Array(_))) {
        hm.insert("skills".into(), Value::Array(Vec::new()));
    }
    let Some(Value::Array(s)) = hm.get_mut("skills") else { unreachable!() };
    s
}

fn skill_ref(id: &str) -> Value {
    let mut one = BTreeMap::new();
    one.insert("skillId".to_string(), Value::String(id.to_string()));
    one.insert("level".to_string(), Value::Number(1.0));
    Value::Object(one)
}

fn sid(v: &Value) -> Option<&str> {
    v.get("skillId").and_then(Value::as_str)
}

fn plan_reason(root: &Value, course: &Value) -> Option<String> {
    let extras = ClassicExtras { raw_power: Vec::new(), strategy: Vec::new(), later_stamina: false };
    skills_compile::plan_retail_with(root, course, &extras).reason
}

struct Field {
    label: String,
    root: Value,
    course: Value,
    seed: i64,
    runners: usize,
}

/// Strip refused skills, then trim the heaviest runner until the tables fit.
fn prepare_field(label: &str, mut root: Value, course: &Value) -> (Value, Vec<String>) {
    let mut dropped = Vec::new();
    for _ in 0..400 {
        let Some(reason) = plan_reason(&root, course) else { break };
        if let Some(id) = skill_id_in(&reason).map(str::to_string).filter(|id| id.chars().all(|c| c.is_ascii_digit())) {
            let mut n = 0;
            for h in horses_mut(&mut root).iter_mut() {
                let s = skills_mut(h);
                let before = s.len();
                s.retain(|x| sid(x) != Some(id.as_str()));
                n += before - s.len();
            }
            if n > 0 {
                dropped.push(format!("{id} (refused: {})", reason.rsplit(": ").next().unwrap_or(&reason)));
                continue;
            }
        }
        // No strippable id: a table overflow. Trim the runner with the most skills.
        let horses = horses_mut(&mut root);
        let (i, _) = horses
            .iter_mut()
            .enumerate()
            .map(|(i, h)| (i, skills_mut(h).len()))
            .max_by_key(|&(i, n)| (n, usize::MAX - i))
            .unwrap();
        match skills_mut(&mut horses[i]).pop() {
            Some(x) => dropped.push(format!("runner {i} {} (table: {reason})", sid(&x).unwrap_or("?"))),
            None => {
                dropped.push(format!("stuck: {reason}"));
                break;
            }
        }
    }
    let _ = label;
    (root, dropped)
}

fn frames_diff(cpu: &[dep_cpu::TickFrame], gpu: &[Vec<[f64; 6]>]) -> Option<String> {
    if cpu.len() != gpu.len() {
        return Some(format!("tick count cpu {} gpu {}", cpu.len(), gpu.len()));
    }
    for (tick, (cf, gf)) in cpu.iter().zip(gpu).enumerate() {
        for (runner, (cr, gr)) in cf.iter().zip(gf).enumerate() {
            for (col, (c, g)) in cr.iter().zip(gr).enumerate() {
                if c.to_bits() != g.to_bits() {
                    return Some(format!("tick {tick} runner {runner} column {col}: cpu {c:?} gpu {g:?}"));
                }
            }
        }
    }
    None
}

/// Run each item on the thread pool, answers in order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<&mut Option<R>>> = out.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                **slots[i].lock().unwrap() = Some(r);
            });
        }
    });
    drop(slots);
    out.into_iter().map(|r| r.expect("par_map slot filled")).collect()
}

/// GPU vs engine for a list of (request, seed); None = bit-exact.
fn gpu_check(cases: &[(String, i64)], batch: usize) -> Vec<Result<(), String>> {
    // One launch over the whole set: run() coalesces races by program bucket,
    // so a case's answer no longer depends on which 48-race chunk it landed in.
    let run = match race_exec::run(cases) {
        Ok(run) => run,
        Err(e) => return cases.iter().map(|_| Err(format!("gpu call failed: {e}"))).collect(),
    };
    let cpu = par_map(cases, |(r, s)| dep_cpu::run_cpu_race(r, *s));
    let mut out = Vec::with_capacity(cases.len());
    for (i, (race, cpu)) in run.races.iter().zip(cpu).enumerate() {
        out.push(match (&race.fallback, cpu) {
            (Some(f), _) => Err(format!("fallback: {f}")),
            (_, Err(e)) => Err(format!("cpu failed: {e}")),
            (None, Ok(c)) => match frames_diff(&c.frames, &race.frames) {
                None => Ok(()),
                Some(d) => Err(d),
            },
        });
        if (i + 1) % batch.max(1) == 0 {
            eprint!(".");
        }
    }
    eprintln!();
    out
}

#[derive(Default)]
struct Opts {
    fields: Vec<String>,
    skills: Option<String>,
    names: Option<String>,
    only: Vec<String>,
    per_skill: usize,
    extra_seeds: i64,
    batch: usize,
    bisect: bool,
    reduce: bool,
    trace: bool,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts { per_skill: 3, extra_seeds: 2, batch: 192, ..Default::default() };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        let mut val = |f: &str| a.next().ok_or(format!("{f} needs a value"));
        match x.as_str() {
            "--skills" => o.skills = Some(val("--skills")?),
            "--names" => o.names = Some(val("--names")?),
            "--only" => o.only = val("--only")?.split(',').map(|s| s.trim().to_string()).collect(),
            "--per-skill" => o.per_skill = val("--per-skill")?.parse().map_err(|_| "--per-skill needs a number")?,
            "--extra-seeds" => o.extra_seeds = val("--extra-seeds")?.parse().map_err(|_| "--extra-seeds needs a number")?,
            "--batch" => o.batch = val("--batch")?.parse().map_err(|_| "--batch needs a number")?,
            "--bisect" => o.bisect = true,
            "--reduce" => o.reduce = true,
            "--trace" => o.trace = true,
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => o.fields.push(x),
        }
    }
    if o.fields.is_empty() {
        return Err("usage: skill_parity [OPTIONS] FIELD.jsonl ...".into());
    }
    Ok(o)
}

enum Outcome {
    Refused(String),
    Cap(String),
    NeverFired,
    Cases(Vec<(usize, usize, i64)>), // (field, runner, seed)
}

fn main() -> Result<(), String> {
    let o = parse_args()?;
    let t0 = std::time::Instant::now();
    let courses = globalsim_gpu::initialization::engine_json("data/course_data.json").map_err(|e| e.to_string())?;
    let names: BTreeMap<String, String> = match &o.names {
        Some(f) => read_json(f)?
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let n = v.as_str().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_str))?;
                        Some((k.clone(), n.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        None => BTreeMap::new(),
    };
    let table_override = match &o.skills {
        Some(f) => Some(read_json(f)?),
        None => None,
    };

    // Fields.
    let mut fields: Vec<Field> = Vec::new();
    for file in &o.fields {
        let text = std::fs::read_to_string(resolve(file)).map_err(|e| format!("{file}: {e}"))?;
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let mut root = json::parse(line).map_err(|e| format!("{file}:{n}: {e}"))?;
            let cid = root.get("courseId").and_then(Value::as_i64).ok_or("field without courseId")?;
            let seed = root.get("seed").and_then(Value::as_i64).unwrap_or(1);
            let course = courses.get(&cid.to_string()).ok_or(format!("course {cid} missing"))?.clone();
            let geometry = std::fs::read_to_string(repo_file(&format!("data/course_geometry/{cid}.json")))
                .map_err(|e| format!("geometry {cid}: {e}"))?;
            globalsim::register_course_geometry(cid as i32, geometry);
            if let (Some(Value::Object(extra)), Value::Object(m)) = (&table_override, &mut root) {
                if let Some(Value::Object(sd)) = m.get_mut("skillData") {
                    for (k, v) in extra {
                        sd.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
            }
            let label = format!("{}#{} (course {cid})", Path::new(file).file_name().unwrap().to_string_lossy(), n + 1);
            let (root, dropped) = prepare_field(&label, root, &course);
            println!("== field {label}: seed {seed}, {} dropped to compile", dropped.len());
            for d in &dropped {
                println!("     dropped {d}");
            }
            let runners = root.get("horses").and_then(Value::as_array).map_or(0, |h| h.len());
            fields.push(Field { label, root, course, seed, runners });
        }
    }

    // Bare check.
    let bare: Vec<(String, i64)> = fields.iter().map(|f| (text_of(&f.root), f.seed)).collect();
    let bare_ok = gpu_check(&bare, o.batch);
    let mut usable = Vec::new();
    for (i, r) in bare_ok.iter().enumerate() {
        match r {
            Ok(()) => {
                println!("  bare {}: bit-exact", fields[i].label);
                usable.push(i);
            }
            Err(e) => println!("  bare {}: FAILS ({e}), not used", fields[i].label),
        }
    }
    if o.reduce {
        // Delta debugging over the field's (runner, skill) pairs: drop chunks
        // while the GPU still diverges (a fallback does not count), down to a
        // set where removing any one pair makes it bit-exact.
        for (i, r) in bare_ok.iter().enumerate() {
            if r.is_ok() {
                continue;
            }
            let f = &fields[i];
            let mut base = f.root.clone();
            let mut keep: Vec<(usize, Value)> = Vec::new();
            for (ri, h) in horses_mut(&mut base).iter_mut().enumerate() {
                keep.extend(skills_mut(h).drain(..).map(|s| (ri, s)));
            }
            let build = |set: &[(usize, Value)]| -> (String, i64) {
                let mut r = base.clone();
                for (ri, s) in set {
                    skills_mut(&mut horses_mut(&mut r)[*ri]).push(s.clone());
                }
                (text_of(&r), f.seed)
            };
            let diverges = |e: &Result<(), String>| matches!(e, Err(m) if !m.starts_with("fallback"));
            let mut last = r.clone();
            let mut n = 2usize;
            println!("== reduce {}: {} skill(s) to start", f.label, keep.len());
            while keep.len() >= 2 {
                let size = keep.len().div_ceil(n);
                let cands: Vec<Vec<(usize, Value)>> = (0..keep.len())
                    .step_by(size)
                    .map(|s| keep.iter().enumerate().filter(|(k, _)| *k < s || *k >= s + size).map(|(_, x)| x.clone()).collect())
                    .collect();
                let res = gpu_check(&cands.iter().map(|c| build(c)).collect::<Vec<_>>(), o.batch);
                if let Some(k) = res.iter().position(diverges) {
                    keep = cands[k].clone();
                    last = res[k].clone();
                    n = (n - 1).max(2);
                    println!("  down to {} skill(s)", keep.len());
                } else if n >= keep.len() {
                    break;
                } else {
                    n = (n * 2).min(keep.len());
                }
            }
            // An empty field can still diverge: that is a mechanics bug.
            if keep.len() == 1 {
                let res = gpu_check(&[build(&[])], o.batch);
                if diverges(&res[0]) {
                    keep.clear();
                    last = res[0].clone();
                }
            }
            println!("  minimal set ({}): {}", keep.len(), last.as_ref().err().cloned().unwrap_or_default());
            for (ri, s) in &keep {
                let id = sid(s).unwrap_or("?");
                let name = names.get(id).map(|n| format!(" {n}")).unwrap_or_default();
                println!("    runner {ri}: {id}{name}");
            }
        }
        return Ok(());
    }
    if o.trace {
        // Every field, side by side: the engine's and the kernel's full-reply
        // event streams and the first reply channel whose content differs, so a
        // missing, extra or late activation shows up without a kernel trace.
        for (i, r) in bare_ok.iter().enumerate() {
            let f = &fields[i];
            let req = text_of(&f.root);
            match r {
                Ok(()) => println!("== trace {}: bit-exact", f.label),
                Err(why) => println!("== trace {}: {why}", f.label),
            }
            let cpu = dep_cpu::run_cpu_binary(&req, f.seed)?;
            let run = race_exec::run_full(&[(req.clone(), f.seed)])?;
            let race = run.races.first().ok_or("no race in the reply run")?;
            if let Some(reason) = &race.fallback {
                println!("  gpu fallback {reason}");
                continue;
            }
            let gpu = full_reply::build(&req, f.seed, race)?;
            let engine: Value = json::parse(&cpu.reply)?;
            let gpu_value: Value = json::parse(&gpu)?;
            let mut text = String::new();
            for key in channel_keys(&engine, &gpu_value) {
                let mut a = String::new();
                let mut b = String::new();
                json::to_string_into(engine.get(&key).unwrap_or(&Value::Null), &mut a);
                json::to_string_into(gpu_value.get(&key).unwrap_or(&Value::Null), &mut b);
                if a == b {
                    text.push_str(&format!("  {key}: same ({} bytes)\n", a.len()));
                } else {
                    let at = a.chars().zip(b.chars()).position(|(x, y)| x != y).unwrap_or(0);
                    let cut = |s: &str| -> String {
                        let tail: String = s.chars().skip(at).take(90).collect();
                        let head: String = s.chars().skip(at.saturating_sub(40)).take(40).collect();
                        format!("...{head}[{tail}]...")
                    };
                    text.push_str(&format!(
                        "  {key}: DIFFERS at char {at}\n    engine {}\n    gpu {}\n",
                        cut(&a),
                        cut(&b)
                    ));
                }
            }
            print!("{text}");
            for (tag, v) in [("engine", &engine), ("gpu", &gpu_value)] {
                let key = |k: &str| v.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
                for field in ["skillDetailEvents", "skillTargetEvents"] {
                    let arr = key(field);
                    println!("  {tag} {field}: {} entries", arr.len());
                    for e in arr.iter().take(120) {
                        let h = e.get("horseIndex").and_then(Value::as_i64).unwrap_or(-1);
                        let id = e.get("skillId").and_then(Value::as_i64).unwrap_or(-1);
                        let d = e.get("detailIndex").and_then(Value::as_i64).unwrap_or(-1);
                        let t = e.get("t").and_then(Value::as_f64).unwrap_or(f64::NAN);
                        let name = names.get(&id.to_string()).map(|n| format!(" {n}")).unwrap_or_default();
                        print!("    horse {h} {id}{name} detail {d} t {t}");
                        if let Some(list) = e.get("abilityTargets") {
                            print!(" targets {}", json_of(list));
                        }
                        println!();
                    }
                }
            }
        }
        return Ok(());
    }
    if o.bisect {
        // A failing bare field: rerun it once per (runner, skill) with that one
        // skill removed, and again with each runner's whole list removed. A
        // removal that makes it bit-exact names a suspect.
        for (i, r) in bare_ok.iter().enumerate() {
            if r.is_ok() {
                continue;
            }
            let f = &fields[i];
            let mut variants: Vec<(String, (String, i64))> = Vec::new();
            for runner in 0..f.runners {
                let mut whole = f.root.clone();
                let list: Vec<String> = skills_mut(&mut horses_mut(&mut whole)[runner]).drain(..).filter_map(|x| sid(&x).map(str::to_string)).collect();
                variants.push((format!("runner {runner} all {} skills", list.len()), (text_of(&whole), f.seed)));
                for id in &list {
                    let mut one = f.root.clone();
                    skills_mut(&mut horses_mut(&mut one)[runner]).retain(|x| sid(x) != Some(id.as_str()));
                    let name = names.get(id).map(|n| format!(" {n}")).unwrap_or_default();
                    variants.push((format!("runner {runner} {id}{name}"), (text_of(&one), f.seed)));
                }
            }
            let reqs: Vec<(String, i64)> = variants.iter().map(|(_, r)| r.clone()).collect();
            let res = gpu_check(&reqs, o.batch);
            println!("== bisect {}: {} variants", f.label, variants.len());
            for ((what, _), r) in variants.iter().zip(&res) {
                match r {
                    Ok(()) => println!("  fixed by removing {what}"),
                    Err(e) if what.contains(" all ") => println!("  still fails without {what}: {e}"),
                    Err(_) => {}
                }
            }
        }
        return Ok(());
    }
    if usable.is_empty() {
        return Err("no field passes the bare check".into());
    }

    // Skills to test.
    let table = table_override.clone().or_else(|| fields[0].root.get("skillData").cloned()).unwrap_or(Value::Null);
    let mut ids: Vec<String> = table.as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default();
    if !o.only.is_empty() {
        ids.retain(|i| o.only.contains(i));
    }
    println!("testing {} skills on {} field(s), {:.0} s in", ids.len(), usable.len(), t0.elapsed().as_secs_f64());

    let with_skill = |f: &Field, runner: usize, id: &str| -> Value {
        let mut r = f.root.clone();
        let s = skills_mut(&mut horses_mut(&mut r)[runner]);
        if !s.iter().any(|x| sid(x) == Some(id)) {
            s.push(skill_ref(id));
        }
        r
    };
    let done = AtomicUsize::new(0);
    let outcomes: Vec<Outcome> = par_map(&ids, |id| {
        let mut cases = Vec::new();
        let mut refusals: Vec<String> = Vec::new();
        let mut caps: Vec<String> = Vec::new();
        let mut compiled_somewhere = false;
        'search: for extra in 0..=o.extra_seeds {
            let max_r = usable.iter().map(|&f| fields[f].runners).max().unwrap_or(0);
            for runner in 0..max_r {
                for &fi in &usable {
                    let f = &fields[fi];
                    if runner >= f.runners {
                        continue;
                    }
                    let req = with_skill(f, runner, id);
                    if extra == 0 {
                        if let Some(reason) = plan_reason(&req, &f.course) {
                            if skill_id_in(&reason) == Some(id.as_str()) {
                                refusals.push(reason);
                            } else {
                                caps.push(format!("{}: {reason}", f.label));
                            }
                            continue;
                        }
                        compiled_somewhere = true;
                    } else if plan_reason(&req, &f.course).is_some() {
                        continue;
                    }
                    let seed = f.seed + extra;
                    let Ok(cpu) = dep_cpu::run_cpu_race(&text_of(&req), seed) else { continue };
                    let want: i64 = id.parse().unwrap_or(-1);
                    if cpu.detail_events.iter().any(|&(h, s, _, _)| h == runner && s == want) {
                        cases.push((fi, runner, seed));
                        if cases.len() >= o.per_skill {
                            break 'search;
                        }
                    }
                }
            }
            if !compiled_somewhere {
                break;
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 50 == 0 {
            eprintln!("  searched {n}/{}", ids.len());
        }
        if !cases.is_empty() {
            Outcome::Cases(cases)
        } else if !compiled_somewhere && !refusals.is_empty() {
            Outcome::Refused(refusals[0].rsplit(": ").next().unwrap_or(&refusals[0]).to_string())
        } else if !compiled_somewhere {
            Outcome::Cap(caps.first().cloned().unwrap_or_default())
        } else {
            Outcome::NeverFired
        }
    });
    println!("search done, {:.0} s in", t0.elapsed().as_secs_f64());

    // GPU check of every firing case.
    let mut flat: Vec<(usize, (String, i64))> = Vec::new();
    for (k, out) in outcomes.iter().enumerate() {
        if let Outcome::Cases(cs) = out {
            for &(fi, runner, seed) in cs {
                flat.push((k, (text_of(&with_skill(&fields[fi], runner, &ids[k])), seed)));
            }
        }
    }
    let reqs: Vec<(String, i64)> = flat.iter().map(|(_, r)| r.clone()).collect();
    let results = gpu_check(&reqs, o.batch);
    let mut per_skill: BTreeMap<usize, Vec<Result<(), String>>> = BTreeMap::new();
    for ((k, _), r) in flat.iter().zip(results) {
        per_skill.entry(*k).or_default().push(r);
    }

    // Report.
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    let mut lines: Vec<(u8, String)> = Vec::new();
    for (k, id) in ids.iter().enumerate() {
        let name = names.get(id).map(|n| format!(" {n}")).unwrap_or_default();
        let (rank, kind, detail) = match &outcomes[k] {
            Outcome::Refused(r) => (2, "refused", r.clone()),
            Outcome::Cap(r) => (1, "cap", r.clone()),
            Outcome::NeverFired => (3, "never fired", String::new()),
            Outcome::Cases(cs) => {
                let rs = &per_skill[&k];
                let bad: Vec<String> = cs
                    .iter()
                    .zip(rs)
                    .filter_map(|(&(fi, runner, seed), r)| {
                        r.as_ref().err().map(|e| format!("{} runner {runner} seed {seed}: {e}", fields[fi].label))
                    })
                    .collect();
                if bad.is_empty() {
                    (9, "match", format!("{} case(s)", cs.len()))
                } else if bad.iter().all(|b| b.contains("fallback:")) {
                    (0, "fallback", format!("{}/{} case(s); {}", bad.len(), cs.len(), bad[0]))
                } else {
                    (0, "diverge", format!("{}/{} case(s); {}", bad.len(), cs.len(), bad[0]))
                }
            }
        };
        *tally.entry(kind).or_default() += 1;
        if kind != "match" {
            lines.push((rank, format!("  {kind:<11} {id}{name}{}", if detail.is_empty() { String::new() } else { format!(": {detail}") })));
        }
    }
    println!("== summary over {} skills ({:.0} s)", ids.len(), t0.elapsed().as_secs_f64());
    for (k, n) in &tally {
        println!("  {n:>4}  {k}");
    }
    lines.sort();
    for (_, l) in lines {
        println!("{l}");
    }
    Ok(())
}
