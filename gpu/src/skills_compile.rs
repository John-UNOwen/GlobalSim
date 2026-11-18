//! Host-side compiler lowering one race's skills into the kernel's numeric
//! instruction table (see gpu/SKILLS_DESIGN.md). An unsupported construct
//! does not throw: it surfaces as a fallback reason that flags the whole race.
//!
//! The engine decides activation two ways. Static region algebra over the
//! condition tree produces the pending-skill window list checked by the scan
//! gate, and only the remaining live predicates (hp_per, is_overtake,
//! is_temptation, order, order_rate, the last-straight entry latch) run per
//! tick. This compiler reproduces the region chain term by term in parse
//! order, with the engine's intersect and rmap filtering.

use crate::json::Value;

pub const SK_STRIDE: usize = 4480;

pub const SK_NREC: usize = 0;
pub const SK_NPROG: usize = 1;
pub const SK_NREG: usize = 2;
pub const SK_NEFF: usize = 3;
pub const SK_NTAG: usize = 4;
pub const SK_NSTR: usize = 5;
pub const SK_NCOR: usize = 6;
pub const SK_NSLOPE: usize = 7;

pub const SK_STRAIGHTS: usize = 16;
pub const SK_CORNERS: usize = 40;
pub const SK_SLOPES: usize = 64;
pub const SK_RECORDS: usize = 112;
pub const SK_EFFECTS: usize = 752;
pub const SK_PROGRAMS: usize = 1232;
pub const SK_REGIONS: usize = 2512;
pub const SK_TAGS: usize = 3536;
pub const SK_TERMS: usize = 4048;

pub const SK_RSTRIDE: usize = 20;
pub const SK_ESTRIDE: usize = 10;
pub const SK_PSTRIDE: usize = 4;
pub const SK_GSTRIDE: usize = 4;
pub const SK_MSTRIDE: usize = 8;

pub const SK_RMAX: usize = 32;
pub const SK_EMAX: usize = 48;
pub const SK_PMAX: usize = 320;
pub const SK_GMAX: usize = 256;
pub const SK_TMAX: usize = 512;
pub const SK_MMAX: usize = 48;
pub const SK_STRMAX: usize = 12;
pub const SK_CORMAX: usize = 12;
pub const SK_SLOPEMAX: usize = 16;
pub const SK_REGMAX: usize = 16;

pub const SK_FLAG_RETAIN: u32 = 1;
pub const SK_FLAG_WISDOM: u32 = 2;
pub const SK_FLAG_SAMPLED: u32 = 4;
pub const SK_FLAG_DEAD: u32 = 8;

pub const SK_TERM_ALL_CORNER: u32 = 0;
pub const SK_TERM_STRAIGHT: u32 = 1;
pub const SK_TERM_DOWN_SLOPE: u32 = 2;
pub const SK_TERM_UP_SLOPE: u32 = 3;

pub const OP_TRUE: u32 = 0;
pub const OP_AND: u32 = 1;
pub const OP_OR: u32 = 2;
pub const OP_FEAT_I: u32 = 4;

pub const FEAT_ORDER: u32 = 1;
pub const FEAT_ORDER_RATE: u32 = 2;
pub const FEAT_HP_PER: u32 = 3;
pub const FEAT_IS_OVERTAKE: u32 = 4;
pub const FEAT_IS_TEMPTATION: u32 = 5;
pub const FEAT_LS_ENTRY: u32 = 6;

/// One race's compiled table. `reason` is Some when the race must run on the
/// CPU, naming the first construct the kernel cannot evaluate.
pub struct SkillsPlan {
    pub words: Vec<u32>,
    pub reason: Option<String>,
}

pub fn plan(root: &Value, course: &Value) -> SkillsPlan {
    let mut c = Compiler {
        words: vec![0; SK_STRIDE],
        tags: Vec::new(),
        reason: None,
        nrec: 0,
        nprog: 0,
        nreg: 0,
        neff: 0,
        nterm: 0,
    };
    let skill_data = match root.get("skillData") {
        Some(v) => v,
        None => return c.finish(),
    };
    let horses = match root.get("horses") {
        Some(v) => v,
        None => return c.finish(),
    };
    c.pack_course(course);
    if c.reason.is_some() {
        return c.finish();
    }
    let entries: Vec<&Value> = horses.as_array().map(|a| a.iter().collect()).unwrap_or_default();
    for (r, horse) in entries.iter().enumerate() {
        let skills = match horse.get("skills") {
            Some(v) => v,
            None => continue,
        };
        let list = match skills.as_array() {
            Some(a) => a,
            None => continue,
        };
        // The engine's skill_enabled is keyed by skill id, so a runner that
        // lists one id twice maps enablement differently than independent
        // kernel records would. Flag the race instead of guessing.
        let mut seen_ids: Vec<i64> = Vec::new();
        for entry in list {
            let Some(sid) = entry.get("skillId").and_then(Value::as_i64) else {
                c.unsupported("skill entry needs a numeric skillId".to_string());
                return c.finish();
            };
            if seen_ids.contains(&sid) {
                c.unsupported(format!("skill {sid} appears twice for one runner"));
                return c.finish();
            }
            seen_ids.push(sid);
            c.pack_skill(entry, r, skill_data, course);
            if c.reason.is_some() {
                return c.finish();
            }
        }
    }
    c.finish()
}

struct Compiler {
    words: Vec<u32>,
    tags: Vec<u8>,
    reason: Option<String>,
    nrec: usize,
    nprog: usize,
    nreg: usize,
    neff: usize,
    nterm: usize,
}

fn f64_words(v: f64) -> [u32; 2] {
    let b = v.to_bits();
    [b as u32, (b >> 32) as u32]
}

/// The engine's Region half-open interval. Intersect collapses the empty
/// result to (-1,-1), and the rmap helpers drop entries with start <= -1.
fn intersect(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let start = a.0.max(b.0);
    let end = a.1.min(b.1);
    if end <= start {
        (-1.0, -1.0)
    } else {
        (start, end)
    }
}

fn rmap_multi(
    regions: &[(f64, f64)],
    f: impl Fn(&(f64, f64)) -> Vec<(f64, f64)>,
) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for r in regions {
        for n in f(r) {
            if n.0 > -1.0 {
                out.push(n);
            }
        }
    }
    out
}

fn phase_start(distance: f64, phase: i64) -> f64 {
    match phase {
        0 => 0.0,
        1 => distance * 1.0 / 6.0,
        2 => distance * 2.0 / 3.0,
        3 => distance * 5.0 / 6.0,
        _ => f64::NAN,
    }
}

fn phase_end(distance: f64, phase: i64) -> f64 {
    match phase {
        0 => distance * 1.0 / 6.0,
        1 => distance * 2.0 / 3.0,
        2 => distance * 5.0 / 6.0,
        3 => distance,
        _ => f64::NAN,
    }
}

/// PHASE_FUDGE_SKILLS in activation_conditions.rs, the ids whose phase==
/// region end stretches by ten metres.
const PHASE_FUDGE_SKILLS: [&str; 9] = [
    "100591", "900591", "110261", "910261", "110191", "910191", "120451", "920451", "101502121",
];

/// Course geometry the region transforms read, in the engine's raw JSON
/// order. The engine reads corners, straights and slopes unsorted and
/// errors on unsorted lists in the filters that require order.
struct Geometry {
    distance: f64,
    straights: Vec<(f64, f64)>,
    corners: Vec<(f64, f64)>,
    slopes: Vec<(f64, f64, f64)>,
}

fn sorted_by_start(items: &[(f64, f64)]) -> bool {
    let mut last = -1.0;
    for it in items {
        if it.0 <= last {
            return false;
        }
        last = it.0;
    }
    true
}

impl Geometry {
    fn new(course: &Value) -> Geometry {
        let empty: Vec<Value> = Vec::new();
        let distance = course.get("distance").and_then(Value::as_f64).unwrap_or(0.);
        let read_pairs = |key: &str, end_key: &str| -> Vec<(f64, f64)> {
            course
                .get(key)
                .and_then(Value::as_array)
                .unwrap_or(&empty)
                .iter()
                .map(|s| {
                    let start = s.get("start").and_then(Value::as_f64).unwrap_or(0.);
                    let end = if end_key.is_empty() {
                        s.get("end").and_then(Value::as_f64).unwrap_or(0.)
                    } else {
                        start + s.get(end_key).and_then(Value::as_f64).unwrap_or(0.)
                    };
                    (start, end)
                })
                .collect()
        };
        let straights = read_pairs("straights", "");
        let corners = read_pairs("corners", "length");
        let slopes = course
            .get("slopes")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(|s| {
                let start = s.get("start").and_then(Value::as_f64).unwrap_or(0.);
                (
                    start,
                    start + s.get("length").and_then(Value::as_f64).unwrap_or(0.),
                    s.get("slope").and_then(Value::as_f64).unwrap_or(0.),
                )
            })
            .collect();
        Geometry { distance, straights, corners, slopes }
    }
}

impl Compiler {
    fn unsupported(&mut self, why: impl Into<String>) {
        if self.reason.is_none() {
            self.reason = Some(why.into());
        }
    }

    fn finish(mut self) -> SkillsPlan {
        self.words[SK_NREC] = self.nrec as u32;
        self.words[SK_NPROG] = self.nprog as u32;
        self.words[SK_NREG] = self.nreg as u32;
        self.words[SK_NEFF] = self.neff as u32;
        self.words[SK_NTAG] = self.tags.len().min(SK_TMAX) as u32;
        if self.tags.len() > SK_TMAX || self.nreg > SK_GMAX {
            if self.tags.len() > SK_TMAX {
                self.unsupported("skill table overflow: too many tag bytes");
            } else {
                self.unsupported("skill table overflow: too many regions");
            }
            self.words[SK_NREC] = 0;
        }
        // Four tag bytes per word, little endian, so the kernel can view the
        // pool as a byte array for r128_substream_hash.
        for (i, chunk) in self.tags.chunks(4).take(SK_TMAX / 4).enumerate() {
            let mut word = 0u32;
            for (j, byte) in chunk.iter().enumerate() {
                word |= (*byte as u32) << (8 * j);
            }
            self.words[SK_TAGS + i] = word;
        }
        SkillsPlan { words: self.words, reason: self.reason }
    }

    fn tag(&mut self, text: &str) -> (usize, usize) {
        let at = self.tags.len();
        self.tags.extend(text.bytes());
        (at, text.len())
    }

    fn pack_course(&mut self, course: &Value) {
        let g = Geometry::new(course);
        if g.straights.len() > SK_STRMAX {
            self.unsupported("skill table overflow: too many straights");
            return;
        }
        self.words[SK_NSTR] = g.straights.len() as u32;
        for (i, (start, end)) in g.straights.iter().enumerate() {
            self.words[SK_STRAIGHTS + 2 * i] = (*start as f32).to_bits();
            self.words[SK_STRAIGHTS + 2 * i + 1] = (*end as f32).to_bits();
        }
        if g.corners.len() > SK_CORMAX {
            self.unsupported("skill table overflow: too many corners");
            return;
        }
        self.words[SK_NCOR] = g.corners.len() as u32;
        for (i, (start, end)) in g.corners.iter().enumerate() {
            self.words[SK_CORNERS + 2 * i] = (*start as f32).to_bits();
            self.words[SK_CORNERS + 2 * i + 1] = (*end as f32).to_bits();
        }
        if g.slopes.len() > SK_SLOPEMAX {
            self.unsupported("skill table overflow: too many slopes");
            return;
        }
        self.words[SK_NSLOPE] = g.slopes.len() as u32;
        for (i, (start, end, value)) in g.slopes.iter().enumerate() {
            self.words[SK_SLOPES + 3 * i] = (*start as f32).to_bits();
            self.words[SK_SLOPES + 3 * i + 1] = (*end as f32).to_bits();
            self.words[SK_SLOPES + 3 * i + 2] = (*value as f32).to_bits();
        }
    }

    fn pack_skill(&mut self, entry: &Value, runner: usize, skill_data: &Value, course: &Value) {
        let sid = entry.get("skillId").and_then(Value::as_i64).unwrap_or(0) as u32;
        let level = entry.get("level").and_then(Value::as_i64).unwrap_or(1);
        let def = match skill_data.get(&sid.to_string()) {
            Some(d) => d,
            None => {
                self.unsupported(format!("skillData has no entry for skill {sid}"));
                return;
            }
        };
        let alts = match def.get("alternatives").and_then(Value::as_array) {
            Some(a) if a.len() == 1 => a,
            _ => {
                self.unsupported(format!("skill {sid} has other than one alternative"));
                return;
            }
        };
        let alt = &alts[0];
        if !alt.get("precondition").and_then(Value::as_str).unwrap_or("").is_empty() {
            self.unsupported(format!("skill {sid} has a precondition column"));
            return;
        }
        let rarity = match def.get("rarity").and_then(Value::as_i64).unwrap_or(0) {
            v @ (1 | 2 | 6) => v as u32,
            3..=5 => 3,
            other => {
                self.unsupported(format!("skill {sid} rarity {other} is not ported"));
                return;
            }
        };
        let wisdom = def.get("wisdomCheck").and_then(Value::as_i64).unwrap_or(0) == 1;
        let priority = def.get("priority").and_then(Value::as_i64).unwrap_or(0);
        if !(0..=1).contains(&priority) {
            self.unsupported(format!("skill {sid} priority {priority} is not ported"));
            return;
        }
        if self.nrec >= SK_RMAX {
            self.unsupported("skill table overflow: too many records");
            return;
        }
        // The engine keeps a primary white/gold/evolution record alive after
        // its trigger ends, which the kernel mirrors as a retained record.
        let retain = matches!(rarity, 1 | 2 | 6);

        let condition = alt.get("condition").and_then(Value::as_str).unwrap_or("");
        let terms = match parse_terms(condition) {
            Ok(t) => t,
            Err(why) => {
                self.unsupported(format!("skill {sid} condition '{condition}': {why}"));
                return;
            }
        };
        let g = Geometry::new(course);
        let leaves: Vec<Leaf> = match terms.iter().map(|t| classify(t, sid, &g)).collect() {
            Ok(l) => l,
            Err(why) => {
                self.unsupported(format!("skill {sid} condition '{condition}': {why}"));
                return;
            }
        };
        let sampled_count = leaves.iter().filter(|l| matches!(l, Leaf::Sampled)).count();
        if sampled_count > 1 {
            self.unsupported(format!("skill {sid} has more than one random term"));
            return;
        }
        if terms.iter().any(|t| t.or) {
            let or_ok = leaves
                .iter()
                .all(|l| matches!(l, Leaf::Dynamic { .. } | Leaf::True));
            if !or_ok {
                self.unsupported(format!(
                    "skill {sid} or-condition mixing region or random terms is not ported"
                ));
                return;
            }
        }

        // Region chain, in parse order: the whole course narrowed by each
        // static term. A random term splits the chain, and the kernel chains
        // the sampled windows at their engine position.
        let mut regions: Vec<(f64, f64)> = vec![(0.0, g.distance)];
        let mut pre_regions: Vec<(f64, f64)> = Vec::new();
        let mut dead = false;
        for (t, leaf) in terms.iter().zip(leaves.iter()) {
            let next = match leaf {
                Leaf::True | Leaf::Dynamic { .. } => regions.clone(),
                Leaf::False => Vec::new(),
                Leaf::Sampled => {
                    pre_regions = regions.clone();
                    regions.clone()
                }
                Leaf::Region(list) | Leaf::LsEntry(list) => {
                    if !sorted_by_start(list) && list.len() > 1 {
                        self.unsupported(format!(
                            "skill {sid} condition '{}' needs a sorted course region list",
                            t.name
                        ));
                        return;
                    }
                    rmap_multi(&regions, |r| list.iter().map(|b| intersect(*r, *b)).collect())
                }
            };
            if next.len() > SK_REGMAX {
                self.unsupported(format!("skill {sid} condition exceeds {SK_REGMAX} regions"));
                return;
            }
            regions = next;
            if regions.is_empty() {
                dead = true;
                break;
            }
        }
        let sampled = sampled_count == 1;
        if sampled {
            let t = terms.iter().find(|t| t.name.contains("random")).unwrap();
            let kind = match t.name.as_str() {
                "all_corner_random" => SK_TERM_ALL_CORNER,
                "straight_random" => SK_TERM_STRAIGHT,
                "down_slope_random" => SK_TERM_DOWN_SLOPE,
                "up_slope_random" => SK_TERM_UP_SLOPE,
                _ => {
                    self.unsupported(format!(
                        "skill {sid} random term '{}' is not ported",
                        t.name
                    ));
                    return;
                }
            };
            if t.op != 0 || t.arg != 1 {
                self.unsupported(format!(
                    "skill {sid} random term needs ==1, got '{}'{}{}",
                    t.name,
                    op_text(t.op),
                    t.arg
                ));
                return;
            }
            if sampled_count == 1
                && pre_regions.len().max(1) * 4 * regions.len().max(1) > SK_REGMAX
            {
                self.unsupported(format!(
                    "skill {sid} random split exceeds {SK_REGMAX} runtime windows"
                ));
                return;
            }
            let (tag_off, tag_len) = self.tag(&format!("T{sid}:0:condition:{}", t.name));
            let at = SK_TERMS + SK_MSTRIDE * self.nterm;
            self.words[at] = kind;
            self.words[at + 1] = t.arg as u32;
            self.words[at + 2] = tag_off as u32;
            self.words[at + 3] = tag_len as u32;
            self.nterm += 1;
        }

        // Region storage: a random record stores the pre-split list and the
        // post-split list, and the kernel chains whole course, sampled
        // windows, then post list. A non-random record stores the final list
        // as the post list with the whole course implicit pre list. A dead
        // record stores nothing.
        let (n_a, reg_a) = if dead {
            (0usize, self.nreg)
        } else if sampled {
            self.push_regions(&pre_regions)
        } else {
            (0usize, self.nreg)
        };
        let (n_b, reg_b) = if dead { (0usize, self.nreg) } else { self.push_regions(&regions) };
        if self.reason.is_some() {
            return;
        }

        let effect_list = match alt.get("effects").and_then(Value::as_array) {
            Some(a) if !a.is_empty() => a,
            _ => {
                self.unsupported(format!("skill {sid} alternative has no effects"));
                return;
            }
        };
        if self.neff + effect_list.len() > SK_EMAX {
            self.unsupported("skill table overflow: too many effects");
            return;
        }
        let eff_base = self.neff;
        // The engine keeps the raw scaled value in f64 and rounds to f32 only
        // where the duration is consumed, so the table stores the f64 product.
        let base_duration = alt.get("baseDuration").and_then(Value::as_f64).unwrap_or(0.) * 0.0001;
        let duration_usage = alt.get("durationUsage").and_then(Value::as_i64).unwrap_or(1) as i32;
        if duration_usage == 4 {
            self.unsupported(format!(
                "skill {sid} effect duration_usage 4 (delayed running instance) is not ported"
            ));
            return;
        }
        let mut kind_seen: Vec<i32> = Vec::new();
        for ef in effect_list {
            let kind = ef.get("type").and_then(Value::as_i64).unwrap_or(-1) as i32;
            if !matches!(kind, 21 | 27 | 31) {
                self.unsupported(format!("skill {sid} effect type {kind} is not ported"));
                return;
            }
            if kind_seen.contains(&kind) {
                self.unsupported(format!(
                    "skill {sid} repeats effect kind {kind}, only one duration per kind is ported"
                ));
                return;
            }
            kind_seen.push(kind);
            if ef.get("target").and_then(Value::as_i64).unwrap_or(1) != 1 {
                self.unsupported(format!(
                    "skill {sid} effect targeting another runner is not ported"
                ));
                return;
            }
            if ef.get("additionalActivate").and_then(Value::as_i64).unwrap_or(0) != 0 {
                self.unsupported(format!("skill {sid} additionalActivate is not ported"));
                return;
            }
            let value_usage = ef.get("valueUsage").and_then(Value::as_i64).unwrap_or(1) as i32;
            if matches!(value_usage, 8 | 9 | 11 | 14) {
                self.unsupported(format!(
                    "skill {sid} effect value_usage {value_usage} is not ported"
                ));
                return;
            }
            let raw = ef.get("modifier").and_then(Value::as_f64).unwrap_or(0.);
            let multiplier = match ef
                .get("levelModifiers")
                .and_then(Value::as_array)
                .map(|a| a.get(level.max(1).min(10) as usize - 1).and_then(Value::as_f64))
            {
                Some(Some(m)) => m,
                _ => level_multiplier(kind, level),
            };
            let modifier = retail_skill_value(raw, multiplier);
            let at = SK_EFFECTS + SK_ESTRIDE * self.neff;
            self.words[at] = kind as u32;
            self.words[at + 1] = 1;
            self.words[at + 2] = value_usage as u32;
            self.words[at + 3] = duration_usage as u32;
            let w = f64_words(modifier);
            self.words[at + 6] = w[0];
            self.words[at + 7] = w[1];
            let w = f64_words(base_duration);
            self.words[at + 8] = w[0];
            self.words[at + 9] = w[1];
            self.neff += 1;
        }
        let cooldown = alt.get("cooldown").and_then(Value::as_f64).unwrap_or(0.) / 10000.0;
        let delay_mult = delay_factor(alt);

        let prog_base = self.nprog;
        let prog_len = match emit_program(self, &terms, &leaves) {
            Ok(len) => len,
            Err(why) => {
                self.unsupported(format!("skill {sid}: {why}"));
                self.nprog = prog_base;
                return;
            }
        };

        let (wis_off, wis_len) = if wisdom { self.tag(&format!("W{sid}")) } else { (0, 0) };
        let at = SK_RECORDS + SK_RSTRIDE * self.nrec;
        self.words[at] = sid;
        self.words[at + 1] = runner as u32;
        self.words[at + 2] = (if retain { SK_FLAG_RETAIN } else { 0 })
            | (if wisdom { SK_FLAG_WISDOM } else { 0 })
            | (if sampled { SK_FLAG_SAMPLED } else { 0 })
            | (if dead { SK_FLAG_DEAD } else { 0 });
        self.words[at + 3] = priority as u32;
        self.words[at + 4] = rarity;
        self.words[at + 5] = (self.neff - eff_base) as u32;
        self.words[at + 6] = eff_base as u32;
        self.words[at + 7] = prog_len as u32;
        self.words[at + 8] = prog_base as u32;
        self.words[at + 9] = sampled as u32;
        self.words[at + 10] = if sampled { (self.nterm - 1) as u32 } else { 0 };
        self.words[at + 11] = n_a as u32;
        self.words[at + 12] = reg_a as u32;
        let w = f64_words(cooldown);
        self.words[at + 13] = w[0];
        self.words[at + 14] = w[1];
        self.words[at + 15] = wis_off as u32;
        self.words[at + 16] = wis_len as u32;
        self.words[at + 17] = delay_mult.to_bits();
        self.words[at + 18] = n_b as u32;
        self.words[at + 19] = reg_b as u32;
        self.nrec += 1;
    }

    fn push_regions(&mut self, list: &[(f64, f64)]) -> (usize, usize) {
        if self.nreg + list.len() > SK_GMAX {
            self.unsupported("skill table overflow: too many regions");
            return (0, self.nreg);
        }
        let base = self.nreg;
        for (s, e) in list {
            let w = f64_words(*s);
            self.words[SK_REGIONS + 4 * self.nreg] = w[0];
            self.words[SK_REGIONS + 4 * self.nreg + 1] = w[1];
            let w = f64_words(*e);
            self.words[SK_REGIONS + 4 * self.nreg + 2] = w[0];
            self.words[SK_REGIONS + 4 * self.nreg + 3] = w[1];
            self.nreg += 1;
        }
        (list.len(), base)
    }
}

/// Retail `retail_skill_value` from the builder: master coefficient /10000
/// through the integral units path the engine takes before the f32 fold.
fn retail_skill_value(raw: f64, multiplier: f64) -> f64 {
    let units = multiplier * 10000.0;
    let integral = units.round();
    let coefficient = if (units - integral).abs() < 1e-7 {
        integral as f32 * 0.0001_f32
    } else {
        multiplier as f32
    };
    ((raw as f32 * 0.0001_f32) * coefficient) as f64
}

fn level_multiplier(kind: i32, level: i64) -> f64 {
    let level = (level.max(1).min(10)) as usize;
    const TARGET_SPEED: [f64; 11] =
        [1.00, 1.00, 1.01, 1.04, 1.07, 1.10, 1.13, 1.16, 1.19, 1.22, 1.25];
    const ACCEL: [f64; 11] = [1.00, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.125, 1.15, 1.175, 1.20];
    const OTHER: [f64; 11] =
        [1.00, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18];
    match kind {
        27 => TARGET_SPEED[level],
        31 => ACCEL[level],
        _ => OTHER[level],
    }
}

/// The always-alternative start-delay factor, read the way
/// `retail_initialization.rs` reads it.
fn delay_factor(alt: &Value) -> f32 {
    if alt.get("condition").and_then(Value::as_str).unwrap_or("") != "always==1" {
        return 1.0;
    }
    if !alt.get("precondition").and_then(Value::as_str).unwrap_or("").is_empty() {
        return 1.0;
    }
    let mut multiplier = 1.0f32;
    if let Some(effects) = alt.get("effects").and_then(Value::as_array) {
        for ef in effects {
            if ef.get("type").and_then(Value::as_i64) == Some(10)
                && ef.get("target").and_then(Value::as_i64) == Some(1)
            {
                let removal =
                    (10000.0 - ef.get("modifier").and_then(Value::as_f64).unwrap_or(0.)) as f32
                        / 10000.0;
                multiplier *= 1.0 - removal;
            }
        }
    }
    multiplier
}

/// One parsed comparison from the engine's condition grammar.
#[derive(Clone)]
pub struct Term {
    pub name: String,
    pub op: i64,
    pub arg: i64,
    pub or: bool,
}

/// `Or := And ('@' And)*`, `And := Cmp ('&' Cmp)*`, `Cmp := ident op integer`.
pub fn parse_terms(text: &str) -> Result<Vec<Term>, String> {
    let bytes = text.as_bytes();
    let mut at = 0usize;
    let mut out: Vec<Term> = Vec::new();
    let mut or_next = false;
    while at < bytes.len() {
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if at >= bytes.len() {
            break;
        }
        if bytes[at] == b'@' {
            or_next = true;
            at += 1;
            continue;
        }
        if bytes[at] == b'&' {
            at += 1;
            continue;
        }
        if !(bytes[at].is_ascii_lowercase() || bytes[at] == b'_') {
            return Err(format!("unexpected byte {:?}", bytes[at] as char));
        }
        let start = at;
        while at < bytes.len() && (bytes[at].is_ascii_lowercase() || bytes[at] == b'_') {
            at += 1;
        }
        let name = text[start..at].to_string();
        let op_at = at;
        while at < bytes.len() && b"<>=!".contains(&bytes[at]) {
            at += 1;
        }
        let op = match &text[op_at..at] {
            "==" => 0i64,
            "!=" => 1,
            "<" => 2,
            "<=" => 3,
            ">" => 4,
            ">=" => 5,
            other => return Err(format!("bad operator {other:?}")),
        };
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        let neg = at < bytes.len() && bytes[at] == b'-';
        if neg {
            at += 1;
        }
        let num_at = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if num_at == at {
            return Err(format!("{name} has no integer argument"));
        }
        let arg: i64 = text[num_at..at].parse().map_err(|_| "argument out of range")?;
        out.push(Term { name, op, arg: if neg { -arg } else { arg }, or: or_next });
        or_next = false;
    }
    if out.is_empty() {
        return Err("empty condition".into());
    }
    if out[0].or {
        return Err("condition starts with an or marker".into());
    }
    if out.len() > 60 {
        return Err("condition has too many comparisons".into());
    }
    Ok(out)
}

/// The engine-side role of one parsed term, after the region algebra.
#[derive(Clone)]
enum Leaf {
    True,
    False,
    Region(Vec<(f64, f64)>),
    /// The last-straight one-time term: a region gate plus the per-tick
    /// entry-latch check combined in the engine's own filter.
    LsEntry(Vec<(f64, f64)>),
    Dynamic { feat: u32, cmp: i64 },
    Sampled,
}

fn classify(t: &Term, sid: u32, g: &Geometry) -> Result<Leaf, String> {
    let arg = t.arg as f64;
    let d = g.distance;
    let sid_text = sid.to_string();
    let unregistered = |what: &str| Err(format!("{what}{} is not a registered comparison", op_text(t.op)));
    match t.name.as_str() {
        "always" => {
            if t.op == 0 && t.arg == 1 {
                return Ok(Leaf::True);
            }
            // always==0 or an odd comparison is a condition the engine
            // resolves statically false, dropping the record.
            return Ok(Leaf::False);
        }
        "phase" => {
            let p = t.arg;
            let fudge = if PHASE_FUDGE_SKILLS.contains(&sid_text.as_str()) { 10.0 } else { 0.0 };
            let bounds = match t.op {
                0 if (0..=3).contains(&p) => (phase_start(d, p), phase_end(d, p) + fudge),
                2 if (1..=3).contains(&p) => (0.0, phase_start(d, p)),
                3 if (0..=3).contains(&p) => (0.0, phase_end(d, p)),
                4 if (-1..=2).contains(&p) => (phase_start(d, p + 1), d),
                5 if (0..=3).contains(&p) => (phase_start(d, p), d),
                _ => return unregistered("phase"),
            };
            Ok(Leaf::Region(vec![bounds]))
        }
        "distance_rate" => {
            let bounds = match t.op {
                3 => (0.0, d * arg / 100.0),
                5 => (d * arg / 100.0, d),
                _ => return unregistered("distance_rate"),
            };
            Ok(Leaf::Region(vec![bounds]))
        }
        "remain_distance" => {
            let bounds = match t.op {
                0 => (d - arg, d - arg + 1.0),
                3 => (d - arg, d),
                5 => (0.0, d - arg),
                _ => return unregistered("remain_distance"),
            };
            Ok(Leaf::Region(vec![bounds]))
        }
        "furlong" => {
            if t.op != 0 {
                return unregistered("furlong");
            }
            Ok(Leaf::Region(vec![((arg - 1.0) * 201.168, arg * 201.168)]))
        }
        "is_finalcorner" => {
            if t.op != 0 || (t.arg != 0 && t.arg != 1) {
                return Err("is_finalcorner expects ==0 or ==1".into());
            }
            if g.corners.is_empty() {
                // The engine passes the regions through for flag 0 and drops
                // the record for flag 1 when the course has no corners.
                return Ok(if t.arg == 0 { Leaf::True } else { Leaf::False });
            }
            let start = g.corners.last().unwrap().0;
            Ok(Leaf::Region(vec![if t.arg == 0 { (0.0, start) } else { (start, d) }]))
        }
        "is_last_straight" => {
            if t.op != 0 || t.arg != 1 {
                return Err("is_last_straight expects ==1".into());
            }
            match g.straights.last() {
                Some((s, e)) => Ok(Leaf::Region(vec![(*s, *e)])),
                None => Err("is_last_straight needs a course with straights".into()),
            }
        }
        "is_last_straight_onetime" => {
            if t.op != 0 || t.arg != 1 {
                return Err("is_last_straight_onetime expects ==1".into());
            }
            match g.straights.last() {
                Some((s, _)) => Ok(Leaf::LsEntry(vec![(*s, s + 10.0)])),
                None => Err("is_last_straight_onetime needs a course with straights".into()),
            }
        }
        "hp_per" => {
            if !(t.op == 3 || t.op == 5) {
                return Err("only <= and >= are registered for hp_per".into());
            }
            Ok(Leaf::Dynamic { feat: FEAT_HP_PER, cmp: t.op })
        }
        "is_overtake" => Ok(Leaf::Dynamic { feat: FEAT_IS_OVERTAKE, cmp: t.op }),
        "is_temptation" => {
            if t.op != 0 {
                return Err("only == is registered for is_temptation".into());
            }
            Ok(Leaf::Dynamic { feat: FEAT_IS_TEMPTATION, cmp: t.op })
        }
        "order" => Ok(Leaf::Dynamic { feat: FEAT_ORDER, cmp: t.op }),
        "order_rate" => Ok(Leaf::Dynamic { feat: FEAT_ORDER_RATE, cmp: t.op }),
        "corner" => {
            if !sorted_by_start(&g.corners) {
                return Err("corner needs a sorted corners list".into());
            }
            let num = t.arg;
            let list = if t.op == 1 {
                if num != 0 {
                    return Err("corner!= only supports 0".into());
                }
                g.corners.clone()
            } else if t.op != 0 {
                return unregistered("corner");
            } else if num == 0 {
                let mut last_end = 0.0;
                let mut gaps = Vec::new();
                for c in &g.corners {
                    gaps.push((last_end, c.0));
                    last_end = c.1;
                }
                if last_end != d {
                    gaps.push((last_end, d));
                }
                gaps
            } else if (g.corners.len() as i64) + num >= 5 {
                let mut idx = g.corners.len() as i64 + num - 5;
                let mut picked = Vec::new();
                while idx >= 0 {
                    picked.push(g.corners[idx as usize]);
                    idx -= 4;
                }
                picked.reverse();
                picked
            } else {
                Vec::new()
            };
            if list.is_empty() {
                return Ok(Leaf::False);
            }
            Ok(Leaf::Region(list))
        }
        "slope" => {
            if t.op != 0 {
                return unregistered("slope");
            }
            if !sorted_by_start(&g.slopes.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>()) {
                return Err("slope needs a sorted slopes list".into());
            }
            let list = match t.arg {
                0 => {
                    let mut last_start = 0.0;
                    let mut gaps = Vec::new();
                    for s in &g.slopes {
                        if s.0 - last_start > 0.0 {
                            gaps.push((last_start, s.0));
                        }
                        last_start = last_start.max(s.1);
                    }
                    if d - last_start > 0.0 {
                        gaps.push((last_start, d));
                    }
                    gaps
                }
                1 => g.slopes.iter().filter(|s| s.2 > 0.0).map(|s| (s.0, s.1)).collect(),
                2 => g.slopes.iter().filter(|s| s.2 < 0.0).map(|s| (s.0, s.1)).collect(),
                _ => return Err(format!("slope type {} is not registered", t.arg)),
            };
            if list.is_empty() {
                return Ok(Leaf::False);
            }
            Ok(Leaf::Region(list))
        }
        "all_corner_random" | "straight_random" | "down_slope_random" | "up_slope_random" => {
            if t.op != 0 || t.arg != 1 {
                return Err(format!("random term '{}' needs ==1", t.name));
            }
            Ok(Leaf::Sampled)
        }
        other => Err(format!("'{other}' is not ported")),
    }
}

fn op_text(op: i64) -> &'static str {
    match op {
        0 => "==",
        1 => "!=",
        2 => "<",
        3 => "<=",
        4 => ">",
        _ => ">=",
    }
}

/// Emits the postfix program over the dynamic leaves only. Static region
/// terms and the sampled term act through the window gate instead, exactly
/// like the engine's None checks. Each group ANDs its leaves left to right,
/// and one OR marker follows every group after the first.
fn emit_program(c: &mut Compiler, terms: &[Term], leaves: &[Leaf]) -> Result<usize, String> {
    let base = c.nprog;
    let mut used = 0usize;
    let mut first_group = true;
    let mut i = 0usize;
    while i < terms.len() {
        let group_start = i;
        let mut group_len = 1usize;
        while i + group_len < terms.len() && !terms[i + group_len].or {
            group_len += 1;
        }
        for j in 0..group_len {
            if base + used >= SK_PMAX {
                return Err("condition program overflow".into());
            }
            emit_leaf(c, base + used, &terms[group_start + j], &leaves[group_start + j])?;
            used += 1;
            if j > 0 {
                if base + used >= SK_PMAX {
                    return Err("condition program overflow".into());
                }
                c.words[SK_PROGRAMS + SK_PSTRIDE * (base + used)] = OP_AND;
                used += 1;
            }
        }
        if !first_group {
            if base + used >= SK_PMAX {
                return Err("condition program overflow".into());
            }
            c.words[SK_PROGRAMS + SK_PSTRIDE * (base + used)] = OP_OR;
            used += 1;
        }
        first_group = false;
        i += group_len;
    }
    if used == 0 {
        return Err("condition has no leaves".into());
    }
    c.nprog += used;
    Ok(used)
}

fn emit_leaf(c: &mut Compiler, slot: usize, t: &Term, leaf: &Leaf) -> Result<(), String> {
    let at = SK_PROGRAMS + SK_PSTRIDE * slot;
    match leaf {
        Leaf::Dynamic { feat, cmp } => {
            c.words[at] = OP_FEAT_I;
            c.words[at + 1] = feat * 8 + *cmp as u32;
            let w = f64_words(t.arg as f64);
            c.words[at + 2] = w[0];
            c.words[at + 3] = w[1];
        }
        Leaf::LsEntry(_) => {
            c.words[at] = OP_FEAT_I;
            c.words[at + 1] = FEAT_LS_ENTRY * 8;
            c.words[at + 2] = 1;
            c.words[at + 3] = 0;
        }
        _ => {
            c.words[at] = OP_TRUE;
            c.words[at + 1] = if matches!(leaf, Leaf::False) { 0 } else { 1 };
        }
    }
    Ok(())
}
