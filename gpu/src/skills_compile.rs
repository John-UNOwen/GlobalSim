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

/// Which runtime the table targets. The classic runtime has no sampled-term
/// sampler and its kernel carries no live-state condition readers, so the
/// classic profile keeps the constant and course-region leaves, takes the
/// classic effect-value chain, and refuses a live-state or sampled leaf by name.
/// The table layout is shared across both lanes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Retail,
    Classic,
}

pub const SK_STRIDE: usize = 55872;

pub const SK_NREC: usize = 0;
pub const SK_NPROG: usize = 1;
pub const SK_NREG: usize = 2;
pub const SK_NEFF: usize = 3;
pub const SK_NTAG: usize = 4;
pub const SK_NSTR: usize = 5;
pub const SK_NCOR: usize = 6;
pub const SK_NSLOPE: usize = 7;
pub const SK_NTERM: usize = 8;

pub const SK_STRAIGHTS: usize = 16;
// The runner the viewer-side leaves read, from the request's playerHorseIndex.
pub const SK_VIDX: usize = 9;
pub const SK_CORNERS: usize = 40;
pub const SK_SLOPES: usize = 64;
pub const SK_RECORDS: usize = 112;
pub const SK_EFFECTS: usize = 26032;
pub const SK_PROGRAMS: usize = 27952;
pub const SK_ARMS: usize = 40240;
pub const SK_REGIONS: usize = 42288;
pub const SK_TERMS: usize = 50480;
pub const SK_TAGS: usize = 51760;

pub const SK_RSTRIDE: usize = 27;
pub const SK_ESTRIDE: usize = 10;
/// Effects per record cap: each effect gets its own duration timer slot and
/// a token of its own, and the token packs the slot index in three bits.
pub const SK_EMAXSLOT: usize = 8;
pub const SK_PSTRIDE: usize = 4;
pub const SK_GSTRIDE: usize = 4;
pub const SK_MSTRIDE: usize = 8;
pub const SK_ASTRIDE: usize = 8;

// Arm flags, mirroring skills.cl SK_ARM_SAMPLED and SK_ARM_DEAD. The dead
// flag marks an arm whose chain is empty or whose sampled term resolved no
// row, so it never passes the runtime gate.
pub const SK_ARM_SAMPLED: u32 = 1;
pub const SK_ARM_DEAD: u32 = 2;

// Record slots per race: the roster cap times the eight-slot window each
// runner can hold (skills.cl SK_RMAX). sk_targets caps live targets at the
// roster, so the raise adds no behavior at small rosters.
pub const SK_RMAX: usize = 960;
pub const SK_EMAX: usize = 192;
pub const SK_PMAX: usize = 3072;
pub const SK_GMAX: usize = 2048;
pub const SK_TMAX: usize = 4096;
pub const SK_MMAX: usize = 160;
pub const SK_AMAX: usize = 256;
pub const SK_STRMAX: usize = 12;
pub const SK_CORMAX: usize = 12;
pub const SK_SLOPEMAX: usize = 16;
pub const SK_REGMAX: usize = 16;
pub const SK_ARMMAX: usize = 4;

pub const SK_FLAG_RETAIN: u32 = 1;
pub const SK_FLAG_WISDOM: u32 = 2;
pub const SK_FLAG_SAMPLED: u32 = 4;
pub const SK_FLAG_DEAD: u32 = 8;
pub const SK_FLAG_ANYACT: u32 = 16;
pub const SK_FLAG_WISDOM_FOLLOW: u32 = 32;
pub const SK_FLAG_INIT_DEFER: u32 = 64;
pub const SK_FLAG_GREEN: u32 = 128;
pub const SK_FLAG_PRIMARY: u32 = 256;
pub const SK_FLAG_RSELIG: u32 = 512;
pub const SK_FLAG_RANGOLD: u32 = 1024;
pub const SK_FLAG_RANWHITE: u32 = 2048;
pub const SK_FLAG_GENERAL: u32 = 4096;
pub const SK_FLAG_BUMPED: u32 = 8192;
/// A wisdom-gated record whose static core fires in the engine's constructor
/// pass (race_solver.rs:1008 before the section draws at lib.rs:760). The
/// kernel bakes its self bumps once the lottery has enabled it.
pub const SK_FLAG_INIT_BUMP: u32 = 16384;

pub const SK_TERM_ALL_CORNER: u32 = 0;
pub const SK_TERM_STRAIGHT: u32 = 1;
pub const SK_TERM_DOWN_SLOPE: u32 = 2;
pub const SK_TERM_UP_SLOPE: u32 = 3;
pub const SK_TERM_PHASE_RANDOM: u32 = 4;
pub const SK_TERM_PHASE_FIRSTHALF: u32 = 5;
pub const SK_TERM_PHASE_LATERHALF: u32 = 6;
pub const SK_TERM_FC_RANDOM: u32 = 7;
pub const SK_TERM_DIST_RATE_AFTER: u32 = 8;
pub const SK_TERM_CORNER_RANDOM: u32 = 9;

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
pub const FEAT_PHASE: u32 = 7;
pub const FEAT_CORNER: u32 = 8;
pub const FEAT_FINALE: u32 = 9;
pub const FEAT_FC_HALF: u32 = 10;
pub const FEAT_DIST_RATE: u32 = 11;
pub const FEAT_REMAIN: u32 = 12;
pub const FEAT_CLOCK: u32 = 13;
pub const FEAT_TEMPT: u32 = 14;
pub const FEAT_ACT0: u32 = 15;
pub const FEAT_ACT1: u32 = 16;
pub const FEAT_BSIDE: u32 = 17;
pub const FEAT_IFNEAR: u32 = 18;
pub const FEAT_BHNEAR: u32 = 19;
pub const FEAT_ORDER_DELTA: u32 = 20;
pub const FEAT_OUTBAND: u32 = 21;
pub const FEAT_NEAR: u32 = 22;
pub const FEAT_BASH_BEHIND: u32 = 23;
pub const FEAT_BASH_INFRONT: u32 = 24;
pub const FEAT_LASTSPURT: u32 = 25;
pub const FEAT_BHNEAR1: u32 = 26;
pub const FEAT_SLOPE: u32 = 27;
pub const FEAT_ANYACT: u32 = 28;
pub const FEAT_OTIME: u32 = 29;
pub const FEAT_DDTOP: u32 = 30;
pub const FEAT_HEAL: u32 = 31;
pub const FEAT_CHGUP_END: u32 = 32;
pub const FEAT_LANETYPE: u32 = 33;
pub const FEAT_FIGHT: u32 = 34;
pub const FEAT_BHIN: u32 = 35;
pub const FEAT_MOVELANE: u32 = 36;
pub const FEAT_BFRONT: u32 = 37;
// Wave-3 leaves. FEAT_INBAND packs the band rate (20/40/50/80) in the
// argument word and FEAT_USED the queried skill id.
pub const FEAT_INBAND: u32 = 38;
pub const FEAT_NEARIF: u32 = 39;
pub const FEAT_VISN: u32 = 40;
pub const FEAT_USED: u32 = 41;
pub const FEAT_ACT2: u32 = 42;
pub const FEAT_ACTLH: u32 = 43;
pub const FEAT_ACTALL: u32 = 44;
pub const FEAT_NOTOP: u32 = 45;
pub const FEAT_NOTOD: u32 = 46;
pub const FEAT_CHGUP_MID: u32 = 47;
pub const FEAT_CHGUP_FC: u32 = 48;
pub const FEAT_BALL: u32 = 49;
pub const FEAT_BFRONTB: u32 = 50;
pub const FEAT_SURR: u32 = 51;
pub const FEAT_TEMPT_B: u32 = 52;
pub const FEAT_TEMPT_F: u32 = 53;
pub const FEAT_DDRATE: u32 = 55;
pub const FEAT_DDTOPF: u32 = 56;
// Retail races compare the phase condition against the runtime phase, which
// keeps 3 while the solver field clamps at 2 (race_solver_retail.rs:1311).
pub const FEAT_RPHASE: u32 = 57;
// running_style_temptation_count_{nige,senko,sashi,oikomi} take 58 to 61, the
// group in the low two id bits, and the _opponent_count_ variants take 62 to
// 65 and leave the querying runner out, like the engine's opponents_only arm.
// The group lives in the feature id because the program slot's one f64
// carries the comparison threshold.
pub const FEAT_RTEMP: u32 = 58;
pub const FEAT_RTEMP_OPP: u32 = 62;
pub const FEAT_RTEMP_END: u32 = 65;
// remain_distance_viewer_id: the course distance minus the floor of the
// viewer row's position, the viewer from the SK_VIDX header word.
pub const FEAT_VREMAIN: u32 = 66;

/// One race's compiled table. `reason` is Some when the race must run on the
/// CPU, naming the first construct the kernel cannot evaluate. `bumps` are
/// construction-time stat deltas, (runner, effect kind, modifier), in wire
/// order, for kinds the kernel cannot apply once the race runs.
pub struct SkillsPlan {
    pub words: Vec<u32>,
    pub reason: Option<String>,
    pub bumps: Vec<(usize, i32, f64)>,
    /// True when some GutsUp arm can fire outside the constructor pass, so
    /// the kernel must recompute the guts-derived powf terms from the live
    /// field instead of reading the prepare-baked words.
    pub live_guts: bool,
}

pub fn plan(root: &Value, course: &Value) -> SkillsPlan {
    plan_lane(root, course, Lane::Retail)
}

/// Compile against the classic (non-retail) runtime profile: the classic
/// effect-value chain and durations, with sampled and live-state condition
/// leaves refused by name.
pub fn plan_classic(root: &Value, course: &Value) -> SkillsPlan {
    plan_lane(root, course, Lane::Classic)
}

fn plan_lane(root: &Value, course: &Value, lane: Lane) -> SkillsPlan {
    let mut c = Compiler {
        words: vec![0; SK_STRIDE],
        tags: Vec::new(),
        reason: None,
        bumps: Vec::new(),
        live_guts: false,
        nrec: 0,
        nprog: 0,
        nreg: 0,
        neff: 0,
        nterm: 0,
        narm: 0,
        team_stats: Vec::new(),
        lane,
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
    c.pack_team_stats(horses);
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
        let mut seen_ids: Vec<String> = Vec::new();
        for entry in list {
            let (sid, level) = match normalize_skill_entry(entry) {
                Some(v) => v,
                None => {
                    c.unsupported("skill entry needs a numeric skillId".to_string());
                    return c.finish();
                }
            };
            if seen_ids.contains(&sid) {
                c.unsupported(format!("skill {sid} appears twice for one runner"));
                return c.finish();
            }
            seen_ids.push(sid.clone());
            c.pack_skill(&sid, level, r, skill_data, course, horse, root);
            if c.reason.is_some() {
                return c.finish();
            }
        }
    }
    // A cross debuff is delivered by the kernel, which drops the same-team
    // receivers exactly like race_field.rs:458-461, so no roster gate remains.
    c.finish()
}

/// `skill_id_and_level` in the engine's request parser: an object carries a
/// string or numeric skillId with a level, a bare string or number is a
/// level 1 skill.
pub fn normalize_skill_entry(entry: &Value) -> Option<(String, i64)> {
    if let Some(obj) = entry.as_object() {
        let id = if let Some(s) = obj.get("skillId").and_then(Value::as_str) {
            s.to_string()
        } else if let Some(n) = obj.get("skillId").and_then(Value::as_f64) {
            (n as i64).to_string()
        } else {
            return None;
        };
        let level = obj
            .get("level")
            .and_then(Value::as_f64)
            .map(|n| n as i64)
            .filter(|&n| n != 0)
            .unwrap_or(1);
        Some((id, level))
    } else if let Some(s) = entry.as_str() {
        Some((s.to_string(), 1))
    } else if let Some(n) = entry.as_f64() {
        Some(((n as i64).to_string(), 1))
    } else {
        None
    }
}

struct Compiler {
    words: Vec<u32>,
    tags: Vec<u8>,
    reason: Option<String>,
    bumps: Vec<(usize, i32, f64)>,
    live_guts: bool,
    nrec: usize,
    nprog: usize,
    nreg: usize,
    neff: usize,
    nterm: usize,
    narm: usize,
    // The engine's per-runner team stat totals (lib.rs PreparedRoster::new,
    // the retail_runtime branch), read by the value_usage 3 to 7 tier fold.
    team_stats: Vec<[f64; 5]>,
    lane: Lane,
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
        self.words[SK_NTERM] = self.nterm as u32;
        if self.tags.len() > SK_TMAX
            || self.nprog > SK_PMAX
            || self.nreg > SK_GMAX
            || self.nterm > SK_MMAX
            || self.narm > SK_AMAX
        {
            self.unsupported("skill table overflow");
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
        let bumps = if self.reason.is_some() { Vec::new() } else { std::mem::take(&mut self.bumps) };
        let live_guts = self.reason.is_none() && self.live_guts;
        SkillsPlan { words: self.words, reason: self.reason, bumps, live_guts }
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

    fn pack_skill(
        &mut self,
        sid: &str,
        level: i64,
        runner: usize,
        skill_data: &Value,
        course: &Value,
        horse: &Value,
        root: &Value,
    ) {
        let sid_num = match sid.parse::<u32>() {
            Ok(v) => v,
            Err(_) => {
                self.unsupported(format!("skill id {sid} is not numeric"));
                return;
            }
        };
        let def = match skill_data.get(sid) {
            Some(d) => d,
            None => {
                self.unsupported(format!("skillData has no entry for skill {sid}"));
                return;
            }
        };
        // build_skill_data installs one record per alternative and skips an
        // alternative whose region algebra is empty. Records share one
        // cooldown key per skill id, and the kernel parks the key across all
        // live records of the group when one fires, so every live alternative
        // lowers and the first to pass its gate blocks the rest like the
        // engine's key map (race_solver.rs:2437). A shadowed duplicate
        // (identical program, no random term) still collapses instead.
        let alts = match def.get("alternatives").and_then(Value::as_array) {
            Some(a) if !a.is_empty() => a,
            _ => {
                self.unsupported(format!("skill {sid} has no alternatives"));
                return;
            }
        };
        let mut live: Option<(String, String)> = None;
        let mut live_rec: Option<usize> = None;
        let mut first_record = true;
        for (alt_index, alt) in alts.iter().enumerate() {
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
            let retain = alt_index == 0 && matches!(rarity, 1 | 2 | 6);
            // The green tag marks the id for the activated_green_skills set,
            // and the master flag gates random re-activation (lib.rs:187-193).
            let green = def
                .get("tags")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_f64).any(|t| (601.0..=615.0).contains(&t)))
                .unwrap_or(false);
            let general = def.get("isGeneralSkill").and_then(Value::as_bool).unwrap_or(true);

            let condition = alt.get("condition").and_then(Value::as_str).unwrap_or("");
            let terms = match parse_terms(condition) {
                Ok(t) => t,
                Err(why) => {
                    self.unsupported(format!("skill {sid} condition '{condition}': {why}"));
                    return;
                }
            };
            let g = Geometry::new(course);
            let statics = Statics::new(course, horse, root, sid, runner);
            let leaves: Vec<Leaf> = match terms.iter().map(|t| classify(t, &g, &statics)).collect() {
                Ok(l) => l,
                Err(why) => {
                    self.unsupported(format!("skill {sid} condition '{condition}': {why}"));
                    return;
                }
            };
            if self.lane == Lane::Classic && self.classic_refuses_leaf(sid, &terms, &leaves) {
                return;
            }
            if leaves.iter().any(|l| leaf_feat(l) == FEAT_VREMAIN) {
                self.words[SK_VIDX] = statics.viewer.unwrap_or(0) as u32;
            }

            // The builder evaluates a precondition over the whole course, clips
            // the main chain to the first precondition region's start, and checks
            // precondition membership plus its dynamics as a latched gate.
            let mut dead = false;
            let mut seed = vec![(0.0f64, g.distance)];
            let mut pre_n = 0usize;
            let mut pre_reg_b = self.nreg;
            let mut pre_prog_b = self.nprog;
            let mut pre_prog_len = 0usize;
            let pre_text = alt.get("precondition").and_then(Value::as_str).unwrap_or("");
            // A precondition that folds to all-True latches every tick, exactly
            // like having no precondition at all, which is what the live-
            // alternative shadow comparison needs.
            let mut pre_trivial = true;
            let mut pre_init_defer = false;
            if !pre_text.is_empty() {
                let pre_terms = match parse_terms(pre_text) {
                    Ok(t) => t,
                    Err(why) => {
                        self.unsupported(format!("skill {sid} precondition '{pre_text}': {why}"));
                        return;
                    }
                };
                let pre_leaves: Vec<Leaf> =
                    match pre_terms.iter().map(|t| classify(t, &g, &statics)).collect() {
                        Ok(l) => l,
                        Err(why) => {
                            self.unsupported(format!("skill {sid} precondition '{pre_text}': {why}"));
                            return;
                        }
                    };
                if self.lane == Lane::Classic
                    && self.classic_refuses_leaf(sid, &pre_terms, &pre_leaves)
                {
                    return;
                }
                if pre_leaves.iter().any(|l| matches!(l, Leaf::Sampled)) {
                    self.unsupported(format!(
                        "skill {sid} precondition random term is not ported"
                    ));
                    return;
                }
                // take_anyskill_nodes drops nodes parsed in the precondition, so
                // the engine never observes them; the kernel has no equivalent.
                if pre_leaves.iter().any(|l| {
                    matches!(l, Leaf::Dynamic { feat, .. } if *feat == FEAT_ANYACT)
                }) {
                    self.unsupported(format!(
                        "skill {sid} precondition 'is_activate_any_skill' is not ported"
                    ));
                    return;
                }
                if pre_leaves.iter().any(|l| leaf_feat(l) == FEAT_VREMAIN) {
                    self.words[SK_VIDX] = statics.viewer.unwrap_or(0) as u32;
                }
                pre_trivial = pre_leaves.iter().all(|l| matches!(l, Leaf::True));
                pre_init_defer = pre_leaves.iter().any(|l| {
                    matches!(leaf_feat(l), FEAT_ORDER_RATE | FEAT_OTIME | FEAT_DDTOP
                        | FEAT_CHGUP_END | FEAT_LANETYPE | FEAT_BHIN | FEAT_MOVELANE
                        | FEAT_BFRONT | FEAT_NEARIF | FEAT_VISN | FEAT_INBAND
                        | FEAT_BALL | FEAT_BFRONTB | FEAT_CHGUP_MID | FEAT_CHGUP_FC
                        | FEAT_NOTOP | FEAT_NOTOD | FEAT_SURR | FEAT_TEMPT_B
                        | FEAT_TEMPT_F | FEAT_RTEMP..=FEAT_RTEMP_END | FEAT_DDRATE
                        | FEAT_DDTOPF | FEAT_VREMAIN)
                });
                let mut pre_regions: Vec<(f64, f64)> = vec![(0.0, g.distance)];
                for leaf in pre_leaves.iter() {
                    let next = match leaf {
                        Leaf::True | Leaf::Dynamic { .. } => pre_regions.clone(),
                        Leaf::False => Vec::new(),
                        Leaf::Sampled => unreachable!(),
                        Leaf::Region(list) | Leaf::RegionDyn(list, _) => {
                            rmap_multi(&pre_regions, |r| list.iter().map(|b| intersect(*r, *b)).collect())
                        }
                    };
                    if next.len() > SK_REGMAX {
                        self.unsupported(format!(
                            "skill {sid} precondition exceeds {SK_REGMAX} regions"
                        ));
                        return;
                    }
                    pre_regions = next;
                }
                if pre_regions.is_empty() {
                    // The builder skips the whole alternative: continue, with no
                    // record to install.
                    dead = true;
                } else {
                    let push = self.push_regions(&pre_regions);
                    pre_n = push.0;
                    pre_reg_b = push.1;
                    seed = vec![(pre_regions[0].0, g.distance)];
                    let base = self.nprog;
                    match emit_program(self, &pre_terms, &pre_leaves) {
                        Ok(len) => {
                            pre_prog_b = base;
                            pre_prog_len = len;
                        }
                        Err(why) => {
                            self.nprog = base;
                            self.unsupported(format!("skill {sid} precondition: {why}"));
                            return;
                        }
                    }
                }
                if self.reason.is_some() {
                    return;
                }
            }

            // A top-level '@' group becomes an arm: its own region chain and
            // dynamic program. At runtime the activation passes when the position
            // is in the merged window list and one arm's membership and program
            // hold, mirroring the Or closure the engine's apply builds.
            let mut groups: Vec<(usize, usize)> = Vec::new();
            for (i, t) in terms.iter().enumerate() {
                if i == 0 || t.or {
                    groups.push((i, i + 1));
                } else {
                    groups.last_mut().unwrap().1 = i + 1;
                }
            }
            if groups.len() > SK_ARMMAX {
                self.unsupported(format!(
                    "skill {sid} condition has more than {} arms",
                    SK_ARMMAX
                ));
                return;
            }
            if self.narm + groups.len().max(1) > SK_AMAX {
                self.unsupported("skill table overflow: too many arms");
                return;
            }
            let mut arms: Vec<([Vec<(f64, f64)>; 2], [u32; 2], usize)> = Vec::new();
            let mut sampled_count = 0usize;
            for (start, end) in groups.iter() {
                let arm_sampled =
                    leaves[*start..*end].iter().filter(|l| matches!(l, Leaf::Sampled)).count();
                if arm_sampled > 1 {
                    self.unsupported(format!(
                        "skill {sid} arm has more than one random term"
                    ));
                    return;
                }
                sampled_count += arm_sampled;
                let mut regions = seed.clone();
                let mut pre_split: Vec<(f64, f64)> = Vec::new();
                for (t, leaf) in terms[*start..*end].iter().zip(leaves[*start..*end].iter()) {
                    let next = match leaf {
                        Leaf::True | Leaf::Dynamic { .. } => regions.clone(),
                        Leaf::False => Vec::new(),
                        Leaf::Sampled => {
                            pre_split = regions.clone();
                            regions.clone()
                        }
                        Leaf::Region(list) | Leaf::RegionDyn(list, _) => {
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
                        break;
                    }
                }
                let sampled_flag =
                    arm_sampled.min(1) as u32 | if regions.is_empty() { SK_ARM_DEAD } else { 0 };
                if regions.is_empty() {
                    // Keep the collapsed arm as a draw-only entry: the engine
                    // still samples its random term at startup, and an empty
                    // chain never passes the runtime membership gate.
                    arms.push(([pre_split, regions], [sampled_flag, 0], 0));
                    continue;
                }
                arms.push(([pre_split, regions], [sampled_flag, 0], 0));
            }
            if !arms.is_empty() && arms.iter().all(|a| a.1[0] & SK_ARM_DEAD != 0) {
                // Every arm collapsed, so the union is empty and the builder
                // skips the alternative.
                dead = true;
            }
            if !dead {
                let pre_sig = if pre_trivial { String::new() } else { pre_text.to_string() };
                if let Some((live_cond, live_pre)) = &live {
                    if *live_cond == condition && *live_pre == pre_sig && sampled_count == 0 {
                        // The engine installs this alternative but the first
                        // alternative shares its cooldown key, is scanned first,
                        // and passes exactly when this one does, so this record
                        // can never be the one that fires. The start-delay
                        // multiplier still folds every alternative's factor of
                        // an enabled skill, in alternative order.
                        if let Some(rec) = live_rec {
                            let at = SK_RECORDS + SK_RSTRIDE * rec + 17;
                            let old = f32::from_bits(self.words[at]);
                            self.words[at] = (old * delay_factor(alt)).to_bits();
                        }
                        continue;
                    }
                    // A second live alternative lowers too: the kernel fans the
                    // shared key's park out over the group, so whichever passes
                    // first blocks the rest exactly like the engine's key map.
                }
                live = Some((condition.to_string(), pre_sig));
                live_rec = Some(self.nrec);
            }
            let anyact =
                leaves.iter().any(|l| matches!(l, Leaf::Dynamic { feat, .. } if *feat == FEAT_ANYACT));

            // Resolve each arm's sampled term. The engine samples one term per
            // arm at init, in text order. An arm whose term resolves to no row
            // spends no draw and can never fire, so it collapses like an empty chain.
            let mut term_word = u32::MAX;
            for ai in 0..arms.len() {
                if arms[ai].1[0] & SK_ARM_SAMPLED == 0 {
                    continue;
                }
                let (gs, ge) = groups[ai];
                let t = terms[gs..ge]
                    .iter()
                    .find(|t| t.name.contains("random"))
                    .expect("a flagged arm carries its sampled term");
                let kind = match t.name.as_str() {
                    "all_corner_random" => SK_TERM_ALL_CORNER,
                    "straight_random" => SK_TERM_STRAIGHT,
                    "down_slope_random" => SK_TERM_DOWN_SLOPE,
                    "up_slope_random" => SK_TERM_UP_SLOPE,
                    "phase_random" => SK_TERM_PHASE_RANDOM,
                    "phase_firsthalf_random" => SK_TERM_PHASE_FIRSTHALF,
                    "phase_laterhalf_random" => SK_TERM_PHASE_LATERHALF,
                    "is_finalcorner_random" => SK_TERM_FC_RANDOM,
                    "distance_rate_after_random" => SK_TERM_DIST_RATE_AFTER,
                    "corner_random" => SK_TERM_CORNER_RANDOM,
                    other => {
                        self.unsupported(format!("skill {sid} random term '{other}' is not ported"));
                        return;
                    }
                };
                let ok_arg = if t.op != 0 {
                    false
                } else if kind == SK_TERM_DIST_RATE_AFTER {
                    // The retail sampler scales the course distance by the
                    // argument percent, so any non-negative integer is valid.
                    t.arg >= 0
                } else if kind == SK_TERM_CORNER_RANDOM {
                    // The row resolution below only reads the argument in its
                    // i64 range, and the kernel word never uses it.
                    true
                } else if matches!(
                    kind,
                    SK_TERM_PHASE_RANDOM | SK_TERM_PHASE_FIRSTHALF | SK_TERM_PHASE_LATERHALF
                ) {
                    (0..=4).contains(&t.arg)
                } else {
                    t.arg == 1
                };
                if !ok_arg {
                    self.unsupported(format!(
                        "skill {sid} random term '{}' needs == with a supported argument, got {}{}",
                        t.name,
                        op_text(t.op),
                        t.arg
                    ));
                    return;
                }
                let corner_idx = match kind {
                    SK_TERM_FC_RANDOM => match fc_random_index(course, g.distance) {
                        Some(i) => i as i32,
                        None => {
                            arms[ai].1[0] = SK_ARM_DEAD;
                            continue;
                        }
                    },
                    SK_TERM_CORNER_RANDOM => match corner_random_row(course, t.arg) {
                        Err(why) => {
                            self.unsupported(format!("skill {sid}: {why}"));
                            return;
                        }
                        Ok(Some(i)) => i as i32,
                        Ok(None) => {
                            arms[ai].1[0] = SK_ARM_DEAD;
                            continue;
                        }
                    },
                    _ => -1,
                };
                if self.nterm >= SK_MMAX {
                    self.unsupported("skill table overflow: too many sampled terms");
                    return;
                }
                // Duplicate tags across one alternative's arms are faithful:
                // the paired mode reseeds a fresh substream per draw, so only
                // the shared solo bank's draw order matters.
                let (tag_off, tag_len) =
                    self.tag(&format!("T{sid}:{alt_index}:condition:{}", t.name));
                let at = SK_TERMS + SK_MSTRIDE * self.nterm;
                self.words[at] = kind;
                self.words[at + 1] = t.arg as u32;
                self.words[at + 2] = tag_off as u32;
                self.words[at + 3] = tag_len as u32;
                self.words[at + 4] = corner_idx as u32;
                self.nterm += 1;
                arms[ai].1[1] = (self.nterm - 1) as u32;
                if term_word == u32::MAX {
                    term_word = (self.nterm - 1) as u32;
                }
            }
            if !dead && !arms.is_empty() && arms.iter().all(|a| a.1[0] & SK_ARM_DEAD != 0) {
                dead = true;
            }

            let arms_base = self.narm;
            let arms_len;
            let mut arm0_prog = (0usize, 0usize);
            let mut arm0_pre = (0usize, 0usize);
            let mut arm0_post = (0usize, 0usize);
            if dead {
                // A dead record still spends its resolved random draws: the
                // engine samples before any region is evaluated, keeping the
                // shared stream aligned. The inert chains stay empty.
                for (_, flags, _) in arms.iter() {
                    let at = SK_ARMS + SK_ASTRIDE * self.narm;
                    if flags[0] & SK_ARM_SAMPLED != 0 {
                        self.words[at + 6] = SK_ARM_SAMPLED;
                        self.words[at + 7] = flags[1];
                    }
                    self.narm += 1;
                }
                arms_len = arms.len().max(1);
            } else {
                arms_len = arms.len();
                for (ai, (lists, flags, _)) in arms.iter().enumerate() {
                    let (gs, ge) = groups[ai];
                    let (n_pre, pre_b) = if flags[0] == SK_ARM_SAMPLED {
                        self.push_regions(&lists[0])
                    } else {
                        (0, self.nreg)
                    };
                    let (n_post, post_b) = self.push_regions(&lists[1]);
                    if self.reason.is_some() {
                        return;
                    }
                    let prog_base = self.nprog;
                    let prog_len = match emit_program(self, &terms[gs..ge], &leaves[gs..ge]) {
                        Ok(len) => len,
                        Err(why) => {
                            self.nprog = prog_base;
                            self.unsupported(format!("skill {sid}: {why}"));
                            return;
                        }
                    };
                    let at = SK_ARMS + SK_ASTRIDE * self.narm;
                    self.words[at] = prog_len as u32;
                    self.words[at + 1] = prog_base as u32;
                    self.words[at + 2] = n_pre as u32;
                    self.words[at + 3] = pre_b as u32;
                    self.words[at + 4] = n_post as u32;
                    self.words[at + 5] = post_b as u32;
                    self.words[at + 6] = flags[0];
                    self.words[at + 7] = flags[1];
                    self.narm += 1;
                    if ai == 0 {
                        arm0_prog = (prog_len, prog_base);
                        arm0_pre = (n_pre, pre_b);
                        arm0_post = (n_post, post_b);
                    }
                }
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
            // Retail folds the ms conversion in f32 with the rounded 0.0001
            // constant (race_solver_builder.rs:416-418 retail branch). The
            // classic else-branch scales the raw ms in plain f64 (:418 else).
            let base_ms = alt.get("baseDuration").and_then(Value::as_f64).unwrap_or(0.);
            let base_duration = if self.lane == Lane::Classic {
                base_ms / 10000.0
            } else {
                ((base_ms as f32) * 0.0001_f32) as f64
            };
            // The engine's `as f64 as i32` cast, so a wire value past i32 reads
            // the same tier on both sides (src/lib.rs:179-187). duration_usage
            // 4 extends the running instance on an overtake (race_solver.rs:2655).
            let duration_usage =
                alt.get("durationUsage").and_then(Value::as_f64).unwrap_or(1.0) as i32;
            if effect_list.len() > SK_EMAXSLOT {
                self.unsupported(format!(
                    "skill {sid} alternative has more than {} effects",
                    SK_EMAXSLOT
                ));
                return;
            }
            // Every term true with no precondition, random term or cooldown
            // fires once in the engine's constructor pass and is gone. Its
            // 4/5/29 bumps are constants the init inputs already consume.
            let cooldown = alt.get("cooldown").and_then(Value::as_f64).unwrap_or(0.) / 10000.0;
            let static_core = !dead
                && cooldown <= 0.0
                && term_word == u32::MAX
                && sampled_count == 0
                && groups.len() <= 1
                && pre_text.is_empty()
                && leaves.iter().all(|l| matches!(l, Leaf::True));
            let static_fire = static_core && !wisdom;
            // The lottery decides a wisdom-gated constructor fire, which the
            // host cannot precompute, so the kernel bakes its self bumps after
            // the roll. Kind 29 and non-1 value usages draw late and stay out.
            let gated_static = static_core
                && wisdom
                && effect_list.iter().all(|ef| {
                    let k = ef.get("type").and_then(Value::as_i64).unwrap_or(-1) as i32;
                    let t = ef.get("target").and_then(Value::as_i64).unwrap_or(1) as i32;
                    let vu = ef.get("valueUsage").and_then(Value::as_f64).unwrap_or(1.0) as i32;
                    matches!(k, 2 | 4 | 5) && t == 1 && vu == 1
                });
            // The random-activation candidate filter keeps a record only when
            // every effect has a non-negative duration and a type past
            // WisdomUp that is not AllStatusUp (race_solver.rs:2710-2711).
            // Mid-race guts and wisdom writes move the live fields the kernel
            // reads each tick, so a kind 4 or 5 fire needs no phase gate: the
            // constructor-pass fire (kernel init pass) applies the bump the
            // same way the engine's constructor pass does.
            let mut bumped = false;
            let mut selig = true;
            for ef in effect_list {
                let kind = ef.get("type").and_then(Value::as_i64).unwrap_or(-1) as i32;
                if !matches!(kind, 1 | 2 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14
                    | 21 | 22 | 27 | 28 | 29 | 31 | 35 | 36 | 37 | 38)
                {
                    let reason = match kind {
                        32 => "all-stat bump is a mid-race stat write with no kernel consumer"
                            .to_string(),
                        33 | 34 => {
                            "has no SkillType variant (retail builder errors, no-ffi degrades to Noop)"
                        }
                            .to_string(),
                        other => format!("type {other} is not ported"),
                    };
                    self.unsupported(format!("skill {sid} effect {reason}"));
                    return;
                }
                let target = ef.get("target").and_then(Value::as_i64).unwrap_or(1) as i32;
                if !(matches!(target, 1 | 4 | 9 | 10) || (target == 21 && kind == 13)) {
                    self.unsupported(format!(
                        "skill {sid} effect target {target} is not ported"
                    ));
                    return;
                }
                // The overtake extension of a duration_usage 4 fire walks the
                // aux entries by skill id, and the kernel keeps one presence
                // cell per runner for these kinds, with nothing to match.
                if duration_usage == 4 && matches!(kind, 11 | 12 | 35 | 38) {
                    self.unsupported(format!(
                        "skill {sid} duration_usage 4 extends type {kind}'s per-fire aux entry, \
                         which the kernel keeps as one presence cell per runner"
                    ));
                    return;
                }
                // The random-activation request is a self-side effect only;
                // delivery to another runner has no reservation owner.
                if matches!(kind, 36 | 37) && target != 1 {
                    self.unsupported(format!(
                        "skill {sid} random-activation effect targeting another runner is not ported"
                    ));
                    return;
                }
                // Cross delivery re-runs refresh_stat_effects on the receiver, and
                // SpeedUp, WisdomUp and GutsUp write its live field there. GutsUp
                // also makes the guts-derived powf terms live for the whole race.
                if target != 1 && kind == 4 && !dead {
                    self.live_guts = true;
                }
                // A follow-up lands on the instance's own turn, re-drawing its
                // targets, and registers one token with one value
                // (race_solver.rs:2479-2537). Only token kinds have a kernel path.
                let aa =
                    ef.get("additionalActivate").and_then(Value::as_f64).unwrap_or(0.0) as i32;
                if aa != 0 && !matches!(kind, 21 | 22 | 27 | 28 | 31) {
                    self.unsupported(format!(
                        "skill {sid} additionalActivate {aa} on type {kind}: a follow-up \
                         registers one token with one value, and type {kind} writes \
                         per-receiver state"
                    ));
                    return;
                }
                if aa != 0 && !(target == 1 || matches!(target, 4 | 9 | 10)) {
                    self.unsupported(format!(
                        "skill {sid} additionalActivate {aa} on type {kind} target {target}: \
                         the turn's fan has no kernel target model"
                    ));
                    return;
                }
                // A usage with no engine arm passes the modifier through: the
                // builder's fold matches 3..=7, 10, 12 and 13
                // (race_solver_builder.rs:915-928), activation_modifier 8, 9 and
                // 14 (race_solver.rs:296-312), the activation read 8, 9 and 11
                // (race_solver.rs:2485-2503). value_fold and sk_vu_factor match
                // the same tiers.
                let value_usage =
                    ef.get("valueUsage").and_then(Value::as_f64).unwrap_or(1.0) as i32;
                let raw = ef.get("modifier").and_then(Value::as_f64).unwrap_or(0.);
                let multiplier = match ef
                    .get("levelModifiers")
                    .and_then(Value::as_array)
                    .map(|a| a.get(level.max(1).min(10) as usize - 1).and_then(Value::as_f64))
                {
                    Some(Some(m)) => m,
                    _ => level_multiplier(kind, level),
                };
                let base_value = if self.lane == Lane::Classic {
                    classic_skill_value(raw, multiplier)
                } else {
                    retail_skill_value(raw, multiplier)
                };
                let modifier = self.value_fold(runner, horse, value_usage, base_value);
                // Mid-race guts fires move the live accumulator and field, and
                // every guts-derived powf term (the spurt maximum, the duel and
                // spot entry words) then recomputes from the field at read
                // time. A construction-time self fire stays a baked constant.
                if kind == 4 && !(static_fire && target == 1) && !dead {
                    self.live_guts = true;
                }
                if static_fire && target == 1 && matches!(kind, 4 | 5 | 29) {
                    self.bumps.push((runner, kind, modifier));
                    bumped = true;
                }
                let target_value =
                    ef.get("targetValue").and_then(Value::as_f64).unwrap_or(0.).max(0.0);
                let target_value = target_value.min(SK_RMAX as f64) as u32;
                if !(base_duration >= 0.0 && kind > 5 && kind != 32) {
                    selig = false;
                }
                let at = SK_EFFECTS + SK_ESTRIDE * self.neff;
                self.words[at] = kind as u32;
                self.words[at + 1] = target as u32;
                self.words[at + 2] = value_usage as u32;
                self.words[at + 3] = duration_usage as u32;
                self.words[at + 4] = target_value;
                self.words[at + 5] = aa as u32;
                let w = f64_words(modifier);
                self.words[at + 6] = w[0];
                self.words[at + 7] = w[1];
                let w = f64_words(base_duration);
                self.words[at + 8] = w[0];
                self.words[at + 9] = w[1];
                self.neff += 1;
            }
            let delay_mult = delay_factor(alt);
            // The engine's constructor pass cannot see field or retail-loop
            // state, so those closures report false at time zero and the record
            // first fires one tick later (retail engine, seed 1..3 probes).
            let init_defer = pre_init_defer || leaves.iter().any(|l| {
                matches!(leaf_feat(l), FEAT_ORDER | FEAT_ORDER_RATE | FEAT_NEAR | FEAT_OTIME
                    | FEAT_DDTOP | FEAT_CHGUP_END | FEAT_LANETYPE | FEAT_BHIN | FEAT_MOVELANE
                    | FEAT_BFRONT | FEAT_NEARIF | FEAT_VISN | FEAT_INBAND
                    | FEAT_BALL | FEAT_BFRONTB | FEAT_CHGUP_MID | FEAT_CHGUP_FC
                    | FEAT_NOTOP | FEAT_NOTOD | FEAT_SURR | FEAT_TEMPT_B
                    | FEAT_TEMPT_F | FEAT_RTEMP..=FEAT_RTEMP_END | FEAT_DDRATE | FEAT_DDTOPF
                    | FEAT_VREMAIN)
            });

            // retail_initialization rolls the wisdom lottery once per skill id;
            // the first record of the (runner, skill) group carries the roll and
            // followers inherit its enablement.
            let (wis_off, wis_len) =
                if wisdom && first_record { self.tag(&format!("W{sid}")) } else { (0, 0) };
            let at = SK_RECORDS + SK_RSTRIDE * self.nrec;
            self.words[at] = sid_num;
            self.words[at + 1] = runner as u32;
            self.words[at + 2] = (if retain { SK_FLAG_RETAIN } else { 0 })
                | (if wisdom && first_record { SK_FLAG_WISDOM } else { 0 })
                | (if wisdom && !first_record { SK_FLAG_WISDOM_FOLLOW } else { 0 })
                | (if term_word != u32::MAX && !dead { SK_FLAG_SAMPLED } else { 0 })
                | (if dead { SK_FLAG_DEAD } else { 0 })
                | (if anyact { SK_FLAG_ANYACT } else { 0 })
                | (if init_defer { SK_FLAG_INIT_DEFER } else { 0 })
                | (if green { SK_FLAG_GREEN } else { 0 })
                | (if alt_index == 0 { SK_FLAG_PRIMARY } else { 0 })
                | (if selig { SK_FLAG_RSELIG } else { 0 })
                | (if matches!(rarity, 2 | 6) { SK_FLAG_RANGOLD } else { 0 })
                | (if rarity == 1 { SK_FLAG_RANWHITE } else { 0 })
                | (if general { SK_FLAG_GENERAL } else { 0 })
                | (if bumped { SK_FLAG_BUMPED } else { 0 })
                | (if gated_static { SK_FLAG_INIT_BUMP | SK_FLAG_BUMPED } else { 0 });
            self.words[at + 3] = priority as u32;
            self.words[at + 4] = rarity;
            self.words[at + 5] = (self.neff - eff_base) as u32;
            self.words[at + 6] = eff_base as u32;
            self.words[at + 7] = arm0_prog.0 as u32;
            self.words[at + 8] = arm0_prog.1 as u32;
            self.words[at + 9] = if sampled_count == 1 && !dead { 1 } else { 0 };
            self.words[at + 10] = term_word;
            self.words[at + 11] = arm0_pre.0 as u32;
            self.words[at + 12] = arm0_pre.1 as u32;
            let w = f64_words(cooldown);
            self.words[at + 13] = w[0];
            self.words[at + 14] = w[1];
            self.words[at + 15] = wis_off as u32;
            self.words[at + 16] = wis_len as u32;
            self.words[at + 17] = delay_mult.to_bits();
            self.words[at + 18] = arm0_post.0 as u32;
            self.words[at + 19] = arm0_post.1 as u32;
            self.words[at + 20] = arms_len as u32;
            self.words[at + 21] = arms_base as u32;
            self.words[at + 22] = pre_n as u32;
            self.words[at + 23] = pre_reg_b as u32;
            self.words[at + 24] = pre_prog_len as u32;
            self.words[at + 25] = pre_prog_b as u32;
            self.words[at + 26] = alt_index as u32;
            self.nrec += 1;
            first_record = false;
        }
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

    fn pack_team_stats(&mut self, horses: &Value) {
        let mut totals: std::collections::BTreeMap<i64, [f64; 5]> =
            std::collections::BTreeMap::new();
        let mut teams: Vec<i64> = Vec::new();
        for h in horses.as_array().map(|a| a.iter()).unwrap_or_default() {
            let mood = horse_mood(h);
            let rank = horse_team_rank(h);
            let team = h.get("teamId").and_then(Value::as_f64).unwrap_or(0.0) as i64;
            let mut own = [0.0f64; 5];
            {
                let total = if team == 0 {
                    &mut own
                } else {
                    totals.entry(team).or_insert([0.0; 5])
                };
                for i in 0..5 {
                    total[i] = (total[i] as f32 + ref_ranked_base_stat(horse_raw_stat(h, i), mood, rank))
                        as f64;
                }
            }
            teams.push(team);
            self.team_stats.push(own);
        }
        for (i, team) in teams.iter().enumerate() {
            if *team != 0 {
                self.team_stats[i] = totals[team];
            }
        }
    }

    /// The classic runtime has no sampled-term sampler and its kernel reads
    /// only constant and course-region condition state, so a sampled or
    /// live-state leaf refuses the skill by name. Returns true when a reason
    /// was set.
    fn classic_refuses_leaf(&mut self, sid: &str, terms: &[Term], leaves: &[Leaf]) -> bool {
        for (t, leaf) in terms.iter().zip(leaves) {
            if matches!(leaf, Leaf::Sampled) {
                self.unsupported(format!(
                    "skill {sid} condition term '{}' is a sampled trigger: no sampler exists in the classic runtime",
                    t.name
                ));
                return true;
            }
            if matches!(leaf, Leaf::Dynamic { .. } | Leaf::RegionDyn(..)) {
                self.unsupported(format!(
                    "skill {sid} condition term '{}' reads live runtime state with no reader in the classic kernel",
                    t.name
                ));
                return true;
            }
        }
        false
    }

    /// `prepare()`'s value_usage tiers (race_solver_builder.rs:908-939),
    /// folded into the level-resolved modifier. Every compiled race runs with
    /// runtime_phase true (the host flags the rest), so the f32 round trip and
    /// the sentinel factors apply. Usages outside the fold tiers pass through
    /// for the kernel's runtime branches.
    fn value_fold(&self, runner: usize, horse: &Value, value_usage: i32, modifier: f64) -> f64 {
        if value_usage == 13 {
            let raw_max = (0..5).fold(0.0f64, |acc, i| acc.max(horse_raw_stat(horse, i)));
            let coef = if raw_max < 600.0 {
                0.800000011920929
            } else if raw_max < 800.0 {
                0.8999999761581421
            } else if raw_max < 1000.0 {
                1.0
            } else if raw_max < 1100.0 {
                1.100000023841858
            } else {
                1.2000000476837158
            };
            return (modifier * coef) as f32 as f64;
        }
        if !matches!(value_usage, 3..=7 | 10 | 12) {
            return modifier;
        }
        let (value, bounds) = match value_usage {
            3..=7 => {
                let own = || {
                    let mood = horse_mood(horse);
                    let rank = horse_team_rank(horse);
                    let mut stats = [0.0f64; 5];
                    for (i, stat) in stats.iter_mut().enumerate() {
                        *stat = ref_ranked_base_stat(horse_raw_stat(horse, i), mood, rank) as f64;
                    }
                    stats
                };
                let stats = self.team_stats.get(runner).copied().unwrap_or_else(own);
                (stats[(value_usage - 3) as usize], [1200.0, 1800.0, 2600.0, 3600.0])
            }
            10 => (
                horse.get("careerWins").and_then(Value::as_f64).unwrap_or(0.0).max(0.0),
                [6.0, 14.0, 18.0, 25.0],
            ),
            _ => (
                horse.get("fans").and_then(Value::as_f64).unwrap_or(0.0).max(0.0),
                [20000.0, 50000.0, 100000.0, 160000.0],
            ),
        };
        let tier = bounds.iter().filter(|b| **b <= value).count();
        let mut factor = match tier {
            0 => 0.8,
            1 => 0.9,
            2 => 1.0,
            3 => 1.1,
            _ => 1.2,
        };
        if (matches!(value_usage, 3..=7) && value >= 99999.0)
            || (value_usage == 12 && value >= 99999999.0)
        {
            factor = 1.0;
        }
        (modifier * factor) as f32 as f64
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

/// The classic (non-retail) `build_skill_effects` else-branch
/// (race_solver_builder.rs:415): the master coefficient scales the raw value
/// through the multiplier in plain f64, no f32 round trip.
fn classic_skill_value(raw: f64, multiplier: f64) -> f64 {
    raw / 10000.0 * multiplier
}

// Source-pinned mirrors of the vendored stat chain the team totals and the
// tier fallback read: race_solver_builder.rs adjust_overcap and
// team_rank_bonus, retail_initialization.rs initial_base_wisdom.
const STAT_KEYS: [&str; 5] = ["speed", "stamina", "power", "guts", "wisdom"];

fn horse_raw_stat(horse: &Value, i: usize) -> f64 {
    horse.get(STAT_KEYS[i]).and_then(Value::as_f64).unwrap_or(0.0)
}

fn horse_mood(horse: &Value) -> i32 {
    horse.get("mood").and_then(Value::as_f64).map(|n| n as i32).unwrap_or(0)
}

fn horse_team_rank(horse: &Value) -> i32 {
    horse.get("teamRank").and_then(Value::as_f64).unwrap_or(0.0) as i32
}

fn ref_adjust_overcap(stat: f64) -> f64 {
    if stat > 1200.0 {
        1200.0 + (stat - 1200.0) / 2.0
    } else {
        stat
    }
}

fn ref_team_rank_bonus(rank: i32) -> f32 {
    if rank <= 0 {
        return 0.0;
    }
    for (threshold, bonus) in [(6, 50.0f32), (11, 30.0), (16, 20.0), (21, 10.0), (26, 0.0), (99999, 0.0)]
    {
        if rank < threshold {
            return bonus;
        }
    }
    0.0
}

fn ref_ranked_base_stat(raw: f64, mood: i32, rank: i32) -> f32 {
    let base = ref_adjust_overcap(raw) as f32 + ref_team_rank_bonus(rank);
    let coefficient = (10000 + 200 * mood.clamp(-2, 2)) as f32 * 0.0001_f32;
    (base * coefficient).clamp(1.0, 2000.0)
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
        while at < bytes.len()
            && (bytes[at].is_ascii_lowercase() || bytes[at] == b'_' || bytes[at].is_ascii_digit())
        {
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
    /// A static region gate combined with a per-tick live comparison, as the
    /// engine's live_with_static and lastspurt filters build them.
    RegionDyn(Vec<(f64, f64)>, u32),
    Dynamic { feat: u32, cmp: i64, arg: Option<f64> },
    Sampled,
}

/// The per-tick feature id a leaf compares against, if any.
fn leaf_feat(l: &Leaf) -> u32 {
    match l {
        Leaf::Dynamic { feat, .. } => *feat,
        Leaf::RegionDyn(_, feat) => *feat,
        _ => u32::MAX,
    }
}

/// Course and runner constants the engine resolves once at trigger build.
struct Statics {
    distance_type: i64,
    turn: i64,
    strategy: i64,
    ground_type: i64,
    track_id: i64,
    popularity: f64,
    ground_condition: Option<f64>,
    weather: Option<f64>,
    season: Option<f64>,
    is_dirtgrade: bool,
    other_style: [f64; 4],
    // Wave-3 roster folds. same_style counts the base-style group including
    // self, skill_count counts horses holding the record's own skill id.
    same_style: Option<f64>,
    n_runners: i64,
    favorite_matches: Option<bool>,
    skill_count: f64,
    skill_ids: Vec<String>,
    // The record's own skill id, the target of is_activate_other_skill_detail.
    sid: f64,
    // Nonzero teams make the opponent counts differ from their plain ones.
    any_team: bool,
    // Wave-9c request folds. v_grade and v_time read the RaceParameters enums;
    // absent strings take the spec defaults "g1" and "midday" (lib.rs:284-285),
    // present unparsable ones fail the engine's own parse, so they refuse.
    grade: Option<f64>,
    race_time: Option<f64>,
    // v_motivation reads the per-runner mood clamped to [-2,2] (lib.rs:505).
    motivation: f64,
    // The runtime_phase trigger base stats are ranked stats truncated to i32
    // (race_solver_builder.rs:872-881), so v_base_power folds per runner.
    base_power: f64,
    // post_number compares the client gate block for the wire gateIndex
    // (default roster index, lib.rs:311) over the roster size.
    post_number: f64,
    // The viewer row remain_distance_viewer_id reads, defaulting to 0. An
    // out-of-range or non-integer present value fails the engine's spec
    // parse (lib.rs:292-300), so it refuses at leaf use.
    viewer: Option<i64>,
}

impl Statics {
    fn new(course: &Value, horse: &Value, root: &Value, sid: &str, runner: usize) -> Statics {
        let track_id = course.get("raceTrackId").and_then(Value::as_i64).unwrap_or(0);
        // The retail builder counts base styles once for the roster
        // (lib.rs:393-401) and hands each runner those counts minus itself
        // (lib.rs:536-538), so every otherself fold is a per-runner constant.
        let own = style_base_index(strategy_code(
            horse.get("strategy").and_then(Value::as_str).unwrap_or(""),
        ));
        let mut counts = [0i64; 4];
        // PreparedRoster::new folds the rest: one entry per horse, the
        // popularity==1 favorite (lib.rs:387-391) and one count per skill id
        // over the per-horse deduplicated skill lists (lib.rs:402-403).
        let mut n_runners = 0i64;
        let mut favorite_style: Option<i64> = None;
        let mut favorite_ambiguous = false;
        let mut skill_ids: Vec<String> = Vec::new();
        let mut skill_count = 0i64;
        let mut any_team = false;
        if let Some(list) = root.get("horses").and_then(Value::as_array) {
            for other in list {
                n_runners += 1;
                let code = strategy_code(other.get("strategy").and_then(Value::as_str).unwrap_or(""));
                if let Some(i) = style_base_index(code) {
                    counts[i] += 1;
                }
                if other.get("popularity").and_then(Value::as_i64).unwrap_or(0) == 1 {
                    if favorite_style.is_some() {
                        favorite_ambiguous = true;
                    }
                    favorite_style = Some(code);
                }
                if other.get("teamId").and_then(Value::as_f64).unwrap_or(0.0) as i64 != 0 {
                    any_team = true;
                }
                let mut holds = false;
                if let Some(sk) = other.get("skills").and_then(Value::as_array) {
                    for entry in sk {
                        // Same two entry shapes the wire pass accepts.
                        let Some((s, _)) = normalize_skill_entry(entry) else { continue };
                        if s == sid && !holds {
                            holds = true;
                            skill_count += 1;
                        }
                        if !skill_ids.iter().any(|x| *x == s) {
                            skill_ids.push(s);
                        }
                    }
                }
            }
        }
        let favorite_matches = match favorite_style {
            Some(f) if !favorite_ambiguous => Some(style_matches(
                strategy_code(horse.get("strategy").and_then(Value::as_str).unwrap_or("")),
                f,
            )),
            _ => None,
        };
        let mut other_style = [0.0f64; 4];
        for i in 0..4 {
            other_style[i] = (counts[i] - i64::from(own == Some(i))) as f64;
        }
        Statics {
            distance_type: course.get("distanceType").and_then(Value::as_i64).unwrap_or(0),
            turn: course.get("turn").and_then(Value::as_i64).unwrap_or(0),
            strategy: strategy_code(horse.get("strategy").and_then(Value::as_str).unwrap_or("")),
            ground_type: course.get("surface").and_then(Value::as_i64).unwrap_or(0),
            track_id,
            // v_popularity: HorseParameters.popularity, i32, missing means 0.
            popularity: horse.get("popularity").and_then(Value::as_i64).unwrap_or(0) as i32 as f64,
            ground_condition: enum_code(
                root.get("groundCondition").and_then(Value::as_str),
                ground_condition_code,
            ),
            weather: enum_code(root.get("weather").and_then(Value::as_str), weather_code),
            season: enum_code(root.get("season").and_then(Value::as_str), season_code),
            // is_dirtgrade_filter_eq: the request flag wins, else the fixed
            // dirt-grade track set (activation_conditions.rs:470,735).
            is_dirtgrade: match root.get("isDirtGrade") {
                Some(v) => match v.as_bool() {
                    Some(b) => b,
                    None => is_dirtgrade_track(track_id),
                },
                None => is_dirtgrade_track(track_id),
            },
            other_style,
            same_style: own.map(|i| counts[i] as f64),
            n_runners,
            favorite_matches,
            skill_count: skill_count as f64,
            skill_ids,
            sid: sid.parse::<u32>().unwrap_or(0) as f64,
            any_team,
            grade: enum_code_or_default(root.get("grade").and_then(Value::as_str), grade_code, 100.0),
            race_time: enum_code_or_default(root.get("time").and_then(Value::as_str), time_code, 2.0),
            motivation: (horse
                .get("mood")
                .and_then(Value::as_f64)
                .map(|n| n as i32)
                .unwrap_or(0)
                .max(-2)
                .min(2)
                + 3) as f64,
            // The runtime_phase builder ranks the wire power and truncates the
            // result to i32 (race_solver_builder.rs:872-881).
            base_power: ref_ranked_base_stat(
                horse_raw_stat(horse, 2),
                horse.get("mood").and_then(Value::as_f64).map(|n| n as i32).unwrap_or(0),
                horse_team_rank(horse),
            ) as i32 as f64,
            post_number: gate_block(
                horse
                    .get("gateIndex")
                    .and_then(Value::as_f64)
                    .map(|n| n as i32)
                    .unwrap_or(runner as i32),
                if n_runners == 0 { 9 } else { n_runners },
            ) as f64,
            viewer: match root.get("playerHorseIndex") {
                None => Some(0),
                Some(v) => v
                    .as_f64()
                    .filter(|x| {
                        let i = *x as i64;
                        *x == i as f64 && i >= 0 && i < n_runners.max(1)
                    })
                    .map(|x| x as i64),
            },
        }
    }
}

/// The client gate block of a wire gate_index, mirroring the engine's
/// post_number filters (activation_conditions.rs:1002-1039). Gates outside the
/// field report block 1.
fn gate_block(gate: i32, num_umas: i64) -> i64 {
    if num_umas < 1 || gate < 0 || (gate as i64) >= num_umas {
        return 1;
    }
    let gate = gate as i64;
    if num_umas <= 8 {
        return gate + 1;
    }
    let per_post = num_umas / 8;
    let small_posts = 8 - num_umas % 8;
    let cutoff = small_posts * per_post;
    if gate < cutoff {
        gate / per_post + 1
    } else {
        small_posts + (gate - cutoff) / (per_post + 1) + 1
    }
}

/// An enum-coded request string with the spec's own default when the key is
/// absent, like the engine's grade and time spec defaults (lib.rs:284-285).
fn enum_code_or_default(
    text: Option<&str>,
    code: fn(&str) -> Option<i64>,
    default: f64,
) -> Option<f64> {
    match text {
        None => Some(default),
        Some(t) => code(t).map(|c| c as f64),
    }
}

/// The race_parameters.rs parse_grade discriminants.
fn grade_code(text: &str) -> Option<i64> {
    match text.to_ascii_uppercase().as_str() {
        "GRADE_0" => Some(0),
        "GRADE_1000" => Some(1000),
        "PRE-OP" | "PREOP" | "PRE_OP" => Some(700),
        "G1" => Some(100),
        "G2" => Some(200),
        "G3" => Some(300),
        "OP" => Some(400),
        "MAIDEN" => Some(800),
        "DEBUT" => Some(900),
        "DAILY" => Some(999),
        _ => None,
    }
}

/// The race_parameters.rs parse_time discriminants. An empty or unparsable
/// non-absent string reports None, refusing the construct the way the
/// engine's parse fails the race.
fn time_code(text: &str) -> Option<i64> {
    match text.to_ascii_uppercase().as_str() {
        "NONE" | "NOTIME" | "NO_TIME" => Some(0),
        "MORNING" => Some(1),
        "MIDDAY" => Some(2),
        "EVENING" => Some(3),
        "NIGHT" => Some(4),
        _ => None,
    }
}

/// The retail corner_random sampler's row for a request argument
/// (retail_initialization.rs:135-166). Any corner carrying an "index" makes
/// every row parse it, and an unparsable one fails the engine's init, so it
/// reports Err. Without metadata the argument is the legacy offset -5 from
/// the row count. Ok(None) means no row, the arm never fires and no draw.
fn corner_random_row(course: &Value, arg: i64) -> Result<Option<usize>, String> {
    let corners = course.get("corners").and_then(Value::as_array);
    let Some(rows) = corners else {
        return Ok(None);
    };
    if arg <= 0 {
        return Ok(None);
    }
    if rows.iter().any(|c| c.get("index").is_some()) {
        let mut found: Option<usize> = None;
        for (i, c) in rows.iter().enumerate() {
            let Some(v) = c.get("index").and_then(Value::as_i64) else {
                return Err(format!(
                    "corner index '{}' is not an integer, the engine's init fails",
                    c.get("index").and_then(Value::as_str).unwrap_or("")
                ));
            };
            if v == arg {
                found = Some(i);
            }
        }
        return Ok(found);
    }
    let legacy = rows.len() as i64 + arg - 5;
    Ok(if legacy >= 0 && (legacy as usize) < rows.len() {
        Some(legacy as usize)
    } else {
        None
    })
}

/// Index into the roster's four base-style counts. The retail builder folds
/// Oonige into Nige, and an unparsable strategy counts nowhere.
fn style_base_index(code: i64) -> Option<usize> {
    match code {
        1 | 5 => Some(0),
        2 => Some(1),
        3 => Some(2),
        4 => Some(3),
        _ => None,
    }
}

fn is_dirtgrade_track(track_id: i64) -> bool {
    matches!(track_id, 10101 | 10103 | 10104 | 10105)
}

/// An enum-coded request string, folded once so a missing or unparsable value
/// refuses the construct instead of guessing (the engine's own parse fails
/// the race on the CPU side too).
fn enum_code(text: Option<&str>, code: fn(&str) -> Option<i64>) -> Option<f64> {
    code(text.unwrap_or("")).map(|c| c as f64)
}

/// The race_parameters.rs enum discriminants, parsed the uppercase-case-
/// insensitive way the engine's parse_* helpers do.
fn ground_condition_code(text: &str) -> Option<i64> {
    match text.to_ascii_uppercase().as_str() {
        "GOOD" => Some(1),
        "YIELDING" => Some(2),
        "SOFT" => Some(3),
        "HEAVY" => Some(4),
        _ => None,
    }
}

fn weather_code(text: &str) -> Option<i64> {
    match text.to_ascii_uppercase().as_str() {
        "SUNNY" => Some(1),
        "CLOUDY" => Some(2),
        "RAINY" => Some(3),
        "SNOWY" => Some(4),
        _ => None,
    }
}

fn season_code(text: &str) -> Option<i64> {
    match text.to_ascii_uppercase().as_str() {
        "SPRING" => Some(1),
        "SUMMER" => Some(2),
        "AUTUMN" => Some(3),
        "WINTER" => Some(4),
        "SAKURA" => Some(5),
        _ => None,
    }
}

fn strategy_code(text: &str) -> i64 {
    match text.to_ascii_uppercase().as_str() {
        "NIGE" => 1,
        "SENKOU" => 2,
        "SASI" | "SASHI" => 3,
        "OIKOMI" => 4,
        "OONIGE" => 5,
        _ => 0,
    }
}

fn dyn_leaf(feat: u32, t: &Term) -> Leaf {
    Leaf::Dynamic { feat, cmp: t.op, arg: None }
}

fn fold(value: f64, t: &Term) -> Leaf {
    fold2(value, t.arg as f64, t.op)
}

/// Compare a folded value against a folded threshold with the term's
/// comparator, the engine's roster and rate filters resolved at build time.
fn fold2(value: f64, threshold: f64, op: i64) -> Leaf {
    let ok = match op {
        0 => value == threshold,
        1 => value != threshold,
        2 => value < threshold,
        3 => value <= threshold,
        4 => value > threshold,
        _ => value >= threshold,
    };
    if ok {
        Leaf::True
    } else {
        Leaf::False
    }
}

fn style_matches(a: i64, b: i64) -> bool {
    a == b || (a == 1 && b == 5) || (a == 5 && b == 1)
}

/// The retail distance_rate comparison runs against an integer threshold:
/// f32 percent of the course distance rounded through a +0.5 truncation.
fn dist_rate_threshold(arg: i64, distance: f64) -> f64 {
    (((arg as f32 / 100.0) * distance as f32 + 0.5f32) as i32) as f64
}

/// The running_style_count_same_rate threshold: the client 0x35c7d30 rounds
/// the percentage of the runner count to a whole through a +0.5 truncation.
fn rate_count_threshold(arg: f64, n: i64) -> f64 {
    (((arg as f32 / 100.0f32) * n as f32 + 0.5f32).trunc()) as f64
}

/// The corner row is_finalcorner_random draws: reverse scan, ids default to
/// position-from-the-last-corner, and only rows numbered 4 or higher whose
/// start sits within the last 1000 m qualify.
fn fc_random_index(course: &Value, distance: f64) -> Option<usize> {
    let rows = course.get("corners").and_then(Value::as_array)?;
    let n = rows.len();
    for (i, row) in rows.iter().enumerate().rev() {
        let id = match row.get("index").and_then(Value::as_i64) {
            Some(v) => v,
            None => i as i64 + 5 - n as i64,
        };
        let start = row.get("start").and_then(Value::as_f64)?;
        if id >= 4 && distance - start <= 1000.0 {
            return Some(i);
        }
    }
    None
}

/// One comparison's engine role. Region-algebra terms the retail runtime
/// swaps to live values lower to pass-through leaves with the runtime
/// feature, and course or runner constants fold at compile time.
fn classify(t: &Term, g: &Geometry, s: &Statics) -> Result<Leaf, String> {
    let d = g.distance;
    match t.name.as_str() {
        "always" => Ok(Leaf::True),
        "accumulatetime" => Ok(dyn_leaf(FEAT_CLOCK, t)),
        "activate_count_start" if t.op == 5 => Ok(dyn_leaf(FEAT_ACT0, t)),
        "activate_count_middle" if t.op == 5 => Ok(dyn_leaf(FEAT_ACT1, t)),
        // The retail filter table only registers >= for the heal count and >
        // for the duel count, so other comparators stay refusals.
        "activate_count_heal" if t.op == 5 => Ok(dyn_leaf(FEAT_HEAL, t)),
        "activate_count_all" if matches!(t.op, 3 | 5) => Ok(dyn_leaf(FEAT_ACTALL, t)),
        "activate_count_end_after" if t.op == 5 => Ok(dyn_leaf(FEAT_ACT2, t)),
        "activate_count_later_half" if t.op == 5 => Ok(dyn_leaf(FEAT_ACTLH, t)),
        // The runtime_phase builder ranks the wire power and truncates to i32,
        // so v_base_power is a per-runner constant (activation_conditions.rs:1233
        // reading the trigger horse built at race_solver_builder.rs:872-881).
        "base_power" => Ok(fold(s.base_power, t)),
        "bashin_diff_behind" => Ok(dyn_leaf(FEAT_BASH_BEHIND, t)),
        "bashin_diff_infront" => Ok(dyn_leaf(FEAT_BASH_INFRONT, t)),
        "behind_near_lane_time" => Ok(dyn_leaf(FEAT_BHNEAR, t)),
        "behind_near_lane_time_set1" => Ok(dyn_leaf(FEAT_BHNEAR1, t)),
        "blocked_all_continuetime" => Ok(dyn_leaf(FEAT_BALL, t)),
        "blocked_front" => Ok(dyn_leaf(FEAT_BFRONTB, t)),
        "blocked_front_continuetime" => Ok(dyn_leaf(FEAT_BFRONT, t)),
        "blocked_side_continuetime" => Ok(dyn_leaf(FEAT_BSIDE, t)),
        "change_order_onetime" => Ok(dyn_leaf(FEAT_ORDER_DELTA, t)),
        "change_order_up_end_after" => Ok(dyn_leaf(FEAT_CHGUP_END, t)),
        "change_order_up_finalcorner_after" => Ok(dyn_leaf(FEAT_CHGUP_FC, t)),
        "change_order_up_middle" => Ok(dyn_leaf(FEAT_CHGUP_MID, t)),
        "compete_fight_count" if t.op == 4 => Ok(dyn_leaf(FEAT_FIGHT, t)),
        "corner" => Ok(dyn_leaf(FEAT_CORNER, t)),
        // The retail sampler resolves the request argument to one corner row
        // at init (retail_initialization.rs:135-166) and draws one distance
        // window there, so the leaf is a sampled term like the other randoms.
        "corner_random" if t.op == 0 => Ok(Leaf::Sampled),
        "distance_diff_rate" => Ok(dyn_leaf(FEAT_DDRATE, t)),
        "distance_diff_top" => Ok(dyn_leaf(FEAT_DDTOP, t)),
        "distance_diff_top_float" => Ok(dyn_leaf(FEAT_DDTOPF, t)),
        "distance_rate" => Ok(Leaf::Dynamic {
            feat: FEAT_DIST_RATE,
            cmp: t.op,
            arg: Some(dist_rate_threshold(t.arg, d)),
        }),
        "distance_type" if matches!(t.op, 0 | 1) => Ok(fold(s.distance_type as f64, t)),
        "grade" => match s.grade {
            Some(v) => Ok(fold(v, t)),
            None => Err("'grade' needs a parsable grade in the request".into()),
        },
        "ground_condition" => match s.ground_condition {
            Some(v) => Ok(fold(v, t)),
            None => Err("'ground_condition' needs a parsable groundCondition".into()),
        },
        // value_filter fields: course or request constants, folded to True/False
        // exactly like the engine's static filter emptying the region list.
        "ground_type" => Ok(fold(s.ground_type as f64, t)),
        "hp_per" if matches!(t.op, 3 | 5) => Ok(dyn_leaf(FEAT_HP_PER, t)),
        "infront_near_lane_time" => Ok(dyn_leaf(FEAT_IFNEAR, t)),
        // v75 retail observe: per-visit changed flag, see sk_feat_value.
        "is_activate_any_skill" if t.op == 0 && t.arg == 1 => Ok(dyn_leaf(FEAT_ANYACT, t)),
        // The engine compares its heal counter boolean against the argument:
        // 1 means at least one heal, any other argument means no heal.
        "is_activate_heal_skill" if t.op == 0 => Ok(if t.arg == 1 {
            Leaf::Dynamic { feat: FEAT_HEAL, cmp: 5, arg: Some(1.0) }
        } else {
            Leaf::Dynamic { feat: FEAT_HEAL, cmp: 0, arg: Some(0.0) }
        }),
        // The engine checks its own skill id in used_skills, and used_skills
        // takes only self-fired ids. The record's first fire is the only way
        // to set the bit, and the same gate blocks that fire, so the check
        // is false forever and the record folds dead.
        "is_activate_other_skill_detail" if t.op == 0 && t.arg == 1 => Ok(Leaf::False),
        "is_surrounded" => Ok(dyn_leaf(FEAT_SURR, t)),
        // used_skills receives exactly the ids the runner self-fired from the
        // table. An id the roster never carries can never be in the set.
        "is_used_skill_id" if t.op == 0 => {
            let want = format!("{}", t.arg);
            if s.skill_ids.iter().any(|x| *x == want) {
                Ok(Leaf::Dynamic { feat: FEAT_USED, cmp: 0, arg: Some(t.arg as f64) })
            } else {
                Ok(Leaf::False)
            }
        }
        "is_behind_in" => Ok(dyn_leaf(FEAT_BHIN, t)),
        "is_dirtgrade" if matches!(t.op, 0 | 1) => {
            Ok(fold(if s.is_dirtgrade { 1.0 } else { 0.0 }, t))
        }
        "is_finalcorner" => Ok(dyn_leaf(FEAT_FINALE, t)),
        "is_finalcorner_laterhalf" => Ok(dyn_leaf(FEAT_FC_HALF, t)),
        "is_last_straight" if t.op == 0 && t.arg == 1 => match g.straights.last() {
            Some((a, b)) => Ok(Leaf::Region(vec![(*a, *b)])),
            None => Err("is_last_straight needs a course with straights".into()),
        },
        "is_last_straight_onetime" if t.op == 0 && t.arg == 1 => match g.straights.last() {
            Some((a, _)) => Ok(Leaf::RegionDyn(vec![(*a, *a + 10.0)], FEAT_LS_ENTRY)),
            None => Err("is_last_straight_onetime needs a course with straights".into()),
        },
        "is_lastspurt" if t.op == 0 && t.arg == 1 => {
            Ok(Leaf::RegionDyn(vec![(phase_start(d, 2), d)], FEAT_LASTSPURT))
        }
        "is_move_lane" => Ok(dyn_leaf(FEAT_MOVELANE, t)),
        // The engine swaps a field-wide set of previous-tick (runner, kind)
        // advantage pairs with same-team exclusion at every field update
        // (race_field.rs:373-398), which no per-runner kernel flag mirrors.
        "is_other_character_activate_advantage_skill" => Err(
            "'is_other_character_activate_advantage_skill' needs the field's team-filtered \
             previous-tick advantage pair set".into(),
        ),
        "is_overtake" => Ok(dyn_leaf(FEAT_IS_OVERTAKE, t)),
        "is_temptation" if t.op == 0 => Ok(dyn_leaf(FEAT_IS_TEMPTATION, t)),
        "lane_type" => Ok(dyn_leaf(FEAT_LANETYPE, t)),
        "motivation" => Ok(fold(s.motivation, t)),
        "near_count" => Ok(dyn_leaf(FEAT_NEAR, t)),
        "near_infront_count" => Ok(dyn_leaf(FEAT_NEARIF, t)),
        "order" => Ok(dyn_leaf(FEAT_ORDER, t)),
        "order_rate" => Ok(dyn_leaf(FEAT_ORDER_RATE, t)),
        name if name.starts_with("order_rate_in") && name.ends_with("_continue") => {
            // order_in_filter (activation_conditions.rs:1604-1607): the retail
            // check reads the in-band cell track_order_bands updates on every
            // rank pass, and accepts only ==1 of its set state.
            let rate = &name["order_rate_in".len()..name.len() - "_continue".len()];
            match (rate.parse::<i64>(), t.op, t.arg) {
                (Ok(r @ (20 | 40 | 50 | 80)), 0, 1) => Ok(Leaf::Dynamic {
                    feat: FEAT_INBAND,
                    cmp: 0,
                    arg: Some(r as f64),
                }),
                _ => Err(format!("'{name}'{}{} is not ported", op_text(t.op), t.arg)),
            }
        }
        name if name.starts_with("order_rate_out") && name.ends_with("_continue") => {
            // order_out_filter (activation_conditions.rs:1608-1611): the rate
            // in the name picks the band cell the retail check reads, and the
            // filter only accepts ==1 of that cell's set state.
            let rate = &name["order_rate_out".len()..name.len() - "_continue".len()];
            match (rate.parse::<i64>(), t.op, t.arg) {
                (Ok(r @ (20 | 40 | 50 | 70)), 0, 1) => Ok(Leaf::Dynamic {
                    feat: FEAT_OUTBAND,
                    cmp: 0,
                    arg: Some(r as f64),
                }),
                _ => Err(format!("'{name}'{}{} is not ported", op_text(t.op), t.arg)),
            }
        }
        "overtake_target_time" => Ok(dyn_leaf(FEAT_OTIME, t)),
        "phase" => Ok(dyn_leaf(FEAT_RPHASE, t)),
        "popularity" => Ok(fold(s.popularity, t)),
        // The registry only registers eq, lte and gte immediate filters
        // (activation_conditions.rs:1020-1039, 1644-1648), and any other
        // comparator fails the engine's build too.
        "post_number" if matches!(t.op, 0 | 3 | 5) => Ok(fold(s.post_number, t)),
        "post_number" => Err("'post_number' only registers == <= and >=".into()),
        "remain_distance" => Ok(dyn_leaf(FEAT_REMAIN, t)),
        // The live check always resolves a viewer row (the spec default is 0),
        // so this is the engine's remain_distance for that runner's position.
        "remain_distance_viewer_id" => match s.viewer {
            Some(_) => Ok(dyn_leaf(FEAT_VREMAIN, t)),
            None => Err("'remain_distance_viewer_id' needs an in-range playerHorseIndex".into()),
        },
        "rotation" => Ok(fold(s.turn as f64, t)),
        "running_style" if t.op == 0 => {
            Ok(if style_matches(s.strategy, t.arg) { Leaf::True } else { Leaf::False })
        }
        "running_style_count_nige_otherself" => Ok(fold(s.other_style[0], t)),
        "running_style_count_senko_otherself" => Ok(fold(s.other_style[1], t)),
        "running_style_count_sashi_otherself" => Ok(fold(s.other_style[2], t)),
        "running_style_count_oikomi_otherself" => Ok(fold(s.other_style[3], t)),
        "running_style_count_same" => match s.same_style {
            Some(v) => Ok(fold(v, t)),
            None => Err("'running_style_count_same' needs a parsable strategy".into()),
        },
        "running_style_count_same_rate" => match s.same_style {
            Some(v) if s.n_runners > 0 => Ok(fold2(v, rate_count_threshold(t.arg as f64, s.n_runners), t.op)),
            Some(_) => Ok(Leaf::False),
            None => Err("'running_style_count_same_rate' needs a parsable strategy".into()),
        },
        "running_style_equal_popularity_one" => match s.favorite_matches {
            Some(v) => Ok(fold(if v { 1.0 } else { 0.0 }, t)),
            None => Ok(Leaf::False),
        },
        name if name.starts_with("running_style_temptation_count_")
            || name.starts_with("running_style_temptation_opponent_count_") =>
        {
            // The opponent variants drop same-team runners, an identity map
            // only while every roster team is zero.
            let opponents = name.contains("_opponent_");
            let group = match name.rsplit('_').next() {
                Some("nige") => Some(0),
                Some("senko") => Some(1),
                Some("sashi") => Some(2),
                Some("oikomi") => Some(3),
                _ => None,
            };
            match group {
                Some(g) if !opponents || !s.any_team => Ok(Leaf::Dynamic {
                    feat: if opponents { FEAT_RTEMP_OPP } else { FEAT_RTEMP } + g,
                    cmp: t.op,
                    arg: None,
                }),
                _ => Err(format!("'{name}'{}{} is not ported", op_text(t.op), t.arg)),
            }
        }
        "same_skill_horse_count" => Ok(fold(s.skill_count, t)),
        "season" => match s.season {
            Some(v) => Ok(fold(v, t)),
            None => Err("'season' needs a parsable season in the request".into()),
        },
        "slope" => Ok(dyn_leaf(FEAT_SLOPE, t)),
        "temptation_count" if t.op == 0 => Ok(dyn_leaf(FEAT_TEMPT, t)),
        name if name == "temptation_count_behind" || name == "temptation_count_infront"
            || name == "temptation_opponent_count_behind"
            || name == "temptation_opponent_count_infront" =>
        {
            // is_opponent keeps every runner an opponent only while every
            // roster team is zero, so the opponent pair needs that gate.
            let opponents = name.contains("opponent");
            if opponents && s.any_team {
                Err(format!(
                    "'{name}'{}{} needs a roster with every team at zero",
                    op_text(t.op),
                    t.arg
                ))
            } else {
                let feat = if name.ends_with("behind") { FEAT_TEMPT_B } else { FEAT_TEMPT_F };
                Ok(dyn_leaf(feat, t))
            }
        }
        "overtake_target_no_order_up_time" => Ok(dyn_leaf(FEAT_NOTOP, t)),
        "overtake_target_no_order_down_time" => Ok(dyn_leaf(FEAT_NOTOD, t)),
        "time" => match s.race_time {
            Some(v) => Ok(fold(v, t)),
            None => Err("'time' needs a parsable time in the request".into()),
        },
        "track_id" => Ok(fold(s.track_id as f64, t)),
        "visiblehorse" => Ok(dyn_leaf(FEAT_VISN, t)),
        "weather" => match s.weather {
            Some(v) => Ok(fold(v, t)),
            None => Err("'weather' needs a parsable weather in the request".into()),
        },
        "all_corner_random" | "straight_random" | "down_slope_random"
        | "up_slope_random"
            if t.op == 0 && t.arg == 1 =>
        {
            Ok(Leaf::Sampled)
        }
        "phase_random" | "phase_firsthalf_random" | "phase_laterhalf_random"
            if t.op == 0 && (0..=4).contains(&t.arg) =>
        {
            Ok(Leaf::Sampled)
        }
        "is_finalcorner_random" if t.op == 0 && t.arg == 1 => Ok(Leaf::Sampled),
        "distance_rate_after_random" if t.op == 0 && t.arg >= 0 => Ok(Leaf::Sampled),
        other => Err(format!(
            "'{other}'{}{} is not ported",
            op_text(t.op),
            t.arg
        )),
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

/// Emits one arm's postfix program: every term in parse order ANDed. The
/// postfix form pushes each operand then folds with AND, so a stack machine
/// accumulates left to right. The '@' split is structural in the arms table,
/// so no OR marker appears here.
fn emit_program(c: &mut Compiler, terms: &[Term], leaves: &[Leaf]) -> Result<usize, String> {
    let base = c.nprog;
    let mut used = 0usize;
    for j in 0..terms.len() {
        if base + used >= SK_PMAX {
            return Err("condition program overflow".into());
        }
        emit_leaf(c, base + used, &terms[j], &leaves[j])?;
        used += 1;
        if j > 0 {
            if base + used >= SK_PMAX {
                return Err("condition program overflow".into());
            }
            c.words[SK_PROGRAMS + SK_PSTRIDE * (base + used)] = OP_AND;
            used += 1;
        }
    }
    c.nprog += used;
    Ok(used)
}

fn emit_leaf(c: &mut Compiler, slot: usize, t: &Term, leaf: &Leaf) -> Result<(), String> {
    let at = SK_PROGRAMS + SK_PSTRIDE * slot;
    match leaf {
        Leaf::Dynamic { feat, cmp, arg } => {
            c.words[at] = OP_FEAT_I;
            c.words[at + 1] = feat * 8 + *cmp as u32;
            let w = f64_words(arg.unwrap_or(t.arg as f64));
            c.words[at + 2] = w[0];
            c.words[at + 3] = w[1];
        }
        Leaf::RegionDyn(_, feat) => {
            c.words[at] = OP_FEAT_I;
            c.words[at + 1] = feat * 8 + t.op as u32;
            let w = f64_words(t.arg as f64);
            c.words[at + 2] = w[0];
            c.words[at + 3] = w[1];
        }
        _ => {
            c.words[at] = OP_TRUE;
            c.words[at + 1] = if matches!(leaf, Leaf::False) { 0 } else { 1 };
        }
    }
    Ok(())
}
