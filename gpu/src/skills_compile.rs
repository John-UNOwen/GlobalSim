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

// Record slots per race (skills.cl SK_RMAX), sized for the eight-slot window
// of a 120-horse grid-mode roster. A larger non-grid roster whose compiled
// records exceed 960 is refused by name at compile time. sk_targets caps
// live targets at the roster, so the cap adds no behavior at small rosters.
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
/// The classic lane's injected asitame conserve release (record carries the
/// engine hook's extra condition instead of a compiled leaf).
pub const SK_FLAG_ASITAME: u32 = 32768;
/// The request flagged this skill id isUnique for the runner, read by the
/// activate hook when it writes skillEvents (lib.rs:620-667).
pub const SK_FLAG_UNIQUE: u32 = 65536;
/// The record gates at least one live condition leaf the classic kernel can
/// read, so the classic scan runs the arm membership and program gate.
pub const SK_FLAG_DYN: u32 = 131072;
/// The classic lane's injected staminasyoubu conserve skill: the scan gate
/// reads live raw stamina and speed, and the TargetSpeed effect replaces its
/// wire modifier 0 with the live formula at activation
/// (race_solver_builder.rs:817-848, race_solver.rs:2496-2498).
pub const SK_FLAG_SYOUBU: u32 = 262144;

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
    /// Classic only: records with a sampled condition leaf, in the engine's
    /// skilldata order. The host replays the builder sampler over these
    /// (race_solver_builder.rs:944-960) and patches the chains.
    pub classic_jobs: Vec<ClassicSampledJob>,
}

/// A classic record whose trigger windows the builder rng resolves before
/// the race starts. `policy` is 0 Random, 1 StraightRandom or 2
/// AllCornerRandom. `regions` is the filter chain the sampler draws from,
/// the engine's sd.regions. `dest_base` is a four-slot region block the
/// windows land in, and `arm_slots` are the arm word slots whose post lists
/// point at it. `cond_bumps` are the construction-fire stat bumps, held back
/// until the sampled window is known to cover the start line.
pub struct ClassicSampledJob {
    pub runner: usize,
    pub pack_seq: usize,
    pub alt_index: usize,
    pub record: usize,
    pub policy: u8,
    pub regions: Vec<(f64, f64)>,
    pub key: String,
    pub dest_base: usize,
    pub arm_slots: Vec<usize>,
    pub cond_bumps: Vec<(i32, f64)>,
    pub fires_start: bool,
}

/// Write one sampled job's windows into the plan words: the dedicated region
/// block, then every live arm's post length and base. An empty window list
/// empties the chains, so the record never arms.
pub fn apply_classic_sampled(
    words: &mut [u32],
    job: &ClassicSampledJob,
    windows: &[(f64, f64)],
) {
    for (i, (s, e)) in windows.iter().enumerate().take(4) {
        let w = f64_words(*s);
        words[SK_REGIONS + 4 * (job.dest_base + i)] = w[0];
        words[SK_REGIONS + 4 * (job.dest_base + i) + 1] = w[1];
        let w = f64_words(*e);
        words[SK_REGIONS + 4 * (job.dest_base + i) + 2] = w[0];
        words[SK_REGIONS + 4 * (job.dest_base + i) + 3] = w[1];
    }
    for slot in &job.arm_slots {
        words[slot + 4] = windows.len().min(4) as u32;
        words[slot + 5] = job.dest_base as u32;
    }
}

pub fn plan(root: &Value, course: &Value) -> SkillsPlan {
    plan_lane(root, course, Lane::Retail, None)
}

/// Compile against the classic (non-retail) runtime profile: the classic
/// effect-value chain and durations, the classic condition leaves, and
/// sampled leaves carried as host sampler jobs.
pub fn plan_classic(root: &Value, course: &Value) -> SkillsPlan {
    plan_lane(root, course, Lane::Classic, None)
}

/// The classic host inputs the classic profile cannot re-derive from the
/// request: the adjusted raw power per runner (initialization::stats_reference)
/// and the resolved strategy ids. The course distance type comes from the
/// course json. With extras, the compiler injects the engine's asitame record
/// for every runner past the 1200 power arm (race_solver_builder.rs:786-815),
/// and the engine's staminasyoubu record for every runner when the request
/// flagged later stamina on a course past 2100 m (lib.rs:546-547).
pub struct ClassicExtras {
    pub raw_power: Vec<f64>,
    pub strategy: Vec<i32>,
    pub later_stamina: bool,
}

pub fn plan_classic_with(root: &Value, course: &Value, extras: &ClassicExtras) -> SkillsPlan {
    plan_lane(root, course, Lane::Classic, Some(extras))
}

fn plan_lane(root: &Value, course: &Value, lane: Lane, extras: Option<&ClassicExtras>) -> SkillsPlan {
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
        wisdom_on: false,
        wisdom_chance: Vec::new(),
        wisdom_hseed: Vec::new(),
        wisdom_rngs: Vec::new(),
        wisdom_picks: Vec::new(),
        pack_seq: 0,
        classic_jobs: Vec::new(),
        drop_skill: false,
    };
    let skill_data = match root.get("skillData") {
        Some(v) => v,
        None => {
            // The construction hooks run per builder regardless of skillData,
            // so the injected records land on a skillData-free request too
            // (lib.rs:545-547, race_solver_builder.rs:940-942).
            let runners = root
                .get("horses")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0);
            c.classic_injections(course, extras, runners);
            return c.finish();
        }
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
    // The engine defaults wisdomChecks to true (lib.rs:341) and the classic
    // builder then rolls the wisdom lottery per wisdomCheck entry
    // (race_solver_builder.rs:1006-1012), so the host replays it per runner.
    c.wisdom_on = lane == Lane::Classic
        && root.get("wisdomChecks").and_then(Value::as_bool).unwrap_or(true);
    if c.wisdom_on {
        let seed = root.get("seed").and_then(Value::as_i64).unwrap_or(0);
        for horse in entries.iter() {
            let wisdom = ref_ranked_base_stat(
                horse_raw_stat(horse, 4),
                horse_mood(horse),
                horse_team_rank(horse),
            ) as f64;
            c.wisdom_chance.push((1.0 - 90.0 / wisdom).max(0.2));
        }
        c.wisdom_hseed =
            (0..entries.len()).map(|r| (seed + r as i64 * 7919) as u32).collect();
    }
    for (r, horse) in entries.iter().enumerate() {
        let skills = match horse.get("skills") {
            Some(v) => v,
            None => continue,
        };
        let list = match skills.as_array() {
            Some(a) => a,
            None => continue,
        };
        // The engine's skill_enabled is a retail-only concept (built and read
        // only with a retail init row), so on retail one runner listing one id
        // twice maps enablement differently than independent kernel records
        // would. Classic has no enablement map: duplicate ids build two
        // skilldata entries that share the sampler cache and the cooldown slot
        // keyed by the cooldown key, exactly like the kernel's slot merge.
        let mut seen_ids: Vec<String> = Vec::new();
        for entry in list {
            let (sid, level) = match normalize_skill_entry(entry) {
                Some(v) => v,
                None => {
                    c.unsupported("skill entry needs a numeric skillId".to_string());
                    return c.finish();
                }
            };
            if seen_ids.contains(&sid) && lane != Lane::Classic {
                c.unsupported(format!("skill {sid} appears twice for one runner"));
                return c.finish();
            }
            seen_ids.push(sid.clone());
            // Classic rolls one skill whose build the engine itself would
            // error on (lib.rs:728-739 keeps the race and drops the skill),
            // so snapshot everything pack_skill appends and roll back on drop.
            let snapshot = (
                c.words.len(),
                c.tags.len(),
                c.bumps.len(),
                c.classic_jobs.len(),
                c.nrec,
                c.nprog,
                c.nreg,
                c.neff,
                c.nterm,
                c.narm,
                c.pack_seq,
                c.live_guts,
                c.words[SK_VIDX],
            );
            c.pack_skill(&sid, level, r, skill_data, course, horse, root);
            if c.reason.is_some() {
                if !(lane == Lane::Classic && c.drop_skill) {
                    return c.finish();
                }
                let (wlen, tlen, blen, jlen, nrec, nprog, nreg, neff, nterm, narm, seq, guts, vidx) =
                    snapshot;
                c.words.truncate(wlen);
                c.tags.truncate(tlen);
                c.bumps.truncate(blen);
                c.classic_jobs.truncate(jlen);
                c.nrec = nrec;
                c.nprog = nprog;
                c.nreg = nreg;
                c.neff = neff;
                c.nterm = nterm;
                c.narm = narm;
                c.pack_seq = seq;
                c.live_guts = guts;
                c.words[SK_VIDX] = vidx;
                c.reason = None;
                c.drop_skill = false;
            }
        }
    }
    // A cross debuff is delivered by the kernel, which drops the same-team
    // receivers exactly like race_field.rs:458-461, so no roster gate remains.
    c.classic_injections(course, extras, entries.len());
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
    // The engine's classic build rolls one wisdom lottery per wisdomCheck
    // entry (race_solver_builder.rs:1006-1012): the draw comes from a
    // per-(horse, skill id) rng against a chance from the horse's ranked
    // wisdom, and a lost draw leaves the entry unbuilt. The host replays it
    // per runner and compiles the lost records dead.
    wisdom_on: bool,
    wisdom_chance: Vec<f64>,
    wisdom_hseed: Vec<u32>,
    wisdom_rngs: Vec<((usize, String), crate::random_gen::Rule30CARng)>,
    wisdom_picks: Vec<((usize, String), bool)>,
    // Counts pack_skill calls so sampled jobs replay the engine's skilldata
    // order: wire skill order, alternatives reversed inside one skill.
    pack_seq: usize,
    classic_jobs: Vec<ClassicSampledJob>,
    // Set by pack_skill alongside reason when the reason is an error the
    // engine's own build raises for this skill. Classic then drops just this
    // skill (lib.rs:728-739) and answers without it, so the runner loop
    // rolls the skill back and continues instead of refusing the race.
    drop_skill: bool,
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

fn phase_end(distance: f64, phase: i64) -> f64 {
    match phase {
        0 => distance * 1.0 / 6.0,
        1 => distance * 2.0 / 3.0,
        2 => distance * 5.0 / 6.0,
        3 => distance,
        _ => f64::NAN,
    }
}

/// The skill ids whose phase== region end moves 10 m later
/// (activation_conditions.rs:468,891).
const PHASE_FUDGE_SKILLS: &[&str] = &[
    "100591", "900591", "110261", "910261", "110191", "910191", "120451", "920451", "101502121",
];

/// The ids whose corner_random resolves the fourth-from-last corner and
/// refuses other args (activation_conditions.rs:462-466,557-564).
const CORNER_RANDOM_QUAD_SKILLS: &[&str] = &[
    "200331", "200332", "200333", "200341", "200342", "200343", "200351", "200352", "200353",
    "200971", "200972", "201041", "201042", "201111", "201112", "201181", "201182", "201251",
    "201252", "201321", "201322", "201391", "201392", "201461", "201462",
];

/// Marks an error whose text the engine raises verbatim. The classic plan
/// reason keeps it unwrapped so it byte-equals the engine error.
const ENGINE_QUOTE: &str = "\u{1}";

fn engine_quote(why: String) -> String {
    format!("{ENGINE_QUOTE}{why}")
}

/// The classic lane's static condition leaves: the engine resolves these to
/// course region lists through the filter table, with no live check on the
/// classic branch (activation_conditions.rs static filter section).
fn classic_regions(t: &Term, g: &Geometry, s: &Statics) -> Result<Vec<(f64, f64)>, String> {
    let d = g.distance;
    match t.name.as_str() {
        "phase" => {
            if t.arg < 0 || t.arg > 3 {
                return Err(format!("phase{}{} is outside the engine's phases", op_text(t.op), t.arg));
            }
            let bounds = match t.op {
                0 => {
                    let fudge =
                        if PHASE_FUDGE_SKILLS.contains(&s.sid_text.as_str()) { 10.0 } else { 0.0 };
                    (phase_start(d, t.arg), phase_end(d, t.arg) + fudge)
                }
                2 => {
                    if t.arg <= 0 {
                        return Err("phase< requires phase>0".to_string());
                    }
                    (0.0, phase_start(d, t.arg))
                }
                3 => (0.0, phase_end(d, t.arg)),
                4 => {
                    if t.arg >= 3 {
                        return Err("phase> requires phase<3".to_string());
                    }
                    (phase_start(d, t.arg + 1), d)
                }
                5 => (phase_start(d, t.arg), d),
                _ => return Err("unsupported comparison".to_string()),
            };
            Ok(vec![bounds])
        }
        "remain_distance" => {
            let r = t.arg as f64;
            let bounds = match t.op {
                0 => (d - r, d - r + 1.0),
                3 => (d - r, d),
                5 => (0.0, d - r),
                _ => return Err("unsupported comparison".to_string()),
            };
            Ok(vec![bounds])
        }
        "corner" => {
            if !sorted_by_start(&g.corners) {
                return Err("corners not sorted".to_string());
            }
            let n = g.corners.len() as i64;
            if t.op == 1 {
                if t.arg != 0 {
                    return Err("corner!= only supports 0".to_string());
                }
                return Ok(g.corners.iter().copied().collect());
            }
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if t.arg == 0 {
                let mut gaps = Vec::new();
                let mut last_end = 0.0f64;
                for &(a, e) in &g.corners {
                    gaps.push((last_end, a));
                    last_end = e;
                }
                if last_end != d {
                    gaps.push((last_end, d));
                }
                return Ok(gaps);
            }
            if n + t.arg < 5 {
                return Ok(Vec::new());
            }
            let mut rows = Vec::new();
            let mut idx = n + t.arg - 5;
            while idx >= 0 {
                rows.push(g.corners[idx as usize]);
                idx -= 4;
            }
            rows.reverse();
            Ok(rows)
        }
        "is_finalcorner" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if t.arg != 0 && t.arg != 1 {
                return Err("is_finalcorner expects 0 or 1".to_string());
            }
            if !sorted_by_start(&g.corners) {
                return Err("corners not sorted".to_string());
            }
            let Some(&(fs, _)) = g.corners.last() else { return Ok(Vec::new()) };
            Ok(vec![if t.arg == 1 { (fs, d) } else { (0.0, fs) }])
        }
        "is_finalcorner_laterhalf" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if t.arg != 1 {
                return Err("is_finalcorner_laterhalf expects arg==1".to_string());
            }
            if !sorted_by_start(&g.corners) {
                return Err("corners not sorted".to_string());
            }
            let Some(&(fs, fe)) = g.corners.last() else { return Ok(Vec::new()) };
            Ok(vec![((fs + fe) / 2.0, fe)])
        }
        "slope" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if !(0..=2).contains(&t.arg) {
                return Err("slope_type must be 0,1,2".to_string());
            }
            if !sorted_by_start(&g.slopes.iter().map(|&(a, e, _)| (a, e)).collect::<Vec<_>>()) {
                return Err("slopes not sorted".to_string());
            }
            let picked: Vec<(f64, f64, f64)> = g
                .slopes
                .iter()
                .copied()
                .filter(|(_, _, s)| (t.arg != 2 && *s > 0.0) || (t.arg != 1 && *s < 0.0))
                .collect();
            if t.arg == 0 {
                let mut out = Vec::new();
                let mut last_end = 0.0f64;
                for (a, e, _) in &picked {
                    out.push((last_end, *a));
                    last_end = *e;
                }
                if last_end != d {
                    out.push((last_end, d));
                }
                return Ok(out);
            }
            Ok(picked.iter().map(|&(a, e, _)| (a, e)).collect())
        }
        "furlong" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let n = t.arg as f64;
            Ok(vec![((n - 1.0) * 201.168, n * 201.168)])
        }
        "is_last_straight" | "is_last_straight_onetime" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if t.arg != 1 {
                return Err(format!("{name} expects arg==1", name = t.name));
            }
            if !sorted_by_start(&g.straights) {
                return Err("straights not sorted".to_string());
            }
            let Some(&(a, e)) = g.straights.last() else {
                return Err("no straights".to_string());
            };
            Ok(vec![if t.name == "is_last_straight" { (a, e) } else { (a, a + 10.0) }])
        }
        other => Err(format!("'{other}' has no classic region filter")),
    }
}

/// The sampler class behind a sampled condition name
/// (SamplePolicy in activation_sample_policy.rs, per registration in the
/// engine's table).
fn classic_sampled_policy(name: &str) -> Option<u8> {
    let p = match name {
        "straight_random" | "last_straight_random" | "phase_straight_random"
        | "phase_first_half_straight_random" | "phase_latter_half_straight_random" => 1,
        "all_corner_random" => 2,
        "corner_random" | "distance_rate_after_random" | "down_slope_random"
        | "up_slope_random" | "up_slope_random_later_half" | "is_finalcorner_random"
        | "phase_corner_random" | "phase_firsthalf_random" | "phase_firstquarter_random"
        | "phase_laterhalf_random" | "phase_random" | "run_at_full_speed_random" => 0,
        _ => return None,
    };
    Some(p)
}

/// The double-dispatch policy table of activation_sample_policy.rs:55-75 on
/// the three sampler kinds the classic table carries: 0 Random, 1
/// StraightRandom, 2 AllCornerRandom. None is the Straight-AllCorner pair
/// the engine rejects.
fn reconcile_policy(a: u8, b: u8) -> Option<u8> {
    match (a, b) {
        (_, 0) => Some(0),
        (1, 2) | (2, 1) => None,
        (_, 1) => Some(1),
        _ => Some(2),
    }
}

/// region.rs:57-83 union, the same fold the kernel's sk_union runs at
/// runtime: stable sort by start, only strict overlaps merge, adjacent
/// regions stay separate.
fn union_regions(a: &[(f64, f64)], b: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut u: Vec<(f64, f64)> = a.iter().chain(b.iter()).copied().collect();
    u.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let mut out = Vec::new();
    let Some(first) = u.first() else { return out };
    let mut acc = *first;
    for &b in &u[1..] {
        let a = acc;
        if a.0 <= b.0 && a.1 >= b.1 {
        } else if a.0 <= b.0 && b.0 < a.1 {
            acc = (a.0, b.1);
        } else if a.0 < b.1 && b.1 <= a.1 {
            acc = (b.0, a.1);
        } else {
            out.push(a);
            acc = b;
        }
    }
    out.push(acc);
    out
}

/// The classic lane's host replay of the builder sampler
/// (activation_sample_policy.rs). `policy` is the reconcile-folded sampler
/// kind, `regions` the filter chain the engine hands to SamplePolicy::sample
/// with nsamples 1 (race_solver_builder.rs:953-958). AllCornerRandom takes
/// the place_all_corner_triggers branch of the builder loop, not the
/// one-window arm of the sample method.
pub fn classic_sample_windows(
    policy: u8,
    regions: &[(f64, f64)],
    rng: &mut crate::random_gen::Rule30CARng,
) -> Vec<(f64, f64)> {
    if regions.is_empty() {
        return Vec::new();
    }
    match policy {
        1 => sample_straight_random(regions, rng),
        2 => place_all_corner_triggers(regions, rng),
        _ => sample_random(regions, rng),
    }
}

/// activation_sample_policy.rs:98-105. Keep the float32 rounding exact:
/// a short region still consumes one draw through its caller.
fn sample_distance_window(r: (f64, f64), rng: &mut crate::random_gen::Rule30CARng) -> (f64, f64) {
    let start = r.0 as f32;
    let upper = (r.1 as f32 - 10.0).max(start);
    let sample = rng.random() as f32;
    let pos = (start + sample * (upper - start)).max(start).min(upper);
    (pos as f64, (pos + 10.0) as f64)
}

/// activation_sample_policy.rs:107-127.
fn sample_random(
    regions: &[(f64, f64)],
    rng: &mut crate::random_gen::Rule30CARng,
) -> Vec<(f64, f64)> {
    let mut acc = 0.0;
    let mut weights = Vec::with_capacity(regions.len());
    for r in regions {
        acc += r.1 - r.0;
        weights.push(acc);
    }
    let picked = if regions.len() == 1 || acc <= 0.0 {
        regions[0]
    } else {
        let threshold = rng.random() * acc;
        regions
            .iter()
            .enumerate()
            .find(|(i, _)| weights[*i] > threshold)
            .map(|(_, r)| *r)
            .unwrap_or(regions[regions.len() - 1])
    };
    vec![sample_distance_window(picked, rng)]
}

/// activation_sample_policy.rs:129-139.
fn sample_straight_random(
    regions: &[(f64, f64)],
    rng: &mut crate::random_gen::Rule30CARng,
) -> Vec<(f64, f64)> {
    let r = regions[rng.uniform(regions.len() as f64) as usize];
    vec![sample_distance_window(r, rng)]
}

/// activation_sample_policy.rs:222-240.
fn place_all_corner_triggers(
    regions: &[(f64, f64)],
    rng: &mut crate::random_gen::Rule30CARng,
) -> Vec<(f64, f64)> {
    let mut triggers: Vec<f64> = Vec::new();
    let mut candidates: Vec<(f64, f64)> = regions.to_vec();
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    while triggers.len() < 4 && !candidates.is_empty() {
        let ci = rng.uniform(candidates.len() as f64) as usize;
        let c = candidates[ci];
        let window = sample_distance_window(c, rng);
        if (c.1 as f32) - (window.1 as f32) >= 10.0 {
            candidates.splice(ci..ci + 1, [(window.1, c.1)]);
        } else {
            candidates.remove(ci);
        }
        candidates.drain(0..ci);
        triggers.push(window.0);
    }
    triggers
        .into_iter()
        .map(|start| (start, (start as f32 + 10.0) as f64))
        .collect()
}

/// The engine's static filter behind a sampled name, the region list the
/// builder sampler draws its window from
/// (activation_conditions.rs, the *_random filter section).
fn classic_sampled_regions(t: &Term, g: &Geometry, s: &Statics) -> Result<Vec<(f64, f64)>, String> {
    let d = g.distance;
    let half = |p: i64, frac: f64| -> Vec<(f64, f64)> {
        let a = phase_start(d, p);
        let e = phase_end(d, p);
        vec![(a, a + (e - a) * frac)]
    };
    match t.name.as_str() {
        "corner_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            if g.corners.is_empty() {
                return Ok(Vec::new());
            }
            if !sorted_by_start(&g.corners) {
                return Err("corners not sorted".to_string());
            }
            if CORNER_RANDOM_QUAD_SKILLS.contains(&s.sid_text.as_str()) {
                if t.arg != 1 {
                    return Ok(Vec::new());
                }
                return Ok(vec![g.corners[(g.corners.len() as i64 - 4).max(0) as usize]]);
            }
            let n = g.corners.len() as i64;
            if n + t.arg < 5 {
                return Ok(Vec::new());
            }
            Ok(vec![g.corners[(n + t.arg - 5) as usize]])
        }
        "distance_rate_after_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            Ok(vec![(d * t.arg as f64 / 100.0, d)])
        }
        "down_slope_random" => {
            sampled_arg_one(t)?;
            Ok(g.slopes.iter().filter(|(_, _, sl)| *sl < 0.0).map(|(a, e, _)| (*a, *e)).collect())
        }
        "up_slope_random" => {
            sampled_arg_one(t)?;
            Ok(g.slopes.iter().filter(|(_, _, sl)| *sl > 0.0).map(|(a, e, _)| (*a, *e)).collect())
        }
        "up_slope_random_later_half" => {
            sampled_arg_one(t)?;
            let picked: Vec<(f64, f64)> = g
                .slopes
                .iter()
                .filter(|(_, _, sl)| *sl > 0.0)
                .map(|(a, e, _)| (*a, *e))
                .collect();
            let half = (d / 2.0, d);
            let mut out = Vec::new();
            for p in picked {
                let r = intersect(p, half);
                if r.0 > -1.0 {
                    out.push(r);
                }
            }
            Ok(out)
        }
        "is_finalcorner_random" => {
            sampled_arg_one(t)?;
            if !sorted_by_start(&g.corners) {
                return Err("corners not sorted".to_string());
            }
            let Some(&(fs, fe)) = g.corners.last() else { return Ok(Vec::new()) };
            Ok(vec![(fs, fe)])
        }
        "phase_corner_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let ps = phase_start(d, t.arg);
            let pe = phase_end(d, t.arg);
            let mut picked = Vec::new();
            for &(cs, ce) in &g.corners {
                if (ps <= cs && cs < pe) || (ps <= ce && ce < pe) {
                    picked.push((cs.max(ps), ce.min(pe)));
                }
            }
            Ok(picked)
        }
        "phase_firsthalf_random" | "phase_firstquarter_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let frac = if t.name == "phase_firstquarter_random" { 0.25 } else { 0.5 };
            Ok(half(t.arg, frac))
        }
        "phase_laterhalf_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let a = phase_start(d, t.arg);
            let e = phase_end(d, t.arg);
            Ok(vec![((a + e) / 2.0, e)])
        }
        "phase_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            Ok(vec![(phase_start(d, t.arg), phase_end(d, t.arg))])
        }
        "run_at_full_speed_random" => {
            sampled_arg_one(t)?;
            Ok(vec![(phase_start(d, 3), d)])
        }
        "straight_random" => {
            sampled_arg_one(t)?;
            Ok(g.straights.clone())
        }
        "last_straight_random" => {
            sampled_arg_one(t)?;
            if !sorted_by_start(&g.straights) {
                return Err("straights not sorted".to_string());
            }
            Ok(g.straights.last().copied().into_iter().collect())
        }
        "phase_straight_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let bounds = (phase_start(d, t.arg), phase_end(d, t.arg));
            Ok(g.straights.iter().map(|st| intersect(*st, bounds)).filter(|r| r.0 > -1.0).collect())
        }
        "phase_first_half_straight_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let a = phase_start(d, t.arg);
            let e = phase_end(d, t.arg);
            let bounds = (a, a + (e - a) / 2.0);
            Ok(g.straights.iter().map(|st| intersect(*st, bounds)).filter(|r| r.0 > -1.0).collect())
        }
        "phase_latter_half_straight_random" => {
            if t.op != 0 {
                return Err("unsupported comparison".to_string());
            }
            let a = phase_start(d, t.arg);
            let e = phase_end(d, t.arg);
            let bounds = (a + (e - a) / 2.0, e);
            Ok(g.straights.iter().map(|st| intersect(*st, bounds)).filter(|r| r.0 > -1.0).collect())
        }
        "all_corner_random" => {
            sampled_arg_one(t)?;
            Ok(g.corners.clone())
        }
        other => Err(format!("'{other}' has no classic sampled filter")),
    }
}

/// The engine's shared arg gate for the sampled filters that accept only
/// arg 1 (activation_conditions.rs, the expects arg==1 messages).
fn sampled_arg_one(t: &Term) -> Result<(), String> {
    if t.op != 0 {
        return Err("unsupported comparison".to_string());
    }
    if t.arg != 1 {
        return Err(format!("{} expects arg==1", t.name));
    }
    Ok(())
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

    /// The engine's own build error for this skill: classic build failures
    /// drop only the failing skill (lib.rs:728-739), so the runner loop
    /// rolls back this pack_skill and answers the rest from the kernel.
    fn engine_error(&mut self, why: impl Into<String>) {
        if self.reason.is_none() {
            self.reason = Some(why.into());
            self.drop_skill = true;
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
        let classic_jobs = if self.reason.is_some() { Vec::new() } else { std::mem::take(&mut self.classic_jobs) };
        SkillsPlan { words: self.words, reason: self.reason, bumps, live_guts, classic_jobs }
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
        let pack_seq = self.pack_seq;
        self.pack_seq += 1;
        self.drop_skill = false;
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
                // The retail config errors first at the init lookup
                // (retail_initialization.rs:222), classic at the builder
                // (race_solver_builder.rs:496) and drops (lib.rs:726-739).
                let retail = [
                    "experimentalRetailInitialization",
                    "experimentalRetailSections",
                    "experimentalRetailRuntime",
                ]
                .iter()
                .any(|k| root.get(k).and_then(Value::as_bool) == Some(true));
                if retail {
                    self.unsupported(format!("Missing retail initialization field {sid}"));
                } else {
                    self.engine_error(format!("bad skill ID {sid}"));
                }
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
        // The wisdom lottery draws once per cooldown key from one per-id rng
        // (builder :589, :1006-1011). Across two keys of one id the draws
        // follow the builder's reversed entry order, so a single-key id is
        // the case whose pick is order-free and provably replayable.
        if self.wisdom_on && def.get("wisdomCheck").and_then(Value::as_i64).unwrap_or(0) == 1 {
            let mut plain = false;
            let mut follow = false;
            for alt in alts.iter() {
                let cond = alt.get("condition").and_then(Value::as_str).unwrap_or("");
                if cond.contains("is_activate_other_skill_detail") {
                    follow = true;
                } else {
                    plain = true;
                }
            }
            if plain && follow {
                self.unsupported(format!(
                    "skill {sid} wisdom lottery across two cooldown keys"
                ));
                return;
            }
        }
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
            // Retail gates the skill through its init lottery row; classic
            // replays the builder's per-entry wisdom lottery in wisdom_pick.
            let wisdom = def.get("wisdomCheck").and_then(Value::as_i64).unwrap_or(0) == 1
                && self.lane != Lane::Classic;
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
            // The parser composes the policy tree before any filter body
            // runs, so the one rejected policy pair errors ahead of filter
            // and runtime-reader refusals.
            if self.lane == Lane::Classic {
                let mut fold: Option<u8> = None;
                for t in terms.iter() {
                    let Some(p) = classic_sampled_policy(&t.name) else { continue };
                    fold = Some(match fold {
                        None => p,
                        Some(prev) => match reconcile_policy(prev, p) {
                            Some(f) => f,
                            None => {
                                self.engine_error(
                                    "cannot reconcile StraightRandomPolicy with AllCornerRandomPolicy"
                                        .to_string(),
                                );
                                return;
                            }
                        },
                    });
                }
            }
            let g = Geometry::new(course);
            let statics = Statics::new(course, horse, root, sid, runner);
            let leaves: Vec<Leaf> = match terms.iter().map(|t| classify(t, &g, &statics, self.lane)).collect() {
                Ok(l) => l,
                Err(why) => {
                    match why.strip_prefix(ENGINE_QUOTE) {
                        Some(raw) => self.engine_error(raw.to_string()),
                        None => self.unsupported(format!("skill {sid} condition '{condition}': {why}")),
                    };
                    return;
                }
            };
            if self.lane == Lane::Classic && self.classic_refuses_leaf(sid, &terms, &leaves, true) {
                return;
            }
            // The classic kernel gates records with a live leaf through the arm
            // programs; without this flag the scan keeps the merged-window-only
            // path every classic race ran before.
            let dyn_gate = self.lane == Lane::Classic
                && leaves.iter().any(|l| matches!(l, Leaf::Dynamic { .. }));
            if leaves.iter().any(|l| leaf_feat(l) == FEAT_VREMAIN) {
                self.words[SK_VIDX] = statics.viewer.unwrap_or(0) as u32;
            }

            // The builder evaluates a precondition over the whole course, clips
            // the main chain to the first precondition region's start, and checks
            // precondition membership plus its dynamics as a latched gate.
            let mut dead = false;
            // A lost classic wisdom draw leaves the entry unbuilt for this
            // runner (builder :1006-1011), which the dead record mirrors. The
            // per-(runner, key) lazy pick is one draw per id for the single
            // key ids accepted above, so its timing never matters.
            if self.wisdom_on
                && def.get("wisdomCheck").and_then(Value::as_i64).unwrap_or(0) == 1
                && !self.wisdom_pick(runner, sid, condition, sid_num)
            {
                dead = true;
            }
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
                    match pre_terms.iter().map(|t| classify(t, &g, &statics, self.lane)).collect() {
                        Ok(l) => l,
                        Err(why) => {
                            match why.strip_prefix(ENGINE_QUOTE) {
                                Some(raw) => self.engine_error(raw.to_string()),
                                None => self.unsupported(format!("skill {sid} precondition '{pre_text}': {why}")),
                            };
                            return;
                        }
                    };
                if self.lane == Lane::Classic
                    && self.classic_refuses_leaf(sid, &pre_terms, &pre_leaves, false)
                {
                    return;
                }
                if pre_leaves.iter().any(|l| matches!(l, Leaf::Sampled))
                    && self.lane != Lane::Classic
                {
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
                for (t, leaf) in pre_terms.iter().zip(pre_leaves.iter()) {
                    let next = match leaf {
                        Leaf::True | Leaf::Dynamic { .. } => pre_regions.clone(),
                        Leaf::False => Vec::new(),
                        // The classic engine applies a sampled leaf's static
                        // filter to the precondition chain and never samples
                        // it: only condition triggers reach the sampler.
                        Leaf::Sampled => match classic_sampled_regions(t, &g, &statics) {
                            Ok(filt) => rmap_multi(&pre_regions, |r| {
                                filt.iter().map(|b| intersect(*r, *b)).collect()
                            }),
                            Err(why) => {
                                self.engine_error(why);
                                return;
                            }
                        },
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
                // One sampled term per arm is the retail term machinery's cap.
                // Classic folds every sampled filter into the arm chain and
                // samples the reconciled policy once, so several are legal,
                // exactly as in the engine's condition tree.
                if arm_sampled > 1 && self.lane != Lane::Classic {
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
                            if self.lane == Lane::Classic {
                                // The classic engine folds the sampled filter
                                // into the chain, and the host sampler later
                                // replaces the whole window list.
                                match classic_sampled_regions(t, &g, &statics) {
                                    Ok(filt) => rmap_multi(&regions, |r| {
                                        filt.iter().map(|b| intersect(*r, *b)).collect()
                                    }),
                                    Err(why) => {
                                        self.engine_error(why);
                                        return;
                                    }
                                }
                            } else {
                                regions.clone()
                            }
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
            // The classic lane resolves these with the host builder sampler.
            let mut term_word = u32::MAX;
            for ai in 0..arms.len() {
                if arms[ai].1[0] & SK_ARM_SAMPLED == 0 || self.lane == Lane::Classic {
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
            // A classic sampled record resolves its windows on the host with
            // the builder sampler. The arms share one reserved four-slot
            // region block; the sampler fills it and repoints the arms at it.
            let mut classic_dest: Option<usize> = None;
            let mut classic_arm_slots: Vec<usize> = Vec::new();
            if self.lane == Lane::Classic && sampled_count > 0 && !dead {
                classic_dest = Some(self.push_regions(&[(0.0, 0.0); 4]).1);
                if self.reason.is_some() {
                    return;
                }
            }
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
                    let sampled_arm = flags[0] & SK_ARM_SAMPLED != 0;
                    let (n_pre, pre_b) =
                        if flags[0] == SK_ARM_SAMPLED && self.lane != Lane::Classic {
                        self.push_regions(&lists[0])
                    } else {
                        (0, self.nreg)
                    };
                    let (n_post, post_b) = match (self.lane, sampled_arm, classic_dest) {
                        (Lane::Classic, true, Some(base)) => (0, base),
                        _ => self.push_regions(&lists[1]),
                    };
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
                    if sampled_arm && flags[0] & SK_ARM_DEAD == 0 {
                        classic_arm_slots.push(at);
                    }
                    self.narm += 1;
                    if ai == 0 {
                        arm0_prog = (prog_len, prog_base);
                        arm0_pre = (n_pre, pre_b);
                        arm0_post = (n_post, post_b);
                    }
                }
            }

            // The engine's sampler entry for this alternative: one sample call
            // over the union of the surviving arms' filter chains, keyed for
            // the per-runner reuse map of race_solver_builder.rs:945.
            let mut classic_job: Option<usize> = None;
            if self.lane == Lane::Classic && sampled_count > 0 && !dead {
                let mut policy: Option<u8> = None;
                let mut bad_pair = false;
                for t in terms.iter() {
                    let Some(p) = classic_sampled_policy(&t.name) else { continue };
                    policy = Some(match policy {
                        None => p,
                        Some(prev) => match reconcile_policy(prev, p) {
                            Some(f) => f,
                            None => {
                                bad_pair = true;
                                break;
                            }
                        },
                    });
                }
                if bad_pair {
                    self.engine_error(
                        "cannot reconcile StraightRandomPolicy with AllCornerRandomPolicy"
                            .to_string(),
                    );
                    return;
                }
                let mut regs: Vec<(f64, f64)> = Vec::new();
                for (lists, flags, _) in arms.iter() {
                    if flags[0] & SK_ARM_DEAD == 0 {
                        regs = union_regions(&regs, &lists[1]);
                    }
                }
                let cdkey = format!(
                    "{}{}",
                    sid,
                    if condition.contains("is_activate_other_skill_detail") { "#followup" } else { "" }
                );
                let regtext = regs
                    .iter()
                    .map(|(s, e)| format!("Region {{ start: {s:?}, end: {e:?} }}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let key = format!(
                    "{cdkey}:{}:[{regtext}]",
                    match policy {
                        Some(1) => "StraightRandom",
                        Some(2) => "AllCornerRandom",
                        _ => "Random",
                    }
                );
                self.classic_jobs.push(ClassicSampledJob {
                    runner,
                    pack_seq,
                    alt_index,
                    record: self.nrec,
                    policy: policy.unwrap_or(0),
                    regions: regs,
                    key,
                    dest_base: classic_dest.unwrap_or(0),
                    arm_slots: classic_arm_slots,
                    cond_bumps: Vec::new(),
                    fires_start: false,
                });
                classic_job = Some(self.classic_jobs.len() - 1);
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
            // The classic kernel folds the du 1 constant and the du 3 HP tiers
            // (race_solver.rs:283-290) at activation. du 2 reads the live
            // leader gap and du 4 extends running instances on an overtake,
            // which the classic lane never enters.
            if self.lane == Lane::Classic && !matches!(duration_usage, 1 | 3) {
                self.unsupported(format!(
                    "skill {sid} duration_usage {duration_usage} reads the live leader gap, \
                     other HP tiers, or the overtake extend, which the classic runtime does not model"
                ));
                return;
            }
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
            // A classic record with a window covering the start line fires in
            // the engine's constructor pass (race_solver.rs:1008), before the
            // host computes the derived constants. Its self stat writes go into
            // the same bump channel the static fires use. The pass sees no
            // field: guarded live readers report false, the clock reads zero
            // and hp_per reads the full-tank ratio (activation_conditions.rs
            // :287-318, :612-624), so an arm with live readers counts only
            // when every live leaf survives that fieldless reading.
            let ctor_leaf = |l: &Leaf| -> bool {
                match l {
                    Leaf::Dynamic { feat, cmp, arg } => {
                        let (v, b) = match *feat {
                            FEAT_CLOCK => (0.0, arg.unwrap_or(0.)),
                            FEAT_HP_PER => (1.0, arg.unwrap_or(0.) / 100.0),
                            _ => return false,
                        };
                        match cmp {
                            0 => v == b,
                            1 => v != b,
                            2 => v < b,
                            3 => v <= b,
                            4 => v > b,
                            _ => v >= b,
                        }
                    }
                    Leaf::False => false,
                    _ => true,
                }
            };
            let fires_at_start = self.lane == Lane::Classic
                && !dead
                && arms.iter().zip(groups.iter()).any(|(a, g)| {
                    a.0[1].iter().any(|&(s, e)| s <= 0.0 && 0.0 < e)
                        && (!leaves[g.0..g.1].iter().any(|l| matches!(l, Leaf::Dynamic { .. }))
                            || leaves[g.0..g.1].iter().all(ctor_leaf))
                });
            for ef in effect_list {
                let kind = ef.get("type").and_then(Value::as_i64).unwrap_or(-1) as i32;
                if !matches!(kind, 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14
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
                if self.lane != Lane::Classic && kind == 3 {
                    self.unsupported(format!(
                        "skill {sid} PowerUp is not ported to the retail kernel"
                    ));
                    return;
                }
                let target = ef.get("target").and_then(Value::as_i64).unwrap_or(1) as i32;
                if !(matches!(target, 1 | 4 | 9 | 10) || (target == 21 && kind == 13)) {
                    self.unsupported(format!(
                        "skill {sid} effect target {target} is not ported"
                    ));
                    return;
                }
                // The classic runtime refuses the random re-activation pair and
                // the kakari extend. The stat kinds apply on both paths: the
                // construction fire lands in a host bump below, a mid-race fire
                // runs the engine's add_stat_modifier and refresh math in the
                // kernel (race_solver.rs:2555-2565, :1330-1340).
                if self.lane == Lane::Classic && matches!(kind, 13 | 36 | 37) && !dead {
                    self.unsupported(format!(
                        "skill {sid} type {kind} is not ported to the classic runtime"
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
                // The classic lane has no running-instance list to carry a
                // follow-up, so the additional-activation machinery is off.
                if self.lane == Lane::Classic && aa != 0 {
                    self.unsupported(format!(
                        "skill {sid} additionalActivate {aa} on type {kind}: the classic \
                         runtime has no running-instance list to re-fire"
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
                // The activation-time rolls and the late-corner tier read live
                // state with no classic reader (race_solver.rs:2485-2503).
                if self.lane == Lane::Classic && matches!(value_usage, 8 | 9 | 11 | 14) {
                    self.unsupported(format!(
                        "skill {sid} value_usage {value_usage} draws live state at activation, \
                         which the classic runtime does not model"
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
                if self.lane != Lane::Classic
                    && static_fire
                    && target == 1
                    && matches!(kind, 4 | 5 | 29)
                {
                    self.bumps.push((runner, kind, modifier));
                    bumped = true;
                }
                if self.lane == Lane::Classic
                    && target == 1
                    && fires_at_start
                    && matches!(kind, 1 | 2 | 3 | 4 | 5 | 29)
                {
                    // A sampled record's construction fire depends on the
                    // drawn window covering the start line, held for the host
                    // sampler to decide.
                    match classic_job {
                        Some(ji) => self.classic_jobs[ji].cond_bumps.push((kind, modifier)),
                        None => self.bumps.push((runner, kind, modifier)),
                    }
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
            if let Some(ji) = classic_job {
                self.classic_jobs[ji].fires_start = fires_at_start;
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
            // The wire's isUnique flag feeds the hook's skillEvents entries
            // (lib.rs:620-667); the request keys it by skill id per horse.
            let unique = horse
                .get("skills")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter().any(|e| {
                        normalize_skill_entry(e).map(|(id, _)| id == sid).unwrap_or(false)
                            && e.get("isUnique").and_then(Value::as_bool).unwrap_or(false)
                    })
                })
                .unwrap_or(false);
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
                | (if gated_static { SK_FLAG_INIT_BUMP | SK_FLAG_BUMPED } else { 0 })
                | (if unique { SK_FLAG_UNIQUE } else { 0 })
                | (if dyn_gate { SK_FLAG_DYN } else { 0 });
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

    // The engine's classic wisdom lottery (race_solver_builder.rs:1006-1011):
    // one draw per cooldown key, from a fresh per-(horse, id) rng, against the
    // horse's wisdom-derived chance. Keyed lazy replay reproduces the engine's
    // pick because each accepted id draws exactly once.
    fn wisdom_pick(&mut self, r: usize, sid: &str, condition: &str, sid_num: u32) -> bool {
        let key = if condition.contains("is_activate_other_skill_detail") {
            format!("{sid}#followup")
        } else {
            sid.to_string()
        };
        if let Some((_, pick)) = self.wisdom_picks.iter().find(|((kr, kk), _)| *kr == r && *kk == key)
        {
            return *pick;
        }
        let hseed = self.wisdom_hseed[r];
        let rng_idx = self.wisdom_rngs.iter().position(|((kr, kk), _)| *kr == r && *kk == sid);
        let mut rng = match rng_idx {
            Some(i) => self.wisdom_rngs.swap_remove(i).1,
            None => crate::random_gen::Rule30CARng::new(
                ref_wisdom_seed(hseed, sid_num),
                ref_wisdom_seed(hseed ^ 0x9e3779b9, sid_num),
            ),
        };
        let pick = rng.random() < self.wisdom_chance[r];
        self.wisdom_rngs.push(((r, sid.to_string()), rng));
        self.wisdom_picks.push(((r, key), pick));
        pick
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

    /// The engine's asitame injection (race_solver_builder.rs:786-815), armed
    /// when the adjusted raw power passes 1200. The record carries one region
    /// chain [phase 2 start, distance] and one Accel effect. The engine's
    /// extra condition (is_last_spurt) is a kernel special case marked by the
    /// ASITAME flag, and the wire modifier 0 is replaced at activation by the
    /// release formula, which depends on no race state here, so the compiled
    /// value is the folded one (race_solver.rs:2489-2495, with the classic
    /// lane's dead cp_struggle and cp_kakari flags).
    fn pack_asitame(
        &mut self,
        runner: usize,
        raw_power: f64,
        strategy: i32,
        distance_type: i32,
        distance: f64,
    ) {
        if self.nrec >= SK_RMAX {
            self.unsupported("skill table overflow: too many records".to_string());
            return;
        }
        if self.nreg + 1 > SK_GMAX || self.nprog + 1 > SK_PMAX || self.narm + 1 > SK_AMAX
            || self.neff + 1 > SK_EMAX
        {
            self.unsupported("skill table overflow: asitame injection".to_string());
            return;
        }
        let region = self.nreg;
        let at = SK_REGIONS + SK_GSTRIDE * self.nreg;
        let w = f64_words(distance * 2.0 / 3.0);
        self.words[at] = w[0];
        self.words[at + 1] = w[1];
        let w = f64_words(distance);
        self.words[at + 2] = w[0];
        self.words[at + 3] = w[1];
        self.nreg += 1;
        let prog = self.nprog;
        self.words[SK_PROGRAMS + SK_PSTRIDE * self.nprog] = OP_TRUE;
        self.nprog += 1;
        let arm = self.narm;
        let at = SK_ARMS + SK_ASTRIDE * self.narm;
        self.words[at] = 1;
        self.words[at + 1] = prog as u32;
        self.words[at + 2] = 0;
        self.words[at + 3] = 0;
        self.words[at + 4] = 1;
        self.words[at + 5] = region as u32;
        self.words[at + 6] = 0;
        self.words[at + 7] = u32::MAX;
        self.narm += 1;
        let effect = self.neff;
        let at = SK_EFFECTS + SK_ESTRIDE * self.neff;
        self.words[at] = 31;
        self.words[at + 1] = 1;
        self.words[at + 2] = 1;
        self.words[at + 3] = 1;
        self.words[at + 4] = 0;
        self.words[at + 5] = 0;
        let w = f64_words(asitame_release_modifier(raw_power, strategy, distance_type));
        self.words[at + 6] = w[0];
        self.words[at + 7] = w[1];
        let w = f64_words(3.0 / (distance / 1000.0));
        self.words[at + 8] = w[0];
        self.words[at + 9] = w[1];
        self.neff += 1;
        let at = SK_RECORDS + SK_RSTRIDE * self.nrec;
        self.words[at] = 0;
        self.words[at + 1] = runner as u32;
        self.words[at + 2] = SK_FLAG_RETAIN | SK_FLAG_PRIMARY | SK_FLAG_ASITAME;
        self.words[at + 3] = 0;
        self.words[at + 4] = 1;
        self.words[at + 5] = 1;
        self.words[at + 6] = effect as u32;
        self.words[at + 7] = 1;
        self.words[at + 8] = prog as u32;
        self.words[at + 9] = 0;
        self.words[at + 10] = u32::MAX;
        self.words[at + 11] = 0;
        self.words[at + 12] = 0;
        self.words[at + 13] = 0;
        self.words[at + 14] = 0;
        self.words[at + 15] = 0;
        self.words[at + 16] = 0;
        self.words[at + 17] = 1.0f32.to_bits();
        self.words[at + 18] = 1;
        self.words[at + 19] = region as u32;
        self.words[at + 20] = 1;
        self.words[at + 21] = arm as u32;
        self.words[at + 22] = 0;
        self.words[at + 23] = 0;
        self.words[at + 24] = 0;
        self.words[at + 25] = 0;
        self.words[at + 26] = 0;
        self.nrec += 1;
    }

    /// The engine's two construction hooks, replayed after the wire skills:
    /// asitame for every runner past the 1200 raw power arm, and staminasyoubu
    /// for every runner when the request flagged later stamina on a course
    /// past 2100 m (lib.rs:545-547, race_solver_builder.rs:817-848).
    fn classic_injections(
        &mut self,
        course: &Value,
        extras: Option<&ClassicExtras>,
        runners: usize,
    ) {
        if self.lane != Lane::Classic {
            return;
        }
        let Some(extras) = extras else {
            return;
        };
        let distance = match course.get_f64("distance") {
            Ok(v) => v,
            Err(why) => {
                self.unsupported(format!("course distance: {why}"));
                return;
            }
        };
        let distance_type = course.get_i64("distanceType").unwrap_or(0) as i32;
        for r in 0..runners {
            let raw_power = extras.raw_power.get(r).copied().unwrap_or(0.);
            if raw_power > 1200.0 {
                let strategy = extras.strategy.get(r).copied().unwrap_or(0);
                self.pack_asitame(r, raw_power, strategy, distance_type, distance);
                if self.reason.is_some() {
                    return;
                }
            }
        }
        if extras.later_stamina && distance > 2100.0 {
            for r in 0..runners {
                self.pack_staminasyoubu(r, distance);
                if self.reason.is_some() {
                    return;
                }
            }
        }
    }

    /// The engine's staminasyoubu hook: one record per runner on a course
    /// past 2100 m, trigger chain [phase 2 start, distance] and one TargetSpeed
    /// effect. The wire modifier 0 is replaced at activation by the live
    /// raw-stamina formula (race_solver.rs:2496-2498), so the compiled
    /// modifier word carries the distance factor alone and the kernel applies
    /// sqrt(raw stamina - 1200) * 0.0085 to it at the fire
    /// (race_solver_builder.rs:181-184, :167-179).
    fn pack_staminasyoubu(&mut self, runner: usize, distance: f64) {
        if self.nrec >= SK_RMAX {
            self.unsupported("skill table overflow: too many records".to_string());
            return;
        }
        if self.nreg + 1 > SK_GMAX || self.nprog + 1 > SK_PMAX || self.narm + 1 > SK_AMAX
            || self.neff + 1 > SK_EMAX
        {
            self.unsupported("skill table overflow: staminasyoubu injection".to_string());
            return;
        }
        let region = self.nreg;
        let at = SK_REGIONS + SK_GSTRIDE * self.nreg;
        let w = f64_words(distance * 2.0 / 3.0);
        self.words[at] = w[0];
        self.words[at + 1] = w[1];
        let w = f64_words(distance);
        self.words[at + 2] = w[0];
        self.words[at + 3] = w[1];
        self.nreg += 1;
        let prog = self.nprog;
        self.words[SK_PROGRAMS + SK_PSTRIDE * self.nprog] = OP_TRUE;
        self.nprog += 1;
        let arm = self.narm;
        let at = SK_ARMS + SK_ASTRIDE * self.narm;
        self.words[at] = 1;
        self.words[at + 1] = prog as u32;
        self.words[at + 2] = 0;
        self.words[at + 3] = 0;
        self.words[at + 4] = 1;
        self.words[at + 5] = region as u32;
        self.words[at + 6] = 0;
        self.words[at + 7] = u32::MAX;
        self.narm += 1;
        let effect = self.neff;
        let at = SK_EFFECTS + SK_ESTRIDE * self.neff;
        self.words[at] = 27;
        self.words[at + 1] = 1;
        self.words[at + 2] = 1;
        self.words[at + 3] = 1;
        self.words[at + 4] = 0;
        self.words[at + 5] = 0;
        let w = f64_words(stamina_syoubu_distance_factor(distance));
        self.words[at + 6] = w[0];
        self.words[at + 7] = w[1];
        let w = f64_words(9999.0);
        self.words[at + 8] = w[0];
        self.words[at + 9] = w[1];
        self.neff += 1;
        let at = SK_RECORDS + SK_RSTRIDE * self.nrec;
        self.words[at] = 0;
        self.words[at + 1] = runner as u32;
        self.words[at + 2] = SK_FLAG_RETAIN | SK_FLAG_PRIMARY | SK_FLAG_SYOUBU;
        self.words[at + 3] = 0;
        self.words[at + 4] = 1;
        self.words[at + 5] = 1;
        self.words[at + 6] = effect as u32;
        self.words[at + 7] = 1;
        self.words[at + 8] = prog as u32;
        self.words[at + 9] = 0;
        self.words[at + 10] = u32::MAX;
        self.words[at + 11] = 0;
        self.words[at + 12] = 0;
        self.words[at + 13] = 0;
        self.words[at + 14] = 0;
        self.words[at + 15] = 0;
        self.words[at + 16] = 0;
        self.words[at + 17] = 1.0f32.to_bits();
        self.words[at + 18] = 1;
        self.words[at + 19] = region as u32;
        self.words[at + 20] = 1;
        self.words[at + 21] = arm as u32;
        self.words[at + 22] = 0;
        self.words[at + 23] = 0;
        self.words[at + 24] = 0;
        self.words[at + 25] = 0;
        self.words[at + 26] = 0;
        self.nrec += 1;
    }

    /// The blocked-side timer is solver state, so the kernel's live reader
    /// serves it in a condition term and in a precondition alike. The eight
    /// field-backed readers stay condition-only, RegionDyn leaves stay refused.
    fn classic_refuses_leaf(
        &mut self,
        sid: &str,
        terms: &[Term],
        leaves: &[Leaf],
        live_ok: bool,
    ) -> bool {
        for (t, leaf) in terms.iter().zip(leaves) {
            if matches!(leaf, Leaf::Sampled) {
                continue;
            }
            let feat = match leaf {
                Leaf::Dynamic { feat, .. } => Some(*feat),
                Leaf::RegionDyn(_, feat) => Some(*feat),
                _ => None,
            };
            let Some(feat) = feat else { continue };
            let reader_ported = matches!(leaf, Leaf::Dynamic { .. })
                && (feat == FEAT_BSIDE
                    || (live_ok
                        && matches!(
                            feat,
                            FEAT_CLOCK
                                | FEAT_HP_PER
                                | FEAT_ORDER
                                | FEAT_ORDER_RATE
                                | FEAT_NEAR
                                | FEAT_NEARIF
                                | FEAT_SURR
                                | FEAT_VISN
                        )));
            if reader_ported {
                continue;
            }
            self.unsupported(format!(
                "skill {sid} condition term '{}' reads live runtime state with no reader in the classic kernel",
                t.name
            ));
            return true;
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

// The engine's asitame coefficient table (race_solver_builder.rs:139-145).
const ASITAME_STRATEGY_DISTANCE_COEFFICIENT: [[f64; 6]; 5] = [
    [0.0; 6],
    [0.0, 1.0, 0.7, 0.75, 0.7, 1.0],
    [0.0, 1.0, 0.8, 0.7, 0.75, 1.0],
    [0.0, 1.0, 0.9, 0.875, 0.86, 1.0],
    [0.0, 1.0, 0.9, 1.0, 0.9, 1.0],
];

/// `asitame_release_modifier` (race_solver_builder.rs:149-156): float operands
/// and a float sqrt, one double expression product rounded to float.
fn asitame_release_modifier(power: f64, strategy: i32, distance_type: i32) -> f64 {
    let strategy = strategy.clamp(0, 5) as usize;
    let coef = ASITAME_STRATEGY_DISTANCE_COEFFICIENT
        .get(distance_type as usize)
        .or(ASITAME_STRATEGY_DISTANCE_COEFFICIENT.last())
        .map(|row| row[strategy])
        .unwrap_or(0.0);
    let arg = ((power as f32 - 1200.0_f32) * 130.0_f32).max(0.);
    let root = (arg as f64).sqrt() as f32;
    (root as f64 * 0.001_f32 as f64 * coef as f32 as f64) as f32 as f64
}

/// `stamina_syoubu_distance_factor` (race_solver_builder.rs:167-179): the
/// course-distance tier of the staminasyoubu activation value.
fn stamina_syoubu_distance_factor(distance: f64) -> f64 {
    if distance < 2101.0 {
        0.0
    } else if distance < 2201.0 {
        0.5
    } else if distance < 2401.0 {
        1.0
    } else if distance < 2601.0 {
        1.2
    } else {
        1.5
    }
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

fn ref_wisdom_seed(horse_seed: u32, skill_id: u32) -> u32 {
    let mut seed = horse_seed ^ skill_id ^ 0x57495344;
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x85ebca6b);
    seed = (seed ^ (seed >> 13)).wrapping_mul(0xc2b2ae35);
    seed ^ (seed >> 16)
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
    // The record's skill id as written, the key of the engine's skill-id
    // tables (phase fudge list, corner-random quad list).
    sid_text: String,
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
            sid_text: sid.to_string(),
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
    Leaf::Dynamic { feat, cmp: t.op, arg: Some(t.arg as f64) }
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
fn classify(t: &Term, g: &Geometry, s: &Statics, lane: Lane) -> Result<Leaf, String> {
    let d = g.distance;
    match t.name.as_str() {
        // The classic lane resolves these course filters as region lists,
        // with the engine filter bodies' own error texts.
        "phase" | "remain_distance" | "corner" | "is_finalcorner" | "is_finalcorner_laterhalf"
        | "slope" | "furlong" | "is_last_straight" | "is_last_straight_onetime"
            if lane == Lane::Classic =>
        {
            Ok(Leaf::Region(classic_regions(t, g, s).map_err(engine_quote)?))
        }
        // Classic sampled leaves become host sampler jobs. Any op or arg is
        // accepted here, and the filter's own error text surfaces at the
        // arms fold.
        "corner_random" | "distance_rate_after_random" | "down_slope_random"
        | "up_slope_random" | "up_slope_random_later_half" | "is_finalcorner_random"
        | "phase_random" | "phase_firsthalf_random" | "phase_firstquarter_random"
        | "phase_laterhalf_random" | "phase_corner_random" | "phase_straight_random"
        | "phase_first_half_straight_random" | "phase_latter_half_straight_random"
        | "run_at_full_speed_random" | "straight_random" | "last_straight_random"
        | "all_corner_random"
            if lane == Lane::Classic =>
        {
            classic_sampled_regions(t, g, s).map_err(engine_quote)?;
            Ok(Leaf::Sampled)
        }
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
        // The engine's distance_rate comparison is pure region algebra, an
        // intersect with [rate*d/100, d] or [0, rate*d/100] and no live check
        // (activation_conditions.rs:574-581, registered at :1552), so the
        // classic runtime ports it as a course window.
        "distance_rate" if lane == Lane::Classic && matches!(t.op, 3 | 5) => Ok(Leaf::Region(vec![
            if t.op == 5 {
                (d * t.arg as f64 / 100.0, d)
            } else {
                (0.0, d * t.arg as f64 / 100.0)
            },
        ])),
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
