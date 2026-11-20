//! Wave-12b classic-profile folds: the compiler's classic lane lowers the
//! classic effect-value chain (`modifier / 10000 * multiplier`, `base_duration
//! / 10000`, race_solver_builder.rs:415 and :418 else branches), resolves
//! sampled triggers through host sampler jobs and static course filters as
//! region lists, and refuses the live-state condition leaves the classic
//! kernel has no reader for. The retail profile of the same skill keeps its
//! f32 chain and its dynamic leaves. Every fold checks both directions: the
//! classic value equals the classic formula and differs from the compiled
//! retail value, and the refusal names the offending leaf.

#![cfg(feature = "race")]

use globalsim_gpu::{json, skills_compile};
use skills_compile::{SK_EFFECTS, SK_FLAG_DEAD, SK_FLAG_DYN, SK_NREC, SK_RECORDS};

/// Hakodate-class turf 2000: surface 1, race_track_id 10005, the course the
/// wave-1 folds use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

/// Compile one synthetic self-target effect under the classic profile.
fn classic_plan(cond: &str, effect: &str, base_duration: i64) -> skills_compile::SkillsPlan {
    compile(cond, effect, base_duration, true)
}

fn retail_plan(cond: &str, effect: &str, base_duration: i64) -> skills_compile::SkillsPlan {
    compile(cond, effect, base_duration, false)
}

fn compile(
    cond: &str,
    effect: &str,
    base_duration: i64,
    classic: bool,
) -> skills_compile::SkillsPlan {
    let skill = format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":0,"baseDuration":{base_duration},"effects":[{effect}]}}]}}"#
    );
    let root_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550001\":{skill}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"550001\",\"level\":2}}]}}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    if classic {
        skills_compile::plan_classic(&root, &course)
    } else {
        skills_compile::plan(&root, &course)
    }
}

fn f64_at(plan: &skills_compile::SkillsPlan, word: usize) -> f64 {
    let lo = u64::from(plan.words[word]);
    let hi = u64::from(plan.words[word + 1]);
    f64::from_bits(lo | (hi << 32))
}

/// The first effect's modifier (words +6/+7) and base duration (words +8/+9).
fn first_effect(plan: &skills_compile::SkillsPlan) -> (f64, f64) {
    assert!(plan.reason.is_none(), "plan refused: {:?}", plan.reason);
    let at = SK_EFFECTS;
    (f64_at(plan, at + 6), f64_at(plan, at + 8))
}

/// Independent mirror of the classic else-branch so a compiler bug cannot make
/// both sides agree.
fn classic_value(raw: f64, multiplier: f64) -> f64 {
    raw / 10000.0 * multiplier
}

#[test]
fn effect_value_uses_the_classic_chain_and_differs_from_retail() {
    // levelModifiers picks the level-2 coefficient 1.02. 12345 is not a whole
    // multiple of 10000, so the f32 retail and f64 classic paths land apart.
    let raw = 12345.0_f64;
    let multiplier = 1.02_f64;
    let effect = format!(
        r#"{{"type":2,"target":1,"valueUsage":1,"modifier":{raw},"levelModifiers":[1,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18]}}"#
    );
    let classic = classic_plan("always==1", &effect, 50000);
    let retail = retail_plan("always==1", &effect, 50000);
    let (classic_mod, _) = first_effect(&classic);
    let (retail_mod, _) = first_effect(&retail);
    assert_eq!(
        classic_mod.to_bits(),
        classic_value(raw, multiplier).to_bits(),
        "classic modifier must be raw / 10000 * multiplier"
    );
    assert_ne!(
        classic_mod.to_bits(),
        retail_mod.to_bits(),
        "classic and retail value chains must differ on this value (classic {classic_mod}, retail {retail_mod})"
    );
}

#[test]
fn duration_uses_the_classic_chain_and_differs_from_retail() {
    // 33333 ms divides exactly in f64 but not through the f32 0.0001 constant,
    // so the two duration chains disagree in the low bits.
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    let classic = classic_plan("always==1", effect, 33333);
    let retail = retail_plan("always==1", effect, 33333);
    let (_, classic_dur) = first_effect(&classic);
    let (_, retail_dur) = first_effect(&retail);
    assert_eq!(
        classic_dur.to_bits(),
        (33333.0_f64 / 10000.0).to_bits(),
        "classic duration must be base_duration / 10000"
    );
    assert_ne!(
        classic_dur.to_bits(),
        retail_dur.to_bits(),
        "classic and retail duration chains must differ here (classic {classic_dur}, retail {retail_dur})"
    );
}

/// A constant-folded course condition survives the classic leaf policy, so the
/// real start-up family that gates on `track_id` compiles live.
#[test]
fn constant_and_region_leaves_survive_the_classic_policy() {
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":600000}"#;
    for cond in [
        "always==1",
        "track_id==10005",
        "ground_type==1",
        "is_last_straight==1",
    ] {
        let plan = classic_plan(cond, effect, 50000);
        assert!(plan.reason.is_none(), "{cond}: classic refusal {:?}", plan.reason);
        let nrec = plan.words[SK_NREC] as usize;
        assert_eq!(nrec, 1, "{cond}: one live record expected");
        // On this course these folds are all true, so the record stays live.
        assert_eq!(plan.words[SK_RECORDS + 2] & SK_FLAG_DEAD, 0, "{cond}: expected a live record");
    }
}

#[test]
fn sampled_leaf_compiles_classic_and_an_empty_chain_folds_dead() {
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    let classic = classic_plan("all_corner_random==1", effect, 50000);
    // The classic lane resolves sampled leaves with the host builder
    // sampler, so the leaf compiles. This course has no corners, so the
    // filter chain folds empty: the engine skips the alternative before
    // sampling, and the kernel keeps the record DEAD with no sampler job.
    assert!(classic.reason.is_none(), "sampled leaf must compile in classic: {:?}", classic.reason);
    assert_eq!(classic.words[SK_NREC], 1, "one record expected");
    assert_ne!(
        classic.words[SK_RECORDS + 2] & SK_FLAG_DEAD,
        0,
        "an empty corner chain folds the record DEAD"
    );
    assert!(classic.classic_jobs.is_empty(), "a folded chain never reaches the sampler");
    // The retail profile still compiles the sampled leaf.
    let retail = retail_plan("all_corner_random==1", effect, 50000);
    assert!(retail.reason.is_none(), "retail keeps the sampled leaf: {:?}", retail.reason);
}

/// On a course with corners the same leaf compiles live and carries one
/// sampler job: the AllCorner policy, the corner chain the windows draw
/// from, and the arm slot the windows repoint at start.
#[test]
fn sampled_leaf_emits_one_classic_sampler_job() {
    let course = json::parse(
        r#"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"straights":[{"start":0.0,"end":450.0},{"start":1100.0,"end":2000.0}],"corners":[{"start":450.0,"length":650.0}],"slopes":[]}"#,
    )
    .expect("course parses");
    let skill = format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"all_corner_random==1","cooldown":0,"baseDuration":50000,"effects":[{{"type":2,"target":1,"valueUsage":1,"modifier":1000}}]}}]}}"#
    );
    let root_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550001\":{skill}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"550001\",\"level\":2}}]}}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let plan = skills_compile::plan_classic(&root, &course);
    assert!(plan.reason.is_none(), "sampled leaf must compile classic: {:?}", plan.reason);
    assert_eq!(plan.words[SK_NREC], 1, "one live record expected");
    assert_eq!(plan.words[SK_RECORDS + 2] & SK_FLAG_DEAD, 0, "the corner chain keeps the record live");
    assert_eq!(plan.classic_jobs.len(), 1, "one sampler job expected");
    let job = &plan.classic_jobs[0];
    assert_eq!(job.policy, 2, "all_corner_random is the AllCorner policy");
    assert_eq!(job.regions, vec![(450.0, 1100.0)], "the job draws from the corner chain, got {:?}", job.regions);
    assert_eq!(job.record, 0, "the job names its record");
    assert_eq!(job.arm_slots.len(), 1, "the live arm slot is repointed by the windows");
}

#[test]
fn a_classic_build_error_drops_only_that_skill() {
    // The engine's classic lane answers the race with the failing skill
    // dropped (lib.rs:728-739): the broken skill's condition errors inside
    // the engine's own build, so the kernel drops it too. Nothing of the
    // broken skill survives, and the next skill compiles untouched.
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    let mk = |cond: &str| {
        format!(
            r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":0,"baseDuration":50000,"effects":[{effect}]}}]}}"#
        )
    };
    let root_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550002\":{},\"550001\":{}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"550002\",\"level\":2}},{{\"skillId\":\"550001\",\"level\":2}}]}}]}}",
        mk("all_corner_random==0"),
        mk("always==1"),
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    let plan = skills_compile::plan_classic(&root, &course);
    assert!(
        plan.reason.is_none(),
        "an engine build error must drop the skill, not refuse the race: {:?}",
        plan.reason
    );
    assert_eq!(plan.words[SK_NREC], 1, "only the healthy skill keeps a record");
    assert_eq!(plan.words[SK_RECORDS], 550001, "the survivor is the healthy skill");
    assert!(plan.classic_jobs.is_empty(), "a dropped sampled leaf never reaches the sampler");
    // Every alternative of a broken skill goes with it: the engine's probe
    // builds the whole skill alone, so a live sibling alternative stays too.
    let alt_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550002\":{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[\
         {{\"precondition\":\"\",\"condition\":\"all_corner_random==0\",\"cooldown\":0,\"baseDuration\":50000,\"effects\":[{effect}]}}\
         ,{{\"precondition\":\"\",\"condition\":\"always==1\",\"cooldown\":0,\"baseDuration\":50000,\"effects\":[{effect}]}}]}}}},\
         \"horses\":[{{\"skills\":[{{\"skillId\":\"550002\",\"level\":2}}]}}]}}"
    );
    let alt_root = json::parse(&alt_text).expect("alt root parses");
    let alt_plan = skills_compile::plan_classic(&alt_root, &course);
    assert!(
        alt_plan.reason.is_none(),
        "the probe errors for the whole skill, so every alternative drops together: {:?}",
        alt_plan.reason
    );
    assert_eq!(alt_plan.words[SK_NREC], 0, "the broken skill's live alternative drops with it");
    // A skill id absent from the request's skillData is the same engine
    // build error (race_solver_builder.rs:496 "bad skill ID {id}"), so the
    // classic lane drops it and answers with the survivors.
    let miss_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550001\":{}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"550003\",\"level\":2}},{{\"skillId\":\"550001\",\"level\":2}}]}}]}}",
        mk("always==1"),
    );
    let miss_root = json::parse(&miss_text).expect("miss root parses");
    let miss_plan = skills_compile::plan_classic(&miss_root, &course);
    assert!(
        miss_plan.reason.is_none(),
        "a missing skillData entry must drop the skill, not refuse the race: {:?}",
        miss_plan.reason
    );
    assert_eq!(miss_plan.words[SK_NREC], 1, "only the listed skill keeps a record");
    assert_eq!(miss_plan.words[SK_RECORDS], 550001, "the survivor is the listed skill");
}

/// A sampled leaf in a classic precondition is a static filter: the engine
/// clips the precondition chain with it and never samples it, so no sampler
/// job appears. Retail still refuses the shape.
#[test]
fn a_classic_sampled_precondition_folds_regions_without_a_job() {
    let course = json::parse(
        r#"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"straights":[{"start":0.0,"end":450.0},{"start":1100.0,"end":2000.0}],"corners":[{"start":450.0,"length":650.0}],"slopes":[]}"#,
    )
    .expect("course parses");
    let skill = r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{"precondition":"all_corner_random==1","condition":"always==1","cooldown":0,"baseDuration":50000,"effects":[{"type":2,"target":1,"valueUsage":1,"modifier":1000}]}]}"#;
    let root_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         ,\"skillData\":{{\"550001\":{skill}}},\"horses\":[{{\"skills\":[{{\"skillId\":\"550001\",\"level\":2}}]}}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let plan = skills_compile::plan_classic(&root, &course);
    assert!(
        plan.reason.is_none(),
        "classic takes the sampled precondition as a static filter: {:?}",
        plan.reason
    );
    assert_eq!(plan.words[SK_NREC], 1, "one live record expected");
    assert_eq!(plan.words[SK_RECORDS + 2] & SK_FLAG_DEAD, 0, "the corner precondition keeps the record live");
    assert_eq!(plan.words[SK_RECORDS + 22], 1, "the precondition chain carries the corner region");
    assert!(plan.classic_jobs.is_empty(), "a precondition never reaches the sampler");
    let retail = skills_compile::plan(&root, &course);
    let reason = retail
        .reason
        .as_deref()
        .expect("retail keeps refusing sampled precondition terms");
    assert!(reason.contains("precondition random term"), "got {reason}");
}

#[test]
fn classic_course_filters_compile_and_live_runtime_reads_refuse() {
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    // The static course filters fold to region lists in classic, so the
    // merged-window membership gate answers them with no runtime reader.
    for cond in ["phase>=2", "remain_distance<=400", "corner==0", "slope==0", "is_finalcorner==1"] {
        let classic = classic_plan(cond, effect, 50000);
        assert!(classic.reason.is_none(), "{cond}: course filter must compile classic: {:?}", classic.reason);
        let retail = retail_plan(cond, effect, 50000);
        assert!(retail.reason.is_none(), "retail keeps {cond}: {:?}", retail.reason);
    }
    // The classic kernel carries live readers for these leaves, so the
    // profile compiles them with the scan-time dynamic gate flag set.
    for cond in [
        "hp_per<=50",
        "order_rate>=40",
        "accumulatetime>=5",
        "visiblehorse>=1",
        "near_count<=2",
        "near_infront_count==0",
        "is_surrounded==1",
        "order<=4",
    ] {
        let classic = classic_plan(cond, effect, 50000);
        assert!(classic.reason.is_none(), "{cond}: classic reader leaf must compile: {:?}", classic.reason);
        assert_eq!(classic.words[SK_NREC], 1, "{cond}: one live record expected");
        assert_eq!(
            classic.words[SK_RECORDS + 2] & SK_FLAG_DYN,
            SK_FLAG_DYN,
            "{cond}: the record must carry the dynamic gate flag"
        );
        let retail = retail_plan(cond, effect, 50000);
        assert!(retail.reason.is_none(), "retail keeps {cond}: {:?}", retail.reason);
        assert_eq!(
            retail.words[SK_RECORDS + 2] & SK_FLAG_DYN,
            0,
            "{cond}: retail never sees the classic gate flag"
        );
    }
    // Live reads the classic kernel has no reader for stay refused by name.
    for cond in ["distance_diff_rate>=20", "blocked_front==1", "change_order_onetime>=1"] {
        let leaf = cond.split(['=', '<', '>']).next().unwrap();
        let classic = classic_plan(cond, effect, 50000);
        let reason = classic
            .reason
            .as_deref()
            .unwrap_or_else(|| panic!("{cond}: classic live-state leaf must refuse"));
        assert!(reason.contains(leaf), "must name the live-state leaf '{leaf}', got {reason}");
        let retail = retail_plan(cond, effect, 50000);
        assert!(retail.reason.is_none(), "retail keeps {leaf}: {:?}", retail.reason);
    }
}

/// The refusal fires even when the offending leaf is not the first term, so a
/// constant guard cannot mask a live-state term that follows it.
#[test]
fn a_later_live_state_leaf_still_refuses() {
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    let plan = classic_plan("track_id==10005&distance_diff_rate>=20", effect, 50000);
    let reason = plan.reason.as_deref().expect("later live-state leaf must refuse");
    assert!(reason.contains("distance_diff_rate"), "must name 'distance_diff_rate', got {reason}");
}

/// A DEAD (folded-false constant) record is still a compiled record, not a
/// refusal: a constant that folds false is a legitimate never-firing skill.
#[test]
fn folded_false_constant_compiles_dead_not_refused() {
    let effect = r#"{"type":2,"target":1,"valueUsage":1,"modifier":1000}"#;
    let plan = classic_plan("track_id==10006", effect, 50000);
    assert!(plan.reason.is_none(), "track_id!=course is a fold, not a refusal: {:?}", plan.reason);
    assert_eq!(plan.words[SK_NREC], 1, "one DEAD record expected");
    assert_ne!(
        plan.words[SK_RECORDS + 2] & SK_FLAG_DEAD,
        0,
        "the false-fold record is DEAD-flagged"
    );
}

/// The real merged-table start-up skill 200111 (track_id-conditioned self
/// SpeedUp) compiles under the classic profile with the classic else-branch
/// value 600000/10000*1 and duration -1/10000.
#[test]
fn merged_startup_skill_200111_compiles_classic_with_classic_values() {
    let root = merged_root_with_200111();
    let course = json::parse(
        r#"{"distance":2000.0,"raceTrackId":10003,"surface":1,"distanceType":3,"turn":1,"straights":[{"start":0.0,"end":2000.0}],"corners":[],"slopes":[]}"#,
    )
    .expect("course parses");
    let plan = skills_compile::plan_classic(&root, &course);
    assert!(plan.reason.is_none(), "200111 must compile classic: {:?}", plan.reason);
    assert_eq!(plan.words[SK_NREC], 1, "one record expected");
    let (modifier, duration) = first_effect(&plan);
    assert_eq!(modifier.to_bits(), (600000.0_f64 / 10000.0 * 1.0).to_bits(), "200111 classic value");
    assert_eq!(duration.to_bits(), (-1.0_f64 / 10000.0).to_bits(), "200111 classic duration");
}

fn merged_root_with_200111() -> json::Value {
    // The definition is the same shape the merged skills.json carries for
    // 200111 (rarity 1, track_id-conditioned, self SpeedUp type 2, duration -1,
    // cooldown 0), inlined so the fold does not depend on the app repo's table.
    let root_text = r##"{"skillData":{"200111":{"rarity":1,"wisdomCheck":0,"priority":0,"tags":[402,604],"alternatives":[{"condition":"track_id==10003","precondition":"","baseDuration":-1,"durationUsage":1,"cooldown":0,"effects":[{"type":2,"target":1,"targetValue":0,"additionalActivate":0,"valueUsage":1,"modifier":600000,"levelModifiers":[1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.1]}]}],"isGeneralSkill":true}},"horses":[{"skills":[{"skillId":"200111","level":1}]}]}"##;
    json::parse(root_text).expect("200111 root parses")
}
