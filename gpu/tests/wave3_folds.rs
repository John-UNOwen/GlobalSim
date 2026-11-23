//! Wave-3 compile gates: the condition leaves this wave ports lower to live
//! records, the roster reads fold per runner, the activation-history folds are
//! provable at build time, and the constructs that stay unported keep refusing
//! by name.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE};

/// Hakodate-class turf 2000, the course the wave-1, wave-2 and wave-2b gates use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_root(skill_data: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text = format!("{{\"skillData\":{{{skill_data}}},\"horses\":[{horses}]}}");
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

fn plan_root_classic(skill_data: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text = format!("{{\"skillData\":{{{skill_data}}},\"horses\":[{horses}]}}");
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan_classic(&root, &course)
}

/// One alternative with the given condition, cooldown and effect list.
fn alt(cond: &str, cooldown: i64, effects: &[&str]) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"effects\":[{}]}}]}}",
        effects.join(",")
    )
}

fn plan_skill(cond: &str) -> skills_compile::SkillsPlan {
    plan_root(
        &format!("\"550001\":{}", alt(cond, 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
        r#"{"skills":["550001"]}"#,
    )
}

fn one_runner(cond: &str) -> skills_compile::SkillsPlan {
    plan_skill(cond)
}

fn flags(plan: &skills_compile::SkillsPlan, rec: usize) -> u32 {
    let nrec = plan.words[SK_NREC] as usize;
    assert!(rec < nrec, "record {rec} of {nrec} expected");
    plan.words[SK_RECORDS + SK_RSTRIDE * rec + 2]
}

fn is_live(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a live record, plan refused: {reason}");
    }
    assert_eq!(flags(plan, 0) & SK_FLAG_DEAD, 0, "{what}: expected the live lowering");
}

fn is_dead(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a DEAD record, plan refused: {reason}");
    }
    assert_ne!(flags(plan, 0) & SK_FLAG_DEAD, 0, "{what}: expected the DEAD lowering");
}

fn refused(plan: &skills_compile::SkillsPlan, name: &str) {
    let reason = plan.reason.as_deref().unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
}

#[test]
fn wave3_dynamic_leaves_stay_live() {
    for cond in [
        "order_rate_in20_continue==1",
        "order_rate_in40_continue==1",
        "order_rate_in50_continue==1",
        "order_rate_in80_continue==1",
        "near_infront_count>=2",
        "visiblehorse>=1",
        "is_activate_heal_skill==1",
        "is_activate_heal_skill==0",
        "activate_count_all>=1",
        "activate_count_all<=5",
        "activate_count_end_after>=2",
        "activate_count_later_half>=2",
        "overtake_target_no_order_up_time>=1",
        "overtake_target_no_order_down_time>=1",
        "change_order_up_middle>=1",
        "change_order_up_finalcorner_after>=1",
        "blocked_all_continuetime>=2",
        "blocked_front==1",
        "is_surrounded==1",
        "temptation_count_behind>=1",
        "temptation_count_infront>=1",
        "temptation_opponent_count_behind>=1",
        "temptation_opponent_count_infront>=1",
        "running_style_temptation_count_nige>=1",
        "running_style_temptation_count_senko>=1",
        "running_style_temptation_count_sashi>=1",
        "running_style_temptation_count_oikomi>=1",
        "running_style_temptation_opponent_count_nige>=1",
        "running_style_temptation_opponent_count_oikomi>=1",
        "distance_diff_rate>=75",
        "distance_diff_top_float>=1",
    ] {
        is_live(&one_runner(cond), cond);
    }
}

#[test]
fn inband_cells_gate_the_registered_forms() {
    // order_in_filter registers == with arg 1 only, and the four rates named
    // in the table; 30 is not one of them.
    refused(&one_runner("order_rate_in30_continue==1"), "order_rate_in30_continue");
    refused(&one_runner("order_rate_in50_continue!=1"), "order_rate_in50_continue");
    refused(&one_runner("order_rate_in50_continue==0"), "order_rate_in50_continue");
    refused(&one_runner("order_rate_in20_continue>=1"), "order_rate_in20_continue");
}

#[test]
fn restricted_comparators_stay_refused() {
    // activate_count_all registers lte and gte, the end_after and later_half
    // filters gte only, is_used_skill_id and the heal / detail checks eq only.
    refused(&one_runner("activate_count_all==3"), "activate_count_all");
    refused(&one_runner("activate_count_end_after==1"), "activate_count_end_after");
    refused(&one_runner("activate_count_later_half==2"), "activate_count_later_half");
    refused(&one_runner("is_used_skill_id>=200001"), "is_used_skill_id");
    refused(&one_runner("is_activate_other_skill_detail==0"), "is_activate_other_skill_detail");
    refused(&one_runner("is_activate_heal_skill>=1"), "is_activate_heal_skill");
}

#[test]
fn activation_history_folds() {
    // Two skills on one runner: the record under test (550001) and a sibling
    // the table can fire first (550002).
    let skill_data = format!(
        "\"550001\":{},\"550002\":{}",
        alt("is_used_skill_id==550002", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"]),
        alt("always>=0", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"]),
    );
    let plan = plan_root(&skill_data, r#"{"skills":["550001","550002"]}"#);
    if let Some(reason) = &plan.reason {
        panic!("sibling used-id leaf expected live, plan refused: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 2, "two records expected");
    assert_eq!(flags(&plan, 0) & SK_FLAG_DEAD, 0, "is_used_skill_id==550002 must stay live");

    // An id the roster never carries cannot enter used_skills.
    is_dead(
        &one_runner("is_used_skill_id==777777"),
        "is_used_skill_id for an absent id",
    );

    // The follow-up arm reads the id group's fired bit, set when an
    // alternative activates (race_solver.rs:2669). The classic kernel tracks
    // no used bit, so the arm folds dead there.
    is_live(
        &one_runner("is_activate_other_skill_detail==1"),
        "is_activate_other_skill_detail self gate",
    );
    is_dead(
        &plan_root_classic(
            &format!(
                "\"550001\":{}",
                alt("is_activate_other_skill_detail==1", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])
            ),
            r#"{"skills":["550001"]}"#,
        ),
        "is_activate_other_skill_detail on the classic lane",
    );
}

#[test]
fn roster_reads_fold() {
    // Nige, Nige, Sashi from runner 0's seat: two runners share the base
    // style including self, out of three.
    let roster = concat!(
        r#"{"skills":["550001"],"strategy":"Nige"},"#,
        r#"{"skills":[],"strategy":"Nige"},"#,
        r#"{"skills":[],"strategy":"Sashi"}"#
    );
    let same = |cond: &str| -> skills_compile::SkillsPlan {
        plan_root(
            &format!("\"550001\":{}", alt(cond, 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
            roster,
        )
    };
    is_live(&same("running_style_count_same>=2"), "two nige including self");
    is_dead(&same("running_style_count_same>=3"), "not three nige");
    // The compared side is a percent of the three runners: 34% rounds to a
    // threshold of 1, 83% to 2 (2.49+0.5 truncates to 2), 84% to 3.
    is_live(&same("running_style_count_same_rate>=34"), "34 percent needs one");
    is_live(&same("running_style_count_same_rate>=83"), "83 percent is reached by two of three");
    is_dead(&same("running_style_count_same_rate>=84"), "84 percent needs a third runner");

    // The favorite flag folds only for a unique popularity==1 runner.
    let fav_match = concat!(
        r#"{"skills":["550001"],"strategy":"Nige","popularity":5},"#,
        r#"{"skills":[],"strategy":"Oonige","popularity":1},"#,
        r#"{"skills":[],"strategy":"Sashi","popularity":9}"#
    );
    let fav = |cond: &str, roster: &str| -> skills_compile::SkillsPlan {
        plan_root(
            &format!("\"550001\":{}", alt(cond, 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
            roster,
        )
    };
    is_live(&fav("running_style_equal_popularity_one==1", fav_match), "oonige matches nige");
    is_dead(&fav("running_style_equal_popularity_one==0", fav_match), "favorite exists");
    let fav_none = concat!(
        r#"{"skills":["550001"],"strategy":"Nige","popularity":5},"#,
        r#"{"skills":[],"strategy":"Oonige","popularity":2}"#
    );
    is_dead(
        &fav("running_style_equal_popularity_one!=1", fav_none),
        "no unique favorite makes every comparison false",
    );

    // same_skill_horse_count counts horses holding the id. The compiler
    // rejects a duplicate id on one runner outright, so the engine's per-horse
    // set dedup never sees one.
    let shared = concat!(
        r#"{"skills":["550001"],"strategy":"Nige"},"#,
        r#"{"skills":["550001"],"strategy":"Nige"},"#,
        r#"{"skills":["550002"],"strategy":"Sashi"}"#
    );
    let shared_data = format!(
        "\"550001\":{},\"550002\":{}",
        alt("same_skill_horse_count>=X", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"]),
        alt("always>=0", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"]),
    );
    let shared_plan = plan_root(
        &shared_data.replace("same_skill_horse_count>=X", "same_skill_horse_count>=2"),
        shared,
    );
    is_live(&shared_plan, "two horses carry the skill");
    is_dead(
        &plan_root(
            &shared_data.replace("same_skill_horse_count>=X", "same_skill_horse_count>=3"),
            shared,
        ),
        "not three horses",
    );

    // With a nonzero team the opponent counts drop same-team receivers. The
    // retail kernel mirrors the engine's is_opponent filter, so they stay
    // live there; the classic kernel has no filtered reader, so the classic
    // lane refuses the behind pair by name there.
    let teams = concat!(
        r#"{"skills":["550001"],"strategy":"Nige","teamId":3},"#,
        r#"{"skills":[],"strategy":"Nige","teamId":3}"#
    );
    let team_plan = |cond: &str| -> skills_compile::SkillsPlan {
        plan_root(
            &format!("\"550001\":{}", alt(cond, 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
            teams,
        )
    };
    is_live(
        &team_plan("temptation_opponent_count_behind>=1"),
        "retail opponent behind keeps working with teams",
    );
    is_live(
        &team_plan("running_style_temptation_opponent_count_nige>=1"),
        "retail opponent style keeps working with teams",
    );
    refused(
        &plan_root_classic(
            &format!("\"550001\":{}", alt("temptation_opponent_count_behind>=1", 500000, &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
            teams,
        ),
        "temptation_opponent_count_behind",
    );
    is_live(&team_plan("temptation_count_behind>=1"), "plain count keeps working");
}

#[test]
fn still_refused_constructs_by_name() {
    // The advantage-skill machine caches the field's previous-tick
    // (runner, kind) advantage pairs with team exclusion (family 2), so the
    // condition now lowers live. remain_distance_viewer_id ported in wave 9c.
    is_live(
        &one_runner("is_other_character_activate_advantage_skill==1"),
        "is_other_character_activate_advantage_skill",
    );
    // Not engine names at all: no registry entry can answer for them.
    refused(&one_runner("behind_time>=1"), "behind_time");
    refused(&one_runner("near_time>=1"), "near_time");
    refused(&one_runner("infront_time>=1"), "infront_time");
    refused(&one_runner("bashin_diff_side>=1"), "bashin_diff_side");
    refused(
        &one_runner("temptation_count_opponent_behind>=1"),
        "temptation_count_opponent_behind",
    );
}

/// One alternative with a precondition, condition and effect list.
fn alt_pre(pre: &str, cond: &str, effects: &[&str]) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"{pre}\",\"condition\":\"{cond}\",\"cooldown\":500000,\"baseDuration\":40000,\"effects\":[{}]}}]}}",
        effects.join(",")
    )
}

fn pre_plan(pre: &str) -> skills_compile::SkillsPlan {
    plan_root(
        &format!("\"550001\":{}", alt_pre(pre, "phase>=2", &["{\"type\":10,\"target\":1,\"modifier\":0}"])),
        r#"{"skills":["550001"]}"#,
    )
}

#[test]
fn precondition_or_groups_fold_to_a_union() {
    // The engine reads an '@' precondition as an Or over its groups. This
    // course's distanceType matches only the second group, and that group
    // folds true, so the watch latches across the whole course.
    is_live(&pre_plan("distance_type==4&phase>=2@distance_type==3"), "true group union");
    // Every group collapses and no alternative is built, the engine's skip.
    is_dead(&pre_plan("distance_type==4&phase>=2@distance_type==4"), "all-false groups");
    // Groups carrying dynamics pair their own chain with their own program,
    // the engine's Or closure, so they compile.
    is_live(&pre_plan("phase>=2@phase>=3"), "multi-group dynamics pair per group");
}

#[test]
fn condition_dead_still_gates_unported_targets() {
    // A condition that folds false inerts only the condition path. The record
    // stays a random-activation candidate (race_solver.rs:2710-2711), fired
    // with targets the engine draws fresh, so the target gate cannot be skipped.
    // The same shape with the team fan behind it: the fan is ported, so the
    // dead mirror still carries the random candidate.
    let dead11 = plan_root(
        &format!("\"550001\":{}", alt_pre("", "distance_type==4", &["{\"type\":27,\"target\":11,\"modifier\":1500}"])),
        r#"{"skills":["550001"]}"#,
    );
    is_dead(&dead11, "the ported target 11 fan keeps the dead mirror");
    // The dead mirror keeps its job for the effects the kernel can deliver.
    is_dead(
        &plan_root(
            &format!("\"550001\":{}", alt_pre("", "distance_type==4", &["{\"type\":27,\"target\":1,\"modifier\":1500}"])),
            r#"{"skills":["550001"]}"#,
        ),
        "self effect keeps the dead mirror",
    );
}
