//! Wave-2 compile gates: the dynamic leaves the fallback histogram found in
//! player skills lower to live records, the roster style counts fold per
//! runner, and the constructs that stay unported keep refusing by name.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{
    SK_FLAG_DEAD, SK_NREC, SK_NTERM, SK_RECORDS, SK_RSTRIDE, SK_TERMS, SK_TERM_DIST_RATE_AFTER,
};

/// Hakodate-class turf 2000, same course as the wave-1 fold gates.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_with(cond: &str, horses: &str) -> skills_compile::SkillsPlan {
    let skill = format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":0,"effects":[{{"type":10,"target":1,"modifier":0}}]}}]}}"#
    );
    let root_text = format!(
        "{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

fn one_runner(cond: &str) -> skills_compile::SkillsPlan {
    plan_with(cond, r#"{"skills":["550001"]}"#)
}

fn is_live(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a live record, plan refused: {reason}");
    }
    let nrec = plan.words[SK_NREC] as usize;
    assert_eq!(nrec, 1, "{what}: one record expected");
    let flags = plan.words[SK_RECORDS + SK_RSTRIDE * (nrec - 1) + 2];
    assert_eq!(flags & SK_FLAG_DEAD, 0, "{what}: expected the live lowering");
}

fn is_dead(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a DEAD record, plan refused: {reason}");
    }
    let nrec = plan.words[SK_NREC] as usize;
    assert_eq!(nrec, 1, "{what}: one record expected");
    let flags = plan.words[SK_RECORDS + SK_RSTRIDE * (nrec - 1) + 2];
    assert_ne!(flags & SK_FLAG_DEAD, 0, "{what}: expected the DEAD lowering");
}

fn refused(plan: &skills_compile::SkillsPlan, name: &str) {
    let reason = plan.reason.as_deref().unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
}

#[test]
fn wave2_dynamic_leaves_stay_live() {
    for cond in [
        "distance_diff_top<=10",
        "lane_type==0",
        "is_move_lane==1",
        "blocked_front_continuetime>=2",
        "is_behind_in==1",
        "compete_fight_count>0",
        "overtake_target_time>=2",
        "change_order_up_end_after>=1",
        "activate_count_heal>=1",
    ] {
        is_live(&one_runner(cond), cond);
    }
}

#[test]
fn distance_rate_after_random_samples_one_window() {
    let plan = one_runner("distance_rate_after_random==50");
    is_live(&plan, "distance_rate_after_random==50");
    assert_eq!(plan.words[SK_NTERM], 1, "one sampled term");
    assert_eq!(plan.words[SK_TERMS], SK_TERM_DIST_RATE_AFTER, "kind");
    assert_eq!(plan.words[SK_TERMS + 1], 50, "rate argument");
}

#[test]
fn restricted_comparators_stay_refused() {
    // The retail filter table registers >= only for the heal count, > only
    // for the duel count, and == only for the random term.
    refused(&one_runner("activate_count_heal==1"), "activate_count_heal");
    refused(&one_runner("activate_count_heal<2"), "activate_count_heal");
    refused(&one_runner("compete_fight_count>=1"), "compete_fight_count");
    refused(&one_runner("distance_rate_after_random!=50"), "distance_rate_after_random");
}

#[test]
fn unported_neighbours_refuse_by_name() {
    // The wave-3 port took change_order_up_*, blocked_all_continuetime,
    // distance_diff_top_float and overtake_target_no_order_* off this list.
    // blocked_side_continuetime_x is not an engine name at all.
    refused(&one_runner("blocked_side_continuetime_x>=1"), "blocked_side_continuetime_x");
}

#[test]
fn roster_style_counts_fold_per_runner() {
    // Nige, Nige, Sashi from runner 0's seat: one other nige, one sashi,
    // no senko or oikomi.
    let roster = concat!(
        r#"{"skills":["550001"],"strategy":"Nige"},"#,
        r#"{"skills":[],"strategy":"Nige"},"#,
        r#"{"skills":[],"strategy":"Sashi"}"#
    );
    is_live(&plan_with("running_style_count_nige_otherself>=1", roster), "one other nige");
    is_dead(&plan_with("running_style_count_nige_otherself>=2", roster), "not two other nige");
    is_live(&plan_with("running_style_count_sashi_otherself==1", roster), "one sashi");
    is_dead(&plan_with("running_style_count_senko_otherself>=1", roster), "no senko");
    is_live(&plan_with("running_style_count_oikomi_otherself==0", roster), "no oikomi");

    // Oonige folds into the nige count, so a lone Oonige has no other nige.
    let oonige = concat!(
        r#"{"skills":["550001"],"strategy":"Oonige"},"#,
        r#"{"skills":[],"strategy":"Sashi"}"#
    );
    is_dead(&plan_with("running_style_count_nige_otherself>=1", oonige), "oonige counts as nige");
    is_live(&plan_with("running_style_count_sashi_otherself==1", oonige), "one sashi");
}

#[test]
fn sampled_precondition_still_refuses() {
    // The engine still draws for a random precondition term, and the runtime
    // has no sampled precondition chain, so the race keeps falling back.
    let skill = r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{"precondition":"distance_rate_after_random==50","condition":"order_rate>=50","cooldown":0,"effects":[{"type":10,"target":1,"modifier":0}]}]}"#;
    let root_text = format!(
        "{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{{\"skills\":[\"550001\"]}}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    refused(&skills_compile::plan(&root, &course), "precondition random term");
}
