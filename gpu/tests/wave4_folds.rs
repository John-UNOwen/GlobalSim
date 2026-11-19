//! Wave-4 compile gates: the team-filtered debuff delivery is now a real port,
//! MultiplyStartDelay on another runner is an inert fire, and the remaining
//! effect families refuse by name at the exact engine boundary the kernel does
//! not model. Each refusal names the construct and the reason.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE};

/// Hakodate-class turf 2000, the same course the earlier fold gates use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_skill(skill: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text = format!("{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]}}");
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

/// One alternative with the given condition, cooldown and effect list.
fn alt(cond: &str, cooldown: i64, effects: &[&str]) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"effects\":[{}]}}]}}",
        effects.join(",")
    )
}

fn plan_effects(cond: &str, cooldown: i64, effects: &[&str], horses: &str) -> skills_compile::SkillsPlan {
    plan_skill(&alt(cond, cooldown, effects), horses)
}

fn one_runner(cond: &str, cooldown: i64, effects: &[&str]) -> skills_compile::SkillsPlan {
    plan_effects(cond, cooldown, effects, r#"{"skills":["550001"]}"#)
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

fn refused(plan: &skills_compile::SkillsPlan, name: &str) {
    let reason = plan
        .reason
        .as_deref()
        .unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
    assert!(reason.len() < 250, "refusal is a compact reason, {reason}");
}

#[test]
fn teamed_cross_debuff_compiles() {
    // The drop moved from a whole-roster flag to a per-receiver runtime gate,
    // so a negative cross target compiles whether the runner has a team or not.
    for roster in [
        r#"{"skills":["550001"]}"#,
        r#"{"skills":["550001"],"teamId":5}"#,
    ] {
        for (kind, target) in [(21i64, 9i64), (22, 10), (8, 4), (9, 10), (31, 9)] {
            let effect = format!("{{\"type\":{kind},\"target\":{target},\"modifier\":-1000,\"targetValue\":2}}");
            is_live(
                &plan_effects("order>=2", 500000, &[effect.as_str()], roster),
                "teamed cross debuff",
            );
        }
    }
}

#[test]
fn start_delay_cross_target_fires_inert() {
    // MultiplyStartDelay on another runner is inert in retail, so it lowers a
    // live record rather than refusing.
    for target in [4i64, 9, 10] {
        let effect = format!("{{\"type\":10,\"target\":{target},\"modifier\":0}}");
        is_live(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "inert start-delay cross",
        );
    }
    // The self target still lowers the init-time start delay factor.
    is_live(
        &one_runner("always>=0", 0, &[r#"{"type":10,"target":1,"modifier":0}"#]),
        "self start-delay",
    );
}

#[test]
fn stat_kinds_still_refuse_cross_targets() {
    // SpeedUp, GutsUp and WisdomUp on another runner write the receiver stats
    // through a refresh the kernel has no consumer for.
    for kind in [1i64, 4, 5] {
        let effect = format!("{{\"type\":{kind},\"target\":9,\"modifier\":1000}}");
        refused(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "targeting another runner",
        );
    }
}

#[test]
fn lane_goal_kinds_compile_both_directions() {
    // 11 and 12 arm a lane-preference presence, 35 a last-set lane goal; all
    // three lower now the passing search consumes them, and the unported
    // duration_usage tiers still refuse.
    for kind in [11i64, 12, 35] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        is_live(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "lane goal",
        );
    }
}

#[test]
fn kakari_allstat_and_variantless_kinds_refuse_by_name() {
    // 13 replaces the rush forced-end and ports; 32 is a mid-race all-stat
    // bump, 33 and 34 have no SkillType variant at all.
    is_live(
        &one_runner("order>=2", 500000, &[r#"{"type":13,"target":1,"modifier":5000}"#]),
        "kakari extend",
    );
    refused(
        &one_runner("order>=2", 500000, &[r#"{"type":32,"target":1,"modifier":5000}"#]),
        "all-stat bump",
    );
    for kind in [33i64, 34] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        refused(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "no SkillType variant",
        );
    }
}

#[test]
fn random_and_immune_kinds_refuse_by_name() {
    // 36 and 37 re-activate another skill through an activation-time draw,
    // still refused; 38 arms the immunity presence now the target filter
    // reads it.
    for kind in [36i64, 37] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        refused(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "random skill activation",
        );
    }
    is_live(
        &one_runner("order>=2", 500000, &[r#"{"type":38,"target":1,"modifier":1}"#]),
        "debuff immunity",
    );
}

#[test]
fn additional_activate_names_the_deferred_turn() {
    let skill = format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"order>=2\",\"cooldown\":60000,\"effects\":[{{\"type\":27,\"target\":1,\"modifier\":1000,\"additionalActivate\":2}}]}}]}}"
    );
    refused(
        &plan_skill(&skill, r#"{"skills":["550001"]}"#),
        "additionalActivate defers the extra activation",
    );
}

#[test]
fn value_usage_tiers_stay_refused() {
    // The fold-from-roster tiers (3 to 7, 10, 12, 13) and the roll or runtime
    // tiers (8, 9, 11, 14) all refuse until their half reaches the table.
    for vu in [3i64, 7, 8, 9, 10, 11, 12, 13, 14] {
        let effect = format!("{{\"type\":27,\"target\":1,\"modifier\":1000,\"valueUsage\":{vu}}}");
        refused(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            &format!("value_usage {vu}"),
        );
    }
}

#[test]
fn two_live_alternatives_lower_together() {
    // 910151's shape: two alternatives, mutually exclusive on a runtime count,
    // each live and unshadowed, sharing the skill cooldown key. The kernel
    // fans the key's park over the group on a fire, so both can lower.
    let skill = concat!(
        r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":["#,
        r#"{"precondition":"","condition":"order>=2","cooldown":50000,"effects":[{"type":27,"target":1,"modifier":2500}]},"#,
        r#"{"precondition":"","condition":"phase==2","cooldown":50000,"effects":[{"type":27,"target":1,"modifier":1500}]}]}"#
    );
    let plan = plan_skill(skill, r#"{"skills":["550001"]}"#);
    if let Some(reason) = &plan.reason {
        panic!("two live alternatives must lower, got refusal: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 2, "both alternatives lower");
    for rec in 0..2usize {
        let flags = plan.words[SK_RECORDS + SK_RSTRIDE * rec + 2];
        assert_eq!(flags & SK_FLAG_DEAD, 0, "alternative {rec} stays live");
    }
}
