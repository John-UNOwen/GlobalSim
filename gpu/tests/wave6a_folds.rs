//! Wave-6a compile gates in both directions: the lane goals (11, 12, 35), the
//! kakari extend (13, including the group target 21) and debuff immunity (38)
//! lower, and the live multi-alternative pair sharing one cooldown key lowers
//! every live alternative. What stays out (targets outside the ported set,
//! still-unported kinds) refuses by name.

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

fn one_runner(cond: &str, cooldown: i64, effects: &[&str]) -> skills_compile::SkillsPlan {
    plan_skill(
        &alt(cond, cooldown, effects),
        r#"{"skills":["550001"]}"#,
    )
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
fn lane_goals_and_immune_presence_compile_self_and_cross() {
    for kind in [11i64, 12, 35, 38] {
        for target in [1i64, 9, 10, 4] {
            let effect =
                format!("{{\"type\":{kind},\"target\":{target},\"modifier\":2000,\"targetValue\":2}}");
            is_live(
                &one_runner("order>=2", 500000, &[effect.as_str()]),
                "lane goal or immunity",
            );
        }
    }
}

#[test]
fn kakari_group_target_compiles_and_strays_refuse() {
    is_live(
        &one_runner("order>=2", 500000, &[r#"{"type":13,"target":21,"targetValue":2,"modifier":50000}"#]),
        "kakari group target",
    );
    is_live(
        &one_runner("order>=2", 500000, &[r#"{"type":13,"target":1,"modifier":50000}"#]),
        "kakari self target",
    );
    refused(
        &one_runner("order>=2", 500000, &[r#"{"type":13,"target":20,"targetValue":2,"modifier":50000}"#]),
        "effect target 20",
    );
    refused(
        &one_runner("order>=2", 500000, &[r#"{"type":27,"target":21,"targetValue":2,"modifier":1000}"#]),
        "effect target 21",
    );
}

#[test]
fn shared_key_pair_lowers_all_live_alternatives() {
    // Two live alternatives plus a shadowed third: both live ones lower (the
    // kernel fans the shared key's park over the group on a fire) while the
    // identical duplicate still collapses into its twin.
    let a = |cond: &str, m: i64| {
        format!(
            "{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":50000,\"effects\":[{{\"type\":27,\"target\":1,\"modifier\":{m}}}]}}"
        )
    };
    let skill = format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{},{},{}]}}",
        a("order>=2", 2500),
        a("phase==2", 1500),
        a("phase==2", 1500)
    );
    let plan = plan_skill(&skill, r#"{"skills":["550001"]}"#);
    if let Some(reason) = &plan.reason {
        panic!("live pair must lower, got refusal: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 2, "both live alternatives lower");
    for rec in 0..2usize {
        let flags = plan.words[SK_RECORDS + SK_RSTRIDE * rec + 2];
        assert_eq!(flags & SK_FLAG_DEAD, 0, "alternative {rec} stays live");
    }
}

#[test]
fn real_merged_shapes_lower() {
    // 201261: the '@' or-arms with the lane-speed plus SetTargetLane pair, and
    // 910151: the two live alternatives sharing one cooldown key. Both paste
    // the app table's definitions verbatim apart from the skill id.
    let lane_goal = concat!(
        r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"running_style==1&phase==0&blocked_front_continuetime>=1@running_style==1&phase==0&blocked_side_continuetime>=1","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":28,"target":1,"targetValue":0,"valueUsage":1,"modifier":350},{"type":35,"target":1,"targetValue":0,"valueUsage":1,"modifier":5000}]}]}"#
    );
    // The nige seat keeps the running_style==1 arms statically true; on any
    // other style both arms collapse and the record correctly lowers DEAD.
    let nige = r#"{"skills":["550001"],"strategy":"Nige"}"#;
    is_live(&plan_skill(lane_goal, nige), "201261 shape");
    let dead_lane = plan_skill(lane_goal, r#"{"skills":["550001"],"strategy":"Senko"}"#);
    assert!(dead_lane.reason.is_none(), "other styles must not refuse");
    assert_ne!(
        dead_lane.words[SK_RECORDS + 2] & SK_FLAG_DEAD,
        0,
        "the collapsed nige-only record lowers DEAD"
    );

    let shared_key = concat!(
        r#"{"rarity":1,"wisdomCheck":1,"priority":0,"alternatives":["#,
        r#"{"condition":"remain_distance<=401&remain_distance>=399&order_rate<=40&activate_count_all>=7","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"valueUsage":1,"modifier":2500}]}"#,
        ",",
        r#"{"condition":"remain_distance<=401&remain_distance>=399&order_rate<=40&activate_count_all<=6","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"valueUsage":1,"modifier":1500}]}"#,
        "]}"
    );
    let plan = plan_skill(shared_key, r#"{"skills":["550001"]}"#);
    if let Some(reason) = &plan.reason {
        panic!("910151 shape must lower, got refusal: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 2, "910151 shape lowers both alternatives");

    let kakari = concat!(
        r#"{"rarity":1,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"running_style_temptation_opponent_count_senko>=1&is_temptation==0","precondition":"","baseDuration":0,"durationUsage":1,"cooldown":5000000,"effects":[{"type":13,"target":21,"targetValue":2,"valueUsage":1,"modifier":50000}]}]}"#
    );
    is_live(
        &plan_skill(kakari, r#"{"skills":["550001"]}"#),
        "200801 shape",
    );
}
