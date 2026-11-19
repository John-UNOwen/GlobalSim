//! Wave-6b compile gates: the value_usage tiers the engine folds at build
//! time (race_solver_builder.rs:908-939) lower into the level-resolved
//! modifier, from the request's team totals, career wins, fans and the
//! displayed-stat tier, each band checked on both sides. The 21 merged-table
//! ids that carry a tier lower, and the runtime usages 8, 9, 11 and 14 refuse
//! by name until the kernel carries their draw and counters.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_EFFECTS, SK_ESTRIDE, SK_FLAG_DEAD, SK_NREC, SK_RECORDS};

/// Hakodate-class turf 2000, the same course the earlier fold gates use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_with(skill: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text = format!("{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]}}");
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

/// One alternative, self-target kind 27, the given value_usage and modifier.
fn skill(vu: i32, modifier: i64) -> String {
    format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"order>=2","cooldown":500000,"durationUsage":1,"effects":[{{"type":27,"target":1,"valueUsage":{vu},"modifier":{modifier}}}]}}]}}"#
    )
}

fn folded(plan: &skills_compile::SkillsPlan, what: &str) -> f64 {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a live record, plan refused: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 1, "{what}: one record expected");
    let flags = plan.words[SK_RECORDS + 2];
    assert_eq!(flags & SK_FLAG_DEAD, 0, "{what}: expected the live record");
    effect_value(plan, 0)
}

fn effect_value(plan: &skills_compile::SkillsPlan, rec: usize) -> f64 {
    let at = SK_EFFECTS + SK_ESTRIDE * rec + 6;
    f64::from_bits(plan.words[at] as u64 | (plan.words[at + 1] as u64) << 32)
}

/// The level-1 kind-27 value 20000 resolves to 2.0 before the tier factor,
/// and the retail fold ends in one f32 round trip.
fn expect(factor: f64) -> f64 {
    ((2.0_f64 * factor) as f32) as f64
}

#[test]
fn team_stat_tiers_fold_both_sides_of_every_band() {
    // value_usage n reads team_stats[n-3]: 3 speed, 4 stamina, 5 power,
    // 6 guts, 7 wisdom. The stat is the overcap-halved, mood-scaled base,
    // and bands [1200,1800,2600,3600] count membership with `<=`.
    let cases: [(i32, &str, f64); 9] = [
        (3, r#"{"skills":["550001"],"speed":1000,"mood":0}"#, 0.8),
        (3, r#"{"skills":["550001"],"speed":1200,"mood":0}"#, 0.9),
        (4, r#"{"skills":["550001"],"stamina":2398,"mood":0}"#, 0.9),
        (4, r#"{"skills":["550001"],"stamina":2400,"mood":0}"#, 1.0),
        (5, r#"{"skills":["550001"],"power":3000,"mood":0}"#, 1.0),
        // The base stat clamps at 2000, so one runner never reaches the 2600
        // band; only a team sum can.
        (5, r#"{"skills":["550001"],"power":6000,"mood":0}"#, 1.0),
        (7, r#"{"skills":["550001"],"wisdom":500,"mood":0}"#, 0.8),
        // Mood multiplies the base stat before the band read: 1180 * 1.02
        // crosses the 1200 band and 1176 * 1.02 stays under it.
        (3, r#"{"skills":["550001"],"speed":1180,"mood":1}"#, 0.9),
        (3, r#"{"skills":["550001"],"speed":1176,"mood":1}"#, 0.8),
    ];
    for (vu, horse, factor) in cases {
        let plan = plan_with(&skill(vu, 20000), horse);
        assert_eq!(folded(&plan, &format!("value_usage {vu}")), expect(factor),
            "value_usage {vu} with horse {horse}");
    }
}

#[test]
fn team_sum_reaches_the_top_band_and_the_sentinel() {
    // 60 clamped-2000 members sum past the 3600 band (factor 1.2) and the
    // 99999 sentinel resets it to 1.0.
    for (count, factor) in [(60usize, 1.2), (84usize, 1.0)] {
        let horses: Vec<String> = (0..count)
            .map(|i| format!(r#"{{"skills":["550001"],"speed":1200,"teamId":1,"gateIndex":{i}}}"#))
            .collect();
        let root_text = format!(
            "{{\"skillData\":{{\"550001\":{}}},\"horses\":[{}]}}",
            skill(3, 20000),
            horses.join(",")
        );
        let root = json::parse(&root_text).expect("root parses");
        let course = json::parse(COURSE).expect("course parses");
        let plan = skills_compile::plan(&root, &course);
        if let Some(reason) = &plan.reason {
            panic!("{count}-member team must compile, got refusal: {reason}");
        }
        assert_eq!(plan.words[SK_NREC] as usize, count);
        assert_eq!(effect_value(&plan, 0), expect(factor), "{count}-member team sum");
    }
}

#[test]
fn team_totals_sum_the_team_and_zero_teams_stay_own() {
    // One runner on team 7 folds from the team sum of both horses' ranked
    // stats, the other team-7 runner reads the same sum. A team-0 runner
    // reads only its own stat.
    let horses = concat!(
        r#"{"skills":["550001"],"speed":800,"teamId":7},"#,
        r#"{"skills":["550001"],"speed":900,"teamId":7},"#,
        r#"{"skills":["550001"],"speed":800}"#
    );
    let root_text = format!("{{\"skillData\":{{\"550001\":{}}},\"horses\":[{}]}}", skill(3, 20000), horses);
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    let plan = skills_compile::plan(&root, &course);
    if let Some(reason) = &plan.reason {
        panic!("team totals must compile, got refusal: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 3);
    let read = |rec: usize| -> f64 {
        let at = SK_EFFECTS + SK_ESTRIDE * rec + 6;
        f64::from_bits(plan.words[at] as u64 | (plan.words[at + 1] as u64) << 32)
    };
    // 800 + 900 = 1700 lands in band tier 1 (>= 1200, < 1800) for team 7,
    // while the lone 800 runner stays in tier 0.
    for rec in 0..2 {
        assert_eq!(read(rec), expect(0.9), "team runner {rec}");
    }
    assert_eq!(read(2), expect(0.8), "team-0 runner");
}

#[test]
fn win_and_fan_tiers_fold_both_sides_of_every_band() {
    // Wins band [6,14,18,25], fans band [20000,50000,100000,160000], each
    // counting membership with `<=`; the fans sentinel is >= 99999999.
    let cases: [(i32, &str, f64); 9] = [
        (10, r#"{"skills":["550001"],"careerWins":5}"#, 0.8),
        (10, r#"{"skills":["550001"],"careerWins":6}"#, 0.9),
        (10, r#"{"skills":["550001"],"careerWins":17}"#, 1.0),
        (10, r#"{"skills":["550001"],"careerWins":18}"#, 1.1),
        (10, r#"{"skills":["550001"],"careerWins":26}"#, 1.2),
        (12, r#"{"skills":["550001"],"fans":19999}"#, 0.8),
        (12, r#"{"skills":["550001"],"fans":50000}"#, 1.0),
        (12, r#"{"skills":["550001"],"fans":99999999}"#, 1.0),
        // A missing field reads 0, the parse default, and stays in tier 0.
        (12, r#"{"skills":["550001"]}"#, 0.8),
    ];
    for (vu, horse, factor) in cases {
        let plan = plan_with(&skill(vu, 20000), horse);
        assert_eq!(folded(&plan, &format!("value_usage {vu}")), expect(factor),
            "value_usage {vu} with horse {horse}");
    }
}

#[test]
fn displayed_stat_tier_reads_the_max_raw_stat() {
    // value_usage 13 bands on the highest displayed raw stat, before the
    // overcap halving: 600, 800, 1000, 1100 with `<` membership.
    let cases: [(&str, f64); 6] = [
        (r#"{"skills":["550001"],"speed":599,"wisdom":599}"#, 0.800000011920929),
        (r#"{"skills":["550001"],"speed":600,"wisdom":599}"#, 0.8999999761581421),
        (r#"{"skills":["550001"],"power":999,"guts":800}"#, 1.0),
        (r#"{"skills":["550001"],"power":1000,"guts":999}"#, 1.100000023841858),
        (r#"{"skills":["550001"],"stamina":1099}"#, 1.100000023841858),
        (r#"{"skills":["550001"],"stamina":1100}"#, 1.2000000476837158),
    ];
    for (horse, factor) in cases {
        let plan = plan_with(&skill(13, 20000), horse);
        assert_eq!(folded(&plan, "value_usage 13"), expect(factor), "horse {horse}");
    }
}

#[test]
fn runtime_usages_compile_live() {
    // Wave 6b: 8 and 9 draw on the race bank at activation and 11 and 14 read
    // live kernel counters, so the compiler keeps the base value in the effect
    // word and the kernel scales it per application.
    for vu in [8i32, 9, 11, 14] {
        let plan = plan_with(&skill(vu, 20000), r#"{"skills":["550001"]}"#);
        assert!(plan.reason.is_none(), "vu {vu}: {:?}", plan.reason);
    }
}

#[test]
fn real_merged_tier_ids_lower() {
    // The gold halves of 210011 (vu3), 210021 (vu4), 210031 (vu5), 210041
    // (vu6), 210051 (vu7), 210061 (vu10), 210071 (vu12) and 210081 (vu13),
    // pasted from the merged table apart from the id.
    let shapes: [(&str, &str, &str); 8] = [
        ("210011",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"phase_random==1","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":3,"modifier":3500}]}]}"#),
         r#"{"skills":["550001"],"speed":1300}"#),
        ("210021",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"phase_random==1","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":9,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":4,"modifier":550}]}]}"#),
         r#"{"skills":["550001"],"stamina":1300}"#),
        ("210031",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"phase_random==2","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":31,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":5,"modifier":4000}]}]}"#),
         r#"{"skills":["550001"],"power":1300}"#),
        ("210041",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"phase_random==2","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":6,"modifier":2500},{"type":31,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":6,"modifier":3000}]}]}"#),
         r#"{"skills":["550001"],"guts":1300}"#),
        ("210051",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"phase_random==0","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":28,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":7,"modifier":350},{"type":8,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":7,"modifier":150000}]}]}"#),
         r#"{"skills":["550001"],"wisdom":1300}"#),
        ("210061",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"distance_rate_after_random==50","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":10,"modifier":2500},{"type":31,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":10,"modifier":3000},{"type":9,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":10,"modifier":350}]}]}"#),
         r#"{"skills":["550001"],"careerWins":20}"#),
        ("210071",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"distance_rate_after_random==50&order_rate<=65","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":12,"modifier":3500},{"type":31,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":12,"modifier":2000}]}]}"#),
         r#"{"skills":["550001"],"fans":60000}"#),
        ("210081",
         concat!(r#"{"rarity":2,"wisdomCheck":1,"priority":0,"alternatives":[{"condition":"is_lastspurt==1&phase_random==3","precondition":"","baseDuration":30000,"durationUsage":1,"cooldown":5000000,"effects":[{"type":27,"target":1,"targetValue":0,"additionalActivate":0,"levelModifiers":[1,1.01,1.04,1.07,1.1,1.13,1.16,1.19,1.22,1.25],"valueUsage":13,"modifier":3500}]}]}"#),
         r#"{"skills":["550001"],"speed":1200}"#),
    ];
    for (sid, shape, horse) in shapes {
        let root_text = format!("{{\"skillData\":{{\"550001\":{shape}}},\"horses\":[{horse}]}}");
        let root = json::parse(&root_text).expect("root parses");
        let course = json::parse(COURSE).expect("course parses");
        let plan = skills_compile::plan(&root, &course);
        if let Some(reason) = &plan.reason {
            panic!("{sid} shape must lower, got refusal: {reason}");
        }
    }
}
