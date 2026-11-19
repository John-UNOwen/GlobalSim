//! Wave-2b compile gates: the effect side of the histogram refusals. Cross
//! runner targets, the new effect kinds, per-record effect repeats, the
//! construction-time bumps and the team gate, each accepted or refused by
//! name at the exact engine boundary the kernel models.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE};

/// Hakodate-class turf 2000, the same course the wave-1 and wave-2 folds use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_skill(skill: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text =
        format!("{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]}}");
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
    let reason = plan.reason.as_deref().unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
}

#[test]
fn cross_runner_targets_compile() {
    // The three rank-window codes, each on a kind the kernel delivers.
    is_live(&one_runner("order>=2", 500000, &["{\"type\":27,\"target\":9,\"modifier\":1000}"]), "kind 27 ahead");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":31,\"target\":10,\"modifier\":1000}"]), "kind 31 behind");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":9,\"target\":4,\"modifier\":1000}"]), "kind 9 visible");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":8,\"target\":9,\"modifier\":-1000,\"targetValue\":2}"]), "kind 8 ahead");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":6,\"target\":10,\"modifier\":1000}"]), "kind 6 behind");
}

#[test]
fn self_only_kinds_refuse_cross_targets() {
    // GutsUp crossed with the wave-8 live field, joining SpeedUp there.
    // Kind 10 stays on the record owner because it is inert in retail.
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":1,\"target\":9,\"modifier\":1000}"]),
        "kind 1 cross is live",
    );
    is_live(
        &one_runner("always>=0", 0, &["{\"type\":4,\"target\":9,\"modifier\":1000}"]),
        "kind 4 cross is live",
    );
    // MultiplyStartDelay on another runner is inert in retail, so the record
    // fires and cools down without a state write.
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":10,\"target\":4,\"modifier\":0}"]),
        "kind 10 cross is inert",
    );
}

#[test]
fn new_kinds_compile_for_self() {
    is_live(&one_runner("always>=0", 0, &["{\"type\":4,\"target\":1,\"modifier\":5000}"]), "static kind 4");
    is_live(&one_runner("always>=0", 0, &["{\"type\":5,\"target\":1,\"modifier\":5000}"]), "static kind 5");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":6,\"target\":1,\"modifier\":1000}"]), "kind 6");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":8,\"target\":1,\"modifier\":1000}"]), "kind 8");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":14,\"target\":1,\"modifier\":1000}"]), "kind 14");
    is_live(&one_runner("order>=2", 500000, &["{\"type\":29,\"target\":1,\"modifier\":1000}"]), "mid-race kind 29");
}

#[test]
fn mid_race_stat_bump_gates() {
    // A guts fire whose condition can hold at time zero would land in the
    // engine's constructor pass ahead of the init draws the kernel bakes.
    refused(
        &one_runner("order>=2", 500000, &["{\"type\":4,\"target\":1,\"modifier\":1000}"]),
        "constructor pass",
    );
    refused(
        &one_runner("order>=2", 500000, &["{\"type\":5,\"target\":1,\"modifier\":1000}"]),
        "constructor pass",
    );
    // A cooldown keeps the record out of the constructor pass.
    refused(
        &one_runner("always>=0", 500000, &["{\"type\":5,\"target\":1,\"modifier\":1000}"]),
        "constructor pass",
    );
    // A phase gate keeps the fire out of the constructor pass in both
    // engines, the shape the kernel ports, for guts and wisdom alike.
    let plan = one_runner("phase>=2", 500000, &["{\"type\":4,\"target\":1,\"modifier\":1000}"]);
    is_live(&plan, "phase-gated kind 4");
    assert!(plan.live_guts, "phase-gated kind 4 flags live guts");
    is_live(
        &one_runner("phase>=2", 500000, &["{\"type\":5,\"target\":1,\"modifier\":1000}"]),
        "phase-gated kind 5",
    );
}

#[test]
fn static_fires_record_bumps() {
    let plan = one_runner("always>=0", 0, &["{\"type\":4,\"target\":1,\"modifier\":5000}"]);
    is_live(&plan, "static kind 4");
    assert_eq!(plan.bumps.len(), 1, "one bump expected");
    assert_eq!((plan.bumps[0].0, plan.bumps[0].1), (0, 4), "runner and kind");
    assert!(plan.bumps[0].2 > 0.0, "bump value is the scaled modifier");

    let plan = one_runner("always>=0", 0, &["{\"type\":5,\"target\":1,\"modifier\":5000}"]);
    assert_eq!(plan.bumps.len(), 1, "kind 5 bumps too");
    assert_eq!(plan.bumps[0].1, 5, "kind");

    let plan = one_runner("always>=0", 0, &["{\"type\":29,\"target\":1,\"modifier\":5000}"]);
    assert_eq!(plan.bumps.len(), 1, "kind 29 bumps too");

    // A cooldown keeps the fire out of the constructor pass, so no bump.
    let plan = one_runner("always>=0", 500000, &["{\"type\":29,\"target\":1,\"modifier\":5000}"]);
    assert!(plan.bumps.is_empty(), "cooled kind 29 records no bump");

    // Two static records keep wire order, the chain order the engine folds.
    let plan = plan_skill(
        &alt("always>=0", 0, &["{\"type\":4,\"target\":1,\"modifier\":5000}", "{\"type\":5,\"target\":1,\"modifier\":6000}"]),
        r#"{"skills":["550001"]}"#,
    );
    let kinds: Vec<i32> = plan.bumps.iter().map(|b| b.1).collect();
    assert_eq!(kinds, vec![4, 5], "wire order preserved");
}

#[test]
fn effect_repeats_within_slot_cap_compile() {
    // The engine queues one activation per effect entry, so repeats of one
    // kind are distinct entries with their own timers.
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":27,\"target\":1,\"modifier\":1000}", "{\"type\":27,\"target\":9,\"modifier\":-1000,\"targetValue\":2}"]),
        "kind 27 twice",
    );
    let mut nine: Vec<String> = Vec::new();
    for _ in 0..9 {
        nine.push("{\"type\":10,\"target\":1,\"modifier\":0}".into());
    }
    refused(
        &one_runner("order>=2", 500000, &nine.iter().map(String::as_str).collect::<Vec<_>>()),
        "more than 8 effects",
    );
}

#[test]
fn cross_debuffs_compile_on_any_roster() {
    // The kernel drops same-team receivers at delivery, so a debuff cross
    // target compiles on a teamed roster as well as a team-free one. The
    // drop is a runtime gate on the delivered list, not a build-time flag.
    is_live(
        &plan_effects("order>=2", 500000, &["{\"type\":27,\"target\":9,\"modifier\":-1000,\"targetValue\":2}"], r#"{"skills":["550001"]}"#),
        "cross debuff, team-free roster",
    );
    is_live(
        &plan_effects("order>=2", 500000, &["{\"type\":27,\"target\":9,\"modifier\":-1000,\"targetValue\":2}"], r#"{"skills":["550001"],"teamId":2}"#),
        "cross debuff, teamed roster",
    );
    is_live(
        &plan_effects("order>=2", 500000, &["{\"type\":27,\"target\":9,\"modifier\":1000,\"targetValue\":2}"], r#"{"skills":["550001"],"teamId":2}"#),
        "cross buff survives teams",
    );
}

#[test]
fn value_usage_runtime_readers_now_live() {
    // Wave 6b: 7 folds at compile, and 14 reads the live green-skill counter
    // through the kernel, so neither refuses.
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":8,\"target\":1,\"modifier\":1000,\"valueUsage\":7}"]),
        "value_usage 7",
    );
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":27,\"target\":1,\"modifier\":1000,\"valueUsage\":14}"]),
        "value_usage 14",
    );
}

#[test]
fn unported_effect_kinds_refuse_by_name() {
    // The deferred random-activation kinds; the lane goals, kakari extend and
    // immune-state writer ported with wave 6a, the random requests with 6b.
    for kind in [27i64, 9, 21, 22, 28, 31] {
        let effect = format!("{{\"type\":{kind},\"target\":9,\"modifier\":1000,\"targetValue\":2}}");
        is_live(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "ported cross kind",
        );
    }
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":37,\"target\":1,\"modifier\":1000}"]),
        "random gold request",
    );
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":38,\"target\":1,\"modifier\":1000}"]),
        "debuff immunity live",
    );
    is_live(
        &one_runner("order>=2", 500000, &["{\"type\":36,\"target\":1,\"modifier\":1000}"]),
        "random any request",
    );
    refused(
        &one_runner("order>=2", 500000, &["{\"type\":36,\"target\":9,\"modifier\":1000,\"targetValue\":2}"]),
        "random-activation effect targeting another runner",
    );
}
