//! Wave-7 compile gates: mid-race stat fires. SpeedUp, GutsUp and WisdomUp
//! cross to other runners through an exact f64 refresh, and a self GutsUp or
//! WisdomUp needs a phase gate so the constructor pass can never fire it.
//! Wave 8 opened the GutsUp cross and mid-race paths, so the guts fires also
//! set the plan's live_guts flag that switches the kernel to field reads.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_FLAG_BUMPED, SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE};

/// Hakodate-class turf 2000, the same course the earlier fold gates use.
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

fn one_runner(cond: &str, cooldown: i64, effects: &[&str]) -> skills_compile::SkillsPlan {
    plan_skill(&alt(cond, cooldown, effects), r#"{"skills":["550001"]}"#)
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

fn flags(plan: &skills_compile::SkillsPlan) -> u32 {
    plan.words[SK_RECORDS + 2]
}

#[test]
fn cross_stat_kinds_gate_by_kind() {
    // SpeedUp, GutsUp and WisdomUp write the receiver through the exact f64
    // refresh. The GutsUp receivers' accumulator and field are live, so the
    // plan carries the live_guts flag.
    for kind in [1i64, 4, 5] {
        for target in [4i64, 9, 10] {
            let effect = format!("{{\"type\":{kind},\"target\":{target},\"modifier\":1000}}");
            let plan = one_runner("order>=2", 500000, &[effect.as_str()]);
            is_live(&plan, &format!("kind {kind} target {target}"));
            assert_eq!(plan.live_guts, kind == 4, "kind {kind} cross live_guts");
        }
    }
}

#[test]
fn stat_bump_self_needs_a_phase_gate() {
    // The engine's constructor pass evaluates the same condition with no
    // field, so a GutsUp or WisdomUp arm that can hold at time zero must
    // stay refused.
    for kind in [4i64, 5] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        refused(
            &one_runner("always>=0", 500000, &[effect.as_str()]),
            "constructor pass",
        );
        refused(
            &one_runner("order>=2", 500000, &[effect.as_str()]),
            "constructor pass",
        );
        // A phase term the field starts at satisfies, so it gates nothing.
        refused(
            &one_runner("phase>=0", 500000, &[effect.as_str()]),
            "constructor pass",
        );
        refused(
            &one_runner("phase>-1", 500000, &[effect.as_str()]),
            "constructor pass",
        );
        // One ungated arm can hold at time zero, so the whole record is refused.
        refused(
            &one_runner("phase>=2@always>=0", 500000, &[effect.as_str()]),
            "constructor pass",
        );
        // Phase gates that read false at phase zero lower live. A mid-race
        // GutsUp fire moves live guts, so the plan flags it.
        for cond in ["phase>=1", "phase>=2", "phase>0", "phase>1", "phase>=2&order>=2"] {
            let plan = one_runner(cond, 500000, &[effect.as_str()]);
            if plan.reason.is_some() {
                refused(&plan, cond);
            }
            is_live(&plan, cond);
            assert_eq!(plan.live_guts, kind == 4, "kind {kind} gated live_guts");
        }
        // A gate also covers the cooldown-zero shape: one loop fire, then dead.
        is_live(
            &one_runner("phase>=1", 0, &[effect.as_str()]),
            &format!("gated kind {kind} cooldown zero"),
        );
    }
}

#[test]
fn bumped_flag_marks_static_stat_bumps() {
    // The host folds a static self fire of 4, 5 and 29 into the init inputs,
    // so the kernel skips the record's own init-pass write only for those.
    for kind in [4i64, 5, 29] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":5000}}");
        let plan = one_runner("always>=0", 0, &[effect.as_str()]);
        is_live(&plan, "static bump");
        assert_ne!(flags(&plan) & SK_FLAG_BUMPED, 0, "kind {kind} static marks BUMPED");
        // A constructor-only fire stays a baked constant, live path off.
        assert!(!plan.live_guts, "kind {kind} static keeps the baked words");
    }
    // A phase-gated fire never reaches the constructor pass, and a static
    // cross fire delivers nothing there, so neither carries the flag.
    let plan = one_runner("phase>=2", 500000, &[r#"{"type":5,"target":1,"modifier":5000}"#]);
    is_live(&plan, "gated kind 5");
    assert_eq!(flags(&plan) & SK_FLAG_BUMPED, 0, "gated kind 5 carries no bump");
    let plan = one_runner("always>=0", 0, &[r#"{"type":5,"target":9,"modifier":5000}"#]);
    is_live(&plan, "static cross kind 5");
    assert_eq!(flags(&plan) & SK_FLAG_BUMPED, 0, "cross static carries no bump");
    // A mid-race fire of another kind keeps its own init-pass write.
    let plan = one_runner("order>=2", 500000, &[r#"{"type":1,"target":1,"modifier":5000}"#]);
    is_live(&plan, "mid-race kind 1");
    assert_eq!(flags(&plan) & SK_FLAG_BUMPED, 0, "kind 1 carries no bump");
}
