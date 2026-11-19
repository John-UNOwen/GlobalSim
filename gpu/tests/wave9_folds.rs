//! Wave-9 compile gates: the duration_usage 4 running instance, the
//! additionalActivate fan outside the firing runner, and the value_usage tiers
//! the engine has no arm for. Each gate states the shape that now lowers, the
//! shape that still refuses, and the authority for the difference.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{
    SK_EFFECTS, SK_ESTRIDE, SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE,
};

/// Hakodate-class turf 2000, the same course the earlier fold gates use.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_with(skill: &str, horses: &str) -> skills_compile::SkillsPlan {
    let root_text = format!("{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]}}");
    let root = json::parse(&root_text).unwrap_or_else(|e| panic!("root parses {e}: {root_text}"));
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

/// One alternative with the given duration_usage and effect list.
fn alt(duration_usage: i64, effects: &str) -> String {
    format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"order>=2","cooldown":500000,"baseDuration":30000,"durationUsage":{duration_usage},"effects":[{effects}]}}]}}"#
    )
}

fn is_live(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a live record, plan refused: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 1, "{what}: one record expected");
    let flags = plan.words[SK_RECORDS + SK_RSTRIDE * 0 + 2];
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

fn effect_word(plan: &skills_compile::SkillsPlan, word: usize) -> u32 {
    plan.words[SK_EFFECTS + SK_ESTRIDE * 0 + word]
}

fn effect_value(plan: &skills_compile::SkillsPlan) -> f64 {
    let at = SK_EFFECTS + SK_ESTRIDE * 0 + 6;
    f64::from_bits(plan.words[at] as u64 | (plan.words[at + 1] as u64) << 32)
}

#[test]
fn duration_usage_four_lowers_with_the_running_instance() {
    // The tier reaches the record's effect word, where the kernel reads the
    // fire's extend_on_overtake flag (race_solver.rs:2655).
    let plan = plan_with(&alt(4, r#"{"type":31,"target":1,"modifier":1000}"#), r#"{"skills":["550001"]}"#);
    is_live(&plan, "duration_usage 4 self accel");
    assert_eq!(effect_word(&plan, 3), 4, "the effect keeps duration_usage 4");
    // A tier no engine arm reads is a coefficient of 1.0, so it lowers too.
    for du in [0i64, 5, 6, 8, -1] {
        let plan = plan_with(&alt(du, r#"{"type":31,"target":1,"modifier":1000}"#), r#"{"skills":["550001"]}"#);
        is_live(&plan, &format!("duration_usage {du}"));
    }
}

#[test]
fn duration_usage_four_names_the_aux_cells_it_cannot_extend() {
    // The overtake extension walks the aux entries by skill id
    // (race_solver.rs:1830-1836). The lane-preference, set-target-lane and
    // debuff-immunity kinds live in one presence cell per runner in the
    // kernel, with nothing to match, so those pairs still refuse.
    for kind in [11i64, 12, 35, 38] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        let plan = plan_with(&alt(4, &effect), r#"{"skills":["550001"]}"#);
        refused(&plan, &format!("duration_usage 4 extends type {kind}"));
        // The same effect with a tier the extension never marks stays live.
        let plan = plan_with(&alt(1, &effect), r#"{"skills":["550001"]}"#);
        is_live(&plan, &format!("type {kind} without duration_usage 4"));
    }
    // Vision and the kakari rewrite keep per-entry or immediate state, so the
    // extension has somewhere to land and the pair lowers.
    for kind in [8i64, 13, 21, 27, 28, 31] {
        let effect = format!("{{\"type\":{kind},\"target\":1,\"modifier\":1000}}");
        let plan = plan_with(&alt(4, &effect), r#"{"skills":["550001"]}"#);
        is_live(&plan, &format!("duration_usage 4 type {kind}"));
    }
}

#[test]
fn additional_activate_fans_outside_the_firing_runner() {
    // The delayed effect re-draws its targets when the turn comes and delivers
    // with the owner's value (race_solver.rs:2479-2484, 2525, race_field.rs:447),
    // so the kinds whose delivery is one timer and one value fan at every
    // ported target.
    for kind in [21i64, 22, 27, 28, 31] {
        for target in [1i64, 4, 9, 10] {
            let effect =
                format!("{{\"type\":{kind},\"target\":{target},\"targetValue\":2,\"modifier\":-1000,\"additionalActivate\":2}}");
            let plan = plan_with(
                &alt(1, &effect),
                r#"{"skills":["550001"],"teamId":2}"#,
            );
            is_live(&plan, &format!("additionalActivate 2 type {kind} target {target}"));
        }
    }
    // A follow-up registers one token with one value, so the kinds that write
    // per-receiver state or per-runner presence still refuse, named by kind.
    for kind in [1i64, 4, 5, 9, 11, 12, 13, 35, 36, 37, 38] {
        let effect =
            format!("{{\"type\":{kind},\"target\":1,\"targetValue\":2,\"modifier\":1000,\"additionalActivate\":3}}");
        let plan = plan_with(&alt(1, &effect), r#"{"skills":["550001"]}"#);
        refused(&plan, &format!("additionalActivate 3 on type {kind}"));
    }
}

#[test]
fn value_usage_without_an_arm_passes_the_modifier_through() {
    // The builder folds 3..=7, 10, 12 and 13 (race_solver_builder.rs:915-928),
    // the runtime reads 8, 9, 11 and 14 (race_solver.rs:296-312, 2485-2503).
    // Every other usage, negative or past the tiers, keeps the wire value, and
    // the engine's saturating cast puts a huge usage in the same class.
    let pass_through = [0i64, 1, 2, 15, 16, 100, -1, -7, 2147483647, 3000000000, -3000000000];
    let base = {
        let plan = plan_with(
            &alt(1, r#"{"type":27,"target":1,"valueUsage":1,"modifier":20000}"#),
            r#"{"skills":["550001"]}"#,
        );
        is_live(&plan, "value_usage 1");
        effect_value(&plan)
    };
    for vu in pass_through {
        let effect = format!("{{\"type\":27,\"target\":1,\"valueUsage\":{vu},\"modifier\":20000}}");
        let plan = plan_with(&alt(1, &effect), r#"{"skills":["550001"]}"#);
        is_live(&plan, &format!("value_usage {vu}"));
        assert_eq!(effect_value(&plan), base, "value_usage {vu} must not rescale");
    }
    // The build-time folds are unchanged: usage 3 reads the speed band.
    let folded = plan_with(
        &alt(1, r#"{"type":27,"target":1,"valueUsage":3,"modifier":20000}"#),
        r#"{"skills":["550001"],"speed":1000,"mood":0}"#,
    );
    is_live(&folded, "value_usage 3 still folds");
    assert_ne!(effect_value(&folded), base, "usage 3 keeps its 0.8 band");
    assert_eq!(effect_value(&folded), ((2.0_f64 * 0.8) as f32) as f64);
}

#[test]
fn real_delayed_instance_ids_compile() {
    // 100531 and 900531 are the merged table's only duration_usage 4 ids, both
    // self kind 31 with one additionalActivate 1 sibling. Before wave 9 a race
    // carrying either flagged to the CPU.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    let merged = json::parse(&text).expect("skills.json parses");
    for id in ["100531", "900531"] {
        let def = merged
            .get(id)
            .unwrap_or_else(|| panic!("merged table has no {id}"))
            .clone();
        let root_text = format!(
            "{{\"skillData\":{{\"{id}\":{}}},\"horses\":[{{\"skills\":[\"{id}\"]}}]}}",
            json::to_string(&def)
        );
        let root = json::parse(&root_text).unwrap_or_else(|e| panic!("root parses {e}: {root_text}"));
        let course = json::parse(COURSE).expect("course parses");
        let plan = skills_compile::plan(&root, &course);
        if let Some(reason) = &plan.reason {
            panic!("{id} must compile after wave 9, got refusal: {reason}");
        }
        assert_eq!(plan.words[SK_NREC], 1, "{id}: one record expected");
        let du = plan.words[SK_EFFECTS + SK_ESTRIDE * 0 + 3];
        assert_eq!(du, 4, "{id} keeps duration_usage 4 in the effect word");
        assert_eq!(
            plan.words[SK_EFFECTS + SK_ESTRIDE * 1 + 5],
            1,
            "{id} keeps its additionalActivate 1 sibling"
        );
    }
}
