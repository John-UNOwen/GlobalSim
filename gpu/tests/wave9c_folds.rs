//! Wave-9c compile gates: the request-condition folds (grade, time,
//! motivation, base_power, post_number), the corner_random sampler, the
//! viewer-row distance leaf, the roll-gated constructor bump, the StaminaUp
//! kind and the four-arm sampled shape. Each gate states the lowered shape,
//! the refused shape, and the engine authority for the difference.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{
    FEAT_VREMAIN, OP_FEAT_I, SK_ARMS, SK_ARM_SAMPLED, SK_ASTRIDE, SK_FLAG_BUMPED,
    SK_FLAG_DEAD, SK_FLAG_INIT_BUMP, SK_FLAG_INIT_DEFER, SK_NREC, SK_RECORDS, SK_RSTRIDE,
    SK_TERM_CORNER_RANDOM, SK_TERMS, SK_VIDX,
};

/// Hakodate-class turf 2000 with the four corner rows the real course 10504
/// carries, none of them tagged with an index.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[],"corners":[{"start":375.0,"end":625.0},{"start":625.0,"end":875.0},"##,
    r##"{"start":1200.0,"end":1450.0},{"start":1450.0,"end":1690.0}],"slopes":[]}"##
);

/// The same course with two index-tagged corner rows, the shape the retail
/// sampler reads the argument through.
const COURSE_INDEXED: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[],"corners":[{"start":100.0,"end":300.0,"index":2},"##,
    r##"{"start":1000.0,"end":1200.0,"index":4}],"slopes":[]}"##
);

/// An index-tagged course whose second row carries a non-integer index, the
/// shape the engine's init errors on.
const COURSE_BAD_INDEX: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[],"corners":[{"start":100.0,"end":300.0,"index":2},"##,
    r##"{"start":1000.0,"end":1200.0,"index":"four"}],"slopes":[]}"##
);

fn plan_course(skill: &str, horses: &str, course: &str, root_extra: &str) -> skills_compile::SkillsPlan {
    let root_text = format!(
        "{{\"skillData\":{{\"550001\":{skill}}},\"horses\":[{horses}]{root_extra}}}"
    );
    let root = json::parse(&root_text).unwrap_or_else(|e| panic!("root parses {e}: {root_text}"));
    let course = json::parse(course).expect("course parses");
    skills_compile::plan(&root, &course)
}

fn plan(skill: &str, horses: &str) -> skills_compile::SkillsPlan {
    plan_course(skill, horses, COURSE, "")
}

/// One alternative on the request's default strings.
fn alt(cond: &str, cooldown: i64, base_duration: i64, effects: &str) -> String {
    format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":{cooldown},"baseDuration":{base_duration},"effects":[{effects}]}}]}}"#
    )
}

fn one(cond: &str) -> skills_compile::SkillsPlan {
    plan(
        &alt(cond, 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
        r#"{"skills":["550001"]}"#,
    )
}

fn flags(plan: &skills_compile::SkillsPlan, rec: usize) -> u32 {
    plan.words[SK_RECORDS + SK_RSTRIDE * rec + 2]
}

fn is_live(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a live record, plan refused: {reason}");
    }
    assert!(plan.words[SK_NREC] >= 1, "{what}: a record expected");
    assert_eq!(flags(plan, 0) & SK_FLAG_DEAD, 0, "{what}: expected the live lowering");
}

fn is_dead(plan: &skills_compile::SkillsPlan, what: &str) {
    if let Some(reason) = &plan.reason {
        panic!("{what}: expected a dead record, plan refused: {reason}");
    }
    assert_eq!(plan.words[SK_NREC], 1, "{what}: one record expected");
    assert_ne!(flags(plan, 0) & SK_FLAG_DEAD, 0, "{what}: expected the dead lowering");
}

fn refused(plan: &skills_compile::SkillsPlan, name: &str) {
    let reason = plan
        .reason
        .as_deref()
        .unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
}

#[test]
fn grade_and_time_fold_the_request_or_their_defaults() {
    // Absent keys take the spec defaults g1 and midday (lib.rs:284-285), so
    // the true side lowers and the false side folds dead exactly like the
    // engine's value_filter emptying the region list.
    is_live(&one("grade==100"), "grade defaults to g1");
    is_dead(&one("grade==200"), "grade is not g2");
    is_live(&one("time==2"), "time defaults to midday");
    is_dead(&one("time==4"), "time is not night");
    // Present parsable strings fold their enum codes.
    let plan = plan_course(
        &alt("grade==200", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
        r#"{"skills":["550001"]}"#,
        COURSE,
        r#","grade":"G2","time":"night""#,
    );
    is_live(&plan, "grade G2");
    is_live(
        &plan_course(
            &alt("time==4", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
            COURSE,
            r#","time":"NIGHT""#,
        ),
        "time NIGHT upper and lower case",
    );
    // A present unparsable string fails the engine's own parse, so the leaf
    // refuses instead of guessing.
    refused(
        &plan_course(
            &alt("grade==100", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
            COURSE,
            r#","grade":"gold""#,
        ),
        "parsable grade",
    );
    refused(
        &plan_course(
            &alt("time==2", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
            COURSE,
            r#","time":"noon""#,
        ),
        "parsable time",
    );
}

#[test]
fn motivation_folds_the_clamped_mood() {
    // v_motivation reads the per-runner mood clamped to [-2,2] plus three
    // (lib.rs:505, activation_conditions.rs:1243).
    is_live(&one("motivation==3"), "absent mood means 3");
    is_dead(&one("motivation==4"), "absent mood is below 4");
    let mood_plan = |mood: i64| {
        plan(
            &alt("motivation==5", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            &format!(r#"{{"skills":["550001"],"mood":{mood}}}"#),
        )
    };
    is_live(&mood_plan(2), "mood 2 means 5");
    is_live(&mood_plan(9), "mood clamps to 2");
    is_dead(&mood_plan(1), "mood 1 means 4");
    is_live(
        &plan(
            &alt("motivation==1", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"],"mood":-9}"#,
        ),
        "mood clamps to -2",
    );
}

#[test]
fn base_power_folds_the_ranked_wire_stat() {
    // The runtime_phase trigger stats are the ranked base stats truncated to
    // i32 (race_solver_builder.rs:872-881), so a raw 1000 power with mood 2
    // sits above 1000 and far below any 2000 band.
    let horse = r#"{"skills":["550001"],"power":1000,"mood":2}"#;
    let power_plan = |cond: &str| {
        plan(
            &alt(cond, 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            horse,
        )
    };
    is_live(&power_plan("base_power>=1000"), "ranked power keeps its floor");
    is_live(&power_plan("base_power<=1200"), "ranked power stays near the raw value");
    is_dead(&power_plan("base_power<=900"), "ranked power is above the 900 band");
    is_dead(&power_plan("base_power>=2000"), "ranked power is below 2000");
}

#[test]
fn post_number_folds_the_gate_block() {
    // The filters are eq, lte and gte immediates over the client gate block
    // (activation_conditions.rs:1002-1039). Nine runners put gates 0..6 in
    // blocks 1..7 and gates 7 and 8 in block 8, so every comparator is
    // checked against the full field at each band edge.
    let blocks = [1i64, 2, 3, 4, 5, 6, 7, 8, 8];
    let horses = (0..9)
        .map(|j| format!(r#"{{"skills":["550001"],"gateIndex":{j}}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let field_plan = |cond: &str| plan(&alt(cond, 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#), &horses);
    for b in 1..=8i64 {
        for op in ["==", ">=", "<="] {
            let plan = field_plan(&format!("post_number{op}{b}"));
            if let Some(reason) = &plan.reason {
                panic!("field fold refused: {reason}");
            }
            for (j, block) in blocks.iter().enumerate() {
                let live = match op {
                    "==" => *block == b,
                    ">=" => *block >= b,
                    _ => *block <= b,
                };
                let dead = flags(&plan, j) & SK_FLAG_DEAD != 0;
                assert_eq!(
                    live,
                    !dead,
                    "gate {j} block {block} vs post_number{op}{b}: {}",
                    if live { "live" } else { "dead" }
                );
            }
        }
    }
    // A gate outside the field reports block 1 (gate_block guard).
    let mut odd = (0..9)
        .map(|j| format!(r#"{{"skills":["550001"],"gateIndex":{j}}}"#))
        .collect::<Vec<_>>();
    odd[0] = r#"{"skills":["550001"],"gateIndex":20}"#.to_string();
    let plan = plan(
        &alt("post_number==1", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
        &odd.join(","),
    );
    if let Some(reason) = &plan.reason {
        panic!("field fold refused: {reason}");
    }
    assert_eq!(flags(&plan, 0) & SK_FLAG_DEAD, 0, "out-of-range gate blocks at 1");
    // The registry has no eq/lte/gte arm for the other comparators, so the
    // engine's build fails on them too.
    for cond in ["post_number!=3", "post_number<3", "post_number>3"] {
        refused(&field_plan(cond), "post_number");
    }
}

#[test]
fn corner_random_resolves_its_row_and_draws_one_window() {
    // The retail sampler turns the argument into one corner row
    // (retail_initialization.rs:135-166) and the kernel draws the window
    // there, so the leaf lowers as a sampled term of kind 9.
    let plan = one("corner_random==1");
    is_live(&plan, "corner_random row 1");
    assert_eq!(plan.words[skills_compile::SK_NTERM], 1, "one sampled term");
    assert_eq!(plan.words[SK_TERMS], SK_TERM_CORNER_RANDOM, "the corner_random kind");
    assert_eq!(
        plan.words[SK_TERMS + 4],
        0,
        "the legacy offset resolves the first row"
    );
    // Row 4 of four, then the legacy offset 4 + 4 - 5 = 3, the last row.
    assert_eq!(
        one("corner_random==4").words[SK_TERMS + 4],
        3,
        "the legacy offset resolves the last row"
    );
    // Offsets at or past the row count resolve no row: the arm can never
    // fire, spends no draw, and a single-arm record folds dead.
    is_dead(&one("corner_random==5"), "no row at the row count");
    is_dead(&one("corner_random==9"), "no row far past the count");
    is_dead(&one("corner_random==0"), "no row below one");
    is_dead(&one("corner_random==-2"), "no row for a negative argument");
    assert_eq!(
        one("corner_random==5").words[SK_ARMS + SK_ASTRIDE * 0 + 6] & SK_ARM_SAMPLED,
        0,
        "an unresolved row takes no draw"
    );
    // Only eq reaches the sampler.
    refused(&one("corner_random!=1"), "corner_random");

    // An index-tagged course resolves the argument through the row metadata.
    let plan = plan_course(
        &alt("corner_random==4", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
        r#"{"skills":["550001"]}"#,
        COURSE_INDEXED,
        "",
    );
    is_live(&plan, "index 4 resolves its row");
    assert_eq!(plan.words[SK_TERMS + 4], 1, "the row carrying index 4");
    is_dead(
        &plan_course(
            &alt("corner_random==3", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
            COURSE_INDEXED,
            "",
        ),
        "index 3 has no row",
    );
    // A non-integer index makes the engine's init fail, so the construct
    // refuses instead of guessing a row.
    refused(
        &plan_course(
            &alt("corner_random==4", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
            COURSE_BAD_INDEX,
            "",
        ),
        "not an integer",
    );
}

#[test]
fn corner_random_arms_resolve_independently() {
    // Every '@' group is its own arm with its own row: the merged table's
    // 200351 and 200352 are four sampled arms, and a mixed arm pair draws
    // for the resolved row only.
    let plan = one("corner_random==1@corner_random==2@corner_random==3@corner_random==4");
    is_live(&plan, "four sampled arms");
    assert_eq!(plan.words[SK_NREC], 1, "one record");
    assert_eq!(plan.words[skills_compile::SK_NTERM], 4, "one draw per arm");
    assert_eq!(plan.words[SK_RECORDS + SK_RSTRIDE * 0 + 20], 4, "four arm entries");
    for a in 0..4 {
        let at = SK_ARMS + SK_ASTRIDE * a;
        assert_ne!(plan.words[at + 6] & SK_ARM_SAMPLED, 0, "arm {a} sampled");
        assert_eq!(
            plan.words[SK_TERMS + skills_compile::SK_MSTRIDE * a],
            SK_TERM_CORNER_RANDOM,
            "arm {a} corner_random term"
        );
    }
    // A missing row collapses only its own arm.
    let plan = one("corner_random==1@corner_random==99");
    is_live(&plan, "the resolved arm survives");
    assert_eq!(plan.words[skills_compile::SK_NTERM], 1, "the missing row draws nothing");
    assert_eq!(plan.words[SK_RECORDS + SK_RSTRIDE * 0 + 20], 2, "both arms stay lowered");
    assert_eq!(
        plan.words[SK_ARMS + SK_ASTRIDE * 1 + 6] & SK_ARM_SAMPLED,
        0,
        "the missing arm takes no draw"
    );
    // Two sampled terms in one arm still refuse: the kernel chains one
    // window set per arm.
    refused(
        &one("corner_random==1&corner_random==2"),
        "more than one random term",
    );
    // The mixed kind pair (a phase sampler beside a corner sampler) lowers.
    is_live(&one("corner_random==1@phase_random==2"), "mixed sampler arms");
}

#[test]
fn viewer_distance_leaf_is_dynamic_deferred_and_headered() {
    // The live check always resolves a viewer row (lib.rs:292-300), so the
    // leaf is the FEAT_VREMAIN dynamic reading the SK_VIDX row.
    let plan = one("remain_distance_viewer_id<=600");
    is_live(&plan, "viewer distance");
    assert_ne!(
        flags(&plan, 0) & SK_FLAG_INIT_DEFER,
        0,
        "the viewer row is unknown at the constructor pass"
    );
    assert_eq!(plan.words[SK_VIDX], 0, "absent playerHorseIndex means row 0");
    let program = &plan.words[skills_compile::SK_PROGRAMS..];
    let leaf = program
        .chunks(skills_compile::SK_PSTRIDE)
        .find(|n| n[0] == OP_FEAT_I)
        .expect("a dynamic leaf");
    assert_eq!(leaf[1], FEAT_VREMAIN * 8 + 3, "FEAT_VREMAIN with the <= comparator");
    // The header carries the request row.
    let four = (0..4)
        .map(|_| r#"{"skills":["550001"]}"#.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let plan = plan_course(
        &alt("remain_distance_viewer_id<=600", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
        &four,
        COURSE,
        r#","playerHorseIndex":3"#,
    );
    is_live(&plan, "viewer row 3");
    assert_eq!(plan.words[SK_VIDX], 3, "the request row reaches the header");
    // An out-of-range row fails the engine's spec parse (lib.rs:292-300).
    refused(
        &plan_course(
            &alt("remain_distance_viewer_id<=600", 500000, 30000, r#"{"type":31,"target":1,"modifier":1000}"#),
            &four,
            COURSE,
            r#","playerHorseIndex":42"#,
        ),
        "in-range playerHorseIndex",
    );
}

#[test]
fn stamina_up_lowers_on_the_record_owner_and_the_fan() {
    // StaminaUp writes the stat accumulator (race_solver.rs:2556-2558) and
    // the fire's refresh_stat_effects (race_solver.rs:2646) rebuilds max hp
    // from the field, which the kernel mirrors for the owner and, through
    // the delivered self-rewrite, for fan receivers.
    is_live(
        &plan(
            &alt("phase>=1", 5000000, 30000, r#"{"type":2,"target":1,"modifier":1000}"#),
            r#"{"skills":["550001"]}"#,
        ),
        "self StaminaUp",
    );
    is_live(
        &plan(
            &alt("phase>=1", 5000000, 30000, r#"{"type":2,"target":9,"modifier":1000,"targetValue":3}"#),
            r#"{"skills":["550001"]}"#,
        ),
        "cross StaminaUp",
    );
    // A static self fire with the bump kinds mixed in (the 210141 shape)
    // lowers: every self stat kind chains through the host bump in effect
    // order, so the shared accumulators add in the engine's order.
    let plan = plan(
        &alt(
            "grade>=0",
            0,
            -1,
            r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#,
        ),
        r#"{"skills":["550001"]}"#,
    );
    is_live(&plan, "static kind 2 with 4 and 5");
    let kinds: Vec<i32> = plan.bumps.iter().map(|b| b.1).collect();
    assert_eq!(kinds, vec![2, 4, 5], "every static self stat kind rides the host bump");
}

#[test]
fn wisdom_gated_static_fires_bake_after_the_lottery() {
    // The lottery decides these fires, so the host pushes no bump and the
    // kernel bakes them once enabled (sk_init_wis_bump, sk_init_stat_bump).
    let wis = |cond: &str, effects: &str| -> String {
        format!(
            r#"{{"rarity":1,"wisdomCheck":1,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":0,"baseDuration":-1,"effects":[{effects}]}}]}}"#
        )
    };
    let gated = plan(
        &wis("grade>=0", r#"{"type":2,"target":1,"modifier":300000},{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#),
        r#"{"skills":["550001"]}"#,
    );
    assert_eq!(gated.words[SK_NREC], 1, "one gated record expected");
    let f = flags(&gated, 0);
    assert_ne!(f & SK_FLAG_INIT_BUMP, 0, "the kernel bakes this fire");
    assert_ne!(f & SK_FLAG_BUMPED, 0, "the live writes must not double the bake");
    assert!(gated.bumps.is_empty(), "the host cannot precompute a rolled fire");
    // A kind-4 gate makes the guts terms live, the roll decides the bump.
    assert!(gated.live_guts, "baked guts terms would miss the roll");

    // The same shape without the lottery stays on the host bump path.
    let plain = plan(
        &alt("grade>=0", 0, -1, r#"{"type":4,"target":1,"modifier":400000},{"type":5,"target":1,"modifier":400000}"#),
        r#"{"skills":["550001"]}"#,
    );
    let f = flags(&plain, 0);
    assert_eq!(f & SK_FLAG_INIT_BUMP, 0, "no lottery, no kernel bake");
    assert_ne!(f & SK_FLAG_BUMPED, 0, "the host bumps stay baked");
    let kinds: Vec<i32> = plain.bumps.iter().map(|b| b.1).collect();
    assert_eq!(kinds, vec![4, 5]);

    // Kind 29 feeds the host-only rush bonus, so it never joins a kernel bake.
    let mixed = plan(
        &wis("grade>=0", r#"{"type":5,"target":1,"modifier":400000},{"type":29,"target":1,"modifier":100000}"#),
        r#"{"skills":["550001"]}"#,
    );
    assert_eq!(flags(&mixed, 0) & SK_FLAG_INIT_BUMP, 0, "kind 29 stays out of the bake");
}

#[test]
fn advantage_skill_machine_lowers_live() {
    // The engine swaps a field-wide previous-tick (runner, kind) advantage set
    // with team exclusion (race_field.rs:373-398). The kernel mirrors it with
    // the cached previous set, so the condition lowers live (family 2).
    is_live(
        &one("is_other_character_activate_advantage_skill==1"),
        "is_other_character_activate_advantage_skill",
    );
}
