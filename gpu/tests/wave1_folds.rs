//! Wave-1 targeted gates: the compile-time folds for the request-static
//! condition fields. Each case compiles one synthetic skill for one runner on
//! a turf course and reads the record table back: a live record proves the
//! True lowering fired, a DEAD-flagged record proves the False lowering fired.
//! The refusal cases keep the shapes the fallback histogram reports.

use globalsim_gpu::{json, skills_compile};

use skills_compile::{SK_FLAG_DEAD, SK_NREC, SK_RECORDS, SK_RSTRIDE};

/// Hakodate-class turf 2000: surface 1, race_track_id 10005, matching the
/// course_data.json entry the app variant jobs run on.
const COURSE: &str = concat!(
    r##"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,"##,
    r##""straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],"##,
    r##""corners":[],"slopes":[]}"##
);

fn plan_for(cond: &str, request_extra: &str, horse_extra: &str) -> skills_compile::SkillsPlan {
    let skill = format!(
        r#"{{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{{"precondition":"","condition":"{cond}","cooldown":0,"effects":[{{"type":10,"target":1,"modifier":0}}]}}]}}"#
    );
    let root_text = format!(
        "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\"\
         {request_extra},\"skillData\":{{\"550001\":{skill}}},\
         \"horses\":[{{\"skills\":[\"550001\"]{horse_extra}}}]}}"
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(COURSE).expect("course parses");
    skills_compile::plan(&root, &course)
}

/// The per-runner record count and the first record's flag word.
fn first_record(plan: &skills_compile::SkillsPlan) -> (usize, u32) {
    if plan.reason.is_some() {
        panic!("plan refused: {}", plan.reason.as_deref().unwrap_or(""));
    }
    let nrec = plan.words[SK_NREC] as usize;
    (nrec, plan.words[SK_RECORDS + SK_RSTRIDE * (nrec - 1) + 2])
}

fn is_live(plan: &skills_compile::SkillsPlan, what: &str) {
    let (nrec, flags) = first_record(plan);
    assert_eq!(nrec, 1, "{what}: one record expected");
    assert_eq!(flags & SK_FLAG_DEAD, 0, "{what}: expected the True lowering (live record)");
}

fn is_dead(plan: &skills_compile::SkillsPlan, what: &str) {
    let (nrec, flags) = first_record(plan);
    assert_eq!(nrec, 1, "{what}: one record expected");
    assert_ne!(flags & SK_FLAG_DEAD, 0, "{what}: expected the False lowering (DEAD record)");
}

fn refused(plan: &skills_compile::SkillsPlan, name: &str) {
    let reason = plan.reason.as_deref().unwrap_or_else(|| panic!("{name}: refusal expected"));
    assert!(reason.contains(name), "{name}: refusal must name it, got {reason}");
}

#[test]
fn value_filter_fields_fold_true_on_the_turf_request() {
    is_live(&plan_for("ground_type==1", "", ""), "ground_type==1 on turf");
    is_live(&plan_for("ground_condition==1", "", ""), "ground_condition==1 on Good");
    is_live(&plan_for("weather==1", "", ""), "weather==1 on Sunny");
    is_live(&plan_for("weather!=4", "", ""), "weather!=4 on Sunny");
    is_live(&plan_for("season==1", "", ""), "season==1 on Spring");
    is_live(&plan_for("track_id==10005", "", ""), "track_id==10005 on this course");
    is_live(&plan_for("is_dirtgrade==0", "", ""), "is_dirtgrade==0 on a non-dirt course");
    is_live(&plan_for("is_dirtgrade!=1", "", ""), "is_dirtgrade!=1 on a non-dirt course");
    is_live(&plan_for("popularity>=4", "", r#","popularity":5"#), "popularity>=4 with 5");
}

#[test]
fn value_filter_fields_fold_false_on_the_turf_request() {
    is_dead(&plan_for("ground_type==2", "", ""), "ground_type==2 on turf");
    is_dead(&plan_for("ground_condition==3", "", ""), "ground_condition==3 on Good");
    is_dead(&plan_for("weather==3", "", ""), "weather==3 on Sunny");
    is_dead(&plan_for("season==4", "", ""), "season==4 on Spring");
    is_dead(&plan_for("track_id==10006", "", ""), "track_id==10006 on this course");
    is_dead(&plan_for("is_dirtgrade==1", "", ""), "is_dirtgrade==1 on a non-dirt course");
    is_dead(&plan_for("popularity>=4", "", ""), "popularity>=4 with the default 0");
}

#[test]
fn request_flag_flips_is_dirtgrade() {
    is_live(&plan_for("is_dirtgrade==1", r#","isDirtGrade":true"#, ""), "isDirtGrade true wins");
    is_dead(
        &plan_for("is_dirtgrade==1", r#","isDirtGrade":false"#, ""),
        "isDirtGrade false is explicit",
    );
}

#[test]
fn order_band_and_anyskill_stay_dynamic() {
    // Not folds: the out-band cell and the observe flag resolve per tick, so
    // a true-static course leaf must keep the record live.
    is_live(&plan_for("order_rate_out70_continue==1", "", ""), "out70 dynamic leaf");
    is_live(&plan_for("order_rate_out50_continue==1", "", ""), "out50 dynamic leaf");
    is_live(&plan_for("is_activate_any_skill==1", "", ""), "anyskill dynamic leaf");
}

#[test]
fn unregistered_or_misshaped_forms_stay_refused() {
    // blocked_all_continuetime stays unported: the min of two timers has no
    // cell of its own in the scratch map.
    refused(&plan_for("blocked_all_continuetime>=1", "", ""), "blocked_all_continuetime");
    // The engine registers only the 20/40/50/70 out rates and ==1 only.
    refused(&plan_for("order_rate_out60_continue==1", "", ""), "order_rate_out60_continue");
    refused(&plan_for("order_rate_out50_continue==2", "", ""), "order_rate_out50_continue");
    refused(&plan_for("order_rate_out50_continue>=1", "", ""), "order_rate_out50_continue");
    // The in-band cells stay untracked in the kernel.
    refused(&plan_for("order_rate_in50_continue==1", "", ""), "order_rate_in50_continue");
    // is_dirtgrade registers eq and neq only.
    refused(&plan_for("is_dirtgrade>=1", "", ""), "is_dirtgrade");
    // The engine's anyskill filter rejects every arg but 1.
    refused(&plan_for("is_activate_any_skill==2", "", ""), "is_activate_any_skill");
    // A precondition anyskill node is dropped by the engine's build.
    let plan = {
        let skill = concat!(
            r#"{"rarity":1,"wisdomCheck":0,"priority":0,"alternatives":[{"precondition":"is_activate_any_skill==1","condition":"always==1","cooldown":0,"effects":[{"type":10,"target":1,"modifier":0}]}]}"#
        );
        let root_text = format!(
            r#"{{"groundCondition":"Good","weather":"Sunny","season":"Spring","skillData":{{"550001":{skill}}},"horses":[{{"skills":["550001"]}}]}}"#
        );
        skills_compile::plan(
            &json::parse(&root_text).unwrap(),
            &json::parse(COURSE).unwrap(),
        )
    };
    refused(&plan, "is_activate_any_skill");
}

/// The multi-alternative install rules: a shadowed alternative (same program,
/// no random term, trivially true precondition) collapses, and the static
/// fold decides which alternative of a pair survives.
#[test]
fn multi_alternative_rules() {
    let course = json::parse(COURSE).unwrap();
    let plan_alt = |alts: &str| -> skills_compile::SkillsPlan {
        let root_text = format!(
            "{{\"groundCondition\":\"Good\",\"weather\":\"Sunny\",\"season\":\"Spring\",\
             \"skillData\":{{\"550001\":{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\
             \"alternatives\":[{alts}]}}}},\"horses\":[{{\"skills\":[\"550001\"]}}]}}"
        );
        skills_compile::plan(&json::parse(&root_text).unwrap(), &course)
    };
    let alt = |pre: &str, cond: &str| {
        format!(
            r#"{{"precondition":"{pre}","condition":"{cond}","cooldown":0,"effects":[{{"type":10,"target":1,"modifier":0}}]}}"#
        )
    };

    // 920011's shape on this course: alt0's track_id precondition folds true,
    // the identical alt1 is shadowed, one live record remains.
    let shadowed = plan_alt(&format!(
        "{},{}",
        alt("track_id==10005", "phase>=2&is_activate_any_skill==1"),
        alt("", "phase>=2&is_activate_any_skill==1")
    ));
    assert_eq!(shadowed.words[SK_NREC], 1, "shadowed alternative collapses");

    // Off that track, alt0's precondition empties it: dead record plus the
    // live alt1, exactly as 920011 lowers on the app variant course's twin.
    let folded_off = plan_alt(&format!(
        "{},{}",
        alt("track_id==10006", "phase>=2&is_activate_any_skill==1"),
        alt("", "phase>=2&is_activate_any_skill==1")
    ));
    assert_eq!(folded_off.words[SK_NREC], 2, "dead alternative keeps its record");
    let dead_flags = folded_off.words[SK_RECORDS + 2];
    assert_ne!(dead_flags & SK_FLAG_DEAD, 0, "folded alternative is DEAD");

    // Two live alternatives that do not shadow stay refused, naming the skill.
    let diverging = plan_alt(&format!("{},{}", alt("", "order_rate>=40"), alt("", "order<=3")));
    let reason = diverging.reason.expect("two live alternatives refused");
    assert!(reason.contains("550001 has other than one alternative"), "got {reason}");
}
