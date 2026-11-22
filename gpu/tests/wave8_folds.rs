//! Wave-8 compile gates: every real GutsUp entry in the shipped catalog
//! compiles, the catalog-wide set stays constructor-time bumps (the live
//! field stays off), and the synthetic mid-race and cross shapes flag the
//! plan's live_guts switch the kernel reads.

use globalsim_gpu::{json, skills_compile};

/// The merged table the runtime gates use, so this audits the shipped ids.
fn merged_skill_data() -> json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../Yet-Another-Umalator-Fork-native/globalsim/skills.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading merged skills at {} failed: {e}", path.display()));
    json::parse(&text).expect("skills.json parses")
}

fn plan_with(skill_data: &json::Value, ids: &[i64]) -> skills_compile::SkillsPlan {
    let horses: Vec<String> = ids
        .iter()
        .map(|id| format!("{{\"skills\":[{{\"skillId\":{id},\"level\":1}}]}}"))
        .collect();
    let root_text = format!(
        "{{\"weather\":\"Sunny\",\"season\":\"Spring\",\"skillData\":{},\"horses\":[{}]}}",
        json::to_string(skill_data),
        horses.join(",")
    );
    let root = json::parse(&root_text).expect("root parses");
    let course = json::parse(
        r#"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,
            "straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],
            "corners":[],"slopes":[]}"#,
    )
    .expect("course parses");
    skills_compile::plan(&root, &course)
}

/// The shipped type-4 skills, eighteen effect entries over sixteen ids.
/// 202441 and 202442 sit apart below: their `random_lot` conditions compile
/// since the E1a lottery sampler, and the kernel bakes the bump at init.
/// 210141 folds dead on its `motivation` condition instead of refusing.
const REAL_GUTS_IDS: [i64; 13] = [
    200211, 200212, 200221, 200222, 200231, 200232, 200233, 200241, 200242, 200291, 200292,
    202181, 1200051,
];

#[test]
fn real_guts_skills_compile_as_bumps() {
    let all = merged_skill_data();
    // Every real GutsUp skill installs and compiles, and none of them is a
    // live fire: the catalog shapes are constructor conditions (weather,
    // motivation, distance and ground type, random lot), so the race keeps
    // the prepare-baked words.
    for id in REAL_GUTS_IDS {
        let mut one = json::obj(vec![]);
        if let json::Value::Object(map) = &mut one {
            map.insert(
                id.to_string(),
                all.get(&id.to_string())
                    .unwrap_or_else(|| panic!("{id} absent from the catalog"))
                    .clone(),
            );
        }
        let plan = plan_with(&one, &[id]);
        if let Some(reason) = &plan.reason {
            panic!("real GutsUp skill {id} refused: {reason}");
        }
        assert!(!plan.live_guts, "real GutsUp skill {id} must stay baked");
    }
    // All of them installed at once is still a compiled, baked race.
    let plan = plan_with(&all, &REAL_GUTS_IDS);
    if let Some(reason) = &plan.reason {
        panic!("all real GutsUp skills at once refused: {reason}");
    }
    assert!(!plan.live_guts, "catalog-wide GutsUp stays baked");
    assert!(plan.bumps.iter().any(|(_, kind, _)| *kind == 4), "some bumps landed");
    // The two lottery-gated skills compile since the sampler landed: the
    // draw decides the init fire inside the kernel, so the bump is a live
    // guts record with no host-baked constant.
    for id in [202441i64, 202442] {
        let mut one = json::obj(vec![]);
        if let json::Value::Object(map) = &mut one {
            map.insert(id.to_string(), all.get(&id.to_string()).expect("id").clone());
        }
        let plan = plan_with(&one, &[id]);
        if let Some(reason) = &plan.reason {
            panic!("{id} should compile with the lottery sampler: {reason}");
        }
        assert!(plan.live_guts, "{id}: the draw-gated guts fire is live, not baked");
        assert!(plan.bumps.is_empty(), "{id}: a draw-gated bump is never host-baked");
    }
    // 210141 folds on its motivation condition instead: the default mood
    // gives motivation 3, so the `motivation>=4` alternative folds dead and
    // the skill compiles without a bump.
    let mut one = json::obj(vec![]);
    if let json::Value::Object(map) = &mut one {
        map.insert("210141".to_string(), all.get("210141").expect("id").clone());
    }
    let plan = plan_with(&one, &[210141]);
    if let Some(reason) = &plan.reason {
        panic!("210141 refused: {reason}");
    }
    assert!(plan.bumps.is_empty(), "210141's dead fold carries no bump");
}

fn def(cond: &str, cooldown: i64, effects: &str) -> String {
    format!(
        "{{\"rarity\":1,\"wisdomCheck\":0,\"priority\":0,\"alternatives\":[{{\"precondition\":\"\",\"condition\":\"{cond}\",\"cooldown\":{cooldown},\"baseDuration\":30000,\"effects\":[{effects}]}}]}}"
    )
}

fn plan_synthetic(sid: i64, def: &str) -> skills_compile::SkillsPlan {
    let text = format!(
        "{{\"skillData\":{{\"{sid}\":{}}},\"horses\":[{{\"skills\":[{{\"skillId\":{sid},\"level\":1}}]}}]}}",
        def
    );
    let root = json::parse(&text).expect("root parses");
    let course = json::parse(
        r#"{"distance":2000.0,"raceTrackId":10005,"surface":1,"distanceType":3,"turn":1,
            "straights":[{"start":0.0,"end":300.0},{"start":400.0,"end":2000.0}],
            "corners":[],"slopes":[]}"#,
    )
    .expect("course parses");
    skills_compile::plan(&root, &course)
}

#[test]
fn live_guts_flags_the_mid_race_shapes_only() {
    // Static self fire: a baked constant, live path stays off.
    let plan = plan_synthetic(669500, &def("always>=0", 0, r#"{"type":4,"target":1,"modifier":4000}"#));
    assert!(plan.reason.is_none(), "static kind 4 refused: {:?}", plan.reason);
    assert!(!plan.live_guts, "static kind 4 stays baked");
    // Phase-gated self fire: live.
    let plan = plan_synthetic(669501, &def("phase>=1", 15000, r#"{"type":4,"target":1,"modifier":4000}"#));
    assert!(plan.reason.is_none(), "gated kind 4 refused: {:?}", plan.reason);
    assert!(plan.live_guts, "gated kind 4 is live");
    // Cross fire of any condition shape: live.
    for cond in ["always>=0", "phase>=1", "order>=2"] {
        let plan = plan_synthetic(
            669502,
            &def(cond, 20000, r#"{"type":4,"target":9,"modifier":4000}"#),
        );
        assert!(plan.reason.is_none(), "cross kind 4 refused: {:?}", plan.reason);
        assert!(plan.live_guts, "cross kind 4 is live");
    }
    // A GutsUp-free race never flags, even with other live stat kinds.
    let plan = plan_synthetic(669503, &def("phase>=1", 15000, r#"{"type":5,"target":1,"modifier":4000}"#));
    assert!(plan.reason.is_none(), "gated kind 5 refused: {:?}", plan.reason);
    assert!(!plan.live_guts, "kind 5 alone keeps the baked guts words");
    // The mixed record flags once from its guts effect.
    let plan = plan_synthetic(
        669504,
        &def(
            "phase>=1",
            15000,
            r#"{"type":5,"target":1,"modifier":4000},{"type":4,"target":1,"modifier":4000}"#,
        ),
    );
    assert!(plan.reason.is_none(), "mixed record refused: {:?}", plan.reason);
    assert!(plan.live_guts, "mixed record flags live guts");
}
