//! Every integration suite in one test binary.
//!
//! `cargo test` runs test binaries one after another, so 63 binaries left the
//! machine mostly idle. In one binary the tests run on every core at once, and
//! their GPU calls share launches (src/coalesce.rs), which is where most of the
//! wait was. The files stay in tests/, and each one is a module here, so a
//! filter like `cargo test --features race --test suite wave3_gate::` runs one
//! file. A file under tests/ that skips its `mod` line below never runs.
mod par;

#[path = "../app_preset_gate.rs"]
mod app_preset_gate;
#[path = "../advantage_fan_gate.rs"]
mod advantage_fan_gate;
#[path = "../badstart_gate.rs"]
mod badstart_gate;
#[path = "../basis_distance_gate.rs"]
mod basis_distance_gate;
#[path = "../carnival_noop_gate.rs"]
mod carnival_noop_gate;
#[path = "../classic_folds.rs"]
mod classic_folds;
#[path = "../compact_gate.rs"]
mod compact_gate;
#[path = "../comparator.rs"]
mod comparator;
#[path = "../constructor_repeat_gate.rs"]
mod constructor_repeat_gate;
#[path = "../dirt_pacemaker_gate.rs"]
mod dirt_pacemaker_gate;
#[path = "../equivalence_gate.rs"]
mod equivalence_gate;
#[path = "../facade_gate.rs"]
mod facade_gate;
#[path = "../fold_gate.rs"]
mod fold_gate;
#[path = "../geometry_gate.rs"]
mod geometry_gate;
#[path = "../gpu_gate.rs"]
mod gpu_gate;
#[path = "../grid_gate.rs"]
mod grid_gate;
#[path = "../initialization_gate.rs"]
mod initialization_gate;
#[path = "../lastspurt_gate.rs"]
mod lastspurt_gate;
#[path = "../laststraight_gate.rs"]
mod laststraight_gate;
#[path = "../random_lot_gate.rs"]
mod random_lot_gate;
#[path = "../sampled_long_gate.rs"]
mod sampled_long_gate;
#[path = "../chara_exist_gate.rs"]
mod chara_exist_gate;
#[path = "../coalesce_gate.rs"]
mod coalesce_gate;
#[path = "../layout.rs"]
mod layout;
#[path = "../paired_race_gate.rs"]
mod paired_race_gate;
#[path = "../parity_audit.rs"]
mod parity_audit;
#[path = "../powerup_gate.rs"]
mod powerup_gate;
#[path = "../preset_dump.rs"]
mod preset_dump;
#[path = "../race_diff.rs"]
mod race_diff;
#[path = "../reference_trace.rs"]
mod reference_trace;
#[path = "../rng_gate.rs"]
mod rng_gate;
#[path = "../sk_emax_gate.rs"]
mod sk_emax_gate;
#[path = "../sk_tmax_gate.rs"]
mod sk_tmax_gate;
#[path = "../sk_afterend_gate.rs"]
mod sk_afterend_gate;
#[path = "../skills_gate.rs"]
mod skills_gate;
#[path = "../slice_gate.rs"]
mod slice_gate;
#[path = "../solo_race_gate.rs"]
mod solo_race_gate;
#[path = "../stall_probe.rs"]
mod stall_probe;
#[path = "../spurt_plan_gate.rs"]
mod spurt_plan_gate;
#[path = "../straightfront_gate.rs"]
mod straightfront_gate;
#[path = "../startup_gate.rs"]
mod startup_gate;
#[path = "../target18_gate.rs"]
mod target18_gate;
#[path = "../temptation_opp_gate.rs"]
mod temptation_opp_gate;
#[path = "../trick_fan_gate.rs"]
mod trick_fan_gate;
#[path = "../wave10_folds.rs"]
mod wave10_folds;
#[path = "../wave10_gate.rs"]
mod wave10_gate;
#[path = "../wave12a_classic_gate.rs"]
mod wave12a_classic_gate;
#[path = "../wave12a_full_gate.rs"]
mod wave12a_full_gate;
#[path = "../wave12b_full_gate.rs"]
mod wave12b_full_gate;
#[path = "../wave12b_gate.rs"]
mod wave12b_gate;
#[path = "../wave12c_classic_gate.rs"]
mod wave12c_classic_gate;
#[path = "../wave12c_modes.rs"]
mod wave12c_modes;
#[path = "../wave13a_later_stamina.rs"]
mod wave13a_later_stamina;
#[path = "../wave17_classic_drift.rs"]
mod wave17_classic_drift;
#[path = "../wave18_gate.rs"]
mod wave18_gate;
#[path = "../wave19_overcap.rs"]
mod wave19_overcap;
#[path = "../wave1_folds.rs"]
mod wave1_folds;
#[path = "../wave1_gate.rs"]
mod wave1_gate;
#[path = "../wave2_folds.rs"]
mod wave2_folds;
#[path = "../wave2_gate.rs"]
mod wave2_gate;
#[path = "../wave2b_folds.rs"]
mod wave2b_folds;
#[path = "../wave2b_gate.rs"]
mod wave2b_gate;
#[path = "../wave3_folds.rs"]
mod wave3_folds;
#[path = "../wave3_gate.rs"]
mod wave3_gate;
#[path = "../wave4_folds.rs"]
mod wave4_folds;
#[path = "../wave4_gate.rs"]
mod wave4_gate;
#[path = "../wave6a_folds.rs"]
mod wave6a_folds;
#[path = "../wave6a_gate.rs"]
mod wave6a_gate;
#[path = "../wave6b_folds.rs"]
mod wave6b_folds;
#[path = "../wave6b_gate.rs"]
mod wave6b_gate;
#[path = "../wave7_folds.rs"]
mod wave7_folds;
#[path = "../wave7_gate.rs"]
mod wave7_gate;
#[path = "../wave8_folds.rs"]
mod wave8_folds;
#[path = "../wave8_gate.rs"]
mod wave8_gate;
#[path = "../wave8_powf_gate.rs"]
mod wave8_powf_gate;
#[path = "../wave9_fan_trace.rs"]
mod wave9_fan_trace;
#[path = "../wave9_folds.rs"]
mod wave9_folds;
#[path = "../wave9_gate.rs"]
mod wave9_gate;
#[path = "../wave9c_folds.rs"]
mod wave9c_folds;
#[path = "../wave9c_gate.rs"]
mod wave9c_gate;
#[path = "../x11_sampled_gate.rs"]
mod x11_sampled_gate;
#[path = "../zero_skill_dump.rs"]
mod zero_skill_dump;
#[path = "../zero_skill_gate.rs"]
mod zero_skill_gate;
