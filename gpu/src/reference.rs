//! The Rust reference for the restricted flat no-skills loop.
//!
//! `tick`, the six cases and the trace emitter reproduce the Jax repo's Rust
//! oracle `tools/restricted_oracle.rs` exactly, so `tests/reference_trace.rs`
//! can byte-compare this output with that repo's `tests/fixtures/restricted_trace.json`.

use super::json;
use super::retail_physics::{self, Parameters};
use std::path::PathBuf;

pub const DT: f32 = 0.066600002348423;
pub const ROWS: usize = 8;

#[derive(Clone, Copy)]
pub struct S {
    pub time: f32,
    pub pos: f32,
    pub speed: f32,
    pub hp: f32,
    pub delay: f32,
    pub phase: usize,
    pub dash: bool,
    pub finished: bool,
    pub finish: f32,
}

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub state: S,
    pub speed: f32,
    pub guts: f32,
    pub power: f32,
    pub strategy: usize,
    pub aptitude: f32,
    pub ground: f32,
    pub distance: f32,
}

/// One restricted tick, ported line for line from the oracle's `tick`.
pub fn tick(
    p: &Parameters,
    mut s: S,
    speed: f32,
    guts: f32,
    power: f32,
    strategy: usize,
    aptitude: f32,
    ground: f32,
    distance: f32,
    dt: f32,
) -> S {
    // M2 wrapper adopts the field scheduling mask: position == goal is no longer stepped.
    if s.finished || s.pos >= distance {
        return s;
    }
    let (delay, gate) = retail_physics::advance_start_delay(s.delay, dt);
    s.delay = delay;
    let Some(fraction) = gate else {
        s.time += dt;
        return s;
    };
    let (_, target_phase) = retail_physics::target_speed_indices(s.pos, distance);
    let base = p.base_target_speed(speed, strategy, target_phase, aptitude, distance);
    let ordinary = p.minimum_speed(guts, distance);
    let minimum = if s.dash { 3. } else { ordinary };
    let target = p.target_speed(
        retail_physics::TargetSpeed {
            base,
            section: 0.,
            pace: 1.,
            force_in: 0.,
            power,
            moving_lane_skill: false,
            slope: 0.,
            downhill: false,
            hp_remaining: s.hp > 0.,
            minimum,
        },
        &[],
    );
    let accel = if s.hp <= 0. {
        -1.2
    } else if s.speed > target {
        [-1.2, -0.8, -1.0][s.phase]
    } else {
        p.acceleration(
            retail_physics::Acceleration {
                power,
                strategy,
                phase: s.phase,
                ground,
                distance: aptitude,
                uphill: false,
                start_dash: s.dash,
            },
            &[],
        )
    };
    let mut next = s.speed + accel * dt;
    next = if s.speed <= target {
        next.min(target)
    } else {
        next.max(target)
    };
    let dash_limit = p.start_dash_speed(distance);
    if s.dash {
        next = next.min(dash_limit)
    };
    next = next.max(minimum);
    s.speed = next;
    let drain = if s.dash {
        20. * ground
    } else {
        p.hp_per_second(next, p.course_speed(distance), ground, guts, s.phase, 1.)
    };
    s.hp = (s.hp - drain * dt).max(0.);
    s.phase = retail_physics::phase_boundaries(distance)
        .iter()
        .filter(|&&b| s.pos >= b)
        .count()
        .min(2);
    let moved = ((next * dt) as f64 * fraction) as f32;
    let previous = s.pos;
    s.pos += moved;
    s.time += dt;
    if previous < distance && s.pos >= distance {
        s.finished = true;
        s.finish = retail_physics::finish_time(s.time, dt, previous, s.pos, distance)
    }
    if s.dash && next >= dash_limit {
        s.dash = false
    };
    s
}

/// The oracle's six cases, in fixture order.
pub fn cases(p: &Parameters) -> Vec<Case> {
    let dash_limit = p.start_dash_speed(1200.);
    vec![
        Case {
            name: "delayed",
            state: S {
                time: 0.,
                pos: 0.,
                speed: 0.,
                hp: p.initial_hp(1000., 1, 1200.),
                delay: 0.04,
                phase: 0,
                dash: true,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.1,
            distance: 1200.,
        },
        Case {
            name: "phase",
            state: S {
                time: 0.,
                pos: 199.99,
                speed: 18.,
                hp: 1000.,
                delay: 0.,
                phase: 0,
                dash: false,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.,
            distance: 1200.,
        },
        Case {
            name: "finish_overshoot",
            state: S {
                time: 0.,
                pos: 1199.8,
                speed: 20.,
                hp: 1000.,
                delay: 0.,
                phase: 2,
                dash: false,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.,
            distance: 1200.,
        },
        Case {
            name: "exact_goal_field_stop",
            state: S {
                time: 0.,
                pos: 1200.,
                speed: 20.,
                hp: 1000.,
                delay: 0.,
                phase: 2,
                dash: false,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.,
            distance: 1200.,
        },
        Case {
            name: "dash_exit",
            state: S {
                time: 0.,
                pos: 1.,
                speed: dash_limit,
                hp: 1000.,
                delay: 0.,
                phase: 0,
                dash: true,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.,
            distance: 1200.,
        },
        Case {
            name: "hp_zero_decel",
            state: S {
                time: 0.,
                pos: 1.,
                speed: 20.,
                hp: 0.,
                delay: 0.,
                phase: 0,
                dash: false,
                finished: false,
                finish: f32::NAN,
            },
            speed: 1000.,
            guts: 900.,
            power: 1000.,
            strategy: 1,
            aptitude: 1.,
            ground: 1.,
            distance: 1200.,
        },
    ]
}

/// The oracle's stdout, byte for byte (including the trailing newline).
pub fn emit_trace(p: &Parameters, cases: &[Case]) -> String {
    let mut out = String::from(
        "{\"provenance\":\"pinned Rust restricted extraction retail_physics.rs\",\"dt\":0.066600002348423,\"cases\":[",
    );
    for (ci, case) in cases.iter().enumerate() {
        if ci > 0 {
            out.push(',');
        }
        let initial = case.state;
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"inputs\":[{},{},{},{},{},{},{}],\"initial\":[{},{},{},{},{},{},{},{}],\"rows\":[",
            case.name,
            case.speed,
            case.guts,
            case.power,
            case.strategy,
            case.aptitude,
            case.ground,
            case.distance,
            initial.time.to_bits(),
            initial.pos.to_bits(),
            initial.speed.to_bits(),
            initial.hp.to_bits(),
            initial.delay.to_bits(),
            initial.phase,
            initial.dash as u8,
            initial.finished as u8
        ));
        let mut s = initial;
        for i in 0..ROWS {
            if i > 0 {
                out.push(',');
            }
            s = tick(
                p,
                s,
                case.speed,
                case.guts,
                case.power,
                case.strategy,
                case.aptitude,
                case.ground,
                case.distance,
                DT,
            );
            out.push_str(&format!(
                "[{}, {}, {}, {}, {}, {}, {}, {}, {}]",
                s.time.to_bits(),
                s.pos.to_bits(),
                s.speed.to_bits(),
                s.hp.to_bits(),
                s.delay.to_bits(),
                s.phase,
                s.dash as u8,
                s.finished as u8,
                s.finish.to_bits()
            ));
        }
        out.push_str("]}");
    }
    out.push_str("]}\n");
    out
}

/// The Jax clone, overridable with `GLOBALSIM_JAX` so the test can point
/// anywhere without hard-coding a user directory.
pub fn jax_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GLOBALSIM_JAX") {
        return dir.into();
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).expect("HOME");
    PathBuf::from(home).join("Documents/GitHub/GlobalSim-Jax")
}

pub fn load_parameters() -> Parameters {
    let path = jax_dir().join("globalsim_jax").join("global_parameters.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()));
    let root = json::parse(&text).expect("global_parameters.json parses");
    Parameters::from_value(root.get("parameters").expect("parameters object")).expect("Parameters")
}

pub fn fixture_bytes() -> Vec<u8> {
    let path = jax_dir().join("tests").join("fixtures").join("restricted_trace.json");
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()))
}

/// One case plus tick inputs, in the `Case` struct of
/// `kernels/restricted_tick.cl`. Field order and widths are the C layout;
/// `tests/layout.rs` checks the sizes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelCase {
    pub time: f32,
    pub pos: f32,
    pub speed: f32,
    pub hp: f32,
    pub delay: f32,
    pub finish: f32,
    pub phase: i32,
    pub dash: i32,
    pub finished: i32,
    pub speed_in: f32,
    pub guts: f32,
    pub power: f32,
    pub strategy: i32,
    pub aptitude: f32,
    pub ground: f32,
    pub distance: f32,
    pub dt: f32,
    pub section: f32,
    pub pace: f32,
    pub force_in: f32,
    pub slope: f32,
    pub moving_lane_skill: i32,
    pub downhill: i32,
    pub uphill: i32,
    pub spot_struggle: f32,
    pub bonus: f32,
}

/// One emitted tick row, the `Row` struct of `kernels/restricted_tick.cl`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelRow {
    pub time: f32,
    pub pos: f32,
    pub speed: f32,
    pub hp: f32,
    pub delay: f32,
    pub finish: f32,
    pub phase: i32,
    pub dash: i32,
    pub finished: i32,
}

/// Every scalar the restricted chain reads from `Parameters`, in the `Params`
/// struct of `kernels/restricted_tick.cl`. The chain never calls
/// `blocking_speed_cap`, `maximum_spurt_speed` (the only `powf`), `hp_state_rate`
/// or `initial_hp`, so blocking, spurt, hp_style, hp_rates and hp_initial are
/// skipped. `accel[2]`/upstream slope coefficients ride along because the
/// acceleration and target-speed ports index them.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelParams {
    pub course: [f32; 3],
    pub phase_speed: [[f32; 4]; 5],
    pub phase_accel: [[f32; 4]; 5],
    pub minimum: [f32; 3],
    pub end_speed_sqrt: f32,
    pub add_speed: f32,
    pub slope: [f32; 3],
    pub target_min: f32,
    pub accel: [f32; 4],
    pub hp_drain: [f32; 5],
}

/// Re-reads the scalars from the same `global_parameters.json`, using the same
/// keys and the same `get_f64(...) as f32` conversions as
/// `Parameters::from_value` (its fields are private to the vendored module).
/// `tests/layout.rs` cross-checks the pack against `Parameters`' public methods.
pub fn kernel_params() -> KernelParams {
    let path = jax_dir().join("globalsim_jax").join("global_parameters.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {} failed: {e}", path.display()));
    kernel_params_from_text(&text)
}

/// Packs the same scalars from any file's text, so tools can read the
/// engine's own `data/global_parameters.json` without the Jax checkout.
pub fn kernel_params_from_text(text: &str) -> KernelParams {
    let root = json::parse(text).expect("global parameters parse");
    let parameters = root.get("parameters").expect("parameters object");
    let group = |key| parameters.get(key).unwrap_or_else(|| panic!("group {key}"));
    let number = |value: &json::Value, key: &str| value.get_f64(key).unwrap_or_else(|_| panic!("number {key}")) as f32;
    let phases = |group: &json::Value, regular: &str, runaway: &str| -> [[f32; 4]; 5] {
        let mut rows = group
            .get_array(regular)
            .unwrap_or_else(|_| panic!("array {regular}"))
            .iter()
            .chain(group.get_array(runaway).unwrap_or_else(|_| panic!("array {runaway}")))
            .map(|row| [
                number(row, "Start"),
                number(row, "Middle"),
                number(row, "End"),
                number(row, "Last"),
            ]);
        let mut result = [[0.; 4]; 5];
        for slot in result.iter_mut() {
            *slot = rows.next().expect("five phase rows");
        }
        result
    };
    let speed = group("Speed");
    let base = group("BasetTargetSpeed");
    KernelParams {
        course: [
            number(parameters, "raceBaseSpeedBaseSpeed"),
            number(parameters, "raceBaseSpeedBaseDistance"),
            number(parameters, "raceBaseSpeedDistanceRate"),
        ],
        phase_speed: phases(
            base,
            "PhaseBaseTargetSpeedPerArray",
            "PhaseBaseTargetSpeedPerExArray",
        ),
        phase_accel: phases(speed, "PhaseAccelCoefArray", "PhaseAccelCoefExArray"),
        minimum: [
            number(speed, "MinSpeedRate"),
            number(speed, "MinSpeedGutsCoefSqrt"),
            number(speed, "MinSpeedGutsCoef"),
        ],
        end_speed_sqrt: number(base, "PhaseEndBaseTargetSpeedCoef"),
        add_speed: number(parameters, "addSpeedParamCoef"),
        slope: [
            number(group("Slope"), "UpSlopeAddSpeedVal1"),
            number(group("Slope"), "DownSlopeAddSpeedVal1"),
            number(group("Slope"), "DownSlopeAddSpeedVal2"),
        ],
        target_min: number(speed, "TargetSpeedMin"),
        accel: [
            number(parameters, "AccelPowCoefSqrt"),
            number(parameters, "accelPowCoef"),
            number(parameters, "accelPowCoefUpSlope"),
            number(parameters, "StartAccelAdd"),
        ],
        hp_drain: [
            number(group("Hp"), "SpeedGapParam1"),
            number(group("Hp"), "SpeedGapParam1Pow"),
            number(group("Hp"), "HpDecBase"),
            number(group("Hp"), "HpGutsCoefSqrt"),
            number(group("Hp"), "HpGutsCoef"),
        ],
    }
}

pub fn kernel_case(case: &Case) -> KernelCase {
    KernelCase {
        time: case.state.time,
        pos: case.state.pos,
        speed: case.state.speed,
        hp: case.state.hp,
        delay: case.state.delay,
        finish: case.state.finish,
        phase: case.state.phase as i32,
        dash: case.state.dash as i32,
        finished: case.state.finished as i32,
        speed_in: case.speed,
        guts: case.guts,
        power: case.power,
        strategy: case.strategy as i32,
        aptitude: case.aptitude,
        ground: case.ground,
        distance: case.distance,
        dt: DT,
        section: 0.,
        pace: 1.,
        force_in: 0.,
        slope: 0.,
        moving_lane_skill: 0,
        downhill: 0,
        uphill: 0,
        spot_struggle: 0.,
        bonus: 0.,
    }
}

/// Deterministic synthetic batch: six fixture cases, then stats/ground/delay
/// varied on a fixed integer pattern so both bench sides see identical inputs.
pub fn bench_cases(p: &Parameters, n: usize) -> Vec<Case> {
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next_value = move |range: f32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f32 / u64::MAX as f32) * range
    };
    let base = cases(p);
    (0..n)
        .map(|index| {
            let mut case = base[index % base.len()];
            case.speed = 200. + next_value(1600.);
            case.guts = 100. + next_value(1800.);
            case.power = 100. + next_value(1800.);
            case.ground = 0.8 + next_value(0.6);
            case.state.delay = if index % 4 == 0 { next_value(0.08) } else { 0. };
            case.state.hp = 300. + next_value(1200.);
            case.state.pos = next_value(case.distance - 1.);
            case.state.speed = 15. + next_value(6.);
            case.state.phase = index % 3;
            case.state.dash = index % 7 == 0;
            case.strategy = 1 + index % 5;
            case
        })
        .collect()
}
