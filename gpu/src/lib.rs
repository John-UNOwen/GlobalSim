//! GPU spike for the GlobalSim restricted flat no-skills loop.
//!
//! The vendored engine files carry dead code (used only by the full engine).
#![allow(dead_code)]

// json.rs's Value::ReplayFrames names crate::replay_frames, so all three
// includes must sit at this crate root, exactly as the engine declares them.
// The engine package root is two levels up from this file (see gpu/README.md).
#[path = "../../src/replay_frames.rs"]
pub mod replay_frames;
#[path = "../../src/json.rs"]
pub mod json;
#[path = "../../src/retail_physics.rs"]
pub mod retail_physics;
#[path = "../../src/binary_replay.rs"]
mod binary_replay;

// The RNG includes stand apart from the json trio above. Neither file names
// crate::replay_frames, so they carry no dependency on Value::ReplayFrames.
#[path = "../../src/random_gen.rs"]
pub mod random_gen;
#[path = "../../src/retail_random.rs"]
pub mod retail_random;

// The engine module's own tests read env!("CARGO_MANIFEST_DIR")/data/..., so
// the include (and the host module over it) stay out of this crate's unit-test
// target, even with the feature on (see gpu/README.md, "Geometry port").
#[cfg(all(feature = "geometry", not(test)))]
pub mod retail_geometry;
#[cfg(all(feature = "geometry", not(test)))]
pub mod geometry;

// These includes need the json trio and retail_random above. Like the
// geometry pair they stay out of the lib's unit-test target (see gpu/README.md,
// "Initialization port").
#[cfg(all(feature = "initialization", not(test)))]
#[path = "../../src/course_data.rs"]
pub mod course_data;
#[cfg(all(feature = "initialization", not(test)))]
#[path = "../../src/retail_sort.rs"]
pub mod retail_sort;
#[cfg(all(feature = "initialization", not(test)))]
#[path = "../../src/retail_initialization.rs"]
pub mod retail_initialization;
#[cfg(all(feature = "initialization", not(test)))]
pub mod initialization;

// The whole-race host reads the geometry and initialization hosts, so it
// shares their not(test) gating (see the notes above).
#[cfg(all(feature = "race", not(test)))]
pub mod race_exec;

// The batch facade builds the compact meta plus the GSIMRP01 packet the app's
// run_compact path consumes, so it needs race_exec, the geometry and
// initialization hosts and the vendored binary_replay above.
#[cfg(all(feature = "race", not(test)))]
pub mod replay_packet;
#[cfg(all(feature = "race", not(test)))]
pub mod facade;

pub mod rng;

pub mod reference;

pub mod comparator;
pub mod dep_cpu;
pub mod race;
pub mod skills_compile;

#[cfg(feature = "gpu")]
pub mod gpu;
