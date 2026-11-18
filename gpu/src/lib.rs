//! GPU spike for the GlobalSim restricted flat no-skills loop.
//!
//! The vendored engine files carry dead code (used only by the full engine).
#![allow(dead_code)]

// json.rs's Value::ReplayFrames names crate::replay_frames, so all three
// includes must sit at this crate root, exactly as the engine declares them.
// The engine package root is two levels up from this file (see gpu/README.md).
#[path = "../../src/replay_frames.rs"]
mod replay_frames;
#[path = "../../src/json.rs"]
mod json;
#[path = "../../src/retail_physics.rs"]
pub mod retail_physics;

// The RNG includes stand apart from the json trio above. Neither file names
// crate::replay_frames, so they carry no dependency on Value::ReplayFrames.
#[path = "../../src/random_gen.rs"]
pub mod random_gen;
#[path = "../../src/retail_random.rs"]
pub mod retail_random;

pub mod rng;

pub mod reference;

pub mod comparator;
pub mod dep_cpu;
pub mod race;

#[cfg(feature = "gpu")]
pub mod gpu;
