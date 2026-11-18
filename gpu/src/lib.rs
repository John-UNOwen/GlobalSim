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

pub mod reference;

#[cfg(feature = "gpu")]
pub mod gpu;
