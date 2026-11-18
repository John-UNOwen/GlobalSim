//! Generates the course-geometry include used by `src/retail_geometry.rs`.
//!
//! `include!` of the engine file directly fails (E0753): an expanded inner
//! doc comment (`//!`) is never module-leading inside a macro expansion. The
//! generated copy turns those lines into plain comments (text preserved) and
//! rewrites the one nested `include!` so it resolves from OUT_DIR. The engine
//! file on disk is never modified.

use std::path::Path;

const ENGINE_FILE: &str = "../src/retail_geometry.rs";
const ASSETS_INCLUDE: &str = "include!(\"retail_geometry_assets.rs\")";
const ASSETS_FIXED: &str =
    "include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../src/retail_geometry_assets.rs\"))";

fn main() {
    println!("cargo::rerun-if-changed=../src/retail_geometry.rs");
    println!("cargo::rerun-if-changed=../src/retail_geometry_assets.rs");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let text = std::fs::read_to_string(Path::new(ENGINE_FILE))
        .unwrap_or_else(|e| panic!("{ENGINE_FILE} unreadable: {e}"));
    if !text.contains(ASSETS_INCLUDE) {
        panic!("{ENGINE_FILE} no longer embeds retail_geometry_assets.rs as expected");
    }
    let mut generated = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("//!") {
            generated.push_str("//");
            generated.push_str(rest);
        } else {
            generated.push_str(&line.replace(ASSETS_INCLUDE, ASSETS_FIXED));
        }
        generated.push('\n');
    }
    let dest = Path::new(&out_dir).join("retail_geometry_engine.rs");
    std::fs::write(&dest, generated).expect("write generated geometry include");
}
