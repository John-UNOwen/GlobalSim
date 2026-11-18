//! Byte-compares the in-crate reference emitter against the Jax repo's fixture.

use globalsim_gpu::reference;

#[test]
fn reference_trace_matches_the_jax_fixture() {
    let parameters = reference::load_parameters();
    let emitted = reference::emit_trace(&parameters, &reference::cases(&parameters));
    let fixture = String::from_utf8(reference::fixture_bytes()).expect("fixture is UTF-8");
    let normalize = |text: &str| text.replace("\r\n", "\n");
    let emitted = normalize(&emitted);
    let fixture = normalize(&fixture);
    if emitted != fixture {
        let differing = emitted
            .chars()
            .zip(fixture.chars())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(index, _)| format!("first difference at byte offset {index}"))
            .unwrap_or_else(|| {
                format!("prefix equal, lengths differ: emitted {} fixture {}", emitted.len(), fixture.len())
            });
        panic!("reference trace differs from restricted_trace.json: {differing}");
    }
}
