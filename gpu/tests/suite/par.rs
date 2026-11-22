//! Run a test's independent cases side by side.
//!
//! A gate that loops over cases makes one GPU launch per case, one after
//! another. Run on their own threads, the cases' launches go out together
//! (src/coalesce.rs) and their CPU reference races use the other cores.

/// `f` over every item on its own thread; answers come back in item order,
/// so problem lists and reports read exactly as the sequential loop's did.
/// A panicking case re-raises its panic here.
pub fn map<T: Send, R: Send>(items: impl IntoIterator<Item = T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items.into_iter().map(|item| s.spawn(move || f(item))).collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            .collect()
    })
}
