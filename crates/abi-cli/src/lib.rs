//! Command metadata, help rendering, and process dispatch for ABI's CLI.
//!
//! During the Rust migration, the help surface is a stable contract boundary
//! checked against output captured from the Zig implementation. Command
//! handlers are attached incrementally behind the same process dispatcher.

pub mod app;
pub mod usage;

mod auth;
mod backends;
mod complete;
mod nn;
mod plugin;
mod train;

/// Serialization for tests that read or write the process-global telemetry table.
///
/// `abi_telemetry` counters are process-wide, and `abi scheduler status` renders
/// the whole table. Any test asserting that output therefore races every other
/// test that records an event — `plugin run` records `plugin.loaded` and
/// `plugin.run`. Cargo runs a crate's tests as threads in one process, so a
/// `reset()` alone is not enough; the readers and writers have to take turns.
#[cfg(test)]
pub(crate) mod telemetry_lock {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Acquire the telemetry lock for the duration of the returned guard.
    ///
    /// A poisoned lock is recovered rather than propagated: the data is a lock
    /// token, so an unwinding test leaves nothing inconsistent behind.
    pub(crate) fn acquire() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
mod scheduler;
mod wdbx;
mod wdbx_simulate;
