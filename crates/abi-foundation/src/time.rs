//! Wall-clock and monotonic timestamps.
//!
//! Ported from `src/foundation/time.zig`. The Zig version had to reach for
//! `std.Options.debug_io` to get a clock without threading an `Io` handle
//! through every call site; Rust's `std::time` needs no such workaround.
//!
//! Signatures keep `i64` to match the Zig contract, since these values are
//! serialized into WDBX segment records and telemetry payloads.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Current Unix time in milliseconds since 1970-01-01.
///
/// Clamped at 0 for pre-epoch clocks rather than returning a negative value, so
/// serialized timestamps stay parseable.
#[must_use]
pub fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Current Unix time in whole seconds.
#[must_use]
pub fn unix_secs() -> i64 {
    unix_ms() / 1_000
}

/// Monotonic timestamp in nanoseconds.
///
/// Measured against a process-lifetime origin, so it is meaningful only as a
/// difference between two readings. Unaffected by wall-clock adjustments, which
/// is why durations and telemetry sampling use it instead of [`unix_ms`].
#[must_use]
pub fn monotonic_ns() -> i64 {
    // A fixed origin captured on first use; `Instant` has no public epoch.
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let origin = ORIGIN.get_or_init(Instant::now);
    i64::try_from(origin.elapsed().as_nanos()).unwrap_or(i64::MAX)
}

/// A simple elapsed-time measurement.
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    start: Instant,
}

impl Stopwatch {
    /// Start measuring from now.
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Nanoseconds elapsed since [`Stopwatch::start`].
    #[must_use]
    pub fn elapsed_ns(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Milliseconds elapsed since [`Stopwatch::start`].
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Fractional seconds elapsed since [`Stopwatch::start`].
    #[must_use]
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::start()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_ms_is_positive() {
        assert!(unix_ms() > 0);
    }

    #[test]
    fn unix_ms_does_not_go_backwards() {
        let first = unix_ms();
        let second = unix_ms();
        assert!(second >= first);
    }

    #[test]
    fn unix_secs_agrees_with_unix_ms() {
        let ms = unix_ms();
        let secs = unix_secs();
        assert!((secs - ms / 1_000).abs() <= 1);
    }

    #[test]
    fn monotonic_ns_advances() {
        let first = monotonic_ns();
        let second = monotonic_ns();
        assert!(second >= first);
    }

    #[test]
    fn stopwatch_measures_a_sleep() {
        let watch = Stopwatch::start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(watch.elapsed_ms() >= 4, "elapsed {}", watch.elapsed_ms());
        assert!(watch.elapsed_ns() > 0);
        assert!(watch.elapsed_secs_f64() > 0.0);
    }
}
