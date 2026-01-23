//! System Information Submodule
//!
//! Provides lightweight runtime information such as process uptime.
//! This is a minimal utility that can be used by applications or
//! other framework components to query basic system metrics.

const std = @import("std");

/// SystemInfo offers static helpers for runtime data.
pub const SystemInfo = struct {
    /// Returns the current monotonic time in nanoseconds.
    /// In Zig 0.16 the preferred high‑resolution timestamp is
    /// `std.time.nanoTimestamp()`.
    pub fn uptimeNs() u64 {
        return std.time.nanoTimestamp();
    }

    /// Returns the current wall‑clock time in milliseconds since the Unix epoch.
    pub fn timestampMs() u64 {
        return std.time.milliTimestamp();
    }
};
