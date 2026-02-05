//! System Information Submodule
//!
//! Provides lightweight runtime information such as process uptime.
//! This is a minimal utility that can be used by applications or
//! other framework components to query basic system metrics.

const std = @import("std");
// Shared utilities for wall‑clock ms timestamps
const utils = @import("../../../services/shared/utils.zig");

/// SystemInfo offers static helpers for runtime data.
pub const SystemInfo = struct {
    /// Returns the current monotonic time in nanoseconds.
    /// In Zig 0.16 use std.time.Timer for high‑resolution timing.
    pub fn uptimeNs() u64 {
        var timer = std.time.Timer.start() catch return 0;
        return timer.read();
    }

    /// Returns the current wall‑clock time in milliseconds since the Unix epoch.
    pub fn timestampMs() u64 {
        return utils.unixMs();
    }
};
