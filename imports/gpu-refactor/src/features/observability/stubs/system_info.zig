//! System Information Submodule (Stub)
//!
//! Stub implementation when observability is disabled.
//! All functions return 0.

const std = @import("std");

/// SystemInfo offers static helpers for runtime data.
/// Stub version returns 0 for all values.
pub const SystemInfo = struct {
    /// Returns 0 when observability is disabled.
    pub fn uptimeNs() u64 {
        return 0;
    }

    /// Returns 0 when observability is disabled.
    pub fn timestampMs() u64 {
        return 0;
    }
};

test {
    std.testing.refAllDecls(@This());
}
