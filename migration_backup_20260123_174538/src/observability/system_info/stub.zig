//! System Information Stub
//!
//! Mirrors the public API of `system_info/mod.zig` when the
//! observability feature is disabled. All operations return the
//! `error.ObservabilityDisabled` error.

const std = @import("std");

pub const SystemInfo = struct {
    /// Stub implementation – always fails with ObservabilityDisabled.
    pub fn uptimeNs() error.ObservabilityDisabled {
        return error.ObservabilityDisabled;
    }

    pub fn timestampMs() error.ObservabilityDisabled {
        return error.ObservabilityDisabled;
    }
};
