//! System Information Stub
//!
//! Mirrors the public API of `system_info/mod.zig` when the
//! observability feature is disabled. All operations return the
//! `error.FeatureDisabled` error.

const std = @import("std");

pub const SystemInfo = struct {
    /// Stub implementation – always fails with FeatureDisabled.
    pub fn uptimeNs() error.FeatureDisabled {
        return error.FeatureDisabled;
    }

    pub fn timestampMs() error.FeatureDisabled {
        return error.FeatureDisabled;
    }
};
