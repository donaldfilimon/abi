//! Core module re-exports
//!
//! This module provides a unified entry point for all core framework components.
//! Submodules are imported and re-exported for use throughout the codebase.

const build_options = @import("build_options");

pub const config = @import("config/mod.zig");
pub const errors = @import("errors.zig");
pub const feature_catalog = @import("feature_catalog.zig");
pub const framework = @import("framework.zig");
pub const registry = @import("registry/mod.zig");
pub const stub_context = @import("stub_helpers.zig");
pub const comptime_meta = @import("comptime_meta.zig");

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
