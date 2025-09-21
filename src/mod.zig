//! ABI Framework - Main Module Interface
//!
//! High level entrypoints and curated re-exports for the reorganised framework
//! runtime. The new `framework` module exposes the orchestration layer that
//! coordinates feature toggles, plugin discovery, and lifecycle management.

const std = @import("std");
const framework = @import("framework/mod.zig");
const core = @import("shared/core/core.zig");
const lifecycle_mod = @import("shared/core/lifecycle.zig");

// =============================================================================
// FEATURE AND FRAMEWORK MODULES
// =============================================================================

/// Grouped feature modules mirroring the documentation structure.
pub const features = @import("features/mod.zig");

/// Framework orchestration layer that coordinates features and plugins.
pub const framework = @import("framework/mod.zig");

// =============================================================================
// SHARED MODULES
// =============================================================================

pub const utils = @import("shared/utils/mod.zig");
pub const core = @import("shared/core/mod.zig");
pub const platform = @import("shared/platform/mod.zig");
pub const logging = @import("shared/logging/mod.zig");
pub const simd = @import("shared/simd.zig");
pub const main = @import("main.zig");
pub const root = @import("root.zig");

// =============================================================================
// PUBLIC API
// =============================================================================

pub const Feature = framework.Feature;
pub const Framework = framework.Framework;
pub const FrameworkOptions = framework.FrameworkOptions;

/// Initialise the ABI framework and return the orchestration handle. Call
/// `Framework.deinit` (or `abi.shutdown`) when finished.
pub fn init(allocator: std.mem.Allocator, options: FrameworkOptions) !Framework {
    return try framework.runtime.Framework.init(allocator, options);
}

/// Convenience wrapper around `Framework.deinit` for callers that prefer the
/// legacy function-style shutdown.
pub fn shutdown(instance: *Framework) void {
    instance.deinit();
}

/// Get framework version information.
pub fn version() []const u8 {
    return "0.1.0-alpha";
}

test {
    std.testing.refAllDecls(@This());
}
