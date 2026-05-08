//! Public core API wiring for `@import("abi")`.
//!
//! This module keeps root-level core exports grouped without changing the
//! public names that `src/root.zig` re-exports.

/// Application configuration: feature flags, platform settings, build profiles.
pub const config = @import("../features/core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

/// Framework error types and error set definitions.
pub const errors = @import("../features/core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

/// Service and plugin registry for runtime module discovery.
pub const registry = @import("../features/core/registry/mod.zig");
pub const Registry = registry.Registry;

/// Framework lifecycle: initialization, shutdown, state management.
pub const framework = @import("../features/core/framework.zig");

/// Framework application type.
pub const App = framework.Framework;
/// Framework builder type.
pub const AppBuilder = framework.FrameworkBuilder;

test {
    @import("std").testing.refAllDecls(@This());
}
