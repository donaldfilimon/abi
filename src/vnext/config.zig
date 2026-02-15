//! vNext application configuration.

const core_config = @import("../core/config/mod.zig");
const capability_mod = @import("capability.zig");

pub const AppConfig = struct {
    /// Underlying framework configuration (compatibility layer for staged migration).
    framework: core_config.Config = .{},

    /// Capabilities that must be enabled for successful startup when strict mode is on.
    required_capabilities: []const capability_mod.Capability = &.{},

    /// When true, `App.init` fails if any required capability is not enabled.
    strict_capability_check: bool = false,

    pub fn defaults() AppConfig {
        return .{
            .framework = core_config.Config.defaults(),
        };
    }

    pub fn minimal() AppConfig {
        return .{
            .framework = core_config.Config.minimal(),
        };
    }
};

const std = @import("std");

test "AppConfig defaults creates valid config" {
    const cfg = AppConfig.defaults();
    try std.testing.expect(cfg.required_capabilities.len == 0);
    try std.testing.expect(!cfg.strict_capability_check);
}

test "AppConfig minimal creates valid config" {
    const cfg = AppConfig.minimal();
    try std.testing.expect(cfg.required_capabilities.len == 0);
    try std.testing.expect(!cfg.strict_capability_check);
}

test "AppConfig strict mode with capabilities" {
    const caps = [_]capability_mod.Capability{ .gpu, .ai };
    const cfg = AppConfig{
        .strict_capability_check = true,
        .required_capabilities = &caps,
    };
    try std.testing.expect(cfg.strict_capability_check);
    try std.testing.expectEqual(@as(usize, 2), cfg.required_capabilities.len);
}
