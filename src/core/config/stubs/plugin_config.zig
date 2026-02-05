const std = @import("std");

pub const PluginConfig = struct {
    paths: []const []const u8 = &.{},
    auto_discover: bool = false,
    load: []const []const u8 = &.{},
    allow_untrusted: bool = false,

    pub fn defaults() PluginConfig {
        return .{};
    }

    pub fn withPaths(paths: []const []const u8) PluginConfig {
        _ = paths;
        return .{};
    }
};
