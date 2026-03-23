//! AI Model management — model registry, info, and lifecycle.

const std = @import("std");
pub const registry = @import("registry.zig");

pub const ModelRegistry = registry.ModelRegistry;
pub const ModelInfo = registry.ModelInfo;

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}

pub fn isEnabled() bool {
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
