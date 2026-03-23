//! AI Models module — model registry and metadata.

const std = @import("std");

pub const registry = @import("registry.zig");
pub const ModelRegistry = registry.ModelRegistry;
pub const ModelInfo = registry.ModelInfo;

test {
    std.testing.refAllDecls(@This());
}
