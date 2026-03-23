//! AI models stub — disabled at compile time.

pub const registry = @import("registry.zig");

pub const ModelRegistry = registry.ModelRegistry;
pub const ModelInfo = registry.ModelInfo;

test {
    @import("std").testing.refAllDecls(@This());
}
