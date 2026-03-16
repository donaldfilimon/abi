//! AI Feature Stub — disabled at compile time.

const std = @import("std");
const config_mod = @import("config.zig");

pub const types = @import("types.zig");
pub const config = config_mod;
pub const registry = @import("registry.zig");
pub const profiles = @import("profiles/stub.zig");

pub const core = @import("core/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const llm = @import("llm/stub.zig");
pub const training = @import("training/stub.zig");
pub const reasoning = @import("reasoning/stub.zig");
pub const explore = @import("explore/stub.zig");

// Compatibility layer
pub const personas = profiles;

pub fn init(_: std.mem.Allocator, _: config.AiConfig) !void {
    return error.AiDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
