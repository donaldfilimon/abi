//! AI Feature Stub — disabled at compile time.

const std = @import("std");
const config_mod = @import("config");

pub const types = @import("types");
pub const config = config_mod;
pub const registry = @import("registry");
pub const profiles = @import("profiles/stub");

pub const core = @import("core/stub");
pub const agents = @import("agents/stub");
pub const llm = @import("llm/stub");
pub const training = @import("training/stub");
pub const reasoning = @import("reasoning/stub");
pub const explore = @import("explore/stub");

// Compatibility layer
pub const personas = profiles;

pub fn init(_: std.mem.Allocator, _: config.AiConfig) !void {
    return error.AiDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
