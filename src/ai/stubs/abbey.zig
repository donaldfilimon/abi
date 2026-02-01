const std = @import("std");
const core = @import("../core/mod.zig");

pub const Abbey = struct {};
pub const Stats = struct {};
pub const ReasoningChain = struct {};
pub const ReasoningStep = struct {};
pub const ConversationContext = struct {};

pub const AbbeyInstance = struct {
    pub fn runRalphLoop(_: *@This(), _: []const u8, _: usize) Error![]const u8 {
        return error.AiDisabled;
    }

    pub fn deinit(_: *@This()) void {}
};

const Error = error{AiDisabled};

pub fn createEngine(allocator: std.mem.Allocator) Error!AbbeyInstance {
    _ = allocator;
    return error.AiDisabled;
}

pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: core.AbbeyConfig) Error!AbbeyInstance {
    _ = allocator;
    _ = config;
    return error.AiDisabled;
}
