//! Helper utilities for applying per-request overrides to agents.

const std = @import("std");
const agent_mod = @import("../agents/agent.zig");
const types = @import("types.zig");

pub const Overrides = struct {
    prev_temperature: f32,
    prev_max_tokens: u32,
    system_index: ?usize = null,
};

pub fn apply(allocator: std.mem.Allocator, agent: *agent_mod.Agent, request: types.PersonaRequest) !Overrides {
    var overrides = Overrides{
        .prev_temperature = agent.config.temperature,
        .prev_max_tokens = agent.config.max_tokens,
    };
    errdefer restore(allocator, agent, overrides);

    if (request.temperature) |temp| {
        try agent.setTemperature(temp);
    }
    if (request.max_tokens) |tokens| {
        try agent.setMaxTokens(tokens);
    }
    if (request.system_instruction) |instruction| {
        const content_copy = try allocator.dupe(u8, instruction);
        errdefer allocator.free(content_copy);
        overrides.system_index = agent.history.items.len;
        try agent.history.append(allocator, .{
            .role = .system,
            .content = content_copy,
        });
    }

    return overrides;
}

pub fn restore(allocator: std.mem.Allocator, agent: *agent_mod.Agent, overrides: Overrides) void {
    agent.config.temperature = overrides.prev_temperature;
    agent.config.max_tokens = overrides.prev_max_tokens;

    if (overrides.system_index) |idx| {
        if (idx < agent.history.items.len) {
            const removed = agent.history.orderedRemove(idx);
            allocator.free(removed.content);
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
