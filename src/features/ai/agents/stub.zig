//! Agents Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
const types = @import("types.zig");

pub const Agent = types.Agent;
pub const AgentBackend = types.AgentBackend;
pub const AgentConfig = types.AgentConfig;
pub const Tool = types.Tool;
pub const ToolResult = types.ToolResult;
pub const ToolRegistry = types.ToolRegistry;

pub const Error = error{
    AgentsDisabled,
    AgentNotFound,
    ToolNotFound,
    ExecutionFailed,
    MaxAgentsReached,
};

pub const Context = struct {
    allocator: std.mem.Allocator = undefined,
    config: config_module.AgentsConfig = .{},
    agents: std.StringHashMapUnmanaged(*Agent) = .{},
    tool_registry: ?*ToolRegistry = null,

    pub fn init(_: std.mem.Allocator, _: config_module.AgentsConfig) !*Context {
        return error.AgentsDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn createAgent(_: *Context, _: []const u8) !*Agent {
        return error.AgentsDisabled;
    }
    pub fn getAgent(_: *Context, _: []const u8) ?*Agent {
        return null;
    }
    pub fn getToolRegistry(_: *Context) !*ToolRegistry {
        return error.AgentsDisabled;
    }
    pub fn registerTool(_: *Context, _: Tool) !void {
        return error.AgentsDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
