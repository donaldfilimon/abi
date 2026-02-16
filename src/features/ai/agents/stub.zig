//! Agents Stub Module

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");

pub const Error = error{ AgentsDisabled, AgentNotFound, ToolNotFound, ExecutionFailed, MaxAgentsReached };

pub const Agent = struct {};
pub const AgentConfig = struct {};
pub const Tool = struct {};
pub const ToolResult = struct {};
pub const ToolRegistry = struct {};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.AgentsConfig) Error!*Context {
        return error.AgentsDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn createAgent(_: *Context, _: []const u8) Error!*Agent {
        return error.AgentsDisabled;
    }
    pub fn getAgent(_: *Context, _: []const u8) ?*Agent {
        return null;
    }
    pub fn getToolRegistry(_: *Context) Error!*ToolRegistry {
        return error.AgentsDisabled;
    }
    pub fn registerTool(_: *Context, _: Tool) Error!void {
        return error.AgentsDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}
