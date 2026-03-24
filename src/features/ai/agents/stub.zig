//! Agents Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
const shared_types = @import("types.zig");

pub const types = shared_types;
pub const Agent = shared_types.Agent;
pub const AgentBackend = shared_types.AgentBackend;
pub const AgentConfig = shared_types.AgentConfig;
pub const AgentError = shared_types.AgentError;
pub const Message = shared_types.Message;
pub const ErrorContext = shared_types.ErrorContext;
pub const OperationContext = shared_types.OperationContext;
pub const BackendMetrics = shared_types.BackendMetrics;
pub const MIN_TEMPERATURE = shared_types.MIN_TEMPERATURE;
pub const MAX_TEMPERATURE = shared_types.MAX_TEMPERATURE;
pub const MIN_TOP_P = shared_types.MIN_TOP_P;
pub const MAX_TOP_P = shared_types.MAX_TOP_P;
pub const MAX_TOKENS_LIMIT = shared_types.MAX_TOKENS_LIMIT;
pub const DEFAULT_TEMPERATURE = shared_types.DEFAULT_TEMPERATURE;
pub const DEFAULT_TOP_P = shared_types.DEFAULT_TOP_P;
pub const DEFAULT_MAX_TOKENS = shared_types.DEFAULT_MAX_TOKENS;
pub const Tool = shared_types.Tool;
pub const ToolResult = shared_types.ToolResult;
pub const ToolRegistry = shared_types.ToolRegistry;

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
    agents: std.StringHashMapUnmanaged(*Agent) = .empty,
    owned_tools: std.ArrayListUnmanaged(*Tool) = .empty,
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
