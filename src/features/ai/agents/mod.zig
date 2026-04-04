//! Agents sub-module facade.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");
const features_agent = @import("agent.zig");
const features_tools = @import("../tools/mod.zig");

pub const types = @import("types.zig");

pub const Agent = features_agent.Agent;
pub const AgentBackend = types.AgentBackend;
pub const AgentConfig = types.AgentConfig;
pub const AgentError = types.AgentError;
pub const Message = types.Message;
pub const ErrorContext = types.ErrorContext;
pub const OperationContext = types.OperationContext;
pub const BackendMetrics = types.BackendMetrics;
pub const MIN_TEMPERATURE = types.MIN_TEMPERATURE;
pub const MAX_TEMPERATURE = types.MAX_TEMPERATURE;
pub const MIN_TOP_P = types.MIN_TOP_P;
pub const MAX_TOP_P = types.MAX_TOP_P;
pub const MAX_TOKENS_LIMIT = types.MAX_TOKENS_LIMIT;
pub const DEFAULT_TEMPERATURE = types.DEFAULT_TEMPERATURE;
pub const DEFAULT_TOP_P = types.DEFAULT_TOP_P;
pub const DEFAULT_MAX_TOKENS = types.DEFAULT_MAX_TOKENS;
pub const Tool = features_tools.Tool;
pub const ToolResult = features_tools.ToolResult;
pub const ToolRegistry = features_tools.ToolRegistry;

pub const Error = error{
    AgentsDisabled,
    AgentNotFound,
    ToolNotFound,
    ExecutionFailed,
    MaxAgentsReached,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AgentsConfig,
    agents: std.StringHashMapUnmanaged(*Agent),
    owned_tools: std.ArrayListUnmanaged(*Tool),
    tool_registry: ?*ToolRegistry = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AgentsConfig) !*Context {
        if (!isEnabled()) return error.AgentsDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .agents = .empty,
            .owned_tools = .empty,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        var it = self.agents.iterator();
        while (it.next()) |entry| {
            // StringHashMap stores keys as []const u8; constCast required to free
            // heap-allocated keys that were duped on insert.
            self.allocator.free(@constCast(entry.key_ptr.*));
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.agents.deinit(self.allocator);

        for (self.owned_tools.items) |tool_ptr| {
            self.allocator.destroy(tool_ptr);
        }
        self.owned_tools.deinit(self.allocator);

        if (self.tool_registry) |registry| {
            registry.deinit();
            self.allocator.destroy(registry);
        }

        self.allocator.destroy(self);
    }

    pub fn createAgent(self: *Context, name: []const u8) !*Agent {
        if (self.agents.count() >= self.config.max_agents) {
            return error.MaxAgentsReached;
        }

        const agent_ptr = try self.allocator.create(Agent);
        errdefer self.allocator.destroy(agent_ptr);

        agent_ptr.* = try Agent.init(self.allocator, .{ .name = name });

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        try self.agents.put(self.allocator, name_copy, agent_ptr);

        return agent_ptr;
    }

    pub fn getAgent(self: *Context, name: []const u8) ?*Agent {
        return self.agents.get(name);
    }

    pub fn getToolRegistry(self: *Context) !*ToolRegistry {
        if (self.tool_registry) |registry| return registry;

        const registry = try self.allocator.create(ToolRegistry);
        registry.* = ToolRegistry.init(self.allocator);
        self.tool_registry = registry;
        return registry;
    }

    pub fn registerTool(self: *Context, tool: Tool) !void {
        const registry = try self.getToolRegistry();
        const tool_ptr = try self.allocator.create(Tool);
        errdefer self.allocator.destroy(tool_ptr);

        tool_ptr.* = tool;
        try self.owned_tools.append(self.allocator, tool_ptr);
        errdefer _ = self.owned_tools.pop();
        try registry.register(tool_ptr);
    }
};

pub fn isEnabled() bool {
    return build_options.feat_ai;
}

test {
    std.testing.refAllDecls(@This());
}
