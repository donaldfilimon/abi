//! Agents Sub-module
//!
//! AI agent runtime with tool support and conversation management.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../config.zig");

// Re-export from existing agent module
const features_agent = @import("../../features/ai/agent.zig");
const features_tools = @import("../../features/ai/tools/mod.zig");

pub const Agent = features_agent.Agent;
pub const AgentConfig = features_agent.AgentConfig;
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

/// Agents context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AgentsConfig,
    agents: std.StringHashMapUnmanaged(*Agent),
    tool_registry: ?*ToolRegistry = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AgentsConfig) !*Context {
        if (!isEnabled()) return error.AgentsDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .agents = .{},
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        // Clean up agents
        var it = self.agents.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.agents.deinit(self.allocator);

        if (self.tool_registry) |r| {
            r.deinit();
            self.allocator.destroy(r);
        }

        self.allocator.destroy(self);
    }

    /// Create a new agent.
    pub fn createAgent(self: *Context, name: []const u8) !*Agent {
        if (self.agents.count() >= self.config.max_agents) {
            return error.MaxAgentsReached;
        }

        const agent_ptr = try self.allocator.create(Agent);
        errdefer self.allocator.destroy(agent_ptr);

        agent_ptr.* = try Agent.init(self.allocator, .{ .name = name });

        const name_copy = try self.allocator.dupe(u8, name);
        try self.agents.put(self.allocator, name_copy, agent_ptr);

        return agent_ptr;
    }

    /// Get an existing agent.
    pub fn getAgent(self: *Context, name: []const u8) ?*Agent {
        return self.agents.get(name);
    }

    /// Get or create the tool registry.
    pub fn getToolRegistry(self: *Context) !*ToolRegistry {
        if (self.tool_registry) |r| return r;

        const registry = try self.allocator.create(ToolRegistry);
        registry.* = ToolRegistry.init(self.allocator);
        self.tool_registry = registry;
        return registry;
    }

    /// Register a tool.
    pub fn registerTool(self: *Context, tool: Tool) !void {
        const registry = try self.getToolRegistry();
        try registry.register(tool);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_ai;
}
