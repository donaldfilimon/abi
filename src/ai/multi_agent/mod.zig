//! Multi‑Agent Coordination Module
//!
//! Provides a lightweight orchestrator that can run a collection of
//! `agents.Agent` instances on a given task. This is useful for
//! collaborative refactoring, code‑base analysis, or any workflow
//! that benefits from multiple specialized agents.
//!
//! The implementation is intentionally minimal – it demonstrates the
//! pattern without pulling in heavy dependencies. Real‑world projects can
//! extend it with richer scheduling, parallel execution, and result
//! aggregation.

const std = @import("std");
const agents = @import("../agents/mod.zig");

pub const Error = error{
    AgentDisabled, // Underlying agents module disabled
    NoAgents, // No agents registered in the coordinator
};

/// Coordinator holds a list of agents that can be invoked sequentially.
pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    agents: std.ArrayListUnmanaged(*agents.Agent) = .{},

    /// Initialise the coordinator with an allocator.
    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return .{ .allocator = allocator };
    }

    /// Deinitialise and free resources.
    pub fn deinit(self: *Coordinator) void {
        // Destroy each agent if owned – here we assume callers manage life‑time.
        self.agents.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register an existing agent instance.
    pub fn register(self: *Coordinator, agent_ptr: *agents.Agent) !void {
        try self.agents.append(self.allocator, agent_ptr);
    }

    /// Run a textual task across all registered agents.
    /// Returns the combined output of each agent concatenated.
    pub fn runTask(self: *Coordinator, task: []const u8) ![]u8 {
        if (self.agents.items.len == 0) return error.NoAgents;
        var builder: std.ArrayListUnmanaged(u8) = .{};
        defer builder.deinit(self.allocator);
        for (self.agents.items) |ag| {
            const response = try ag.process(task, self.allocator);
            defer self.allocator.free(response);
            try builder.appendSlice(self.allocator, response);
            try builder.appendSlice(self.allocator, "\n---\n");
        }
        return builder.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "coordinator init and deinit" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Coordinator starts with no agents
    try std.testing.expectEqual(@as(usize, 0), coord.agents.items.len);
}

test "coordinator runTask with no agents returns error" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Running task with no agents should return NoAgents error
    const result = coord.runTask("test task");
    try std.testing.expectError(error.NoAgents, result);
}
