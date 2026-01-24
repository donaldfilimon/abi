//! Multi‑Agent Stub Module
//!
//! Mirrors the public API of `multi_agent/mod.zig` when the AI feature
//! is disabled. All operations return `error.AgentDisabled`.

const std = @import("std");

pub const Error = error{ AgentDisabled, NoAgents };

pub const Coordinator = struct {
    allocator: std.mem.Allocator = undefined,
    agents: std.ArrayListUnmanaged(*anyopaque) = .{},

    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *Coordinator) void {
        self.agents.deinit(self.allocator);
    }
    pub fn register(_: *Coordinator, _: *anyopaque) Error!void {
        return error.AgentDisabled;
    }
    pub fn runTask(_: *Coordinator, _: []const u8) Error![]u8 {
        return error.AgentDisabled;
    }
};
