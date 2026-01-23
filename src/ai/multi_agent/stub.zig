//! Multi‑Agent Stub Module
//!
//! Mirrors the public API of `multi_agent/mod.zig` when the AI feature
//! is disabled. All operations return `error.AgentDisabled`.

const std = @import("std");

pub const Error = error{ AgentDisabled, NoAgents };

pub const Coordinator = struct {
    pub fn init(_: std.mem.Allocator) Coordinator {
        return .{};
    }
    pub fn deinit(_: *Coordinator) void {}
    pub fn register(_: *Coordinator, _: *anyopaque) Error!void {
        return error.AgentDisabled;
    }
    pub fn runTask(_: *Coordinator, _: []const u8) Error![]u8 {
        return error.AgentDisabled;
    }
};
