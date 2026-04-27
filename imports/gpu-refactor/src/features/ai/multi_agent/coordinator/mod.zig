//! Multi-agent coordinator runtime surface.

const std = @import("std");
const sync = @import("../../../../foundation/mod.zig").sync;
const agents_mod = @import("../../agents/mod.zig");
const messaging = @import("../messaging.zig");
const types = @import("../types.zig");
const runtime = @import("runtime.zig");

pub const Error = types.Error;
pub const AgentHealth = types.AgentHealth;
pub const AgentResult = types.AgentResult;
pub const CoordinatorConfig = types.CoordinatorConfig;
pub const CoordinatorStats = types.CoordinatorStats;

pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    config: CoordinatorConfig,
    agents: std.ArrayListUnmanaged(*agents_mod.Agent) = .empty,
    health: std.ArrayListUnmanaged(AgentHealth) = .empty,
    mailboxes: std.ArrayListUnmanaged(messaging.AgentMailbox) = .empty,
    results: std.ArrayListUnmanaged(AgentResult) = .empty,
    mutex: sync.Mutex = .{},
    event_bus: ?messaging.EventBus = null,

    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return initWithConfig(allocator, .{});
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, config: CoordinatorConfig) Coordinator {
        return .{
            .allocator = allocator,
            .config = config,
            .agents = .empty,
            .health = .empty,
            .mailboxes = .empty,
            .results = .empty,
            .mutex = .{},
            .event_bus = if (config.enable_events) messaging.EventBus.init(allocator) else null,
        };
    }

    pub fn deinit(self: *Coordinator) void {
        runtime.deinit(self);
    }

    pub fn register(self: *Coordinator, agent_ptr: *agents_mod.Agent) Error!void {
        return runtime.register(self, agent_ptr);
    }

    pub fn getAgentHealth(self: *const Coordinator, index: usize) ?AgentHealth {
        return runtime.getAgentHealth(self, index);
    }

    pub fn sendMessage(self: *Coordinator, msg: messaging.AgentMessage) Error!void {
        return runtime.sendMessage(self, msg);
    }

    pub fn pendingMessages(self: *const Coordinator, agent_index: usize) ?usize {
        return runtime.pendingMessages(self, agent_index);
    }

    pub fn agentCount(self: *const Coordinator) usize {
        return runtime.agentCount(self);
    }

    pub fn onEvent(self: *Coordinator, event_type: messaging.EventType, callback: messaging.EventCallback) !void {
        return runtime.onEvent(self, event_type, callback);
    }

    pub fn runTask(self: *Coordinator, task: []const u8) Error![]u8 {
        return runtime.runTask(self, task);
    }

    pub fn getStats(self: *const Coordinator) CoordinatorStats {
        return runtime.getStats(self);
    }
};

test {
    std.testing.refAllDecls(@This());
}
