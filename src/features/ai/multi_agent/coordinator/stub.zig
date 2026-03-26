//! Multi-agent coordinator stub surface.

const std = @import("std");
const sync = @import("../../../../foundation/mod.zig").sync;
const agents_mod = @import("../../agents/stub.zig");
const messaging = @import("../messaging.zig");
const types = @import("../types.zig");

pub const Error = types.Error;
pub const AgentHealth = types.AgentHealth;
pub const AgentResult = types.AgentResult;
pub const CoordinatorConfig = types.CoordinatorConfig;
pub const CoordinatorStats = types.CoordinatorStats;

pub const Coordinator = struct {
    allocator: std.mem.Allocator = undefined,
    config: CoordinatorConfig = .{},
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
            .event_bus = if (config.enable_events) messaging.EventBus.init(allocator) else null,
        };
    }

    pub fn deinit(self: *Coordinator) void {
        self.results.deinit(self.allocator);
        for (self.mailboxes.items) |*mailbox| mailbox.deinit();
        self.mailboxes.deinit(self.allocator);
        self.health.deinit(self.allocator);
        self.agents.deinit(self.allocator);
        if (self.event_bus) |*bus| bus.deinit();
        self.* = undefined;
    }

    pub fn register(_: *Coordinator, _: *agents_mod.Agent) Error!void {
        return error.FeatureDisabled;
    }

    pub fn getAgentHealth(_: *const Coordinator, _: usize) ?AgentHealth {
        return null;
    }

    pub fn sendMessage(_: *Coordinator, _: messaging.AgentMessage) Error!void {
        return error.FeatureDisabled;
    }

    pub fn pendingMessages(_: *const Coordinator, _: usize) ?usize {
        return null;
    }

    pub fn agentCount(_: *const Coordinator) usize {
        return 0;
    }

    pub fn onEvent(_: *Coordinator, _: messaging.EventType, _: messaging.EventCallback) !void {
        return error.FeatureDisabled;
    }

    pub fn runTask(_: *Coordinator, _: []const u8) Error![]u8 {
        return error.FeatureDisabled;
    }

    pub fn getStats(_: *const Coordinator) CoordinatorStats {
        return .{};
    }
};


test {
    std.testing.refAllDecls(@This());
}
