//! Stub for multi-profile orchestration when feat_ai is disabled.

const std = @import("std");
pub const types = @import("types.zig");
const pipeline_types = @import("../pipeline/types.zig");

pub const ProfileId = types.ProfileId;
pub const ProfileState = types.ProfileState;
pub const RoutingStrategy = types.RoutingStrategy;
pub const RoutingDecision = types.RoutingDecision;
pub const ProfileResponse = types.ProfileResponse;
pub const ProfileMessage = types.ProfileMessage;
pub const MessageKind = types.MessageKind;
pub const RoutingConfig = types.RoutingConfig;
pub const ProfileError = types.ProfileError;

pub const MultiProfileConfig = struct {
    routing: RoutingConfig = .{},
};

pub const ProfileInstance = struct {
    id: ProfileId,
    state: ProfileState = .uninitialized,
    allocator: std.mem.Allocator,

    pub fn process(_: *ProfileInstance, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn getName(self: *const ProfileInstance) []const u8 {
        return self.id.name();
    }
    pub fn isAvailable(_: *const ProfileInstance) bool {
        return false;
    }
};

pub const ProfileRegistry = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, _: MultiProfileConfig) ProfileRegistry {
        return .{ .allocator = allocator };
    }
    pub fn initAll(_: *ProfileRegistry) ProfileError!void {
        return error.ProfileNotInitialized;
    }
    pub fn getProfile(_: *ProfileRegistry, _: ProfileId) ?*ProfileInstance {
        return null;
    }
    pub fn getAbiRouter(_: *ProfileRegistry) ?*anyopaque {
        return null;
    }
    pub fn suspendProfile(_: *ProfileRegistry, _: ProfileId) void {}
    pub fn resumeProfile(_: *ProfileRegistry, _: ProfileId) void {}
    pub fn listAvailable(_: *ProfileRegistry) [3]?ProfileId {
        return .{ null, null, null };
    }
    pub fn deinit(_: *ProfileRegistry) void {}
};

pub const MultiProfileRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: *ProfileRegistry, _: RoutingConfig) MultiProfileRouter {
        return .{ .allocator = allocator };
    }
    pub fn route(_: *MultiProfileRouter, _: []const u8) RoutingDecision {
        return .{
            .primary = .abbey,
            .weights = .{},
            .strategy = .single,
            .confidence = 0.0,
            .reason = "AI disabled",
        };
    }
    pub fn execute(_: *MultiProfileRouter, _: RoutingDecision, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn routeAndExecute(_: *MultiProfileRouter, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn deinit(_: *MultiProfileRouter) void {}
};

pub const ProfileBus = struct {
    pub fn init(_: std.mem.Allocator) ProfileBus {
        return .{};
    }
    pub fn deinit(_: *ProfileBus) void {}
};

pub const ConversationMemory = struct {
    pub fn init(_: std.mem.Allocator) ConversationMemory {
        return .{};
    }
    pub fn asStoreStep(_: *const ConversationMemory) pipeline_types.StoreConfig {
        return .{ .target = .wdbx };
    }
    pub fn deinit(_: *ConversationMemory) void {}
};

pub const registry = struct {
    pub const ProfileRegistry_ = ProfileRegistry;
    pub const ProfileInstance_ = ProfileInstance;
    pub const MultiProfileConfig_ = MultiProfileConfig;
};

pub const router = struct {
    pub const MultiProfileRouter_ = MultiProfileRouter;
};

pub const bus = struct {
    pub const ProfileBus_ = ProfileBus;
};

pub const memory = struct {
    pub const ConversationMemory_ = ConversationMemory;
};

test {
    std.testing.refAllDecls(@This());
}
