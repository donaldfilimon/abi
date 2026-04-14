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

pub const MultiProfileConfig_Internal = struct {
    routing: RoutingConfig = .{},
};

pub const ProfileInstance_Internal = struct {
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

pub const ProfileRegistry_Internal = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, _: MultiProfileConfig_Internal) ProfileRegistry_Internal {
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

pub const MultiProfileRouter_Internal = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: *ProfileRegistry_Internal, _: RoutingConfig) MultiProfileRouter_Internal {
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

pub const ProfileBus_Internal = struct {
    pub fn init(_: std.mem.Allocator) ProfileBus_Internal {
        return .{};
    }
    pub fn deinit(_: *ProfileBus_Internal) void {}
};

pub const ConversationMemory_Internal = struct {
    pub fn init(_: std.mem.Allocator) ConversationMemory_Internal {
        return .{};
    }
    pub fn asStoreStep(_: *const ConversationMemory_Internal) pipeline_types.StoreConfig {
        return .{ .target = .wdbx };
    }
    pub fn deinit(_: *ConversationMemory_Internal) void {}
};

pub const MultiProfileConfig = MultiProfileConfig_Internal;
pub const ProfileInstance = ProfileInstance_Internal;
pub const ProfileRegistry = ProfileRegistry_Internal;
pub const MultiProfileRouter = MultiProfileRouter_Internal;
pub const ProfileBus = ProfileBus_Internal;
pub const ConversationMemory = ConversationMemory_Internal;

pub const registry = struct {
    pub const ProfileRegistry = ProfileRegistry_Internal;
    pub const ProfileInstance = ProfileInstance_Internal;
    pub const MultiProfileConfig = MultiProfileConfig_Internal;
};

pub const router = struct {
    pub const MultiProfileRouter = MultiProfileRouter_Internal;
};

pub const bus = struct {
    pub const ProfileBus = ProfileBus_Internal;
};

pub const memory = struct {
    pub const ConversationMemory = ConversationMemory_Internal;
};

test {
    std.testing.refAllDecls(@This());
}
