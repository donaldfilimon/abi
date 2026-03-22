//! Coordination stub surface when AI features are disabled.

const std = @import("std");
const types = @import("../types.zig");
const config = @import("../config.zig");
const profiles = @import("../profiles/stub.zig");
const semantic_store = @import("../../database/stub.zig").semantic_store;

pub const InteractionRequest = types.ProfileRequest;
pub const InteractionResponse = types.ProfileResponse;
/// Stub coordination context.
pub const CoordinationContext = struct {
    allocator: std.mem.Allocator,
    profile: ?profiles.BehaviorProfile = null,

    pub fn init(allocator: std.mem.Allocator) CoordinationContext {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *CoordinationContext) void {}
};

/// Stub interaction coordinator.
pub const InteractionCoordinator = struct {
    allocator: std.mem.Allocator,
    context: CoordinationContext,

    pub fn init(allocator: std.mem.Allocator) InteractionCoordinator {
        return .{ .allocator = allocator, .context = CoordinationContext.init(allocator) };
    }
    pub fn deinit(self: *InteractionCoordinator) void {
        self.context.deinit();
    }
};

pub const CoordinationConfig = config.MultiProfileConfig;
pub const LegacyRoutingDecision = types.RoutingDecision;
pub const PolicyFlags = types.PolicyFlags;

pub const ProfileSelection = struct {
    selected_profile: profiles.BehaviorProfile = .governance,
    legacy_profile: profiles.LegacyProfileType = .abi,
    confidence: f32 = 0.0,
    policy_flags: PolicyFlags = .{},
    reasoning: []const u8 = "",
    influence_trace: ?semantic_store.InfluenceTrace = null,

    pub fn deinit(self: *ProfileSelection, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const PolicyRouter = struct {
    pub fn init(_: std.mem.Allocator, _: config.AbiConfig) error{FeatureDisabled}!*PolicyRouter {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *PolicyRouter) void {}
    pub fn addRoutingRule(_: *PolicyRouter, _: anytype) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn getRuleCount(_: *const PolicyRouter) usize {
        return 0;
    }
    pub fn routeLegacy(_: *PolicyRouter, _: InteractionRequest) error{FeatureDisabled}!LegacyRoutingDecision {
        return error.FeatureDisabled;
    }
    pub fn routeProfile(
        _: *PolicyRouter,
        _: InteractionRequest,
        _: ?semantic_store.InfluenceTrace,
    ) error{FeatureDisabled}!ProfileSelection {
        return error.FeatureDisabled;
    }
    pub fn validateResponse(_: *PolicyRouter, _: InteractionResponse) error{FeatureDisabled}!PolicyFlags {
        return error.FeatureDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
