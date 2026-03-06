//! Coordination stub — disabled at compile time.

const std = @import("std");
const profiles = @import("../profiles/stub.zig");

pub const InteractionRequest = struct {};
pub const InteractionResponse = struct {};
pub const CoordinationContext = struct {};
pub const InteractionCoordinator = struct {};
pub const CoordinationConfig = struct {};
pub const PolicyFlags = struct {
    pub fn deinit(_: *PolicyFlags, _: std.mem.Allocator) void {}
};
pub const LegacyRoutingDecision = struct {};

pub const ProfileSelection = struct {
    selected_profile: profiles.BehaviorProfile = .governance,
    legacy_persona: profiles.LegacyPersonaType = .abi,
    confidence: f32 = 0.0,
    policy_flags: PolicyFlags = .{},
    reasoning: []const u8 = "",
    influence_trace: ?@import("../../database/stub.zig").InfluenceTrace = null,

    pub fn deinit(_: *ProfileSelection, _: std.mem.Allocator) void {}
};

pub const PolicyRouter = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) error{FeatureDisabled}!*PolicyRouter {
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
    pub fn routeProfile(_: *PolicyRouter, _: InteractionRequest, _: ?@import("../../database/stub.zig").InfluenceTrace) error{FeatureDisabled}!ProfileSelection {
        return error.FeatureDisabled;
    }
    pub fn validateResponse(_: *PolicyRouter, _: InteractionResponse) error{FeatureDisabled}!PolicyFlags {
        return error.FeatureDisabled;
    }
};
