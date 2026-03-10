//! Coordination stub surface when AI features are disabled.

const std = @import("std");
const legacy_personas = @import("../profiles/stub");
const legacy_types = @import("../types");
const legacy_config = @import("../config");
const profiles = @import("../profiles/stub");
const semantic_store = @import("../../../core/database/semantic_store/stub");

pub const InteractionRequest = legacy_types.PersonaRequest;
pub const InteractionResponse = legacy_types.PersonaResponse;
pub const CoordinationContext = legacy_personas.Context;
pub const InteractionCoordinator = legacy_personas.MultiPersonaSystem;
pub const CoordinationConfig = legacy_config.MultiPersonaConfig;
pub const LegacyRoutingDecision = legacy_types.RoutingDecision;
pub const PolicyFlags = legacy_types.PolicyFlags;

pub const ProfileSelection = struct {
    selected_profile: profiles.BehaviorProfile = .governance,
    legacy_persona: profiles.LegacyPersonaType = .abi,
    confidence: f32 = 0.0,
    policy_flags: PolicyFlags = .{},
    reasoning: []const u8 = "",
    influence_trace: ?semantic_store.InfluenceTrace = null,

    pub fn deinit(self: *ProfileSelection, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

pub const PolicyRouter = struct {
    pub fn init(_: std.mem.Allocator, _: legacy_config.AbiConfig) error{FeatureDisabled}!*PolicyRouter {
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
