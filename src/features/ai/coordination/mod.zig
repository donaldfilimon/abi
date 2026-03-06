//! Canonical coordination surface layered over the legacy persona router/system.

const std = @import("std");
const legacy_personas = @import("../personas/mod.zig");
const legacy_types = @import("../personas/types.zig");
const legacy_config = @import("../personas/config.zig");
const legacy_router = @import("../personas/abi/mod.zig");
const profiles = @import("../profiles/mod.zig");
const semantic_store = @import("../../database/semantic_store/mod.zig");

pub const InteractionRequest = legacy_types.PersonaRequest;
pub const InteractionResponse = legacy_types.PersonaResponse;
pub const CoordinationContext = legacy_personas.Context;
pub const InteractionCoordinator = legacy_personas.MultiPersonaSystem;
pub const CoordinationConfig = legacy_config.MultiPersonaConfig;
pub const LegacyRoutingDecision = legacy_types.RoutingDecision;
pub const PolicyFlags = legacy_types.PolicyFlags;

pub const ProfileSelection = struct {
    selected_profile: profiles.BehaviorProfile,
    legacy_persona: legacy_types.PersonaType,
    confidence: f32,
    policy_flags: PolicyFlags = .{},
    reasoning: []const u8,
    influence_trace: ?semantic_store.InfluenceTrace = null,

    pub fn deinit(self: *ProfileSelection, allocator: std.mem.Allocator) void {
        self.policy_flags.deinit(allocator);
        allocator.free(self.reasoning);
    }

    pub fn fromLegacy(
        allocator: std.mem.Allocator,
        decision: legacy_types.RoutingDecision,
        trace: ?semantic_store.InfluenceTrace,
    ) !ProfileSelection {
        return .{
            .selected_profile = profiles.fromLegacyPersona(decision.selected_persona) orelse .governance,
            .legacy_persona = decision.selected_persona,
            .confidence = decision.confidence,
            .policy_flags = try clonePolicyFlags(allocator, decision.policy_flags),
            .reasoning = try allocator.dupe(u8, decision.routing_reason),
            .influence_trace = trace,
        };
    }
};

pub const PolicyRouter = struct {
    allocator: std.mem.Allocator,
    inner: *legacy_router.AbiRouter,

    pub fn init(allocator: std.mem.Allocator, cfg: legacy_config.AbiConfig) !*PolicyRouter {
        const self = try allocator.create(PolicyRouter);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .inner = try legacy_router.AbiRouter.init(allocator, cfg),
        };

        return self;
    }

    pub fn deinit(self: *PolicyRouter) void {
        self.inner.deinit();
        self.allocator.destroy(self);
    }

    pub fn addRoutingRule(self: *PolicyRouter, rule: legacy_router.RoutingRule) !void {
        try self.inner.addRoutingRule(rule);
    }

    pub fn getRuleCount(self: *const PolicyRouter) usize {
        return self.inner.getRuleCount();
    }

    pub fn routeLegacy(self: *PolicyRouter, request: InteractionRequest) !LegacyRoutingDecision {
        return self.inner.route(request);
    }

    pub fn routeProfile(
        self: *PolicyRouter,
        request: InteractionRequest,
        trace: ?semantic_store.InfluenceTrace,
    ) !ProfileSelection {
        var decision = try self.inner.route(request);
        defer {
            decision.policy_flags.deinit(self.allocator);
            decision.deinit(self.allocator);
        }
        return ProfileSelection.fromLegacy(self.allocator, decision, trace);
    }

    pub fn validateResponse(self: *PolicyRouter, response: InteractionResponse) !PolicyFlags {
        return self.inner.validateResponse(response);
    }
};

fn clonePolicyFlags(allocator: std.mem.Allocator, flags: PolicyFlags) !PolicyFlags {
    return .{
        .is_safe = flags.is_safe,
        .requires_moderation = flags.requires_moderation,
        .sensitive_topic = flags.sensitive_topic,
        .pii_detected = flags.pii_detected,
        .violation_details = if (flags.violation_details) |details|
            try allocator.dupe(u8, details)
        else
            null,
    };
}

test "profile selection maps branded routing decision" {
    const allocator = std.testing.allocator;
    const selection = try ProfileSelection.fromLegacy(allocator, .{
        .selected_persona = .abbey,
        .confidence = 0.75,
        .emotional_context = .{},
        .policy_flags = .{},
        .routing_reason = try allocator.dupe(u8, "test"),
    }, semantic_store.InfluenceTrace.forRetrieval(7, 0.8, 0.5));
    defer {
        var owned = selection;
        owned.deinit(allocator);
    }

    try std.testing.expectEqual(profiles.BehaviorProfile.collaborative, selection.selected_profile);
    try std.testing.expectEqual(@as(?u64, 7), selection.influence_trace.?.block_id);
}
