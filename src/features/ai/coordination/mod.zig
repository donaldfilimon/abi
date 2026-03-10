//! Canonical coordination surface over the legacy multi-persona system.

const std = @import("std");
const legacy_personas = @import("../profiles");
const legacy_types = @import("../types");
const legacy_config = @import("../config");
const legacy_abi = @import("../abi_logic");
const profiles = @import("../profiles");
const semantic_store = @import("wdbx").semantic_store;

pub const InteractionRequest = legacy_types.PersonaRequest;
pub const InteractionResponse = legacy_types.PersonaResponse;
/// Coordination context wrapping profile selection state.
pub const CoordinationContext = struct {
    allocator: std.mem.Allocator,
    profile: ?profiles.BehaviorProfile = null,

    pub fn init(allocator: std.mem.Allocator) CoordinationContext {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *CoordinationContext) void {}
};

/// Interaction coordinator that routes requests to behavior profiles.
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

pub const CoordinationConfig = legacy_config.MultiPersonaConfig;
pub const LegacyRoutingDecision = legacy_types.RoutingDecision;
pub const PolicyFlags = legacy_types.PolicyFlags;

pub const ProfileSelection = struct {
    selected_profile: profiles.BehaviorProfile,
    legacy_persona: profiles.LegacyPersonaType,
    confidence: f32,
    policy_flags: PolicyFlags,
    reasoning: []const u8,
    influence_trace: ?semantic_store.InfluenceTrace = null,

    pub fn deinit(self: *ProfileSelection, allocator: std.mem.Allocator) void {
        allocator.free(self.reasoning);
        self.policy_flags.deinit(allocator);
        self.* = undefined;
    }

    pub fn fromLegacy(
        allocator: std.mem.Allocator,
        decision: LegacyRoutingDecision,
        trace: ?semantic_store.InfluenceTrace,
    ) !ProfileSelection {
        return .{
            .selected_profile = profiles.fromLegacyPersona(decision.selected_persona),
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
    inner: *legacy_abi.AbiRouter,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, cfg: legacy_config.AbiConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .inner = try legacy_abi.AbiRouter.init(allocator, cfg),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.inner.deinit();
        self.allocator.destroy(self);
    }

    pub fn addRoutingRule(self: *Self, rule: legacy_abi.RoutingRule) !void {
        try self.inner.addRoutingRule(rule);
    }

    pub fn getRuleCount(self: *const Self) usize {
        return self.inner.getRuleCount();
    }

    pub fn routeLegacy(self: *Self, request: InteractionRequest) !LegacyRoutingDecision {
        return self.inner.route(request);
    }

    pub fn routeProfile(
        self: *Self,
        request: InteractionRequest,
        trace: ?semantic_store.InfluenceTrace,
    ) !ProfileSelection {
        var legacy = try self.inner.route(request);
        defer legacy.deinit(self.allocator);
        return ProfileSelection.fromLegacy(self.allocator, legacy, trace);
    }

    pub fn validateResponse(self: *Self, response: InteractionResponse) !PolicyFlags {
        return self.inner.validateResponse(response);
    }
};

fn clonePolicyFlags(
    allocator: std.mem.Allocator,
    flags: PolicyFlags,
) !PolicyFlags {
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

test "profile selection maps legacy routing decisions" {
    const allocator = std.testing.allocator;
    const decision = LegacyRoutingDecision{
        .selected_persona = .abbey,
        .confidence = 0.9,
        .emotional_context = .{},
        .policy_flags = .{},
        .routing_reason = try allocator.dupe(u8, "Test routing"),
    };
    defer allocator.free(decision.routing_reason);

    var selection = try ProfileSelection.fromLegacy(
        allocator,
        decision,
        semantic_store.InfluenceTrace.forRetrieval(7, 0.7, 0.6),
    );
    defer selection.deinit(allocator);

    try std.testing.expectEqual(profiles.BehaviorProfile.collaborative, selection.selected_profile);
    try std.testing.expectEqual(@as(?u64, 7), selection.influence_trace.?.block_id);
}
