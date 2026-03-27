//! MultiProfileRouter: extends AbiRouter with multi-profile orchestration.
//!
//! Uses Abi's sentiment analysis, policy checking, and rules engine to
//! produce weighted routing decisions, then dispatches to the appropriate
//! profile(s) via the ProfileRegistry.
//!
//! Pipeline per spec:
//!   User Input → Abi Analysis → Modulation → Routing Decision → Execution
//!   → Constitution Validation → WDBX Memory Storage → Response

const std = @import("std");
const types = @import("types.zig");
const registry_mod = @import("registry.zig");
const memory_mod = @import("memory.zig");
const ProfileId = types.ProfileId;
const RoutingDecision = types.RoutingDecision;
const RoutingStrategy = types.RoutingStrategy;
const RoutingConfig = types.RoutingConfig;
const ProfileResponse = types.ProfileResponse;
const ProfileError = types.ProfileError;
const ProfileRegistry = registry_mod.ProfileRegistry;

const ai_types = @import("../types.zig");
const abi_mod = @import("../abi/mod.zig");
const modulation_mod = @import("../modulation.zig");
const constitution_mod = @import("../constitution/mod.zig");
const build_options = @import("build_options");
const pipeline_mod = if (build_options.feat_reasoning) @import("../pipeline/mod.zig") else @import("../pipeline/stub.zig");

/// Multi-profile router that wraps AbiRouter for intelligent dispatch.
///
/// Implements the full Abbey-Aviva-Abi pipeline:
/// 1. Abi analyzes input (sentiment + policy + rules)
/// 2. Translates to weighted routing decision
/// 3. Executes via appropriate profile(s)
/// 4. Stores interaction in WDBX conversation memory
pub const MultiProfileRouter = struct {
    allocator: std.mem.Allocator,
    registry: *ProfileRegistry,
    config: RoutingConfig,
    memory: ?memory_mod.ConversationMemory = null,
    modulator: ?*modulation_mod.AdaptiveModulator = null,
    constitution: ?constitution_mod.Constitution = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, registry: *ProfileRegistry, config: RoutingConfig) Self {
        return .{
            .allocator = allocator,
            .registry = registry,
            .config = config,
        };
    }

    /// Attach conversation memory for WDBX block chain storage.
    pub fn attachMemory(self: *Self, session_id: []const u8) void {
        self.memory = memory_mod.ConversationMemory.init(self.allocator, session_id);
    }

    /// Attach adaptive modulator for user preference learning.
    pub fn attachModulator(self: *Self, mod: *modulation_mod.AdaptiveModulator) void {
        self.modulator = mod;
    }

    /// Attach constitution for post-generation ethical validation.
    pub fn attachConstitution(self: *Self, constitution: constitution_mod.Constitution) void {
        self.constitution = constitution;
    }

    /// Route a request: analyze content via Abi, apply modulation, and decide routing.
    pub fn route(self: *Self, input: []const u8) RoutingDecision {
        // Get initial routing decision (Abi-backed or heuristic fallback)
        var decision = blk: {
            if (self.registry.getAbiRouter()) |abi_router| {
                if (self.abiBackedRoute(abi_router, input)) |d| break :blk d;
            }
            break :blk self.heuristicRoute(input);
        };

        // Apply adaptive modulation based on user preference history
        if (self.modulator) |mod| {
            const abbey_result = mod.modulate("default", .abbey, decision.weights.abbey);
            const aviva_result = mod.modulate("default", .aviva, decision.weights.aviva);
            const abi_result = mod.modulate("default", .abi, decision.weights.abi);

            decision.weights.abbey = abbey_result.modulated_score;
            decision.weights.aviva = aviva_result.modulated_score;
            decision.weights.abi = abi_result.modulated_score;
            decision.weights.normalize();

            // Recalculate primary after modulation
            decision.primary = if (decision.weights.abbey >= decision.weights.aviva and decision.weights.abbey >= decision.weights.abi)
                ProfileId.abbey
            else if (decision.weights.aviva >= decision.weights.abi)
                ProfileId.aviva
            else
                ProfileId.abi;

            decision.confidence = decision.weights.forProfile(decision.primary);
        }

        return decision;
    }

    /// Route using Abi's sentiment analysis, policy checking, and rules engine.
    /// Translates Abi's RoutingDecision (ai_types) to profile RoutingDecision.
    fn abiBackedRoute(self: *Self, abi_router: *abi_mod.AbiRouter, input: []const u8) ?RoutingDecision {
        // Build a ProfileRequest for Abi
        const request = ai_types.ProfileRequest{
            .content = input,
            .session_id = "default",
        };

        // Call Abi's route() which runs sentiment + policy + rules
        var abi_decision = abi_router.route(request) catch |err| {
            std.log.warn("profile: abiBackedRoute failed: {s}", .{@errorName(err)});
            return null;
        };
        defer abi_decision.deinit(self.allocator);

        // Translate ai_types.RoutingDecision → profile types.RoutingDecision
        var weights = RoutingDecision.Weights{};

        // Map selected_profile to weights
        switch (abi_decision.selected_profile) {
            .abbey => {
                weights.abbey = abi_decision.confidence;
                weights.aviva = (1.0 - abi_decision.confidence) * 0.6;
                weights.abi = (1.0 - abi_decision.confidence) * 0.4;
            },
            .aviva => {
                weights.aviva = abi_decision.confidence;
                weights.abbey = (1.0 - abi_decision.confidence) * 0.6;
                weights.abi = (1.0 - abi_decision.confidence) * 0.4;
            },
            .abi => {
                weights.abi = abi_decision.confidence;
                weights.abbey = (1.0 - abi_decision.confidence) * 0.5;
                weights.aviva = (1.0 - abi_decision.confidence) * 0.5;
            },
            else => {
                // Other profile types default to Abbey
                weights.abbey = 0.5;
                weights.aviva = 0.3;
                weights.abi = 0.2;
            },
        }

        weights.normalize();

        // Determine primary profile from weights
        const primary = if (weights.abbey >= weights.aviva and weights.abbey >= weights.abi)
            ProfileId.abbey
        else if (weights.aviva >= weights.abi)
            ProfileId.aviva
        else
            ProfileId.abi;

        // Determine strategy based on confidence and policy
        const strategy: RoutingStrategy = if (!abi_decision.policy_flags.is_safe)
            .single // Policy violations always route to single profile (Abi)
        else if (abi_decision.confidence < 0.5)
            .parallel // Low confidence → try multiple profiles
        else if (abi_decision.confidence < 0.7 and primary != .abi)
            .consensus // Medium confidence → blend responses
        else
            .single; // High confidence → single profile

        return .{
            .primary = primary,
            .weights = weights,
            .strategy = strategy,
            .confidence = abi_decision.confidence,
            .reason = switch (primary) {
                .abbey => "Abi analysis: empathetic/conversational query routed to Abbey",
                .aviva => "Abi analysis: technical/factual query routed to Aviva",
                .abi => "Abi analysis: policy/compliance concern routed to Abi",
            },
        };
    }

    /// Heuristic routing based on content keywords and patterns.
    /// Used as fallback when AbiRouter is not available.
    fn heuristicRoute(self: *Self, input: []const u8) RoutingDecision {
        _ = self;
        var weights = RoutingDecision.Weights{};

        // Lowercase first 512 chars for analysis
        var buf: [512]u8 = undefined;
        const len = @min(input.len, buf.len);
        for (0..len) |i| {
            buf[i] = std.ascii.toLower(input[i]);
        }
        const lower = buf[0..len];

        // Code/technical patterns → Aviva
        const code_patterns = [_][]const u8{ "code", "function", "implement", "debug", "error", "compile", "api", "syntax" };
        for (code_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                weights.aviva += 0.3;
                break;
            }
        }

        // Emotional/conversational patterns → Abbey
        const abbey_patterns = [_][]const u8{ "feel", "think", "opinion", "help me", "explain", "understand", "why" };
        for (abbey_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                weights.abbey += 0.3;
                break;
            }
        }

        // Compliance/policy patterns → Abi
        const abi_patterns = [_][]const u8{ "policy", "privacy", "comply", "regulate", "moderate", "safe", "filter" };
        for (abi_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) {
                weights.abi += 0.3;
                break;
            }
        }

        // Default: slight Abbey preference for general queries
        if (weights.abbey == 0 and weights.aviva == 0 and weights.abi == 0) {
            weights.abbey = 0.5;
            weights.aviva = 0.3;
            weights.abi = 0.2;
        }

        weights.normalize();

        // Determine primary profile
        const primary = if (weights.abbey >= weights.aviva and weights.abbey >= weights.abi)
            ProfileId.abbey
        else if (weights.aviva >= weights.abi)
            ProfileId.aviva
        else
            ProfileId.abi;

        const confidence = weights.forProfile(primary);

        return .{
            .primary = primary,
            .weights = weights,
            .strategy = if (confidence < 0.5) .parallel else .single,
            .confidence = confidence,
            .reason = switch (primary) {
                .abbey => "Conversational or exploratory query routed to Abbey",
                .aviva => "Technical or factual query routed to Aviva",
                .abi => "Compliance or policy query routed to Abi",
            },
        };
    }

    /// Execute a routed request through the chosen profile(s).
    pub fn execute(self: *Self, decision: RoutingDecision, input: []const u8) ProfileError!ProfileResponse {
        return switch (decision.strategy) {
            .single => self.executeSingle(decision.primary, input),
            .parallel => self.executeParallel(decision, input),
            .consensus => self.executeConsensus(decision, input),
        };
    }

    /// Route, execute, validate, and store — the full pipeline.
    /// Pipeline: Abi Analysis → Modulation → Execution → Constitution Check → WDBX Store
    pub fn routeAndExecute(self: *Self, input: []const u8) ProfileError!ProfileResponse {
        const decision = self.route(input);
        const response = try self.execute(decision, input);

        // Post-generation: validate response against Constitution
        if (self.constitution) |c| {
            if (!c.isCompliant(response.content)) {
                // Response violates ethical principles — return safe fallback
                const safe_msg = "I cannot provide this response as it may violate safety guidelines. Please rephrase your request.";
                const safe_response = ProfileResponse{
                    .profile = .abi,
                    .content = try self.allocator.dupe(u8, safe_msg),
                    .confidence = 1.0,
                    .allocator = self.allocator,
                };

                // Store the blocked interaction in memory
                if (self.memory) |*mem| {
                    mem.recordInteraction(decision, input, safe_response, null) catch |err| {
                        std.log.warn("profile: failed to record blocked interaction: {s}", .{@errorName(err)});
                    };
                }

                return safe_response;
            }
        }

        // Store interaction in WDBX memory (best-effort, don't fail the response)
        if (self.memory) |*mem| {
            mem.recordInteraction(decision, input, response, null) catch |err| {
                std.log.warn("profile: failed to record memory interaction: {s}", .{@errorName(err)});
            };
        }

        // Record interaction for modulator preference learning
        if (self.modulator) |mod| {
            const profile_profile: ai_types.ProfileType = switch (response.profile) {
                .abbey => .abbey,
                .aviva => .aviva,
                .abi => .abi,
            };
            mod.recordInteraction("default", profile_profile, true) catch |err| {
                std.log.warn("profile: failed to record modulator interaction: {s}", .{@errorName(err)});
            };
        }

        return response;
    }

    /// Execute the full pipeline using the Abbey Dynamic Model DSL.
    /// Equivalent to routeAndExecute but expressed as a composable pipeline
    /// with every step recorded as a WDBX block.
    pub fn routeAndExecutePipeline(self: *Self, input: []const u8) !pipeline_mod.PipelineResult {
        const session_id = if (self.memory) |m| m.chain.session_id else "default";
        var builder = pipeline_mod.chain(self.allocator, session_id);
        defer builder.deinit();

        // Wire up the WDBX chain from memory if available
        if (self.memory) |*mem| {
            _ = builder.withChain(&mem.chain);
        }

        var p = builder
            .retrieve(.wdbx, .{ .k = 5 })
            .template("Given {context}, respond to: {input}")
            .route(.adaptive)
            .modulate()
            .generate(.{})
            .validate(.constitution)
            .store(.wdbx)
            .build();
        defer p.deinit();

        return p.run(input);
    }

    fn executeSingle(self: *Self, profile_id: ProfileId, input: []const u8) ProfileError!ProfileResponse {
        const profile = self.registry.getProfile(profile_id);
        return profile.process(input);
    }

    fn executeParallel(self: *Self, decision: RoutingDecision, input: []const u8) ProfileError!ProfileResponse {
        // Try primary first, fall back to next highest weight
        const primary_result = self.executeSingle(decision.primary, input);
        if (primary_result) |_| return primary_result else |_| {
            // Determine fallback profile
            const fallback: ProfileId = if (decision.primary != .abbey and decision.weights.abbey > 0)
                .abbey
            else if (decision.primary != .aviva and decision.weights.aviva > 0)
                .aviva
            else
                .abi;
            return self.executeSingle(fallback, input);
        }
    }

    /// Execute consensus routing: run two profiles and blend results.
    /// Per spec Section 3.1.1 — Dynamic Profile Blending:
    ///   α > 0.8 → primary only
    ///   α < 0.2 → secondary only
    ///   otherwise → blend both responses
    fn executeConsensus(self: *Self, decision: RoutingDecision, input: []const u8) ProfileError!ProfileResponse {
        const alpha = decision.weights.forProfile(decision.primary);

        // High confidence in primary → single dispatch
        if (alpha > 0.8) {
            return self.executeSingle(decision.primary, input);
        }

        // Determine secondary profile (highest weight after primary)
        const secondary: ProfileId = blk: {
            if (decision.primary == .abbey) {
                break :blk if (decision.weights.aviva >= decision.weights.abi) ProfileId.aviva else ProfileId.abi;
            } else if (decision.primary == .aviva) {
                break :blk if (decision.weights.abbey >= decision.weights.abi) ProfileId.abbey else ProfileId.abi;
            } else {
                break :blk if (decision.weights.abbey >= decision.weights.aviva) ProfileId.abbey else ProfileId.aviva;
            }
        };

        // Low primary weight → secondary only
        if (alpha < 0.2) {
            return self.executeSingle(secondary, input);
        }

        // Blend: execute primary, then if secondary succeeds, annotate blend
        const primary_response = self.executeSingle(decision.primary, input) catch {
            return self.executeSingle(secondary, input);
        };

        // Try secondary — if it fails, return primary alone
        var secondary_response = self.executeSingle(secondary, input) catch {
            return primary_response;
        };

        const secondary_weight = decision.weights.forProfile(secondary);
        const secondary_confidence = secondary_response.confidence;

        // Build blended response: primary content + secondary perspective
        const blended = std.fmt.allocPrint(
            self.allocator,
            "{s}\n\n[{s} perspective (weight {d:.0}%)]: {s}",
            .{
                primary_response.content,
                secondary.name(),
                secondary_weight * 100.0,
                secondary_response.content,
            },
        ) catch {
            secondary_response.deinit();
            return primary_response;
        };

        secondary_response.deinit();

        return ProfileResponse{
            .profile = decision.primary,
            .content = blended,
            .confidence = (alpha * primary_response.confidence + secondary_weight * secondary_confidence),
            .allocator = self.allocator,
        };
    }

    /// Get conversation memory (for external access to history).
    pub fn getMemory(self: *const Self) ?*const memory_mod.ConversationMemory {
        return if (self.memory != null) &self.memory.? else null;
    }

    pub fn deinit(self: *Self) void {
        if (self.memory) |*mem| {
            mem.deinit();
        }
    }
};

test "heuristic routing - code query" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("How do I implement a function in Zig?");

    try std.testing.expectEqual(ProfileId.aviva, decision.primary);
    try std.testing.expect(decision.weights.aviva > decision.weights.abbey);
}

test "heuristic routing - emotional query" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("I feel stuck, can you help me understand this?");

    try std.testing.expectEqual(ProfileId.abbey, decision.primary);
}

test "heuristic routing - compliance query" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("What is our privacy policy for user data?");

    try std.testing.expectEqual(ProfileId.abi, decision.primary);
}

test "heuristic routing - default" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("Hello there");

    // Default routing favors Abbey
    try std.testing.expectEqual(ProfileId.abbey, decision.primary);
    try std.testing.expect(decision.confidence > 0.0);
}

test "memory attachment" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    router.attachMemory("test-session");
    try std.testing.expect(router.memory != null);
}

test "constitution attachment" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    router.attachConstitution(constitution_mod.Constitution.init());
    try std.testing.expect(router.constitution != null);
}

test "constitution validates safe content" {
    const c = constitution_mod.Constitution.init();
    try std.testing.expect(c.isCompliant("Hello, how can I help you today?"));
}

test "constitution blocks harmful content" {
    const c = constitution_mod.Constitution.init();
    try std.testing.expect(!c.isCompliant("run rm -rf / to clean up"));
}

test "abiBackedRoute failure falls back to heuristic" {
    var registry = ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiProfileRouter.init(std.testing.allocator, &registry, .{});

    // No AbiRouter attached — route() should fall through to heuristicRoute
    const decision = router.route("How do I implement error handling?");

    // Should still produce a valid decision via heuristic fallback
    try std.testing.expect(decision.confidence > 0.0);
    try std.testing.expect(decision.reason.len > 0);
    // "implement" keyword should trigger Aviva routing
    try std.testing.expectEqual(ProfileId.aviva, decision.primary);
}

test {
    std.testing.refAllDecls(@This());
}
