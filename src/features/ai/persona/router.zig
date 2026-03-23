//! MultiPersonaRouter: extends AbiRouter with multi-persona orchestration.
//!
//! Uses Abi's sentiment analysis, policy checking, and rules engine to
//! produce weighted routing decisions, then dispatches to the appropriate
//! persona(s) via the PersonaRegistry.
//!
//! Pipeline per spec:
//!   User Input → Abi Analysis → Modulation → Routing Decision → Execution
//!   → Constitution Validation → WDBX Memory Storage → Response

const std = @import("std");
const types = @import("types.zig");
const registry_mod = @import("registry.zig");
const memory_mod = @import("memory.zig");
const PersonaId = types.PersonaId;
const RoutingDecision = types.RoutingDecision;
const RoutingStrategy = types.RoutingStrategy;
const RoutingConfig = types.RoutingConfig;
const PersonaResponse = types.PersonaResponse;
const PersonaError = types.PersonaError;
const PersonaRegistry = registry_mod.PersonaRegistry;

const ai_types = @import("../types.zig");
const abi_mod = @import("../abi/mod.zig");
const modulation_mod = @import("../modulation.zig");
const constitution_mod = @import("../constitution/mod.zig");

/// Multi-persona router that wraps AbiRouter for intelligent dispatch.
///
/// Implements the full Abbey-Aviva-Abi pipeline:
/// 1. Abi analyzes input (sentiment + policy + rules)
/// 2. Translates to weighted routing decision
/// 3. Executes via appropriate persona(s)
/// 4. Stores interaction in WDBX conversation memory
pub const MultiPersonaRouter = struct {
    allocator: std.mem.Allocator,
    registry: *PersonaRegistry,
    config: RoutingConfig,
    memory: ?memory_mod.ConversationMemory = null,
    modulator: ?*modulation_mod.AdaptiveModulator = null,
    constitution: ?constitution_mod.Constitution = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, registry: *PersonaRegistry, config: RoutingConfig) Self {
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
                PersonaId.abbey
            else if (decision.weights.aviva >= decision.weights.abi)
                PersonaId.aviva
            else
                PersonaId.abi;

            decision.confidence = decision.weights.forPersona(decision.primary);
        }

        return decision;
    }

    /// Route using Abi's sentiment analysis, policy checking, and rules engine.
    /// Translates Abi's RoutingDecision (ai_types) to persona RoutingDecision.
    fn abiBackedRoute(self: *Self, abi_router: *abi_mod.AbiRouter, input: []const u8) ?RoutingDecision {
        // Build a ProfileRequest for Abi
        const request = ai_types.ProfileRequest{
            .content = input,
            .session_id = "default",
        };

        // Call Abi's route() which runs sentiment + policy + rules
        var abi_decision = abi_router.route(request) catch return null;
        defer @constCast(&abi_decision).deinit(self.allocator);

        // Translate ai_types.RoutingDecision → persona types.RoutingDecision
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

        // Determine primary persona from weights
        const primary = if (weights.abbey >= weights.aviva and weights.abbey >= weights.abi)
            PersonaId.abbey
        else if (weights.aviva >= weights.abi)
            PersonaId.aviva
        else
            PersonaId.abi;

        // Determine strategy based on confidence and policy
        const strategy: RoutingStrategy = if (!abi_decision.policy_flags.is_safe)
            .single // Policy violations always route to single persona (Abi)
        else if (abi_decision.confidence < 0.5)
            .parallel // Low confidence → try multiple personas
        else if (abi_decision.confidence < 0.7 and primary != .abi)
            .consensus // Medium confidence → blend responses
        else
            .single; // High confidence → single persona

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

        // Determine primary persona
        const primary = if (weights.abbey >= weights.aviva and weights.abbey >= weights.abi)
            PersonaId.abbey
        else if (weights.aviva >= weights.abi)
            PersonaId.aviva
        else
            PersonaId.abi;

        const confidence = weights.forPersona(primary);

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

    /// Execute a routed request through the chosen persona(s).
    pub fn execute(self: *Self, decision: RoutingDecision, input: []const u8) PersonaError!PersonaResponse {
        return switch (decision.strategy) {
            .single => self.executeSingle(decision.primary, input),
            .parallel => self.executeParallel(decision, input),
            .consensus => self.executeConsensus(decision, input),
        };
    }

    /// Route, execute, validate, and store — the full pipeline.
    /// Pipeline: Abi Analysis → Modulation → Execution → Constitution Check → WDBX Store
    pub fn routeAndExecute(self: *Self, input: []const u8) PersonaError!PersonaResponse {
        const decision = self.route(input);
        const response = try self.execute(decision, input);

        // Post-generation: validate response against Constitution
        if (self.constitution) |c| {
            if (!c.isCompliant(response.content)) {
                // Response violates ethical principles — return safe fallback
                const safe_response = PersonaResponse{
                    .persona = .abi,
                    .content = "I cannot provide this response as it may violate safety guidelines. Please rephrase your request.",
                    .confidence = 1.0,
                    .allocator = self.allocator,
                };

                // Store the blocked interaction in memory
                if (self.memory) |*mem| {
                    mem.recordInteraction(decision, input, safe_response) catch |err| {
                        std.log.warn("persona: failed to record blocked interaction: {s}", .{@errorName(err)});
                    };
                }

                return safe_response;
            }
        }

        // Store interaction in WDBX memory (best-effort, don't fail the response)
        if (self.memory) |*mem| {
            mem.recordInteraction(decision, input, response) catch |err| {
                std.log.warn("persona: failed to record memory interaction: {s}", .{@errorName(err)});
            };
        }

        // Record interaction for modulator preference learning
        if (self.modulator) |mod| {
            const persona_profile: ai_types.ProfileType = switch (response.persona) {
                .abbey => .abbey,
                .aviva => .aviva,
                .abi => .abi,
            };
            mod.recordInteraction("default", persona_profile, true) catch |err| {
                std.log.warn("persona: failed to record modulator interaction: {s}", .{@errorName(err)});
            };
        }

        return response;
    }

    fn executeSingle(self: *Self, persona_id: PersonaId, input: []const u8) PersonaError!PersonaResponse {
        const persona = self.registry.getPersona(persona_id);
        return persona.process(input);
    }

    fn executeParallel(self: *Self, decision: RoutingDecision, input: []const u8) PersonaError!PersonaResponse {
        // Try primary first, fall back to next highest weight
        const primary_result = self.executeSingle(decision.primary, input);
        if (primary_result) |_| return primary_result else |_| {
            // Determine fallback persona
            const fallback: PersonaId = if (decision.primary != .abbey and decision.weights.abbey > 0)
                .abbey
            else if (decision.primary != .aviva and decision.weights.aviva > 0)
                .aviva
            else
                .abi;
            return self.executeSingle(fallback, input);
        }
    }

    /// Execute consensus routing: run two personas and blend results.
    /// Per spec Section 3.1.1 — Dynamic Persona Blending:
    ///   α > 0.8 → primary only
    ///   α < 0.2 → secondary only
    ///   otherwise → blend both responses
    fn executeConsensus(self: *Self, decision: RoutingDecision, input: []const u8) PersonaError!PersonaResponse {
        const alpha = decision.weights.forPersona(decision.primary);

        // High confidence in primary → single dispatch
        if (alpha > 0.8) {
            return self.executeSingle(decision.primary, input);
        }

        // Determine secondary persona (highest weight after primary)
        const secondary: PersonaId = blk: {
            if (decision.primary == .abbey) {
                break :blk if (decision.weights.aviva >= decision.weights.abi) PersonaId.aviva else PersonaId.abi;
            } else if (decision.primary == .aviva) {
                break :blk if (decision.weights.abbey >= decision.weights.abi) PersonaId.abbey else PersonaId.abi;
            } else {
                break :blk if (decision.weights.abbey >= decision.weights.aviva) PersonaId.abbey else PersonaId.aviva;
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
        const secondary_response = self.executeSingle(secondary, input) catch {
            return primary_response;
        };

        // Build blended response: primary content + secondary perspective
        const blended = std.fmt.allocPrint(
            self.allocator,
            "{s}\n\n[{s} perspective (weight {d:.0}%)]: {s}",
            .{
                primary_response.content,
                secondary.name(),
                (1.0 - alpha) * 100.0,
                secondary_response.content,
            },
        ) catch return primary_response;

        return PersonaResponse{
            .persona = decision.primary,
            .content = blended,
            .confidence = (alpha * primary_response.confidence + (1.0 - alpha) * secondary_response.confidence),
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
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("How do I implement a function in Zig?");

    try std.testing.expectEqual(PersonaId.aviva, decision.primary);
    try std.testing.expect(decision.weights.aviva > decision.weights.abbey);
}

test "heuristic routing - emotional query" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("I feel stuck, can you help me understand this?");

    try std.testing.expectEqual(PersonaId.abbey, decision.primary);
}

test "heuristic routing - compliance query" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("What is our privacy policy for user data?");

    try std.testing.expectEqual(PersonaId.abi, decision.primary);
}

test "heuristic routing - default" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    const decision = router.route("Hello there");

    // Default routing favors Abbey
    try std.testing.expectEqual(PersonaId.abbey, decision.primary);
    try std.testing.expect(decision.confidence > 0.0);
}

test "memory attachment" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    router.attachMemory("test-session");
    try std.testing.expect(router.memory != null);
}

test "constitution attachment" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
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

test {
    std.testing.refAllDecls(@This());
}
