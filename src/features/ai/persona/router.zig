//! MultiPersonaRouter: extends AbiRouter with multi-persona orchestration.
//!
//! Uses Abi's sentiment analysis, policy checking, and rules engine to
//! produce weighted routing decisions, then dispatches to the appropriate
//! persona(s) via the PersonaRegistry.

const std = @import("std");
const types = @import("types.zig");
const registry_mod = @import("registry.zig");
const PersonaId = types.PersonaId;
const RoutingDecision = types.RoutingDecision;
const RoutingStrategy = types.RoutingStrategy;
const RoutingConfig = types.RoutingConfig;
const PersonaResponse = types.PersonaResponse;
const PersonaError = types.PersonaError;
const PersonaRegistry = registry_mod.PersonaRegistry;

const ai_types = @import("../types.zig");
const abi_mod = @import("../abi/mod.zig");

/// Multi-persona router that wraps AbiRouter for intelligent dispatch.
pub const MultiPersonaRouter = struct {
    allocator: std.mem.Allocator,
    registry: *PersonaRegistry,
    config: RoutingConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, registry: *PersonaRegistry, config: RoutingConfig) Self {
        return .{
            .allocator = allocator,
            .registry = registry,
            .config = config,
        };
    }

    /// Route a request: analyze content and decide which persona(s) should handle it.
    pub fn route(self: *Self, input: []const u8) RoutingDecision {
        // Use Abi's analysis capabilities if available
        if (self.registry.getAbiRouter()) |abi_router| {
            _ = abi_router;
            // In a full implementation, we'd call abi_router.route() here
            // and translate its RoutingDecision to our format.
            // For now, use heuristic routing based on content analysis.
        }

        return self.heuristicRoute(input);
    }

    /// Heuristic routing based on content keywords and patterns.
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

    /// Route and execute in one call.
    pub fn routeAndExecute(self: *Self, input: []const u8) PersonaError!PersonaResponse {
        const decision = self.route(input);
        return self.execute(decision, input);
    }

    fn executeSingle(self: *Self, persona_id: PersonaId, input: []const u8) PersonaError!PersonaResponse {
        const persona = self.registry.getPersona(persona_id);
        return persona.process(input);
    }

    fn executeParallel(self: *Self, decision: RoutingDecision, input: []const u8) PersonaError!PersonaResponse {
        // Simplified: try primary, fall back to next highest weight
        const primary_result = self.executeSingle(decision.primary, input);
        if (primary_result) |_| return primary_result else |_| {
            // Try fallback
            const fallback: PersonaId = if (decision.primary != .abbey and decision.weights.abbey > 0) .abbey else if (decision.primary != .aviva and decision.weights.aviva > 0) .aviva else .abi;
            return self.executeSingle(fallback, input);
        }
    }

    fn executeConsensus(self: *Self, decision: RoutingDecision, input: []const u8) PersonaError!PersonaResponse {
        // Simplified: route to primary (full consensus requires async)
        return self.executeSingle(decision.primary, input);
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        // Router doesn't own any resources — registry owns the personas
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

test {
    std.testing.refAllDecls(@This());
}
