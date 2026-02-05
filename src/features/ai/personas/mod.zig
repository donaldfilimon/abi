//! Multi-Persona AI Assistant Module
//!
//! This module implements a multi-layer, multi-persona AI system that routes
//! user queries through specialized interaction models.
//!
//! Architecture:
//! - Abi: Content moderation, sentiment analysis, and routing layer.
//! - Abbey: Empathetic polymath for supportive, deep technical assistance.
//! - Aviva: Direct expert for concise, factual, and technically forceful output.
//! - WDBX: Distributed neural database for long-term memory and context continuity.

const std = @import("std");
const build_options = @import("build_options");
const obs = @import("../../observability/mod.zig");

pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");
const health = @import("health.zig");

// Persona sub-modules
pub const abi = @import("abi/mod.zig");
pub const abbey = @import("abbey/mod.zig");
pub const aviva = @import("aviva/mod.zig");
pub const embeddings = @import("embeddings/mod.zig");
pub const enhanced = @import("routing/enhanced.zig");
pub const metrics = @import("metrics.zig");
pub const loadbalancer = @import("loadbalancer.zig");
pub const generic = @import("generic.zig");

// Re-export core types for stable public API
pub const PersonaType = types.PersonaType;
pub const PersonaRequest = types.PersonaRequest;
pub const PersonaResponse = types.PersonaResponse;
pub const RoutingDecision = types.RoutingDecision;
pub const PersonaInterface = types.PersonaInterface;
pub const PolicyFlags = types.PolicyFlags;
pub const ReasoningStep = types.ReasoningStep;
pub const CodeBlock = types.CodeBlock;
pub const Source = types.Source;
pub const RoutingScore = loadbalancer.PersonaScore;

// Re-export configuration types
pub const MultiPersonaConfig = config.MultiPersonaConfig;
pub const AbiConfig = config.AbiConfig;
pub const AbbeyConfig = config.AbbeyConfig;
pub const AvivaConfig = config.AvivaConfig;
pub const LoadBalancingConfig = config.LoadBalancingConfig;

// Re-export registry
pub const PersonaRegistry = registry.PersonaRegistry;

/// Personas context for framework integration.
/// Manages the registration, selection, and coordination of AI personas.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MultiPersonaConfig,
    registry: PersonaRegistry,
    embeddings_module: ?*embeddings.EmbeddingsModule = null,
    metrics_manager: ?*metrics.PersonaMetrics = null,
    load_balancer: ?*loadbalancer.PersonaLoadBalancer = null,

    const Self = @This();

    /// Initialize the personas context with configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: MultiPersonaConfig) !*Context {
        const self = try allocator.create(Context);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .registry = PersonaRegistry.init(allocator),
        };

        return self;
    }

    /// Shutdown the personas context and free resources.
    pub fn deinit(self: *Self) void {
        if (self.load_balancer) |lb| {
            lb.deinit();
            self.allocator.destroy(lb);
        }
        if (self.metrics_manager) |m| {
            m.deinit();
        }
        if (self.embeddings_module) |m| {
            m.deinit();
        }
        self.registry.deinit();
        self.allocator.destroy(self);
    }

    /// Initialize the embeddings sub-feature for the multi-persona system.
    pub fn initEmbeddings(self: *Self, db_ctx: anytype, emb_ctx: anytype) !void {
        if (self.embeddings_module != null) return;
        self.embeddings_module = try embeddings.EmbeddingsModule.init(self.allocator, db_ctx, emb_ctx);
    }

    /// Initialize the metrics manager for the persona system.
    pub fn initMetrics(self: *Self, collector: anytype) !void {
        if (self.metrics_manager != null) return;
        self.metrics_manager = try metrics.PersonaMetrics.init(self.allocator, collector);

        // Register existing personas for metrics tracking.
        const registered = try self.registry.listRegisteredTypes(self.allocator);
        defer self.allocator.free(registered);
        for (registered) |persona_type| {
            try self.metrics_manager.?.registerPersona(persona_type);
        }
    }

    /// Initialize the load balancer for the persona system.
    pub fn initLoadBalancer(self: *Self, cfg: config.LoadBalancingConfig) !void {
        if (self.load_balancer != null) return;
        const lb = try self.allocator.create(loadbalancer.PersonaLoadBalancer);
        errdefer self.allocator.destroy(lb);
        lb.* = loadbalancer.PersonaLoadBalancer.init(self.allocator, cfg);
        self.load_balancer = lb;

        // Register existing personas for load balancing.
        const registered = try self.registry.listRegisteredTypes(self.allocator);
        defer self.allocator.free(registered);
        for (registered) |persona_type| {
            try lb.registerPersona(persona_type, 1.0);
        }
    }

    /// Register a persona implementation.
    pub fn registerPersona(self: *Self, persona_type: PersonaType, persona: PersonaInterface) !void {
        try self.registry.registerPersona(persona_type, persona);

        if (self.metrics_manager) |m| {
            try m.registerPersona(persona_type);
        }
        if (self.load_balancer) |lb| {
            try lb.registerPersona(persona_type, 1.0);
        }
    }

    /// Get a registered persona by type.
    pub fn getPersona(self: *Self, persona_type: PersonaType) ?PersonaInterface {
        return self.registry.getPersona(persona_type);
    }

    /// Configure a specific persona's behavior.
    pub fn configurePersona(self: *Self, persona_type: PersonaType, cfg: MultiPersonaConfig) !void {
        try self.registry.configurePersona(persona_type, cfg);
    }

    /// List all registered persona types.
    pub fn listRegisteredTypes(self: *Self, allocator: std.mem.Allocator) ![]PersonaType {
        return self.registry.listRegisteredTypes(allocator);
    }

    /// Retrieve any registered persona type (for fallback selection).
    pub fn getAnyPersonaType(self: *Self) ?PersonaType {
        return self.registry.getAnyPersonaType();
    }
};

/// High-level orchestrator for the multi-persona assistant.
/// Handles the end-to-end request lifecycle: routing, persona selection, and metrics.
pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    router: *abi.AbiRouter,
    owned_metrics_collector: ?*obs.MetricsCollector = null,
    health_checker: ?*health.HealthChecker = null,
    owned_abbey: ?*abbey.AbbeyPersona = null,
    owned_aviva: ?*aviva.AvivaPersona = null,
    owned_abi: ?abi.AbiPersona = null,
    owned_generic: std.AutoHashMapUnmanaged(PersonaType, *generic.GenericPersona) = .{},

    const Self = @This();

    /// Initialize the multi-persona system.
    pub fn init(allocator: std.mem.Allocator, cfg: MultiPersonaConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const ctx = try Context.init(allocator, cfg);
        errdefer ctx.deinit();

        const router = try abi.AbiRouter.init(allocator, cfg.abi);
        errdefer router.deinit();

        self.* = .{
            .allocator = allocator,
            .ctx = ctx,
            .router = router,
        };

        return self;
    }

    /// Initialize the system with default personas, metrics, and load balancing.
    pub fn initWithDefaults(allocator: std.mem.Allocator, cfg: MultiPersonaConfig) !*Self {
        const self = try init(allocator, cfg);
        errdefer self.deinit();

        // Enable load balancing for resilience.
        try self.enableLoadBalancer(cfg.load_balancing);

        // Enable metrics with an owned collector if none provided.
        try self.enableMetrics(null);

        // Register default personas.
        try self.registerDefaultPersonas();

        return self;
    }

    /// Shutdown the system and free resources.
    pub fn deinit(self: *Self) void {
        if (self.health_checker) |checker| {
            checker.deinit();
            self.allocator.destroy(checker);
        }
        var generic_it = self.owned_generic.valueIterator();
        while (generic_it.next()) |persona| {
            persona.*.deinit();
            self.allocator.destroy(persona.*);
        }
        self.owned_generic.deinit(self.allocator);
        if (self.owned_abbey) |persona| {
            persona.deinit();
            self.allocator.destroy(persona);
        }
        if (self.owned_aviva) |persona| {
            persona.deinit();
            self.allocator.destroy(persona);
        }
        self.router.deinit();
        self.ctx.deinit();
        if (self.owned_metrics_collector) |collector| {
            collector.deinit();
            self.allocator.destroy(collector);
        }
        self.allocator.destroy(self);
    }

    /// Process a user request through the persona pipeline.
    pub fn process(self: *Self, request: PersonaRequest) !PersonaResponse {
        var timer = std.time.Timer.start() catch return error.TimerFailed;

        var selected_persona: PersonaType = self.ctx.config.default_persona;
        var primary_score: f32 = 1.0;
        var decision: ?RoutingDecision = null;
        defer if (decision) |*d| d.deinit(self.allocator);

        const forced_request = request.preferred_persona != null;

        // 1. Forced persona overrides routing.
        if (request.preferred_persona) |forced| {
            selected_persona = forced;
            primary_score = 1.0;
        } else if (self.ctx.config.enable_dynamic_routing) {
            // 2. Determine routing via Abi.
            var route_decision = try self.router.route(request);
            decision = route_decision;
            selected_persona = route_decision.selected_persona;
            primary_score = route_decision.confidence;

            // 3. If confidence is too low, fall back to embeddings or default persona.
            if (route_decision.confidence < self.ctx.config.routing_confidence_threshold) {
                if (self.ctx.embeddings_module) |emb| {
                    const matches = emb.findBestPersonaWithLearning(
                        self.allocator,
                        request.content,
                        null,
                        request.user_id,
                        1,
                    ) catch null;
                    if (matches) |list| {
                        defer self.allocator.free(list);
                        if (list.len > 0) {
                            selected_persona = list[0].persona;
                            primary_score = list[0].similarity;
                        } else {
                            selected_persona = self.ctx.config.default_persona;
                            primary_score = 1.0 - route_decision.confidence;
                        }
                    } else {
                        selected_persona = self.ctx.config.default_persona;
                        primary_score = 1.0 - route_decision.confidence;
                    }
                } else {
                    selected_persona = self.ctx.config.default_persona;
                    primary_score = 1.0 - route_decision.confidence;
                }
            }
        } else {
            selected_persona = self.ctx.config.default_persona;
            primary_score = 1.0;
        }

        // 4. Resolve to an available persona.
        var resolved_persona: PersonaType = selected_persona;
        if (!forced_request) {
            resolved_persona = self.resolvePersonaFallback(selected_persona);
        } else if (self.ctx.getPersona(selected_persona) == null) {
            return error.PersonaNotFound;
        }

        // 5. Apply load balancer if available.
        if (self.ctx.load_balancer) |lb| {
            var scores: [2]loadbalancer.PersonaScore = undefined;
            var count: usize = 0;
            scores[count] = .{ .persona_type = selected_persona, .score = primary_score };
            count += 1;
            if (!forced_request and selected_persona != self.ctx.config.default_persona) {
                scores[count] = .{
                    .persona_type = self.ctx.config.default_persona,
                    .score = @max(0.1, 1.0 - primary_score),
                };
                count += 1;
            }

            resolved_persona = lb.selectWithScores(scores[0..count]) catch |err| blk: {
                if (forced_request) return err;
                break :blk self.resolvePersonaFallback(selected_persona);
            };
        }

        // 6. Track request start.
        if (self.ctx.metrics_manager) |m| {
            m.recordRequest(resolved_persona);
        }

        // 7. Get persona implementation.
        const persona = self.ctx.getPersona(resolved_persona) orelse return error.PersonaNotFound;

        // 8. Generate response.
        var response = persona.process(request) catch |err| {
            if (self.ctx.metrics_manager) |m| {
                m.recordError(resolved_persona);
            }
            if (self.ctx.load_balancer) |lb| {
                lb.recordFailure(resolved_persona);
            }
            return err;
        };
        response.generation_time_ms = timer.read() / std.time.ns_per_ms;

        // 9. Update observability data.
        if (self.ctx.metrics_manager) |m| {
            m.recordSuccess(resolved_persona, response.generation_time_ms);
        }
        if (self.ctx.load_balancer) |lb| {
            lb.recordSuccessWithLatency(resolved_persona, response.generation_time_ms);
        }

        return response;
    }

    /// Fallback logic when a routing decision cannot be fulfilled.
    fn processFallback(self: *Self, request: PersonaRequest) !PersonaResponse {
        const default_type = self.ctx.config.default_persona;
        const persona = self.ctx.getPersona(default_type) orelse return error.PersonaNotFound;
        return persona.process(request);
    }

    /// Force processing with a specific persona.
    pub fn processWithPersona(self: *Self, persona_type: PersonaType, request: PersonaRequest) !PersonaResponse {
        var forced = request;
        forced.preferred_persona = persona_type;
        return self.process(forced);
    }

    /// Get a registered persona interface.
    pub fn getPersona(self: *Self, persona_type: PersonaType) ?PersonaInterface {
        return self.ctx.getPersona(persona_type);
    }

    /// Get metrics manager if enabled.
    pub fn getMetrics(self: *Self) ?*metrics.PersonaMetrics {
        return self.ctx.metrics_manager;
    }

    /// Get health checker if enabled.
    pub fn getHealthChecker(self: *Self) ?*health.HealthChecker {
        return self.health_checker;
    }

    /// Register a persona implementation with the system.
    pub fn registerPersona(self: *Self, persona_type: PersonaType, persona: PersonaInterface) !void {
        try self.ctx.registerPersona(persona_type, persona);
        if (self.health_checker) |checker| {
            try checker.registerPersona(persona_type);
        }
    }

    /// Enable metrics using the provided collector or an owned one if null.
    pub fn enableMetrics(self: *Self, collector: ?*obs.MetricsCollector) !void {
        if (self.ctx.metrics_manager != null) return;

        if (collector) |external| {
            try self.ctx.initMetrics(external);
            return;
        }

        const owned = try self.allocator.create(obs.MetricsCollector);
        errdefer {
            owned.deinit();
            self.allocator.destroy(owned);
        }
        owned.* = obs.MetricsCollector.init(self.allocator);
        try self.ctx.initMetrics(owned);
        self.owned_metrics_collector = owned;
    }

    /// Enable load balancing with the provided configuration.
    pub fn enableLoadBalancer(self: *Self, cfg: config.LoadBalancingConfig) !void {
        try self.ctx.initLoadBalancer(cfg);
    }

    /// Enable health checks for registered personas.
    pub fn enableHealthChecks(self: *Self, cfg: health.HealthCheckerConfig) !void {
        if (self.health_checker != null) return;

        const checker = try self.allocator.create(health.HealthChecker);
        errdefer {
            checker.deinit();
            self.allocator.destroy(checker);
        }

        if (self.ctx.metrics_manager != null and self.ctx.load_balancer != null) {
            checker.* = health.HealthChecker.initWithDependencies(
                self.allocator,
                cfg,
                self.ctx.metrics_manager.?,
                self.ctx.load_balancer.?,
            );
        } else {
            checker.* = health.HealthChecker.initWithConfig(self.allocator, cfg);
        }

        const registered = try self.ctx.listRegisteredTypes(self.allocator);
        defer self.allocator.free(registered);
        for (registered) |persona_type| {
            try checker.registerPersona(persona_type);
        }

        self.health_checker = checker;
    }

    fn resolvePersonaFallback(self: *Self, selected: PersonaType) PersonaType {
        if (self.ctx.getPersona(selected) != null) return selected;

        const fallback = self.ctx.config.default_persona;
        if (self.ctx.getPersona(fallback) != null) return fallback;

        if (self.ctx.getAnyPersonaType()) |any_type| {
            return any_type;
        }

        return selected;
    }

    fn registerDefaultPersonas(self: *Self) !void {
        if (self.owned_abbey == null) {
            const abbey_persona = try abbey.AbbeyPersona.init(self.allocator, self.ctx.config.abbey);
            errdefer {
                abbey_persona.deinit();
                self.allocator.destroy(abbey_persona);
            }
            self.owned_abbey = abbey_persona;
            try self.registerPersona(.abbey, abbey_persona.interface());
        }

        if (self.owned_aviva == null) {
            const aviva_persona = try aviva.AvivaPersona.init(self.allocator, self.ctx.config.aviva);
            errdefer {
                aviva_persona.deinit();
                self.allocator.destroy(aviva_persona);
            }
            self.owned_aviva = aviva_persona;
            try self.registerPersona(.aviva, aviva_persona.interface());
        }

        if (self.owned_abi == null) {
            self.owned_abi = abi.AbiPersona.init(self.router);
            try self.registerPersona(.abi, self.owned_abi.?.interface());
        }

        for (types.allPersonaTypes()) |persona_type| {
            if (self.ctx.getPersona(persona_type) != null) continue;
            if (!generic.supports(persona_type)) continue;

            const persona = try generic.GenericPersona.init(self.allocator, persona_type);
            errdefer {
                persona.deinit();
                self.allocator.destroy(persona);
            }
            try self.registerPersona(persona_type, persona.interface());
            try self.owned_generic.put(self.allocator, persona_type, persona);
        }
    }
};

/// Check if the personas module is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

test {
    std.testing.refAllDecls(@This());
    // Integration and unit tests
    _ = @import("tests/integration_test.zig");
    _ = @import("tests/abbey_test.zig");
    _ = @import("tests/abi_test.zig");
    _ = @import("tests/aviva_test.zig");
    _ = @import("tests/benchmark_test.zig");
}
