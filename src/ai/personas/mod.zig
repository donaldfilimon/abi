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

pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");

// Persona sub-modules
pub const abi = @import("abi/mod.zig");
pub const abbey = @import("abbey/mod.zig");
pub const aviva = @import("aviva/mod.zig");
pub const embeddings = @import("embeddings/mod.zig");
pub const metrics = @import("metrics.zig");
pub const loadbalancer = @import("loadbalancer.zig");

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
    }

    /// Initialize the load balancer for the persona system.
    pub fn initLoadBalancer(self: *Self, cfg: config.LoadBalancingConfig) !void {
        if (self.load_balancer != null) return;
        const lb = try self.allocator.create(loadbalancer.PersonaLoadBalancer);
        errdefer self.allocator.destroy(lb);
        lb.* = loadbalancer.PersonaLoadBalancer.init(self.allocator, cfg);
        self.load_balancer = lb;
    }

    /// Register a persona implementation.
    pub fn registerPersona(self: *Self, persona_type: PersonaType, persona: PersonaInterface) !void {
        try self.registry.registerPersona(persona_type, persona);
    }

    /// Get a registered persona by type.
    pub fn getPersona(self: *Self, persona_type: PersonaType) ?PersonaInterface {
        return self.registry.getPersona(persona_type);
    }

    /// Configure a specific persona's behavior.
    pub fn configurePersona(self: *Self, persona_type: PersonaType, cfg: MultiPersonaConfig) !void {
        try self.registry.configurePersona(persona_type, cfg);
    }
};

/// High-level orchestrator for the multi-persona assistant.
/// Handles the end-to-end request lifecycle: routing, persona selection, and metrics.
pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    router: *abi.AbiRouter,

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

    /// Shutdown the system and free resources.
    pub fn deinit(self: *Self) void {
        self.router.deinit();
        self.ctx.deinit();
        self.allocator.destroy(self);
    }

    /// Process a user request through the persona pipeline.
    pub fn process(self: *Self, request: PersonaRequest) !PersonaResponse {
        var timer = std.time.Timer.start() catch return error.TimerFailed;

        // 1. Determine routing via Abi
        const decision = try self.router.route(request);
        defer @constCast(&decision).deinit(self.allocator);

        // 2. Track request start
        if (self.ctx.metrics_manager) |m| {
            m.recordRequest(decision.selected_persona);
        }

        // 3. Get the selected persona implementation
        const persona = self.ctx.getPersona(decision.selected_persona) orelse {
            // Fallback to default persona if the selected one is not registered
            return self.processFallback(request);
        };

        // 4. Generate response
        var response = try persona.process(request);
        response.generation_time_ms = timer.read() / 1_000_000;

        // 5. Update observability data
        if (self.ctx.metrics_manager) |m| {
            m.recordSuccess(decision.selected_persona, response.generation_time_ms);
        }
        if (self.ctx.load_balancer) |lb| {
            lb.recordSuccess(decision.selected_persona);
        }

        return response;
    }

    /// Fallback logic when a routing decision cannot be fulfilled.
    fn processFallback(self: *Self, request: PersonaRequest) !PersonaResponse {
        const default_type = self.ctx.config.default_persona;
        const persona = self.ctx.getPersona(default_type) orelse return error.PersonaNotFound;
        return persona.process(request);
    }
};

/// Check if the personas module is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests/integration_test.zig");
}
