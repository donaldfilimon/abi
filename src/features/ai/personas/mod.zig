//! Multi-Persona AI Assistant Module
//!
//! This module implements a multi-layer, multi-persona AI system that routes
//! user queries through specialized interaction models.
//!
//! Canonical public entrypoints now live under `abi.features.ai.profiles` and
//! `abi.features.ai.coordination`. This module remains the branded implementation
//! and compatibility layer for the current migration wave.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
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
pub const swap = @import("swap.zig");
pub const modulation = @import("modulation.zig");
pub const templates = @import("templates/mod.zig");

const profiles = @import("../profiles/mod.zig");

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

/// Personas context for framework integration (Legacy Shim).
pub const Context = profiles.Context(MultiPersonaConfig);

/// High-level orchestrator for the multi-persona assistant (Legacy Shim).
pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    impl: *profiles.ProfileSystem(MultiPersonaConfig),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, cfg: MultiPersonaConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const impl = try profiles.ProfileSystem(MultiPersonaConfig).init(allocator, cfg);
        errdefer impl.deinit();

        self.* = .{
            .allocator = allocator,
            .ctx = impl.ctx,
            .impl = impl,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.impl.deinit();
        self.allocator.destroy(self);
    }

    pub fn getMetrics(self: *Self) ?*metrics.PersonaMetrics {
        const opaque_metrics = self.impl.ctx.metrics_manager orelse return null;
        return @ptrCast(@alignCast(opaque_metrics));
    }

    pub fn process(self: *Self, request: PersonaRequest) !PersonaResponse {
        return self.impl.process(request);
    }

    pub fn processWithPersona(self: *Self, persona_type: PersonaType, request: PersonaRequest) !PersonaResponse {
        var forced = request;
        forced.preferred_persona = persona_type;
        return self.process(forced);
    }

    pub fn registerPersona(self: *Self, persona_type: PersonaType, persona: PersonaInterface) !void {
        try self.impl.ctx.registerProfile(persona_type, persona);
    }
};

/// Check if the personas module is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.feat_ai;
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
