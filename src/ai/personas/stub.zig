//! Multi-Persona AI Assistant Module (Stub)
//!
//! This module provides a compile-time compatible API for the personas module
//! when the AI feature is disabled. All operations return error.AiDisabled.

const std = @import("std");

// Re-import types for API parity
pub const types = @import("types.zig");
pub const config = @import("config.zig");

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

/// Stub persona registry.
pub const PersonaRegistry = struct {
    pub fn init(_: std.mem.Allocator) PersonaRegistry {
        return .{};
    }
    pub fn deinit(_: *PersonaRegistry) void {}

    pub fn registerPersona(_: *PersonaRegistry, _: PersonaType, _: PersonaInterface) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn getPersona(_: *PersonaRegistry, _: PersonaType) ?PersonaInterface {
        return null;
    }

    pub fn configurePersona(_: *PersonaRegistry, _: PersonaType, _: MultiPersonaConfig) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn get(_: *PersonaRegistry, _: PersonaType) ?PersonaInterface {
        return null;
    }
};

/// Personas context for framework integration (Stub).
pub const Context = struct {
    allocator: std.mem.Allocator,
    config_: MultiPersonaConfig,
    registry: PersonaRegistry,
    embeddings_module: ?*anyopaque = null,
    metrics_manager: ?*anyopaque = null,
    load_balancer: ?*anyopaque = null,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: MultiPersonaConfig) error{AiDisabled}!*Context {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn initEmbeddings(_: *Self, _: anytype, _: anytype) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn initMetrics(_: *Self, _: anytype) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn initLoadBalancer(_: *Self, _: LoadBalancingConfig) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn registerPersona(_: *Self, _: PersonaType, _: PersonaInterface) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn getPersona(_: *Self, _: PersonaType) ?PersonaInterface {
        return null;
    }

    pub fn configurePersona(_: *Self, _: PersonaType, _: MultiPersonaConfig) error{AiDisabled}!void {
        return error.AiDisabled;
    }
};

/// High-level orchestrator for the multi-persona assistant (Stub).
pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: ?*Context = null,
    router: ?*anyopaque = null,
    metrics: ?*anyopaque = null,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: MultiPersonaConfig) error{AiDisabled}!*Self {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn process(_: *Self, _: PersonaRequest) error{AiDisabled}!PersonaResponse {
        return error.AiDisabled;
    }
};

// Stub submodule namespaces for API parity
pub const abi = struct {
    pub const AbiRouter = struct {
        pub fn init(_: std.mem.Allocator, _: AbiConfig) error{AiDisabled}!*AbiRouter {
            return error.AiDisabled;
        }
        pub fn deinit(_: *AbiRouter) void {}
        pub fn route(_: *AbiRouter, _: PersonaRequest) error{AiDisabled}!RoutingDecision {
            return error.AiDisabled;
        }
    };
};

pub const abbey = struct {
    pub const AbbeyPersona = struct {
        pub fn init(_: std.mem.Allocator, _: AbbeyConfig) error{AiDisabled}!*AbbeyPersona {
            return error.AiDisabled;
        }
        pub fn deinit(_: *AbbeyPersona) void {}
    };
};

pub const aviva = struct {
    pub const AvivaPersona = struct {
        pub fn init(_: std.mem.Allocator, _: AvivaConfig) error{AiDisabled}!*AvivaPersona {
            return error.AiDisabled;
        }
        pub fn deinit(_: *AvivaPersona) void {}
    };
};

pub const embeddings = struct {
    pub const EmbeddingsModule = struct {
        pub fn init(_: std.mem.Allocator, _: anytype, _: anytype) error{AiDisabled}!*EmbeddingsModule {
            return error.AiDisabled;
        }
        pub fn deinit(_: *EmbeddingsModule) void {}
    };
};

pub const metrics = struct {
    pub const PersonaMetrics = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) error{AiDisabled}!*PersonaMetrics {
            return error.AiDisabled;
        }
        pub fn deinit(_: *PersonaMetrics) void {}
        pub fn getStats(_: *PersonaMetrics, _: PersonaType) ?anyopaque {
            return null;
        }
    };
};

pub const loadbalancer = struct {
    pub const PersonaLoadBalancer = struct {
        pub fn init(_: std.mem.Allocator, _: LoadBalancingConfig) PersonaLoadBalancer {
            return .{};
        }
        pub fn deinit(_: *PersonaLoadBalancer) void {}
        pub fn recordSuccess(_: *PersonaLoadBalancer, _: PersonaType) void {}
    };
};

/// Always returns false in the stub.
pub fn isEnabled() bool {
    return false;
}
