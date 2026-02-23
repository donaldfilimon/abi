//! Multi-Persona stub — disabled at compile time.

const std = @import("std");
const obs = @import("../../observability/mod.zig");
const health = @import("health.zig");
const metrics_mod = @import("metrics.zig");

pub const types = @import("types.zig");
pub const config = @import("config.zig");

// Core type re-exports
pub const PersonaType = types.PersonaType;
pub const PersonaRequest = types.PersonaRequest;
pub const PersonaResponse = types.PersonaResponse;
pub const RoutingDecision = types.RoutingDecision;
pub const PersonaInterface = types.PersonaInterface;
pub const PolicyFlags = types.PolicyFlags;
pub const ReasoningStep = types.ReasoningStep;
pub const CodeBlock = types.CodeBlock;
pub const Source = types.Source;
pub const RoutingScore = struct { persona_type: PersonaType, score: f32 };

pub const generic = struct {
    pub const GenericPersona = struct {
        pub fn init(_: std.mem.Allocator, _: PersonaType) error{FeatureDisabled}!*GenericPersona {
            return error.FeatureDisabled;
        }
        pub fn deinit(_: *GenericPersona) void {}
    };
    pub fn supports(_: PersonaType) bool {
        return false;
    }
};

pub const MultiPersonaConfig = config.MultiPersonaConfig;
pub const AbiConfig = config.AbiConfig;
pub const AbbeyConfig = config.AbbeyConfig;
pub const AvivaConfig = config.AvivaConfig;
pub const LoadBalancingConfig = config.LoadBalancingConfig;

pub const PersonaRegistry = struct {
    pub fn init(_: std.mem.Allocator) PersonaRegistry {
        return .{};
    }
    pub fn deinit(_: *PersonaRegistry) void {}
    pub fn registerPersona(_: *PersonaRegistry, _: PersonaType, _: PersonaInterface) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn getPersona(_: *PersonaRegistry, _: PersonaType) ?PersonaInterface {
        return null;
    }
    pub fn configurePersona(_: *PersonaRegistry, _: PersonaType, _: MultiPersonaConfig) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn get(_: *PersonaRegistry, _: PersonaType) ?PersonaInterface {
        return null;
    }
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MultiPersonaConfig,
    registry: PersonaRegistry,
    embeddings_module: ?*anyopaque = null,
    metrics_manager: ?*anyopaque = null,
    load_balancer: ?*anyopaque = null,
    const Self = @This();
    pub fn init(_: std.mem.Allocator, _: MultiPersonaConfig) error{FeatureDisabled}!*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Self) void {}
    pub fn initEmbeddings(_: *Self, _: anytype, _: anytype) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn initMetrics(_: *Self, _: anytype) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn initLoadBalancer(_: *Self, _: LoadBalancingConfig) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn registerPersona(_: *Self, _: PersonaType, _: PersonaInterface) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn getPersona(_: *Self, _: PersonaType) ?PersonaInterface {
        return null;
    }
    pub fn configurePersona(_: *Self, _: PersonaType, _: MultiPersonaConfig) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn listRegisteredTypes(_: *Self, _: std.mem.Allocator) error{FeatureDisabled}![]PersonaType {
        return error.FeatureDisabled;
    }
    pub fn getAnyPersonaType(_: *Self) ?PersonaType {
        return null;
    }
};

pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: ?*Context = null,
    router: ?*anyopaque = null,
    metrics: ?*anyopaque = null,
    const Self = @This();
    pub fn init(_: std.mem.Allocator, _: MultiPersonaConfig) error{FeatureDisabled}!*Self {
        return error.FeatureDisabled;
    }
    pub fn initWithDefaults(_: std.mem.Allocator, _: MultiPersonaConfig) error{FeatureDisabled}!*Self {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Self) void {}
    pub fn process(_: *Self, _: PersonaRequest) error{FeatureDisabled}!PersonaResponse {
        return error.FeatureDisabled;
    }
    pub fn processWithPersona(_: *Self, _: PersonaType, _: PersonaRequest) error{FeatureDisabled}!PersonaResponse {
        return error.FeatureDisabled;
    }
    pub fn getPersona(_: *Self, _: PersonaType) ?PersonaInterface {
        return null;
    }
    pub fn getMetrics(_: *Self) ?*metrics_mod.PersonaMetrics {
        return null;
    }
    pub fn getHealthChecker(_: *Self) ?*health.HealthChecker {
        return null;
    }
    pub fn registerPersona(_: *Self, _: PersonaType, _: PersonaInterface) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn enableMetrics(_: *Self, _: ?*obs.MetricsCollector) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn enableLoadBalancer(_: *Self, _: LoadBalancingConfig) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn enableHealthChecks(_: *Self, _: health.HealthCheckerConfig) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
};

// Submodule stubs
pub const abi = struct {
    pub const AbiRouter = struct {
        pub fn init(_: std.mem.Allocator, _: AbiConfig) error{FeatureDisabled}!*AbiRouter {
            return error.FeatureDisabled;
        }
        pub fn deinit(_: *AbiRouter) void {}
        pub fn route(_: *AbiRouter, _: PersonaRequest) error{FeatureDisabled}!RoutingDecision {
            return error.FeatureDisabled;
        }
    };
};
pub const abbey = struct {
    pub const AbbeyPersona = struct {
        pub fn init(_: std.mem.Allocator, _: AbbeyConfig) error{FeatureDisabled}!*AbbeyPersona {
            return error.FeatureDisabled;
        }
        pub fn deinit(_: *AbbeyPersona) void {}
    };
};
pub const aviva = struct {
    pub const AvivaPersona = struct {
        pub fn init(_: std.mem.Allocator, _: AvivaConfig) error{FeatureDisabled}!*AvivaPersona {
            return error.FeatureDisabled;
        }
        pub fn deinit(_: *AvivaPersona) void {}
    };
};
pub const embeddings = struct {
    pub const EmbeddingsModule = struct {
        pub fn init(_: std.mem.Allocator, _: anytype, _: anytype) error{FeatureDisabled}!*EmbeddingsModule {
            return error.FeatureDisabled;
        }
        pub fn deinit(_: *EmbeddingsModule) void {}
    };
};
pub const metrics = struct {
    pub const PersonaMetrics = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) error{FeatureDisabled}!*PersonaMetrics {
            return error.FeatureDisabled;
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

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
