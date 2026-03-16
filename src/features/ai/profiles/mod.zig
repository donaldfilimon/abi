//! Canonical behavior profiles for routed AI interactions.
//!
//! Standardizes interaction patterns and manages the registration,
//! selection, and coordination of AI behavior profiles.

const std = @import("std");
const time = @import("../../../services/shared/mod.zig").time;
const obs = @import("../../observability/mod.zig");

// Relative imports to flattened feature root
const types = @import("../types.zig");
const registry = @import("../registry.zig");
const abi = @import("../abi/mod.zig");
const abbey = @import("../abbey/persona.zig");
const aviva = @import("../aviva/mod.zig");
const generic = @import("../generic.zig");
const health = @import("../health.zig");
const loadbalancer = @import("../loadbalancer.zig");

pub const BehaviorProfile = enum {
    collaborative,
    direct,
    governance,
    iterative,

    pub fn displayName(self: BehaviorProfile) []const u8 {
        return switch (self) {
            .collaborative => "Collaborative",
            .direct => "Direct",
            .governance => "Governance",
            .iterative => "Iterative",
        };
    }
};

pub const LegacyPersonaType = types.PersonaType;
pub const ProfileRegistry = registry.PersonaRegistry;

pub fn fromLegacyPersona(persona: LegacyPersonaType) BehaviorProfile {
    return switch (persona) {
        .assistant, .companion, .docs, .abbey => .collaborative,
        .coder, .analyst, .reviewer, .minimal, .aviva => .direct,
        .abi => .governance,
        .writer, .ralph => .iterative,
    };
}

pub fn defaultLegacyPersona(profile: BehaviorProfile) LegacyPersonaType {
    return switch (profile) {
        .collaborative => .abbey,
        .direct => .aviva,
        .governance => .abi,
        .iterative => .ralph,
    };
}

/// Profiles context for framework integration.
pub fn Context(comptime Config: type) type {
    return struct {
        allocator: std.mem.Allocator,
        config: Config,
        registry: ProfileRegistry,
        metrics_manager: ?*anyopaque = null,
        load_balancer: ?*anyopaque = null,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, cfg: Config) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .allocator = allocator,
                .config = cfg,
                .registry = ProfileRegistry.init(allocator),
            };

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.registry.deinit();
            self.allocator.destroy(self);
        }

        pub fn registerPersona(self: *Self, persona_type: LegacyPersonaType, persona: types.PersonaInterface) !void {
            try self.registry.registerPersona(persona_type, persona);
        }

        pub fn getPersona(self: *Self, persona_type: LegacyPersonaType) ?types.PersonaInterface {
            return self.registry.getPersona(persona_type);
        }
    };
}

/// High-level orchestrator for behavior profiles.
pub fn ProfileSystem(comptime Config: type) type {
    return struct {
        allocator: std.mem.Allocator,
        ctx: *Context(Config),
        router: *abi.AbiRouter,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, cfg: Config) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            const ctx = try Context(Config).init(allocator, cfg);
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

        pub fn deinit(self: *Self) void {
            self.router.deinit();
            self.ctx.deinit();
            self.allocator.destroy(self);
        }

        pub fn process(self: *Self, request: types.PersonaRequest) !types.PersonaResponse {
            _ = self;
            _ = request;
            return error.NotImplemented; // Stubbed for close-out wave
        }
    };
}

/// Concrete orchestrator facade used by web handlers.
///
/// Wraps the generic `ProfileSystem` behind a non-generic interface so that
/// HTTP handlers can hold a `?*MultiPersonaSystem` pointer without knowing
/// the concrete config type at compile time.
pub const MultiPersonaSystem = struct {
    allocator: std.mem.Allocator,
    ctx: MultiPersonaContext,
    _metrics: ?*MetricsManager = null,

    pub const MultiPersonaContext = struct {
        registry: ProfileRegistry,

        pub fn getPersona(self: *MultiPersonaContext, persona_type: LegacyPersonaType) ?types.PersonaInterface {
            return self.registry.getPersona(persona_type);
        }
    };

    /// Stub metrics manager for web handler compatibility.
    pub const MetricsManager = struct {
        pub const PersonaStats = struct {
            total_requests: u64 = 0,
            success_rate: f32 = 1.0,
            error_count: u64 = 0,
            latency: ?LatencyStats = null,
        };

        pub const LatencyStats = struct {
            p50: f64 = 0,
            p99: f64 = 0,
        };

        pub fn getStats(_: *MetricsManager, _: LegacyPersonaType) ?PersonaStats {
            return null;
        }
    };

    pub fn process(self: *MultiPersonaSystem, request: types.PersonaRequest) !types.PersonaResponse {
        _ = self;
        _ = request;
        return error.NotImplemented;
    }

    pub fn getMetrics(self: *MultiPersonaSystem) ?*MetricsManager {
        return self._metrics;
    }
};

test "behavior profiles normalize branded personas" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyPersona(.abbey));
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyPersona(.aviva));
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyPersona(.abi));
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyPersona(.ralph));
}
