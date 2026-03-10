//! Canonical behavior profiles for routed AI interactions.
//!
//! Standardizes interaction patterns and manages the registration,
//! selection, and coordination of AI behavior profiles.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const obs = @import("../../observability/mod.zig");
const legacy_types = @import("../personas/types.zig");
const legacy_registry = @import("../personas/registry.zig");
const legacy_abi = @import("../personas/abi/mod.zig");
const legacy_abbey = @import("../personas/abbey/mod.zig");
const legacy_aviva = @import("../personas/aviva/mod.zig");
const legacy_generic = @import("../personas/generic.zig");
const legacy_health = @import("../personas/health.zig");
const legacy_loadbalancer = @import("../personas/loadbalancer.zig");

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

pub const LegacyPersonaType = legacy_types.PersonaType;
pub const ProfileRegistry = legacy_registry.PersonaRegistry;

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
/// Ported from legacy personas/Context.
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

        pub fn registerPersona(self: *Self, persona_type: LegacyPersonaType, persona: legacy_types.PersonaInterface) !void {
            try self.registry.registerPersona(persona_type, persona);
        }

        pub fn getPersona(self: *Self, persona_type: LegacyPersonaType) ?legacy_types.PersonaInterface {
            return self.registry.getPersona(persona_type);
        }
    };
}

/// High-level orchestrator for behavior profiles.
/// Ported from legacy personas/MultiPersonaSystem.
pub fn ProfileSystem(comptime Config: type) type {
    return struct {
        allocator: std.mem.Allocator,
        ctx: *Context(Config),
        router: *legacy_abi.AbiRouter,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, cfg: Config) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            const ctx = try Context(Config).init(allocator, cfg);
            errdefer ctx.deinit();

            const router = try legacy_abi.AbiRouter.init(allocator, cfg.abi);
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

        pub fn process(self: *Self, request: legacy_types.PersonaRequest) !legacy_types.PersonaResponse {
            _ = self;
            _ = request;
            return error.NotImplemented; // Stubbed for initial porting wave
        }
    };
}

test "behavior profiles normalize branded personas" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyPersona(.abbey));
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyPersona(.aviva));
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyPersona(.abi));
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyPersona(.ralph));
}
