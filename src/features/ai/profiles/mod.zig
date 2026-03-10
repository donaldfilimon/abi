//! Canonical behavior profiles for routed AI interactions.
//!
//! Standardizes interaction patterns and manages the registration,
//! selection, and coordination of AI behavior profiles.

const std = @import("std");
const time = @import("shared_services").time;
const obs = @import("../../observability");

// Relative imports to flattened feature root
const types = @import("types");
const registry = @import("../registry");
const abi_logic = @import("../abi_logic");
const abbey_logic = @import("../abbey_logic");
const aviva_logic = @import("aviva_logic");
const generic = @import("../generic");
const health = @import("../health");
const loadbalancer = @import("../loadbalancer");

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
        router: *abi_logic.AbiRouter,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, cfg: Config) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            const ctx = try Context(Config).init(allocator, cfg);
            errdefer ctx.deinit();

            const router = try abi_logic.AbiRouter.init(allocator, cfg.abi);
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

test "behavior profiles normalize branded personas" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyPersona(.abbey));
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyPersona(.aviva));
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyPersona(.abi));
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyPersona(.ralph));
}
