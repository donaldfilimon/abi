//! Canonical behavior profiles for routed AI interactions.
//!
//! Standardizes interaction patterns and manages the registration,
//! selection, and coordination of AI behavior profiles.

const std = @import("std");
const build_options = @import("build_options");
const time = @import("../../../foundation/mod.zig").time;
const obs = if (build_options.feat_observability)
    @import("../../observability/mod.zig")
else
    @import("../../observability/stub.zig");

// Relative imports to flattened feature root
const types = @import("../types.zig");
const registry = @import("../registry.zig");
const abi = @import("../abi/mod.zig");

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

pub const LegacyProfileType = types.ProfileType;
pub const ProfileRegistry = registry.ProfileRegistry;

pub fn fromLegacyProfile(profile: LegacyProfileType) BehaviorProfile {
    return switch (profile) {
        .assistant, .companion, .docs, .abbey => .collaborative,
        .coder, .analyst, .reviewer, .minimal, .aviva, .ava => .direct,
        .abi => .governance,
        .writer, .ralph => .iterative,
    };
}

pub fn defaultLegacyProfile(profile: BehaviorProfile) LegacyProfileType {
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

        pub fn registerProfile(self: *Self, profile_type: LegacyProfileType, profile: types.ProfileInterface) !void {
            try self.registry.registerProfile(profile_type, profile);
        }

        pub fn getProfile(self: *Self, profile_type: LegacyProfileType) ?types.ProfileInterface {
            return self.registry.getProfile(profile_type);
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

        pub fn process(self: *Self, request: types.ProfileRequest) !types.ProfileResponse {
            // 1. If the request explicitly specifies a profile, honour it.
            //    Otherwise use the Abi router to choose one.
            const target_profile: types.ProfileType = if (request.preferred_profile) |pref|
                pref
            else blk: {
                var decision = try self.router.route(request);
                defer decision.deinit(self.allocator);

                // Safety violation — refuse immediately.
                if (!decision.policy_flags.is_safe) {
                    return types.ProfileResponse{
                        .content = try self.allocator.dupe(u8, "Request blocked by safety policy."),
                        .profile = .abi,
                        .confidence = 1.0,
                    };
                }

                break :blk decision.selected_profile;
            };

            // 2. Look up the profile implementation in the registry.
            if (self.ctx.getProfile(target_profile)) |profile_impl| {
                return profile_impl.process(request);
            }

            // 3. Fallback — no implementation registered for the chosen
            //    profile.  Return a minimal acknowledgement so callers
            //    always receive a valid response.
            return types.ProfileResponse{
                .content = try self.allocator.dupe(u8, "No profile handler registered for the selected profile."),
                .profile = target_profile,
                .confidence = 0.0,
            };
        }
    };
}

/// Concrete orchestrator facade used by web handlers.
///
/// Wraps the generic `ProfileSystem` behind a non-generic interface so that
/// HTTP handlers can hold a `?*MultiProfileSystem` pointer without knowing
/// the concrete config type at compile time.
pub const MultiProfileSystem = struct {
    allocator: std.mem.Allocator,
    ctx: MultiProfileContext,
    _metrics: ?*MetricsManager = null,

    pub const MultiProfileContext = struct {
        registry: ProfileRegistry,

        pub fn getProfile(self: *MultiProfileContext, profile_type: LegacyProfileType) ?types.ProfileInterface {
            return self.registry.getProfile(profile_type);
        }
    };

    /// Stub metrics manager for web handler compatibility.
    pub const MetricsManager = struct {
        pub const ProfileStats = struct {
            total_requests: u64 = 0,
            success_rate: f32 = 1.0,
            error_count: u64 = 0,
            latency: ?LatencyStats = null,
        };

        pub const LatencyStats = struct {
            p50: f64 = 0,
            p99: f64 = 0,
        };

        pub fn getStats(_: *MetricsManager, _: LegacyProfileType) ?ProfileStats {
            return null;
        }
    };

    pub fn process(self: *MultiProfileSystem, request: types.ProfileRequest) !types.ProfileResponse {
        // 1. Determine target profile: explicit preference > default (.abbey).
        const target_profile: types.ProfileType = request.preferred_profile orelse .abbey;

        // 2. Look up the profile implementation.
        if (self.ctx.getProfile(target_profile)) |profile_impl| {
            return profile_impl.process(request);
        }

        // 3. Fallback: try any registered profile.
        if (self.ctx.registry.getAnyProfileType()) |fallback_type| {
            if (self.ctx.getProfile(fallback_type)) |fallback_impl| {
                return fallback_impl.process(request);
            }
        }

        // 4. No profiles registered at all — return a minimal response.
        return types.ProfileResponse{
            .content = try self.allocator.dupe(u8, "No profile handler available."),
            .profile = target_profile,
            .confidence = 0.0,
        };
    }

    pub fn getMetrics(self: *MultiProfileSystem) ?*MetricsManager {
        return self._metrics;
    }
};

test "behavior profiles normalize branded profiles" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyProfile(.abbey));
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyProfile(.aviva));
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyProfile(.abi));
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyProfile(.ralph));
}

test {
    std.testing.refAllDecls(@This());
}
