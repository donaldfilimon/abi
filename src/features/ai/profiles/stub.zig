//! Profiles Stub — disabled at compile time.

const std = @import("std");
const types = @import("../types.zig");
const registry = @import("../registry.zig");

pub const BehaviorProfile = enum { collaborative, direct, governance, iterative };
pub const LegacyProfileType = types.ProfileType;
pub const ProfileRegistry = registry.ProfileRegistry;

pub fn fromLegacyProfile(_: LegacyProfileType) BehaviorProfile {
    return .collaborative;
}

pub fn defaultLegacyProfile(_: BehaviorProfile) LegacyProfileType {
    return .assistant;
}

pub fn Context(comptime Config: type) type {
    return struct {
        const Self = @This();
        pub fn init(_: std.mem.Allocator, _: Config) !*Self {
            return error.AiDisabled;
        }
        pub fn deinit(_: *Self) void {}
        pub fn registerProfile(_: *Self, _: LegacyProfileType, _: types.ProfileInterface) !void {}
        pub fn getProfile(_: *Self, _: LegacyProfileType) ?types.ProfileInterface {
            return null;
        }
    };
}

pub fn ProfileSystem(comptime Config: type) type {
    return struct {
        const Self = @This();
        pub fn init(_: std.mem.Allocator, _: Config) !*Self {
            return error.AiDisabled;
        }
        pub fn deinit(_: *Self) void {}
        pub fn process(_: *Self, _: types.ProfileRequest) !types.ProfileResponse {
            return error.AiDisabled;
        }
    };
}

pub const MultiProfileSystem = struct {
    pub const MetricsManager = struct {
        pub const LatencyStats = struct {
            p50: f64 = 0,
            p99: f64 = 0,
        };

        pub const ProfileStats = struct {
            total_requests: u64 = 0,
            success_rate: f32 = 1.0,
            error_count: u64 = 0,
            latency: ?LatencyStats = null,
        };

        pub fn getStats(_: *MetricsManager, _: LegacyProfileType) ?ProfileStats {
            return null;
        }
    };

    pub const MultiProfileContext = struct {
        pub fn getProfile(_: *MultiProfileContext, _: LegacyProfileType) ?types.ProfileInterface {
            return null;
        }
    };

    allocator: std.mem.Allocator,
    ctx: MultiProfileContext = .{},
    _metrics: ?*MetricsManager = null,

    pub fn process(_: *MultiProfileSystem, _: types.ProfileRequest) !types.ProfileResponse {
        return error.AiDisabled;
    }

    pub fn getMetrics(self: *MultiProfileSystem) ?*MetricsManager {
        return self._metrics;
    }
};
