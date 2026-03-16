//! Profiles Stub — disabled at compile time.

const std = @import("std");
const types = @import("../types.zig");
const registry = @import("../registry.zig");

pub const BehaviorProfile = enum { collaborative, direct, governance, iterative };
pub const LegacyPersonaType = types.PersonaType;
pub const ProfileRegistry = registry.PersonaRegistry;

pub fn fromLegacyPersona(_: LegacyPersonaType) BehaviorProfile {
    return .collaborative;
}

pub fn defaultLegacyPersona(_: BehaviorProfile) LegacyPersonaType {
    return .assistant;
}

pub fn Context(comptime Config: type) type {
    return struct {
        const Self = @This();
        pub fn init(_: std.mem.Allocator, _: Config) !*Self {
            return error.AiDisabled;
        }
        pub fn deinit(_: *Self) void {}
        pub fn registerPersona(_: *Self, _: LegacyPersonaType, _: types.PersonaInterface) !void {}
        pub fn getPersona(_: *Self, _: LegacyPersonaType) ?types.PersonaInterface {
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
        pub fn process(_: *Self, _: types.PersonaRequest) !types.PersonaResponse {
            return error.AiDisabled;
        }
    };
}

pub const MultiPersonaSystem = struct {
    pub const MetricsManager = struct {
        pub const LatencyStats = struct {
            p50: f64 = 0,
            p99: f64 = 0,
        };

        pub const PersonaStats = struct {
            total_requests: u64 = 0,
            success_rate: f32 = 1.0,
            error_count: u64 = 0,
            latency: ?LatencyStats = null,
        };

        pub fn getStats(_: *MetricsManager, _: LegacyPersonaType) ?PersonaStats {
            return null;
        }
    };

    pub const MultiPersonaContext = struct {
        pub fn getPersona(_: *MultiPersonaContext, _: LegacyPersonaType) ?types.PersonaInterface {
            return null;
        }
    };

    allocator: std.mem.Allocator,
    ctx: MultiPersonaContext = .{},
    _metrics: ?*MetricsManager = null,

    pub fn process(_: *MultiPersonaSystem, _: types.PersonaRequest) !types.PersonaResponse {
        return error.AiDisabled;
    }

    pub fn getMetrics(self: *MultiPersonaSystem) ?*MetricsManager {
        return self._metrics;
    }
};
