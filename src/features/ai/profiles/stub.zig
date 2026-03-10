//! Profiles Stub — disabled at compile time.

const std = @import("std");
const types = @import("types");
const registry = @import("../registry");

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
