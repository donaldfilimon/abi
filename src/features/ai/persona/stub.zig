//! Stub for multi-persona orchestration when feat_ai is disabled.

const std = @import("std");
const types = @import("types.zig");

pub const PersonaId = types.PersonaId;
pub const PersonaState = types.PersonaState;
pub const RoutingStrategy = types.RoutingStrategy;
pub const RoutingDecision = types.RoutingDecision;
pub const PersonaResponse = types.PersonaResponse;
pub const PersonaMessage = types.PersonaMessage;
pub const MessageKind = types.MessageKind;
pub const RoutingConfig = types.RoutingConfig;
pub const PersonaError = types.PersonaError;

pub const MultiPersonaConfig = struct {
    routing: RoutingConfig = .{},
};

pub const PersonaInstance = struct {
    id: PersonaId,
    state: PersonaState = .uninitialized,
    allocator: std.mem.Allocator,

    pub fn process(_: *PersonaInstance, _: []const u8) PersonaError!PersonaResponse {
        return error.PersonaNotInitialized;
    }
    pub fn getName(self: *const PersonaInstance) []const u8 {
        return self.id.name();
    }
    pub fn isAvailable(_: *const PersonaInstance) bool {
        return false;
    }
};

pub const PersonaRegistry = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, _: MultiPersonaConfig) PersonaRegistry {
        return .{ .allocator = allocator };
    }
    pub fn initAll(_: *PersonaRegistry) PersonaError!void {
        return error.PersonaNotInitialized;
    }
    pub fn getPersona(_: *PersonaRegistry, _: PersonaId) ?*PersonaInstance {
        return null;
    }
    pub fn getAbiRouter(_: *PersonaRegistry) ?*anyopaque {
        return null;
    }
    pub fn suspendPersona(_: *PersonaRegistry, _: PersonaId) void {}
    pub fn resumePersona(_: *PersonaRegistry, _: PersonaId) void {}
    pub fn listAvailable(_: *PersonaRegistry) [3]?PersonaId {
        return .{ null, null, null };
    }
    pub fn deinit(_: *PersonaRegistry) void {}
};

pub const MultiPersonaRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: *PersonaRegistry, _: RoutingConfig) MultiPersonaRouter {
        return .{ .allocator = allocator };
    }
    pub fn route(_: *MultiPersonaRouter, _: []const u8) RoutingDecision {
        return .{
            .primary = .abbey,
            .weights = .{},
            .strategy = .single,
            .confidence = 0.0,
            .reason = "AI disabled",
        };
    }
    pub fn execute(_: *MultiPersonaRouter, _: RoutingDecision, _: []const u8) PersonaError!PersonaResponse {
        return error.PersonaNotInitialized;
    }
    pub fn routeAndExecute(_: *MultiPersonaRouter, _: []const u8) PersonaError!PersonaResponse {
        return error.PersonaNotInitialized;
    }
    pub fn deinit(_: *MultiPersonaRouter) void {}
};

pub const PersonaBus = struct {
    pub fn init(_: std.mem.Allocator) PersonaBus {
        return .{};
    }
    pub fn deinit(_: *PersonaBus) void {}
};

test {
    std.testing.refAllDecls(@This());
}
