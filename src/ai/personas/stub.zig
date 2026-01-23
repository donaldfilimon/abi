//! Multi-Persona AI Assistant Module (Stub)
//!
//! This module provides a compile-time compatible API for the personas module
//! when the AI feature is disabled. All operations return error.AiDisabled.

const std = @import("std");

// Re-import types for API parity
const types = @import("types.zig");
const config = @import("config.zig");

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

/// Stub persona registry.
pub const PersonaRegistry = struct {
    pub fn init(_: std.mem.Allocator) PersonaRegistry {
        return .{};
    }
    pub fn deinit(_: *PersonaRegistry) void {}
};

/// Personas context for framework integration (Stub).
pub const Context = struct {
    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: MultiPersonaConfig) error{AiDisabled}!*Context {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

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

/// Always returns false in the stub.
pub fn isEnabled() bool {
    return false;
}
