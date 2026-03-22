//! Multi-Persona Orchestration Layer
//!
//! Provides unified lifecycle management, routing, and collaboration for
//! the Abbey-Aviva-Abi multi-persona AI system.
//!
//! ## Quick Start
//!
//! ```zig
//! const persona = abi.ai.persona;
//!
//! // Create registry with default config
//! var registry = persona.PersonaRegistry.init(allocator, .{});
//! defer registry.deinit();
//!
//! // Initialize all persona engines
//! try registry.initAll();
//!
//! // Create router
//! var router = persona.MultiPersonaRouter.init(allocator, &registry, .{});
//!
//! // Route and execute a query
//! const response = try router.routeAndExecute("Explain how HNSW indexing works");
//! defer response.deinit();
//! ```

const std = @import("std");

// ── Sub-modules ─────────────────────────────────────────────────────────
pub const types = @import("types.zig");
pub const registry = @import("registry.zig");
pub const router = @import("router.zig");
pub const bus = @import("bus.zig");

// ── Type re-exports ─────────────────────────────────────────────────────
pub const PersonaId = types.PersonaId;
pub const PersonaState = types.PersonaState;
pub const RoutingStrategy = types.RoutingStrategy;
pub const RoutingDecision = types.RoutingDecision;
pub const PersonaResponse = types.PersonaResponse;
pub const PersonaMessage = types.PersonaMessage;
pub const MessageKind = types.MessageKind;
pub const RoutingConfig = types.RoutingConfig;
pub const PersonaError = types.PersonaError;

pub const PersonaRegistry = registry.PersonaRegistry;
pub const PersonaInstance = registry.PersonaInstance;
pub const MultiPersonaConfig = registry.MultiPersonaConfig;

pub const MultiPersonaRouter = router.MultiPersonaRouter;
pub const PersonaBus = bus.PersonaBus;

test {
    std.testing.refAllDecls(@This());
}
