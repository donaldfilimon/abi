//! Multi-Profile Orchestration Layer
//!
//! Provides unified lifecycle management, routing, and collaboration for
//! the Abbey-Aviva-Abi multi-profile AI system.
//!
//! ## Quick Start
//!
//! ```zig
//! const profile = abi.ai.profile;
//!
//! // Create registry with default config
//! var registry = profile.ProfileRegistry.init(allocator, .{});
//! defer registry.deinit();
//!
//! // Initialize all profile engines
//! try registry.initAll();
//!
//! // Create router
//! var router = profile.MultiProfileRouter.init(allocator, &registry, .{});
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
pub const memory = @import("memory.zig");

// ── Type re-exports ─────────────────────────────────────────────────────
pub const ProfileId = types.ProfileId;
pub const ProfileState = types.ProfileState;
pub const RoutingStrategy = types.RoutingStrategy;
pub const RoutingDecision = types.RoutingDecision;
pub const ProfileResponse = types.ProfileResponse;
pub const ProfileMessage = types.ProfileMessage;
pub const MessageKind = types.MessageKind;
pub const RoutingConfig = types.RoutingConfig;
pub const ProfileError = types.ProfileError;

pub const ProfileRegistry = registry.ProfileRegistry;
pub const ProfileInstance = registry.ProfileInstance;
pub const MultiProfileConfig = registry.MultiProfileConfig;

pub const MultiProfileRouter = router.MultiProfileRouter;
pub const ProfileBus = bus.ProfileBus;
pub const ConversationMemory = memory.ConversationMemory;

test {
    std.testing.refAllDecls(@This());
}
