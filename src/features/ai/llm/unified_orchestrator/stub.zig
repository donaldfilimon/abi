//! Unified orchestrator stub — disabled when LLM feature is off.
//! Mirrors public API of unified_orchestrator/mod.zig for stub parity.

const std = @import("std");

pub const Error = error{FeatureDisabled};

// Stub submodule namespaces (same names as mod.zig; empty so no real deps)
pub const types = struct {};
pub const backend = struct {};
pub const claude = struct {};
pub const codex = struct {};
pub const ollama = struct {};
pub const mlx = struct {};
pub const opencode = struct {};
pub const skill_registry = struct {};
pub const sync_manager = struct {};
pub const orchestrator = struct {};
pub const http = struct {};
pub const json_util = struct {};
pub const process_util = struct {};

// Re-exports for convenience (minimal types so API surface matches)
pub const Orchestrator = struct {};
pub const BackendType = enum(u8) { placeholder = 0 };
pub const BackendConfig = struct {};
pub const InferenceRequest = struct {};
pub const InferenceResponse = struct {};
pub const Skill = struct {};
pub const SkillVisibility = enum(u8) { placeholder = 0 };
pub const SyncEvent = struct {};
pub const StreamCallback = struct {};
pub const BackendInterface = struct {};
pub const SkillRegistry = struct {};
pub const SyncManager = struct {};

/// Stub: returns FeatureDisabled when LLM is disabled.
pub fn createDefaultOrchestrator(allocator: std.mem.Allocator) Error!Orchestrator {
    _ = allocator;
    return error.FeatureDisabled;
}
