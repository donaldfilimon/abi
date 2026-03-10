// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  LLM Orchestrator — Multi-Backend Inference & Skill Synchronization        ║
// ║  Zig 0.16                                                                   ║
// ║                                                                            ║
// ║  Unified abstraction over: Claude CLI, Codex CLI, Open Code, Ollama, MLX   ║
// ║  with cross-frontend plugin/skill sync backed by WDBX vector storage.      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

const std = @import("std");

pub const types = @import("protocol/types");
pub const backend = @import("backends/backend");
pub const claude = @import("backends/claude");
pub const codex = @import("backends/codex");
pub const ollama = @import("backends/ollama");
pub const mlx = @import("backends/mlx");
pub const opencode = @import("backends/opencode");
pub const skill_registry = @import("skills/registry");
pub const sync_manager = @import("sync/manager");
pub const orchestrator = @import("orchestrator");
pub const http = @import("utils/http");
pub const json_util = @import("utils/json");
pub const process_util = @import("utils/process");

// ── Re-exports for convenience ───────────────────────────────────────────────

pub const Orchestrator = orchestrator.Orchestrator;
pub const BackendType = types.BackendType;
pub const BackendConfig = types.BackendConfig;
pub const InferenceRequest = types.InferenceRequest;
pub const InferenceResponse = types.InferenceResponse;
pub const Skill = types.Skill;
pub const SkillVisibility = types.SkillVisibility;
pub const SyncEvent = types.SyncEvent;
pub const StreamCallback = types.StreamCallback;
pub const BackendInterface = backend.BackendInterface;
pub const SkillRegistry = skill_registry.SkillRegistry;
pub const SyncManager = sync_manager.SyncManager;

/// Create a default orchestrator with all backends pre-registered.
pub fn createDefaultOrchestrator(allocator: std.mem.Allocator) !Orchestrator {
    return Orchestrator.init(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
