// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  LLM Orchestrator — Multi-Backend Inference & Skill Synchronization        ║
// ║  Zig 0.16                                                                   ║
// ║                                                                            ║
// ║  Unified abstraction over: Claude CLI, Codex CLI, Open Code, Ollama, MLX   ║
// ║  with cross-frontend plugin/skill sync backed by WDBX vector storage.      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

const std = @import("std");

pub const types = @import("protocol/types.zig");
pub const backend = @import("backends/backend.zig");
pub const claude = @import("backends/claude.zig");
pub const codex = @import("backends/codex.zig");
pub const ollama = @import("backends/ollama.zig");
pub const mlx = @import("backends/mlx.zig");
pub const opencode = @import("backends/opencode.zig");
pub const skill_registry = @import("skills/registry.zig");
pub const sync_manager = @import("sync/manager.zig");
pub const orchestrator = @import("orchestrator.zig");
pub const http = @import("utils/http.zig");
pub const json_util = @import("utils/json.zig");
pub const process_util = @import("utils/process.zig");

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
