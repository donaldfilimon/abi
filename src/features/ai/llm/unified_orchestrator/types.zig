//! Protocol types for the LLM unified orchestrator.

const std = @import("std");

pub const BackendType = enum {
    claude,
    codex,
    ollama,
    mlx,
    opencode,
};

pub const BackendConfig = struct {
    backend_type: BackendType,
    endpoint: []const u8 = "",
    api_key: []const u8 = "",
};

pub const InferenceRequest = struct {
    /// Model identifier (e.g. "gpt-4", "claude-3")
    model: []const u8 = "",
    /// System prompt (optional)
    system: []const u8 = "",
    /// User prompt / messages payload
    prompt: []const u8 = "",
    /// Max tokens to generate
    max_tokens: u32 = 4096,
    /// Temperature (0.0 = deterministic)
    temperature: f32 = 0.7,
};

pub const InferenceResponse = struct {
    /// Generated text
    text: []const u8 = "",
    /// Tokens used (if reported)
    tokens_used: u32 = 0,
    /// Finish reason (e.g. "stop", "length")
    finish_reason: []const u8 = "",
};

pub const SkillVisibility = enum {
    /// Visible only to this frontend
    private,
    /// Synced to WDBX and visible to linked frontends
    shared,
};

pub const Skill = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    content: []const u8 = "",
    visibility: SkillVisibility = .private,
};

pub const SyncEvent = struct {
    kind: enum { skill_created, skill_updated, skill_deleted } = .skill_created,
    skill_id: []const u8 = "",
    payload: []const u8 = "",
};

/// Callback for streaming token delivery.
pub const StreamCallback = *const fn (chunk: []const u8, done: bool) void;
