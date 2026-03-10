//! Provider-scoped connector stubs.
//!
//! Mirrors the public API surface of each real connector but returns
//! `error.ConnectorsDisabled` for all operations. Used when connectors
//! are compiled out or for type-checking without network dependencies.
const std = @import("std");

pub const openai = @import("openai");
pub const codex = @import("codex");
pub const opencode = @import("opencode");
pub const claude = @import("claude");
pub const gemini = @import("gemini");
pub const anthropic = @import("anthropic");
pub const ollama = @import("ollama");
pub const ollama_passthrough = @import("ollama_passthrough");
pub const huggingface = @import("huggingface");
pub const mistral = @import("mistral");
pub const cohere = @import("cohere");
pub const lm_studio = @import("lm_studio");
pub const vllm = @import("vllm");
pub const mlx = @import("mlx");
pub const llama_cpp = @import("llama_cpp");
pub const discord = @import("discord");
pub const local_scheduler = @import("local_scheduler");
pub const contract = @import("contract");

test {
    std.testing.refAllDecls(@This());
}
