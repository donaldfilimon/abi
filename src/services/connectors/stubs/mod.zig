//! Provider-scoped connector stubs.
//!
//! Mirrors the public API surface of each real connector but returns
//! `error.ConnectorsDisabled` for all operations. Used when connectors
//! are compiled out or for type-checking without network dependencies.

pub const openai = @import("openai.zig");
pub const codex = @import("codex.zig");
pub const opencode = @import("opencode.zig");
pub const claude = @import("claude.zig");
pub const gemini = @import("gemini.zig");
pub const anthropic = @import("anthropic.zig");
pub const ollama = @import("ollama.zig");
pub const ollama_passthrough = @import("ollama_passthrough.zig");
pub const huggingface = @import("huggingface.zig");
pub const mistral = @import("mistral.zig");
pub const cohere = @import("cohere.zig");
pub const lm_studio = @import("lm_studio.zig");
pub const vllm = @import("vllm.zig");
pub const mlx = @import("mlx.zig");
pub const llama_cpp = @import("llama_cpp.zig");
pub const discord = @import("discord.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const contract = @import("contract.zig");
