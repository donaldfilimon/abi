//! Connector configuration loaders and auth helpers.
//!
//! This module provides unified access to various AI service connectors including:
//!
//! - **OpenAI**: GPT models via the Chat Completions API
//! - **Anthropic**: Claude models via the Messages API
//! - **Ollama**: Local LLM inference server
//! - **HuggingFace**: Hosted inference API
//! - **Mistral**: Mistral AI models with OpenAI-compatible API
//! - **Cohere**: Chat, embeddings, and reranking
//! - **LM Studio**: Local LLM inference with OpenAI-compatible API
//! - **vLLM**: High-throughput local LLM serving with OpenAI-compatible API
//! - **MLX**: Apple Silicon-optimized inference via mlx-lm server
//! - **Discord**: Bot integration for Discord
//!
//! ## Usage
//!
//! Each connector can be loaded from environment variables:
//!
//! ```zig
//! const connectors = @import("abi").connectors;
//!
//! // Load and create clients
//! if (try connectors.tryLoadOpenAI(allocator)) |config| {
//!     var client = try connectors.openai.Client.init(allocator, config);
//!     defer client.deinit();
//!     // Use client...
//! }
//! ```
//!
//! ## Security
//!
//! All connectors securely wipe API keys from memory using `std.crypto.secureZero`
//! before freeing to prevent memory forensics attacks.

const std = @import("std");

pub const openai = @import("openai.zig");
pub const codex = @import("codex.zig");
pub const opencode = @import("opencode.zig");
pub const claude = @import("claude.zig");
pub const gemini = @import("gemini.zig");
pub const huggingface = @import("huggingface.zig");
pub const ollama = @import("ollama.zig");
pub const ollama_passthrough = @import("ollama_passthrough.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const discord = @import("discord/mod.zig");
pub const anthropic = @import("anthropic.zig");
pub const mistral = @import("mistral.zig");
pub const cohere = @import("cohere.zig");
pub const lm_studio = @import("lm_studio.zig");
pub const vllm = @import("vllm.zig");
pub const mlx = @import("mlx.zig");
pub const llama_cpp = @import("llama_cpp.zig");
pub const shared = @import("shared.zig");
pub const auth = @import("auth.zig");
pub const env = @import("env.zig");
pub const providers = @import("providers.zig");
pub const loaders = @import("loaders.zig");

pub const AuthHeader = auth.AuthHeader;
pub const buildBearerHeader = auth.buildBearerHeader;
pub const getEnvOwned = env.getEnvOwned;
pub const getFirstEnvOwned = env.getFirstEnvOwned;
pub const ENV_VARS = env.ENV_VARS;
pub const ProviderInfo = providers.ProviderInfo;
pub const ProviderRegistry = providers.ProviderRegistry;
pub const loadOpenAI = loaders.loadOpenAI;
pub const tryLoadOpenAI = loaders.tryLoadOpenAI;
pub const loadCodex = loaders.loadCodex;
pub const tryLoadCodex = loaders.tryLoadCodex;
pub const loadOpenCode = loaders.loadOpenCode;
pub const tryLoadOpenCode = loaders.tryLoadOpenCode;
pub const loadClaude = loaders.loadClaude;
pub const tryLoadClaude = loaders.tryLoadClaude;
pub const loadGemini = loaders.loadGemini;
pub const tryLoadGemini = loaders.tryLoadGemini;
pub const loadHuggingFace = loaders.loadHuggingFace;
pub const tryLoadHuggingFace = loaders.tryLoadHuggingFace;
pub const loadOllama = loaders.loadOllama;
pub const tryLoadOllama = loaders.tryLoadOllama;
pub const loadOllamaPassthrough = loaders.loadOllamaPassthrough;
pub const tryLoadOllamaPassthrough = loaders.tryLoadOllamaPassthrough;
pub const loadLocalScheduler = loaders.loadLocalScheduler;
pub const loadDiscord = loaders.loadDiscord;
pub const tryLoadDiscord = loaders.tryLoadDiscord;
pub const loadAnthropic = loaders.loadAnthropic;
pub const tryLoadAnthropic = loaders.tryLoadAnthropic;
pub const loadMistral = loaders.loadMistral;
pub const tryLoadMistral = loaders.tryLoadMistral;
pub const loadCohere = loaders.loadCohere;
pub const tryLoadCohere = loaders.tryLoadCohere;
pub const loadLMStudio = loaders.loadLMStudio;
pub const tryLoadLMStudio = loaders.tryLoadLMStudio;
pub const loadVLLM = loaders.loadVLLM;
pub const tryLoadVLLM = loaders.tryLoadVLLM;
pub const loadMLX = loaders.loadMLX;
pub const tryLoadMLX = loaders.tryLoadMLX;
pub const loadLlamaCpp = loaders.loadLlamaCpp;
pub const tryLoadLlamaCpp = loaders.tryLoadLlamaCpp;

var initialized = std.atomic.Value(bool).init(false);

/// Initialize the connectors subsystem (idempotent; no-op if already initialized).
pub fn init(_: std.mem.Allocator) !void {
    if (initialized.load(.acquire)) return;
    initialized.store(true, .release);
}

/// Tear down the connectors subsystem; safe to call multiple times.
pub fn deinit() void {
    initialized.store(false, .release);
}

/// Returns true; connectors are always available when this module is compiled in.
pub fn isEnabled() bool {
    return true;
}

/// Returns true after `init()` has been called.
pub fn isInitialized() bool {
    return initialized.load(.acquire);
}

test "connectors init toggles state" {
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "isEnabled always returns true" {
    try std.testing.expect(isEnabled());
}

test {
    std.testing.refAllDecls(@This());
}
