//! # Connectors
//!
//! Integration points to external AI services, platforms, and communication APIs.
//!
//! ## Supported Connectors
//!
//! | Connector | Description | Environment Variable |
//! |-----------|-------------|---------------------|
//! | OpenAI | GPT-4, GPT-3.5, embeddings | `ABI_OPENAI_API_KEY` |
//! | Codex | OpenAI-compatible Codex endpoint | `ABI_CODEX_API_KEY` |
//! | OpenCode | OpenAI-compatible OpenCode endpoint | `ABI_OPENCODE_API_KEY` |
//! | Claude | Claude API with Anthropic fallback envs | `ABI_CLAUDE_API_KEY` |
//! | Gemini | Google Gemini native API | `ABI_GEMINI_API_KEY` |
//! | Anthropic | Claude models | `ABI_ANTHROPIC_API_KEY` |
//! | Ollama | Local LLM inference | `ABI_OLLAMA_HOST` |
//! | Ollama Passthrough | OpenAI-compatible Ollama passthrough | `ABI_OLLAMA_PASSTHROUGH_URL` |
//! | HuggingFace | Inference API | `ABI_HF_API_TOKEN` |
//! | Mistral | Mistral AI models | `ABI_MISTRAL_API_KEY` |
//! | Cohere | Cohere models | `ABI_COHERE_API_KEY` |
//! | LM Studio | Local server (OpenAI-compat) | `ABI_LM_STUDIO_HOST` |
//! | vLLM | Local server (OpenAI-compat) | `ABI_VLLM_HOST` |
//! | MLX | Apple MLX inference | `ABI_MLX_HOST` |
//! | Discord | Discord Bot API | `DISCORD_BOT_TOKEN` |
//! | Local Scheduler | Local task scheduling | `ABI_LOCAL_SCHEDULER_URL` |
//!
//! ## Usage
//!
//! ```zig
//! const connectors = @import("abi").connectors;
//!
//! // OpenAI
//! var openai = try connectors.openai.Client.init(allocator);
//! defer openai.deinit();
//! const response = try openai.chat(.{ .model = "gpt-4", .messages = &.{...} });
//!
//! // Ollama (local)
//! var ollama = try connectors.ollama.Client.init(allocator);
//! defer ollama.deinit();
//! const result = try ollama.generate(.{ .model = "llama3.2", .prompt = "Hello" });
//!
//! // Discord
//! var discord = try connectors.discord.createClient(allocator);
//! defer discord.deinit();
//! const user = try discord.getCurrentUser();
//! const guilds = try discord.getCurrentUserGuilds();
//! ```
//!
//! ## Sub-modules
//!
//! - `openai.zig` - OpenAI API client
//! - `codex.zig` - Codex OpenAI-compatible client
//! - `opencode.zig` - OpenCode OpenAI-compatible client
//! - `claude.zig` - Claude API client
//! - `gemini.zig` - Gemini API client
//! - `anthropic.zig` - Anthropic/Claude API client
//! - `ollama.zig` - Ollama local inference
//! - `ollama_passthrough.zig` - Ollama OpenAI-compatible passthrough
//! - `huggingface.zig` - HuggingFace Inference API
//! - `mistral.zig` - Mistral AI client
//! - `cohere.zig` - Cohere API client
//! - `lm_studio.zig` - LM Studio local server (OpenAI-compatible)
//! - `vllm.zig` - vLLM local server (OpenAI-compatible)
//! - `mlx.zig` - Apple MLX inference
//! - `discord/` - Discord Bot API (REST, webhooks, interactions)
//! - `local_scheduler.zig` - Local task scheduler
//! - `shared.zig` - Shared types and discovery
//!
//! ## Adding a Connector
//!
//! 1. Implement a struct with request/response methods
//! 2. Register it in `shared.zig` for discovery
//! 3. Add tests in `../tests`
//!
//! ## See Also
//!
//! - [AI Module](../ai/README.md)
//! - [Agent Documentation](../../docs/_docs/ai-overview.md)


## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
