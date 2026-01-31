---
title: "README"
tags: []
---
//! # AI
//!
//! > **Codebase Status:** Synced with repository as of 2026-01-30.
//!
//! AI module providing LLM inference, agents, embeddings, and training capabilities.
//!
//! ## Features
//!
//! - **LLM Inference**: Local and API-based language model inference
//! - **Agents**: Autonomous AI agents with tool use and memory
//! - **Embeddings**: Vector embedding generation for semantic search
//! - **Training**: Model fine-tuning and training pipelines
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Public API entry point with `Context` struct |
//! | `stub.zig` | Feature-disabled stubs for API parity |
//! | `core/` | Shared AI types and configuration |
//! | `llm/` | Local LLM inference (GGUF, tokenization, sampling) |
//! | `embeddings/` | Embedding generation |
//! | `agents/` | Autonomous agent runtime |
//! | `personas/` | Multi-persona assistant system |
//! | `orchestration/` | Multi-model routing and ensemble logic |
//! | `streaming/` | SSE/WebSocket streaming inference |
//! | `models/` | Model management + downloads |
//! | `documents/` | Document parsing + understanding |
//! | `rag/` | Retrieval-augmented generation pipeline |
//! | `templates/` | Prompt/template rendering |
//! | `explore/` | Codebase exploration tooling |
//! | `memory/` | Conversation memory systems |
//! | `prompts/` | Prompt builders and persona prompts |
//! | `tools/` | Agent tool registry and helpers |
//! | `vision/` | Vision processing and ViT training |
//!
//! ## Architecture
//!
//! The AI module is implemented directly in `src/ai/` (no `features/ai` bridge).
//! Feature gating is handled via `stub.zig` and submodule-specific stubs:
//!
//! ```
//! src/ai/
//! ├── mod.zig          # Public API + Context
//! ├── stub.zig         # Feature-disabled stubs
//! ├── llm/             # Local GGUF inference
//! ├── agents/          # Agent runtime
//! ├── embeddings/      # Embedding generation
//! ├── training/        # Training pipelines
//! ├── streaming/       # SSE/WebSocket streaming
//! └── orchestration/   # Multi-model routing
//! ```
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with AI
//! var fw = try abi.init(allocator, .{
//!     .ai = .{
//!         .llm = .{ .model_path = "./models/llama.gguf" },
//!         .embeddings = .{},
//!     },
//! });
//! defer fw.deinit();
//!
//! // Get AI context
//! const ai = try fw.getAi();
//!
//! // Generate text
//! const response = try ai.llm.generate("Hello, world!", .{});
//! defer allocator.free(response);
//! ```
//!
//! ## LLM Connectors
//!
//! | Connector | Description | Config |
//! |-----------|-------------|--------|
//! | Local GGUF | Load local GGUF models | `model_path` |
//! | Ollama | Connect to Ollama server | `OLLAMA_HOST` env |
//! | OpenAI | OpenAI API | `OPENAI_API_KEY` env |
//! | HuggingFace | HuggingFace API | `HF_API_TOKEN` env |
//!
//! ## Build Options
//!
//! Enable with `-Denable-ai=true` (default: true).
//!
//! Sub-features:
//! - `-Denable-llm=true` - LLM inference (requires `-Denable-ai`)
//! - `-Denable-explore=true` - Codebase exploration (requires `-Denable-ai`)
//!
//! ## See Also
//!
//! - [AI Documentation](../../docs/ai.md)
//! - [API Reference](../../docs/ai.md#api-reference)
//! - [Training Guide](../../docs/ai.md#training)


