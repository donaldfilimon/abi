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
//! | `mod.zig` | Public API entry point with Context struct |
//! | `core/` | Shared AI types, config, and utilities |
//! | `llm/` | Local LLM inference engine (GGUF) |
//! | `embeddings/` | Embedding generation |
//! | `agents/` + `agent.zig` | Agent runtime + simple agent API |
//! | `memory/` | Memory systems (short/long-term, summaries) |
//! | `training/` | Training pipelines and checkpoints |
//! | `personas/` | Multi-persona profiles and configs |
//! | `streaming/` | SSE/WebSocket streaming responses |
//! | `orchestration/` | Multi-model routing, fallback, ensembles |
//! | `rag/` | Retrieval-augmented generation |
//! | `explore/` | Codebase exploration tooling |
//! | `documents/` | Document parsing and layout analysis |
//! | `models/` + `model_registry.zig` | Model registry + downloads |
//! | `vision/` | Vision models (ViT) + preprocessing |
//! | `eval/` | Metrics (BLEU/ROUGE/perplexity) |
//! | `templates/` + `prompts/` | Prompt templates and builders |
//! | `tools/` | Agent tools (filesystem, discord, OS, search) |
//! | `abbey/` | Abbey persona subsystem |
//!
//! ## Architecture
//!
//! `src/ai/mod.zig` is the public API and framework integration layer.
//! Sub-modules live directly under `src/ai/` and are compiled in-place
//! (no legacy wrapper indirection).
//!
//! Feature gating:
//! - `-Denable-ai` toggles the AI module
//! - `-Denable-llm`, `-Denable-vision`, `-Denable-explore` toggle sub-features
//!
//! When a feature is disabled, `src/ai/stub.zig` (and per-submodule stubs
//! like `llm/stub.zig`) provide API-compatible no-op implementations.
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with AI
//! var fw = try abi.initWithConfig(allocator, .{
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
//! - `-Denable-vision=true` - Vision processing (requires `-Denable-ai`)
//! - `-Denable-explore=true` - Codebase exploration (requires `-Denable-ai`)
//!
//! ## See Also
//!
//! - [AI Documentation](../../docs/ai.md)
//! - [API Reference](../../docs/ai.md#api-reference)
//! - [Training Guide](../../docs/ai.md#training)


