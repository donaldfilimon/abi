---
title: "README"
tags: []
---
//! # AI
//!
//! > **Codebase Status:** Synced with repository as of 2026-01-31.
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
//! | `llm/` | Local LLM inference (GGUF), tokenizers, sampling |
//! | `embeddings/` | Embedding generation |
//! | `agents/` | Agent runtime and tool use |
//! | `training/` | Training pipelines (LLM, vision, multimodal) |
//! | `streaming/` | SSE/WebSocket streaming server |
//! | `orchestration/` | Multi-model routing, ensemble, fallback |
//! | `rag/` | Retrieval-augmented generation |
//! | `documents/` | Document parsing, segmentation, entities |
//! | `memory/` | Agent memory + persistence |
//! | `models/` | Model management and downloads |
//! | `explore/` | Codebase exploration |
//!
//! ## Architecture
//!
//! This module is a primary feature module. The implementation lives directly in
//! `src/ai/` with a feature-gated stub in `src/ai/stub.zig` for disabled builds.
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
//! - `-Denable-vision=true` - Vision processing (requires `-Denable-ai`)
//! - `-Denable-explore=true` - Codebase exploration (requires `-Denable-ai`)
//!
//! ## See Also
//!
//! - [AI Documentation](../../docs/content/ai.html)
//! - [API Reference](../../docs/content/api.html)
//! - [Training Guide](../../docs/content/ai.html)


