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
//! | `llm/` | LLM inference (local GGUF, tokenizers, sampling) |
//! | `embeddings/` | Embedding generation |
//! | `agents/` | Autonomous agent system |
//! | `training/` | Training pipelines and trainers |
//! | `personas/` | Multi-persona assistant layer |
//! | `orchestration/` | Multi-model routing, fallback, and ensembles |
//! | `multi_agent/` | Multi-agent coordination |
//! | `rag/` | Retrieval-augmented generation pipeline |
//! | `templates/` | Prompt template management |
//! | `eval/` | Evaluation and metrics |
//! | `explore/` | Codebase exploration (feature gated) |
//! | `models/` | Model registry and download utilities |
//! | `streaming/` | Streaming generation + server |
//! | `memory/` | Memory stores and summarization |
//! | `documents/` | Document understanding pipeline |
//! | `vision/` | Vision/image processing |
//! | `tools/` | Tool registry and built-in tools |
//! | `prompts/` | Prompt builders and persona formats |
//! | `abbey/` | Abbey persona subsystem |
//! | `federated/` | Federated coordination and training |
//! | `discovery.zig` | Model discovery and adaptive configuration |
//! | `gpu_agent.zig` | GPU-aware agent scheduling |
//!
//! ## Architecture
//!
//! AI functionality lives directly under `src/ai/` (no indirection through a
//! legacy `features/` tree). The module exposes a stable public API in `mod.zig`
//! and organizes implementation by sub-feature:
//!
//! ```
//! src/ai/
//! ├── mod.zig          - Public API entry point + Context
//! ├── core/            - Shared AI types and config
//! ├── llm/             - LLM inference engine
//! ├── embeddings/      - Embedding generation
//! ├── agents/          - Agent runtime
//! ├── training/        - Training pipelines
//! ├── personas/        - Multi-persona assistant system
//! ├── orchestration/   - Multi-model routing and fallback
//! ├── rag/             - Retrieval-augmented generation
//! └── streaming/       - Streaming output and server helpers
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
//! | Ollama | Connect to Ollama server | `ABI_OLLAMA_HOST` env |
//! | OpenAI | OpenAI API | `ABI_OPENAI_API_KEY` env |
//! | HuggingFace | HuggingFace API | `ABI_HF_API_TOKEN` env |
//!
//! ## Build Options
//!
//! Enable with `-Denable-ai=true` (default: true).
//!
//! Sub-features:
//! - `-Denable-llm=true` - LLM inference (requires `-Denable-ai`)
//! - `-Denable-vision=true` - Vision/image processing (requires `-Denable-ai`)
//! - `-Denable-explore=true` - Codebase exploration (requires `-Denable-ai`)
//!
//! ## See Also
//!
//! - [AI Documentation](../../docs/ai.md)
//! - [API Reference](../../docs/ai.md#api-reference)
//! - [Training Guide](../../docs/ai.md#training)


