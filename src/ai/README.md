//! # AI
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
//! | `llm/` | LLM inference (local GGUF, Ollama, OpenAI) |
//! | `agents/` | Autonomous agent system |
//! | `embeddings/` | Embedding generation |
//! | `training/` | Training pipelines |
//!
//! ## Architecture
//!
//! This module uses the wrapper pattern - thin wrappers in `src/ai/` delegate
//! to full implementations in `src/features/ai/`:
//!
//! ```
//! src/ai/mod.zig (wrapper)
//!        ↓
//! src/features/ai/mod.zig (implementation)
//!        ↓
//! ├── agent.zig    - Agent runtime
//! ├── llm/         - LLM connectors
//! ├── embeddings/  - Embedding models
//! └── training/    - Training loops
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
//! - [API Reference](../../docs/api_ai.md)
//! - [Training Guide](../../docs/ai.md#training)

