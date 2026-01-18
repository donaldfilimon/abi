> **Codebase Status:** Synced with repository as of 2026-01-18.

//! # AI Feature Module
//!
//! Comprehensive AI capabilities including local LLM inference, embeddings,
//! retrieval-augmented generation, and API connectors.
//!
//! ## Features
//!
//! - **Local LLM Inference**: GGUF model loading, tokenization, text generation
//! - **Embeddings**: Vector embeddings with caching
//! - **RAG**: Retrieval-augmented generation pipeline
//! - **Code Exploration**: AST parsing, call graph, dependency analysis
//! - **Streaming**: Token streaming for real-time output
//! - **Memory**: Conversation memory management
//! - **Connectors**: OpenAI, HuggingFace, Ollama integrations
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Public API aggregation |
//! | `agent.zig` | Agent orchestration primitives |
//! | `model_registry.zig` | Runtime model lookup and versioning |
//! | `llm/` | Local LLM inference (GGUF, tokenizer, transformers) |
//! | `embeddings/` | Embedding models and caching |
//! | `rag/` | Retrieval-augmented generation |
//! | `explore/` | Code exploration (AST, callgraph, dependencies) |
//! | `streaming/` | Token streaming |
//! | `memory/` | Conversation memory |
//! | `training/` | Training helpers (checkpointing, gradients) |
//! | `transformer/` | Core transformer building blocks |
//!
//! ## Usage
//!
//! ### Chat Completion
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var connector = try abi.ai.connectors.openai.init(allocator, .{});
//! defer connector.deinit();
//!
//! const response = try connector.chat(.{
//!     .model = "gpt-4",
//!     .messages = &.{
//!         .{ .role = .user, .content = "Hello!" },
//!     },
//! });
//! defer allocator.free(response.content);
//! ```
//!
//! ### Local LLM
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var model = try abi.ai.llm.loadModel(allocator, "model.gguf", .{});
//! defer model.deinit();
//!
//! const output = try model.generate("Once upon a time", .{
//!     .max_tokens = 100,
//!     .temperature = 0.7,
//! });
//! defer allocator.free(output);
//! ```
//!
//! ### Code Exploration
//!
//! ```zig
//! const explore = abi.ai.explore;
//!
//! var agent = explore.ExploreAgent.init(allocator, .medium);
//! defer agent.deinit();
//!
//! const result = try agent.explore("src/", "find HTTP handlers");
//! defer result.deinit();
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Description |
//! |----------|-------------|
//! | `ABI_OPENAI_API_KEY` | OpenAI API key |
//! | `ABI_HF_API_TOKEN` | HuggingFace API token |
//! | `ABI_OLLAMA_HOST` | Ollama server URL (default: `http://127.0.0.1:11434`) |
//!
//! ## Feature Flag
//!
//! Requires `-Denable-ai=true` (default: enabled).
//!
//! When disabled, stub module returns `error.AiDisabled`.
//!
//! ## See Also
//!
//! - [AI Documentation](../../../docs/ai.md)
//! - [Explore Documentation](../../../docs/explore.md)
//! - [Connectors](../connectors/README.md)

