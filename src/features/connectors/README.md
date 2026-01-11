//! # Connectors
//!
//! Integration points to external AI services and platforms.
//!
//! ## Supported Connectors
//!
//! | Connector | Description | Environment Variable |
//! |-----------|-------------|---------------------|
//! | OpenAI | GPT-4, GPT-3.5, embeddings | `ABI_OPENAI_API_KEY` |
//! | Ollama | Local LLM inference | `ABI_OLLAMA_HOST` |
//! | HuggingFace | Inference API | `ABI_HF_API_TOKEN` |
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
//! ```
//!
//! ## Sub-modules
//!
//! - `openai.zig` - OpenAI API client
//! - `ollama.zig` - Ollama local inference
//! - `huggingface.zig` - HuggingFace Inference API
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
//! - [Agent Documentation](../../../docs/ai.md)
