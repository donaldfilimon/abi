//! # Features Module
//!
//! Implementation layer for optional features. Each feature provides the underlying
//! logic that is re-exported by top-level modules (src/ai/, src/database/, etc.)
//! with added Framework integration.
//!
//! ## Architecture
//!
//! Features are now accessible via two paths:
//!
//! 1. **Top-level modules** (preferred): `src/gpu/`, `src/ai/`, `src/database/`, etc.
//!    These re-export from features/ while adding Context structs for Framework integration.
//!
//! 2. **Direct access** (implementation details): `src/features/ai/`, `src/features/database/`, etc.
//!    Use these when you need implementation-level access or are extending the framework.
//!
//! ```
//! src/ai/mod.zig         ->  re-exports from  ->  src/features/ai/mod.zig
//! src/database/mod.zig   ->  primary implementation (no longer in features/)
//! src/network/mod.zig    ->  primary implementation (no longer in features/)
//! src/web/mod.zig        ->  primary implementation (no longer in features/)
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Flag | Default | Description |
//! |---------|------|---------|-------------|
//! | AI | `-Denable-ai` | true | LLM inference, embeddings, RAG, connectors |
//! | Database | `-Denable-database` | true | WDBX vector database, HNSW indexing |
//! | GPU | `-Denable-gpu` | true | GPU compute backends, memory pools |
//! | Monitoring | `-Denable-profiling` | true | Metrics, tracing, OpenTelemetry |
//! | Network | `-Denable-network` | true | Distributed compute, node discovery |
//! | Web | `-Denable-web` | true | HTTP utilities, server helpers |
//!
//! ## Sub-modules
//!
//! | Directory | Top-Level Module | Description |
//! |-----------|------------------|-------------|
//! | `ai/` | `src/ai/` | AI features (LLM, embeddings, RAG, explore, streaming) |
//! | `connectors/` | (via ai) | API connectors (OpenAI, HuggingFace, Ollama) |
//! | `ha/` | (internal) | High availability components |
//! | `monitoring/` | `src/observability/` | Observability and metrics |
//!
//! ## Stub Modules
//!
//! When a feature is disabled, stub modules return `error.*Disabled`:
//!
//! ```zig
//! const impl = if (build_options.enable_ai)
//!     @import("ai/mod.zig")
//! else
//!     @import("ai/stub.zig");
//! ```
//!
//! ## Usage
//!
//! **Preferred: Using top-level modules with Framework**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with configuration
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .vulkan })
//!     .withAi(.{ .llm = .{} })
//!     .withDatabase(.{ .path = "./data" })
//!     .build();
//! defer fw.deinit();
//!
//! // Access features via Framework
//! const gpu_ctx = try fw.getGpu();
//! const ai_ctx = try fw.getAi();
//! const db_ctx = try fw.getDatabase();
//! ```
//!
//! **Direct access (legacy/advanced)**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // AI (if enabled)
//! const response = try abi.ai.inferText(allocator, "Hello!");
//!
//! // Database (if enabled)
//! var db = try abi.wdbx.createDatabase(allocator, "vectors.db", .{});
//! defer abi.wdbx.closeDatabase(db);
//! ```
//!
//! ## See Also
//!
//! - [Build Documentation](../../README.md)
//! - [API Reference](../../API_REFERENCE.md)
//! - [src/README.md](../README.md) - Source overview with module hierarchy
