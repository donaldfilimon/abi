//! # Features Module
//!
//! Optional capabilities toggled via build flags. Each feature provides a `mod.zig` for public API.
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
//! | Directory | Description |
//! |-----------|-------------|
//! | `ai/` | AI features (LLM, embeddings, RAG, explore, streaming) |
//! | `connectors/` | API connectors (OpenAI, HuggingFace, Ollama) |
//! | `database/` | WDBX vector database with HNSW |
//! | `gpu/` | GPU backend stubs and feature detection |
//! | `monitoring/` | Observability and metrics |
//! | `network/` | Network features (discovery, HA, circuit breaker) |
//! | `web/` | Web utilities and HTTP helpers |
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
//! ```zig
//! const abi = @import("abi");
//!
//! // AI (if enabled)
//! const response = try abi.ai.chat("Hello!");
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

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
[Main Workspace](MAIN_WORKSPACE.md)
