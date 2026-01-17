//! # Source Directory
//!
//! Core source modules of the ABI framework organized by function.
//!
//! ## Structure
//!
//! The codebase uses a modular architecture with top-level modules that re-export
//! from their implementation directories while adding Framework integration via
//! Context structs.
//!
//! | Directory | Description |
//! |-----------|-------------|
//! | `abi.zig` | Public API entry point with curated re-exports |
//! | `config.zig` | Unified configuration system (struct literal + builder APIs) |
//! | `framework.zig` | Framework orchestration and lifecycle management |
//! | `runtime/` | Always-on infrastructure (scheduling, concurrency, memory) |
//! | `gpu/` | GPU acceleration (re-exports from `compute/gpu/`) |
//! | `ai/` | AI module with sub-features (llm, embeddings, agents, training) |
//! | `database/` | Vector database (re-exports from `features/database/`) |
//! | `network/` | Distributed compute (re-exports from `features/network/`) |
//! | `observability/` | Metrics, tracing, profiling |
//! | `web/` | Web/HTTP utilities (re-exports from `features/web/`) |
//! | `internal/` | Shared utilities (re-exports from `shared/`) |
//! | `core/` | Core infrastructure, hardware helpers |
//! | `compute/` | Compute engine implementation (runtime, concurrency, gpu, memory) |
//! | `features/` | Feature implementations (ai, database, network, web, monitoring) |
//! | `shared/` | Cross-cutting utilities (logging, platform, utils) |
//! | `tests/` | Test utilities and property-based testing |
//!
//! ## Module Hierarchy
//!
//! ```
//! src/
//! ├── abi.zig              # Public API
//! ├── config.zig           # Unified configuration
//! ├── framework.zig        # Framework orchestration
//! │
//! ├── runtime/             # Always-on infrastructure
//! │   └── mod.zig          # Re-exports from compute/runtime + Context
//! │
//! ├── gpu/                 # GPU acceleration
//! │   ├── mod.zig          # Re-exports from compute/gpu + Context
//! │   └── stub.zig         # Feature-disabled stub
//! │
//! ├── ai/                  # AI module
//! │   ├── mod.zig          # Re-exports from features/ai + Context
//! │   ├── llm/             # LLM inference sub-feature
//! │   ├── embeddings/      # Embeddings generation sub-feature
//! │   ├── agents/          # Agent runtime sub-feature
//! │   └── training/        # Training pipelines sub-feature
//! │
//! ├── database/            # Vector database
//! │   ├── mod.zig          # Re-exports from features/database + Context
//! │   └── stub.zig         # Feature-disabled stub
//! │
//! ├── network/             # Distributed compute
//! │   ├── mod.zig          # Re-exports from features/network + Context
//! │   └── stub.zig         # Feature-disabled stub
//! │
//! ├── observability/       # Metrics and tracing
//! │   ├── mod.zig          # Re-exports + Context
//! │   └── stub.zig         # Feature-disabled stub
//! │
//! ├── web/                 # Web utilities
//! │   ├── mod.zig          # Re-exports from features/web + Context
//! │   └── stub.zig         # Feature-disabled stub
//! │
//! ├── internal/            # Shared internal utilities
//! │   └── mod.zig          # Re-exports from shared/
//! │
//! ├── compute/             # Implementation layer
//! │   ├── runtime/         # Engine, scheduler, NUMA, futures
//! │   ├── concurrency/     # Lock-free data structures
//! │   ├── memory/          # Arena allocators, pools
//! │   ├── gpu/             # GPU backends
//! │   └── profiling/       # Metrics collection
//! │
//! ├── features/            # Feature implementations
//! │   ├── ai/              # AI (LLM, embeddings, RAG)
//! │   ├── connectors/      # API connectors
//! │   ├── database/        # WDBX vector database
//! │   ├── monitoring/      # Observability
//! │   ├── network/         # Network features
//! │   └── web/             # Web utilities
//! │
//! └── shared/              # Cross-cutting concerns
//!     ├── logging/         # Logging infrastructure
//!     ├── observability/   # Tracing, metrics
//!     ├── platform/        # OS abstractions
//!     ├── plugins/         # Plugin system
//!     ├── security/        # API keys, auth
//!     └── utils/           # General utilities
//! ```
//!
//! ## Key Entry Points
//!
//! - **Public API**: `abi.zig` - Use `abi.init()`, `abi.shutdown()`, `abi.version()`
//! - **Configuration**: `config.zig` - Unified `Config` struct with `Builder` API
//! - **Framework**: `framework.zig` - `Framework` struct manages feature lifecycle
//! - **Runtime**: `runtime/mod.zig` - Always-available scheduling and concurrency
//!
//! ## Module Pattern
//!
//! Top-level modules (gpu, ai, database, network, observability, web) follow this pattern:
//!
//! 1. Re-export types and functions from implementation directories
//! 2. Add a `Context` struct for Framework integration
//! 3. Provide `isEnabled()` function for compile-time feature detection
//! 4. Have a corresponding `stub.zig` for when the feature is disabled
//!
//! ```zig
//! // Example: src/gpu/mod.zig
//! const compute_gpu = @import("../compute/gpu/mod.zig");
//!
//! // Re-exports
//! pub const Gpu = compute_gpu.Gpu;
//! pub const Buffer = compute_gpu.GpuBuffer;
//!
//! // Context for Framework integration
//! pub const Context = struct {
//!     allocator: std.mem.Allocator,
//!     config: config_module.GpuConfig,
//!     gpu: ?Gpu = null,
//!
//!     pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context { ... }
//!     pub fn deinit(self: *Context) void { ... }
//! };
//!
//! pub fn isEnabled() bool {
//!     return build_options.enable_gpu;
//! }
//! ```
//!
//! ## See Also
//!
//! - [CLAUDE.md](../CLAUDE.md) - Full project documentation
//! - [API Reference](../API_REFERENCE.md)
//! - [docs/intro.md](../docs/intro.md) - Architecture overview
