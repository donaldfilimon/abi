//! # Source Directory
//!
//! Core source modules of the ABI framework organized by function.
//!
//! ## Structure
//!
//! | Directory | Description |
//! |-----------|-------------|
//! | `abi.zig` | Public API entry point with curated re-exports |
//! | `root.zig` | Root module entrypoint |
//! | `core/` | Core infrastructure, hardware helpers, profiling |
//! | `compute/` | Compute engine, concurrency, memory, GPU integration |
//! | `features/` | Optional features (AI, database, GPU, network, web) |
//! | `framework/` | Lifecycle management, feature orchestration |
//! | `shared/` | Cross-cutting utilities (logging, platform, utils) |
//! | `tests/` | Test utilities and property-based testing |
//!
//! ## Module Hierarchy
//!
//! ```
//! src/
//! ├── abi.zig              # Public API
//! ├── root.zig             # Root module
//! ├── core/                # Core infrastructure
//! ├── compute/
//! │   ├── runtime/         # Engine, scheduler, NUMA
//! │   ├── concurrency/     # Lock-free data structures
//! │   ├── memory/          # Arena allocators, pools
//! │   ├── gpu/             # GPU backends
//! │   ├── network/         # Distributed compute
//! │   └── profiling/       # Metrics collection
//! ├── features/
//! │   ├── ai/              # AI (LLM, embeddings, RAG)
//! │   ├── connectors/      # API connectors
//! │   ├── database/        # WDBX vector database
//! │   ├── gpu/             # GPU feature stubs
//! │   ├── monitoring/      # Observability
//! │   ├── network/         # Network features
//! │   └── web/             # Web utilities
//! ├── framework/           # Orchestration layer
//! ├── shared/
//! │   ├── logging/         # Logging infrastructure
//! │   ├── observability/   # Tracing, metrics
//! │   ├── platform/        # OS abstractions
//! │   ├── plugins/         # Plugin system
//! │   ├── security/        # API keys, auth
//! │   └── utils/           # General utilities
//! └── tests/               # Test utilities
//! ```
//!
//! ## Key Entry Points
//!
//! - **Public API**: `abi.zig` - Use `abi.init()`, `abi.shutdown()`, `abi.version()`
//! - **Compute**: `compute/mod.zig` - Work-stealing scheduler, GPU integration
//! - **Features**: `features/*/mod.zig` - Feature-specific APIs
//!
//! ## See Also
//!
//! - [CLAUDE.md](../CLAUDE.md) - Full project documentation
//! - [API Reference](../API_REFERENCE.md)
//! - [docs/intro.md](../docs/intro.md) - Architecture overview

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
