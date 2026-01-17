//! # Framework Module
//!
//! High-level orchestration layer for lifecycle management and feature coordination.
//!
//! ## Features
//!
//! - **Lifecycle Management**: Init/shutdown coordination
//! - **Feature Orchestration**: Build-time feature flag handling
//! - **Runtime Configuration**: Dynamic configuration options
//! - **Plugin System**: Extensible plugin architecture
//!
//! ## Usage
//!
//! ### Initialization
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var framework = try abi.init(allocator, abi.FrameworkOptions{
//!     .log_level = .info,
//!     .enable_telemetry = true,
//! });
//! defer abi.shutdown(&framework);
//!
//! std.debug.print("ABI version: {s}\n", .{abi.version()});
//! ```
//!
//! ### Configuration Options
//!
//! | Option | Type | Default | Description |
//! |--------|------|---------|-------------|
//! | `log_level` | LogLevel | `.info` | Logging verbosity |
//! | `enable_telemetry` | bool | `false` | Enable metrics collection |
//! | `worker_threads` | ?u32 | `null` | Worker thread count (auto-detect if null) |
//!
//! ## Feature Flags
//!
//! Features are enabled at build time and coordinated by the framework:
//!
//! - `-Denable-ai` - AI features
//! - `-Denable-gpu` - GPU acceleration
//! - `-Denable-database` - Vector database
//! - `-Denable-network` - Distributed compute
//! - `-Denable-profiling` - Metrics collection
//!
//! ## See Also
//!
//! - [Framework Documentation](../../docs/framework.md)
//! - [API Reference](../../API_REFERENCE.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
[Main Workspace](MAIN_WORKSPACE.md)
