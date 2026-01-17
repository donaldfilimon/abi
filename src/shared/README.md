//! # Shared Utilities
//!
//! Cross-cutting concerns used throughout the ABI framework.
//! These utilities are re-exported via `src/internal/` for cleaner imports.
//!
//! ## Architecture
//!
//! The shared module provides foundational utilities that are re-exported by:
//!
//! - `src/internal/` - Re-exports all shared utilities for internal framework use
//!
//! ```zig
//! // Instead of:
//! const logging = @import("shared/logging/mod.zig");
//!
//! // Use:
//! const internal = @import("internal/mod.zig");
//! internal.logging.info("message", .{});
//! ```
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `logging/` | Structured, leveled logging with backends |
//! | `observability/` | Metrics, tracing, and diagnostics |
//! | `platform/` | OS-specific abstractions (threads, files) |
//! | `plugins/` | Plugin registration and discovery |
//! | `security/` | API keys, authentication |
//! | `simd.zig` | SIMD vector operations |
//! | `utils/` | General utilities (crypto, encoding, fs, http, json, math, net, string) |
//!
//! ## Usage
//!
//! **Via internal module (preferred for framework code)**
//!
//! ```zig
//! const internal = @import("internal/mod.zig");
//!
//! // Logging
//! internal.logging.info("Starting service", .{});
//!
//! // Platform
//! const cpu_count = internal.platform.getCpuCount();
//!
//! // Utils
//! const hash = internal.utils.crypto.sha256(data);
//!
//! // SIMD
//! const dot = internal.vectorDot(a, b);
//! ```
//!
//! **Via abi public API**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Logging
//! abi.shared.logging.info("Starting service", .{});
//!
//! // Platform
//! const cpu_count = abi.shared.platform.getCpuCount();
//!
//! // Utils
//! const hash = abi.shared.utils.crypto.sha256(data);
//! const json = try abi.shared.utils.json.parse(allocator, text);
//! ```
//!
//! ## Import Pattern
//!
//! Each subdirectory has a `mod.zig` that re-exports its symbols.
//! The top-level `src/shared/mod.zig` aggregates all sub-modules.
//! The `src/internal/mod.zig` re-exports from shared/ with convenience aliases.
//!
//! ## Convenience Exports in internal/
//!
//! The internal module provides shorthand access to commonly used utilities:
//!
//! ```zig
//! const internal = @import("internal/mod.zig");
//!
//! // Direct access to common types
//! internal.Logger       // -> logging.Logger
//! internal.LogLevel     // -> logging.LogLevel
//! internal.Platform     // -> platform.Platform
//!
//! // SIMD operations
//! internal.vectorAdd()
//! internal.vectorDot()
//! internal.cosineSimilarity()
//!
//! // Utility shortcuts
//! internal.config       // -> utils.config
//! internal.memory       // -> utils.memory
//! internal.time         // -> utils.time
//! internal.retry        // -> utils.retry
//! ```
//!
//! ## See Also
//!
//! - [src/internal/](../internal/) - Re-export module with convenience aliases
//! - [Logging](logging/README.md)
//! - [Platform](platform/README.md)
//! - [Utils](utils/README.md)
