---
title: "README"
tags: []
---
//! # Shared Utilities
//!
//! > **Codebase Status:** Synced with repository as of 2026-01-22.
//!
//! Cross-cutting concerns used throughout the ABI framework.
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
//! **Via abi public API**
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Logging
//! abi.logging.info("Starting service", .{});
//!
//! // Platform
//! const cpu_count = abi.platform.getCpuCount();
//!
//! // Utils
//! const hash = abi.utils.crypto.sha256(data);
//! const json = try abi.utils.json.parse(allocator, text);
//!
//! // SIMD operations
//! const dot = abi.vectorDot(a, b);
//! ```
//!
//! **Direct imports (for internal framework code)**
//!
//! ```zig
//! const logging = @import("shared/logging/mod.zig");
//! const platform = @import("shared/platform/mod.zig");
//! const simd = @import("shared/simd.zig");
//! const utils = @import("shared/utils_combined.zig");
//! ```
//!
//! ## Import Pattern
//!
//! Each subdirectory has a `mod.zig` that re-exports its symbols.
//! The top-level `src/shared/mod.zig` aggregates all sub-modules.
//! The public API in `src/abi.zig` imports directly from shared/.
//!
//! ## See Also
//!
//! - [Logging](logging/README.md)
//! - [Platform](platform/README.md)
//! - [Utils](utils/README.md)

