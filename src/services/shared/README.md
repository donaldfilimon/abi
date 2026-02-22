//! # Shared Utilities
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
//! const dot = abi.simd.vectorDot(a, b);
//! ```
//!
//! **Direct imports (for internal framework code)**
//!
//! ```zig
//! const logging = @import("shared/logging/mod.zig");
//! const platform = @import("shared/platform/mod.zig");
//! const simd = @import("shared/simd/mod.zig");
//! const utils = @import("shared/utils.zig");
//! ```
//!
//! ## Import Pattern
//!
//! Each subdirectory has a `mod.zig` that re-exports its symbols.
//! The top-level `src/services/shared/mod.zig` aggregates all sub-modules.
//! The public API is exposed via `src/abi.zig` and imports directly from shared/.
//!
//! ## See Also
//!
//! - [Logging](logging.zig)
//! - [Platform](platform.zig)
//! - [Utils](utils/README.md)

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
