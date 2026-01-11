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
//! | `utils/` | General utilities (crypto, encoding, fs, http, json, math, net, string) |
//!
//! ## Usage
//!
//! ```zig
//! const shared = @import("abi").shared;
//!
//! // Logging
//! shared.logging.info("Starting service", .{});
//!
//! // Platform
//! const cpu_count = shared.platform.getCpuCount();
//!
//! // Utils
//! const hash = shared.utils.crypto.sha256(data);
//! const json = try shared.utils.json.parse(allocator, text);
//! ```
//!
//! ## Import Pattern
//!
//! Each subdirectory has a `mod.zig` that re-exports its symbols.
//! The top-level `src/shared/mod.zig` aggregates all sub-modules.
//!
//! ## See Also
//!
//! - [Logging](logging/README.md)
//! - [Platform](platform/README.md)
//! - [Utils](utils/README.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

