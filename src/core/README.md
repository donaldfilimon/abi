//! # Core Module
//!
//! Core infrastructure providing hardware helpers, cache-aligned buffers, and profiling support.
//!
//! ## Features
//!
//! - **Hardware Helpers**: CPU feature detection, cache line alignment
//! - **Profiling Support**: Performance measurement utilities
//! - **Buffer Management**: Cache-aligned allocations
//! - **Platform Abstractions**: Cross-platform primitives
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Core definitions and public API |
//! | `profile.zig` | Profiling and timing helpers |
//!
//! ## Usage
//!
//! ### Cache-Aligned Buffers
//!
//! ```zig
//! const core = @import("core");
//!
//! // Allocate cache-line aligned buffer
//! const buffer = try core.alignedAlloc(u8, 64, 4096);
//! defer core.alignedFree(buffer);
//! ```
//!
//! ### Profiling
//!
//! ```zig
//! const profile = @import("core").profile;
//!
//! var timer = profile.Timer.start();
//! // ... work ...
//! const elapsed_ns = timer.read();
//! ```
//!
//! ## See Also
//!
//! - [Compute Module](../compute/README.md)
//! - [Framework Module](../framework/README.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

