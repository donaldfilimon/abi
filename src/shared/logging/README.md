//! Logging Utilities
//!
//! Centralised logging support used throughout the ABI framework. Provides a
//! lightweight wrapper around `std.log` with configurable log levels and
//! output redirection. The `mod.zig` file defines the public logger API.
//!
//! To add structured logging, extend `mod.zig` with additional helpers.
