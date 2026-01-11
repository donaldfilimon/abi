//! # Logging Utilities
//!
//! Centralized logging support with configurable levels and structured output.
//!
//! ## Features
//!
//! - **Configurable Levels**: Debug, info, warn, error
//! - **Structured Logging**: Key-value pair support
//! - **Output Redirection**: File, stderr, custom writers
//! - **Thread-Safe**: Safe for concurrent use
//!
//! ## Log Levels
//!
//! | Level | Description |
//! |-------|-------------|
//! | `.debug` | Detailed debugging information |
//! | `.info` | General informational messages |
//! | `.warn` | Warning conditions |
//! | `.err` | Error conditions |
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Basic logging
//! abi.log.info("Server started on port {d}", .{port});
//! abi.log.warn("Connection timeout", .{});
//! abi.log.err("Failed to connect: {}", .{err});
//!
//! // Debug logging (filtered in release builds)
//! abi.log.debug("Request details: {s}", .{request_id});
//! ```
//!
//! ## Configuration
//!
//! ```zig
//! const config = abi.LogConfig{
//!     .level = .info,
//!     .output = .stderr,
//!     .include_timestamp = true,
//!     .include_source_location = false,
//! };
//! ```
//!
//! ## See Also
//!
//! - [Monitoring Documentation](../../../docs/monitoring.md)
//! - [Observability Module](../observability/README.md)
