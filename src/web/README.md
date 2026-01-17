//! # Web
//!
//! HTTP client utilities and web integrations.
//!
//! ## Features
//!
//! - **HTTP Client**: Async HTTP/1.1 and HTTP/2 client
//! - **Request Building**: Fluent API for request construction
//! - **Response Parsing**: JSON, text, and binary response handling
//! - **Timeouts**: Configurable connection and request timeouts
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Public API aggregation |
//! | `client.zig` | Async HTTP client |
//! | `weather.zig` | Example external service |
//!
//! ## Usage
//!
//! ```zig
//! const web = @import("abi").web;
//!
//! // Simple GET request
//! var client = try web.Client.init(allocator);
//! defer client.deinit();
//!
//! const response = try client.get("https://api.example.com/data");
//! defer allocator.free(response.body);
//!
//! // POST with JSON body
//! const result = try client.post("https://api.example.com/submit", .{
//!     .body = "{\"key\": \"value\"}",
//!     .content_type = "application/json",
//! });
//! ```
//!
//! ## Build Options
//!
//! Enable with `-Denable-web=true` (default: true).
//!
//! ## See Also
//!
//! - [HTTP Utilities](../../shared/utils/http/README.md)
//! - [CLAUDE.md Web Section](../../../CLAUDE.md)

