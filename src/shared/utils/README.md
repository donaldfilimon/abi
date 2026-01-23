//! # Utility Modules
//!
//! > **Codebase Status:** Synced with repository as of 2026-01-22.
//!
//! Reusable components designed to be independent and avoid circular dependencies.
//!
//! ## Modules
//!
//! | Module | Focus |
//! |--------|-------|
//! | `crypto/` | Cryptographic primitives, hash functions, random generators |
//! | `encoding/` | Data encoding/decoding (base64, hex, protobuf helpers) |
//! | `fs/` | File-system abstractions, path utilities, safe I/O wrappers |
//! | `http/` | Minimal HTTP client/server utilities |
//! | `json/` | JSON parsing/serialization with Zig allocator support |
//! | `math/` | Numeric helpers, statistics, SIMD operations |
//! | `net/` | Low-level socket wrappers, address handling |
//! | `string/` | String manipulation, UTF-8 utilities |
//! | `errors.zig` | ErrorContext for structured error tracking |
//!
//! ## Usage Pattern
//!
//! All modules follow the ABI style guide:
//!
//! ```zig
//! const utils = @import("shared").utils;
//!
//! // JSON parsing
//! const data = try utils.json.parse(allocator, json_string);
//! defer utils.json.free(allocator, data);
//!
//! // HTTP request
//! const response = try utils.http.get(allocator, "https://api.example.com");
//! defer allocator.free(response.body);
//!
//! // Crypto
//! const hash = utils.crypto.sha256(data);
//! ```
//!
//! ## Conventions
//!
//! - `//!` file header documentation
//! - `pub const` re-exports in `mod.zig`
//! - `snake_case` naming
//! - Accept `std.mem.Allocator` as first argument where allocation occurs
//!
//! ## See Also
//!
//! - [Shared Module](../README.md)

