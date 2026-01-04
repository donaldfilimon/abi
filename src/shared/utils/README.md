//! Utility Modules Overview

The **utils** package bundles small, reusable components that are deliberately independent to avoid circular dependencies.

| Module | Focus |
|--------|-------|
| `crypto` | Cryptographic primitives, hash functions, random generators |
| `encoding` | Data encoding/decoding (base64, hex, protobuf helpers) |
| `fs` | File‑system abstractions, path utilities, safe I/O wrappers |
| `http` | Minimal HTTP client/server utilities used by higher‑level APIs |
| `json` | JSON parsing/serialization with Zig allocator support |
| `math` | Numeric helpers, statistics, algebraic functions |
| `net` | Low‑level socket wrappers, address handling |
| `string` | String manipulation, UTF‑8 utilities, formatting helpers |

All modules follow the ABI style guide: `//!` file header, `pub const` re‑exports in `mod.zig`, snake_case naming, and accept an `std.mem.Allocator` as the first argument where allocation occurs.

