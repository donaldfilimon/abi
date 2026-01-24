//! Consolidated Utilities Wrapper
//!
//! This file re‑exports all utility sub‑modules under `src/shared/utils/` and the
//! top‑level utility files. It provides a single import point for any code that
//! previously imported a specific utility file (e.g. `../shared/utils/http/mod.zig`).
//!
//! Example usage:
//! ```zig
//! const utils = @import("../shared/utils_combined.zig");
//! const http = utils.http; // same as importing the http module directly
//! const async_http = utils.async_http;
//! ```

// Sub‑module re‑exports
pub const crypto = @import("utils/crypto/mod.zig");
pub const encoding = @import("utils/encoding/mod.zig");
pub const fs = @import("utils/fs/mod.zig");
pub const http = @import("utils/http/mod.zig");
pub const async_http = @import("utils/http/async_http.zig");
pub const json = @import("utils/json/mod.zig");
pub const memory = @import("utils/memory/mod.zig");
pub const net = @import("utils/net/mod.zig");

// Top‑level utility files
pub const binary = @import("utils/binary.zig");
pub const config = @import("utils/config.zig");
pub const retry = @import("utils/retry.zig");
pub const logging = @import("logging.zig");
pub const os = @import("os.zig");
pub const platform = @import("platform.zig");
pub const plugins = @import("plugins.zig");
pub const simd = @import("simd.zig");
pub const time = @import("time.zig");
pub const utils = @import("utils.zig");

// Direct re-exports from utils for convenience
pub const unixSeconds = utils.unixSeconds;
pub const unixMs = utils.unixMs;
pub const unixMilliseconds = utils.unixMs; // Alias for compatibility
pub const nowSeconds = utils.nowSeconds;
pub const nowMs = utils.nowMs;
pub const nowMilliseconds = utils.nowMilliseconds;
pub const nowNanoseconds = utils.nowNanoseconds;
pub const sleepMs = utils.sleepMs;
pub const sleepNs = utils.sleepNs;
