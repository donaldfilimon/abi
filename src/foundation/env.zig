//! Portable process-environment access.
//!
//! This std exposes environment variables only through `std.process.Init`
//! (`init.environ_map`) — there is no global `getenv`/`std.posix.environ` in
//! this toolchain, and the previous libc `std.c.getenv` path is neither
//! Windows-portable nor available without linking libc. Each executable
//! entrypoint captures the process environment once via `install`, and all
//! env-dependent code reads through `get`.
//!
//! Lookups return a *borrowed* slice owned by the captured map (valid for the
//! whole process lifetime); callers must not free it. Before `install` runs —
//! e.g. unit tests that never go through `main` — every lookup reports "unset",
//! so callers fall back to their documented defaults.

const std = @import("std");

/// Process environment captured at startup. Null until `install` runs.
var g_environ: ?*std.process.Environ.Map = null;

/// Capture the process environment. Call once from each executable entrypoint
/// (`main`) before any env-dependent work. Reads are not threadsafe with
/// concurrent map mutation, but the captured map is never mutated, so the
/// read-only lookups below are safe across threads.
pub fn install(environ_map: *std.process.Environ.Map) void {
    g_environ = environ_map;
}

pub fn resetForTesting() void {
    g_environ = null;
}

/// Look up an environment variable. Returns a borrowed, non-empty value (owned
/// by the captured map, process-lifetime) or null when unset, empty, or the
/// environment was never captured. Do NOT free the result.
pub fn get(key: []const u8) ?[]const u8 {
    const map = g_environ orelse return null;
    const v = map.get(key) orelse return null;
    return if (v.len == 0) null else v;
}

// --- Canonical environment variable names (shared across MCP, WDBX REST, connectors) ---

pub const MCP_HTTP_PORT_ENV = "ABI_MCP_HTTP_PORT";
pub const MCP_HTTP_TOKEN_ENV = "ABI_MCP_HTTP_TOKEN";
pub const WDBX_REST_TOKEN_ENV = "ABI_WDBX_REST_TOKEN";
pub const WDBX_REST_PORT_ENV = "ABI_WDBX_REST_PORT";
pub const WDBX_CLUSTER_TOKEN_ENV = "ABI_WDBX_CLUSTER_TOKEN";
pub const WDBX_CLUSTER_PEERS_ENV = "ABI_WDBX_CLUSTER_PEERS";
pub const WDBX_TLS_CERT_ENV = "ABI_WDBX_TLS_CERT";
pub const WDBX_TLS_KEY_ENV = "ABI_WDBX_TLS_KEY";
pub const WDBX_PATH_ENV = "ABI_WDBX_PATH";
pub const WDBX_PERSIST_ENV = "ABI_WDBX_PERSIST";
pub const WDBX_RATE_LIMIT_CAPACITY_ENV = "ABI_WDBX_RATE_LIMIT_CAPACITY";
pub const WDBX_RATE_LIMIT_REFILL_ENV = "ABI_WDBX_RATE_LIMIT_REFILL";
pub const LLAMA_CPP_ENDPOINT_ENV = "ABI_LLAMA_CPP_ENDPOINT";
pub const MLX_ENDPOINT_ENV = "ABI_MLX_ENDPOINT";

test "get reports unset before install" {
    // No install() in this unit-test process, so every key resolves to null.
    try std.testing.expect(get("ABI_DEFINITELY_UNSET_VAR_XYZ") == null);
}

test {
    std.testing.refAllDecls(@This());
}
