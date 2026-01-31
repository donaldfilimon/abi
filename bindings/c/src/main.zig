//! Main entry point for ABI C bindings library.
//! Re-exports all C-compatible functions from submodules.

const std = @import("std");

// Import all submodules
pub const errors = @import("errors.zig");
pub const framework = @import("framework.zig");
pub const simd = @import("simd.zig");
pub const database = @import("database.zig");
pub const gpu = @import("gpu.zig");
pub const agent = @import("agent.zig");

// Memory management - search results are caller-allocated, so this is a no-op
pub export fn abi_free_results(results: ?[*]database.SearchResult, count: usize) void {
    _ = results;
    _ = count;
    // Results array is caller-allocated, individual vectors point to database storage
    // No cleanup needed from caller
}

test "all modules compile" {
    _ = errors;
    _ = framework;
    _ = simd;
    _ = database;
    _ = gpu;
    _ = agent;
}

test "all tests" {
    std.testing.refAllDecls(@This());
}
