//! Cache Example
//!
//! Demonstrates the in-memory LRU/LFU cache with TTL support.
//! Shows put/get operations, eviction, and statistics.
//!
//! Run with: `zig build run-cache`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);

    var framework = try builder
        .with(.cache, .{})
        .build();
    defer framework.deinit();

    if (!abi.cache.isEnabled()) {
        std.debug.print("Cache feature is disabled. Enable with -Denable-cache=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Cache Example ===\n\n", .{});

    // Basic put/get operations
    abi.cache.put("user:1", "Alice") catch |err| {
        std.debug.print("Failed to put: {t}\n", .{err});
        return;
    };
    abi.cache.put("user:2", "Bob") catch |err| {
        std.debug.print("Failed to put: {t}\n", .{err});
        return;
    };
    std.debug.print("Stored 2 entries\n", .{});

    // Retrieve
    if (abi.cache.get("user:1") catch null) |value| {
        std.debug.print("user:1 = {s}\n", .{value});
    }

    // Put with TTL (5 seconds)
    abi.cache.putWithTtl("session:abc", "token-xyz", 5000) catch |err| {
        std.debug.print("Failed to put with TTL: {t}\n", .{err});
        return;
    };
    std.debug.print("Stored session with 5s TTL\n", .{});

    // Check existence
    const exists = abi.cache.contains("user:2");
    std.debug.print("user:2 exists: {}\n", .{exists});

    // Delete
    const deleted = abi.cache.delete("user:2") catch false;
    std.debug.print("user:2 deleted: {}\n", .{deleted});

    // Stats
    const s = abi.cache.stats();
    std.debug.print("\nCache stats: {} entries, {} hits, {} misses\n", .{
        s.entries, s.hits, s.misses,
    });
}
