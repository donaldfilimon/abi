//! Object Storage Example
//!
//! Demonstrates the unified object storage abstraction with
//! metadata, listing, and the memory backend.
//!
//! Run with: `zig build run-storage`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);

    var framework = try builder
        .withStorage(.{})
        .build();
    defer framework.deinit();

    if (!abi.storage.isEnabled()) {
        std.debug.print("Storage feature is disabled. Enable with -Denable-storage=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Object Storage Example ===\n\n", .{});

    // Store objects
    abi.storage.putObject(allocator, "images/logo.png", "PNG binary data here") catch |err| {
        std.debug.print("Failed to store object: {t}\n", .{err});
        return;
    };
    std.debug.print("Stored: images/logo.png\n", .{});

    // Store with metadata
    abi.storage.putObjectWithMetadata(
        allocator,
        "docs/readme.md",
        "# README\nProject documentation",
        .{
            .content_type = "text/markdown",
            .custom = blk: {
                var entries: [4]abi.storage.ObjectMetadata.MetadataEntry = [_]abi.storage.ObjectMetadata.MetadataEntry{.{}} ** 4;
                entries[0] = .{ .key = "author", .value = "abi-framework" };
                break :blk entries;
            },
            .custom_count = 1,
        },
    ) catch |err| {
        std.debug.print("Failed to store with metadata: {t}\n", .{err});
        return;
    };
    std.debug.print("Stored: docs/readme.md (with metadata)\n", .{});

    // Retrieve
    const data = abi.storage.getObject(allocator, "docs/readme.md") catch |err| {
        std.debug.print("Failed to retrieve: {t}\n", .{err});
        return;
    };
    defer allocator.free(data);
    std.debug.print("\nRetrieved docs/readme.md ({} bytes)\n", .{data.len});

    // Check existence
    const exists = abi.storage.objectExists("images/logo.png") catch false;
    std.debug.print("images/logo.png exists: {}\n", .{exists});

    // List objects with prefix
    const objects = abi.storage.listObjects(allocator, "docs/") catch |err| {
        std.debug.print("List failed: {t}\n", .{err});
        return;
    };
    defer allocator.free(objects);

    std.debug.print("\nObjects under 'docs/':\n", .{});
    for (objects) |obj| {
        std.debug.print("  {s} ({} bytes)\n", .{ obj.key, obj.size });
    }

    // Stats
    const s = abi.storage.stats();
    std.debug.print("\nStorage stats: {} objects, {} bytes total\n", .{
        s.total_objects, s.total_bytes,
    });
}
