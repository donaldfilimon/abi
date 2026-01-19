//! Vector Database Tutorial - Example 4: Advanced Operations
//!
//! Run with: zig run docs/tutorials/code/vector-database/04-advanced-operations.zig

const std = @import("std");
// In a real project, you would use: const abi = @import("abi");
// For tutorial purposes, we use a relative path.
const abi = @import("../../../../src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    var db = try abi.database.openOrCreate(allocator, "advanced_db");
    defer abi.database.close(&db);

    const embedding1 = [_]f32{ 1.0, 2.0, 3.0 };
    try abi.database.insert(&db, 1, &embedding1, "Original data");
    std.debug.print("Inserted vector 1\n", .{});

    const embedding1_updated = [_]f32{ 1.5, 2.5, 3.5 };
    const updated = try abi.database.update(&db, 1, &embedding1_updated);
    if (updated) {
        std.debug.print("Updated vector 1\n", .{});
    } else {
        std.debug.print("Vector 1 not found for update\n", .{});
    }

    const retrieved = abi.database.get(&db, 1) orelse {
        std.debug.print("Vector 1 not found\n", .{});
        return;
    };
    std.debug.print("Retrieved: [{d:.1}, {d:.1}, {d:.1}]\n", .{
        retrieved.vector[0],
        retrieved.vector[1],
        retrieved.vector[2],
    });

    const stats = abi.database.stats(&db);
    const views = try abi.database.list(&db, allocator, stats.count);
    defer allocator.free(views);
    std.debug.print("Total vectors: {d}\n", .{views.len});

    const removed = abi.database.remove(&db, 1);
    if (removed) {
        std.debug.print("Deleted vector 1\n", .{});
    } else {
        std.debug.print("Vector 1 already removed\n", .{});
    }

    try abi.database.optimize(&db);
    std.debug.print("Database optimized\n", .{});
}
