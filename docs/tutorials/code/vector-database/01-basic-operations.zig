//! Vector Database Tutorial - Example 1: Basic Operations
//!
//! Run with: zig run docs/tutorials/code/vector-database/01-basic-operations.zig

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    if (!framework.isEnabled(.database)) {
        std.debug.print("Error: Database disabled\n", .{});
        std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
        return error.DatabaseDisabled;
    }

    var db = try abi.database.openOrCreate(allocator, "my_vectors");
    defer abi.database.close(&db);

    std.debug.print("Database 'my_vectors' ready\n", .{});

    const stats = abi.database.stats(&db);
    std.debug.print("  Vectors: {d}\n", .{stats.count});
    std.debug.print("  Dimensions: {d}\n", .{stats.dimension});
}
