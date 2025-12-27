const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const handle = try abi.database.createDatabase(allocator, "example");
    defer abi.database.closeDatabase(&handle);

    try abi.database.insertVector(handle, 1, &.{ 1.0, 0.0, 0.0 }, null);
    try abi.database.insertVector(handle, 2, &.{ 0.0, 1.0, 0.0 }, null);
    try abi.database.insertVector(handle, 3, &.{ 0.0, 0.0, 1.0 }, null);

    const query = &.{ 1.0, 0.0, 0.0 };
    const results = try abi.database.searchVectors(handle, query, 2);
    defer allocator.free(results);

    std.debug.print("Found {d} results:\n", .{results.len});
    for (results) |r| {
        std.debug.print("  ID {d}: score={d:.3}\n", .{ r.id, r.score });
    }

    const stats = abi.database.getStats(&handle);
    std.debug.print("Database: {d} vectors, dimension={d}\n", .{ stats.count, stats.dimension });
}
