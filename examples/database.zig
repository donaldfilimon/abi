const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_database = true,
        .enable_gpu = false,
    });
    defer abi.shutdown(&framework);

    if (!abi.database.isEnabled()) {
        std.debug.print("Database feature is disabled. Enable with -Denable-database=true\n", .{});
        return;
    }

    var handle = abi.database.openOrCreate(allocator, "example") catch |err| {
        std.debug.print("Failed to open/create database: {}\n", .{err});
        return err;
    };
    defer abi.database.close(&handle);

    // Insert test vectors
    const test_vectors = [_][3]f32{
        [_]f32{ 1.0, 0.0, 0.0 },
        [_]f32{ 0.0, 1.0, 0.0 },
        [_]f32{ 0.0, 0.0, 1.0 },
    };

    for (test_vectors, 1..) |vec, i| {
        abi.database.insert(&handle, @intCast(i), &vec, null) catch |err| {
            std.debug.print("Failed to insert vector {}: {}\n", .{ i, err });
            return err;
        };
    }
    std.debug.print("Inserted {} test vectors\n", .{test_vectors.len});

    // Perform similarity search
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const results = abi.database.search(&handle, allocator, &query, 2) catch |err| {
        std.debug.print("Failed to search database: {}\n", .{err});
        return err;
    };
    defer allocator.free(results);

    if (results.len == 0) {
        std.debug.print("No search results found\n", .{});
    } else {
        std.debug.print("Found {} similar vectors:\n", .{results.len});
        for (results) |r| {
            std.debug.print("  ID {}: similarity={d:.3}\n", .{ r.id, r.score });
        }
    }

    const stats = abi.database.stats(&handle);
    std.debug.print("Database contains {} vectors of dimension {}\n", .{ stats.count, stats.dimension });
}
