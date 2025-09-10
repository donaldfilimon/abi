const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var args = try std.process.argsWithAllocator(alloc);
    defer args.deinit();
    _ = args.next(); // exe
    const threshold_arg = args.next() orelse "20000000"; // 20ms default
    const threshold = try std.fmt.parseInt(u64, threshold_arg, 10);

    const test_file = "perf_guard.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    const database = @import("database");
    var db = try database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);

    // Load dataset
    const count: usize = 5000;
    for (0..count) |i| {
        var emb = try alloc.alloc(f32, 128);
        defer alloc.free(emb);
        for (0..128) |j| emb[j] = @as(f32, @floatFromInt((i * 128 + j) % 97)) * 0.001;
        _ = try db.addEmbedding(emb);
    }

    // Prepare query
    var query = try alloc.alloc(f32, 128);
    defer alloc.free(query);
    for (0..128) |i| query[i] = @as(f32, @floatFromInt(i)) * 0.01;

    // Time searches
    const iters: usize = 50;
    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        const results = try db.search(query, 10, alloc);
        alloc.free(results);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    const avg: u64 = @as(u64, @intCast(elapsed)) / @as(u64, iters);

    if (avg > threshold) {
        std.log.err("PerfGuard: average search ns {} exceeds threshold {}", .{ avg, threshold });
        std.process.exit(1);
    } else {
        std.log.info("PerfGuard: average search ns {} within threshold {}", .{ avg, threshold });
    }
}
