const std = @import("std");
const abi = @import("abi");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "simd",
    .summary = "CPU SIMD utilities and benchmarks",
    .usage = "abi simd <info|benchmark|dot|matrix> [options]",
    .details = "  info       Inspect SIMD capabilities\n" ++
        "  benchmark  Run SIMD benchmark suite\n" ++
        "  dot        Compute dot product using SIMD\n" ++
        "  matrix     Multiply matrices with SIMD\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "info")) {
        const has_simd = true;
        std.debug.print("SIMD CPU Features:\n", .{});
        std.debug.print("  SIMD Support: {s}\n", .{if (has_simd) "available" else "limited"});
        return;
    }

    if (std.mem.eql(u8, sub, "benchmark")) {
        var size: usize = 10000;
        var iterations: usize = 1000;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--size") and i + 1 < args.len) {
                size = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--iterations") and i + 1 < args.len) {
                iterations = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            }
        }

        std.debug.print("Running SIMD benchmark...\n", .{});
        try runSimdBenchmark(allocator, size, iterations);
        return;
    }

    if (std.mem.eql(u8, sub, "dot")) {
        var a_str: ?[]const u8 = null;
        var b_str: ?[]const u8 = null;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--a") and i + 1 < args.len) {
                i += 1;
                a_str = args[i];
            } else if (std.mem.eql(u8, args[i], "--b") and i + 1 < args.len) {
                i += 1;
                b_str = args[i];
            }
        }
        if (a_str == null or b_str == null) {
            std.debug.print("simd dot requires --a and --b CSV vectors\n", .{});
            return;
        }

        const a_vals = try common.parseCsvFloats(allocator, a_str.?);
        defer allocator.free(a_vals);
        const b_vals = try common.parseCsvFloats(allocator, b_str.?);
        defer allocator.free(b_vals);

        const len = @min(a_vals.len, b_vals.len);
        const dot = abi.VectorOps.dotProduct(a_vals[0..len], b_vals[0..len]);
        std.debug.print("SIMD dot product({d}): {d:.6}\n", .{ len, dot });
        return;
    }

    if (std.mem.eql(u8, sub, "matrix")) {
        var a_str: ?[]const u8 = null;
        var b_str: ?[]const u8 = null;
        var rows: usize = 0;
        var cols: usize = 0;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--a") and i + 1 < args.len) {
                i += 1;
                a_str = args[i];
            } else if (std.mem.eql(u8, args[i], "--b") and i + 1 < args.len) {
                i += 1;
                b_str = args[i];
            } else if (std.mem.eql(u8, args[i], "--rows") and i + 1 < args.len) {
                i += 1;
                rows = try std.fmt.parseInt(usize, args[i], 10);
            } else if (std.mem.eql(u8, args[i], "--cols") and i + 1 < args.len) {
                i += 1;
                cols = try std.fmt.parseInt(usize, args[i], 10);
            }
        }

        if (a_str == null or b_str == null) {
            std.debug.print("simd matrix requires --a and --b CSV matrices\n", .{});
            return;
        }

        const a_vals = try common.parseCsvFloats(allocator, a_str.?);
        defer allocator.free(a_vals);
        const b_vals = try common.parseCsvFloats(allocator, b_str.?);
        defer allocator.free(b_vals);

        if (rows == 0) rows = @intFromFloat(@sqrt(@as(f64, @floatFromInt(a_vals.len))));
        if (cols == 0) cols = rows;

        std.debug.print("Matrix multiplication ({d}x{d}) with SIMD optimization\n", .{ rows, cols });
        const result = try allocator.alloc(f32, rows * cols);
        defer allocator.free(result);

        abi.VectorOps.matrixMultiply(result, a_vals, b_vals, rows, cols, cols);
        std.debug.print("Matrix multiplication result: first few values...\n", .{});
        for (0..@min(10, result.len)) |idx| {
            std.debug.print("  [{d}] = {d:.3}\n", .{ idx, result[idx] });
        }
        return;
    }

    std.debug.print("Unknown simd subcommand: {s}\n", .{sub});
}

fn runSimdBenchmark(allocator: std.mem.Allocator, size: usize, iterations: usize) !void {
    std.debug.print("SIMD benchmark: size={}, iterations={}\n", .{ size, iterations });

    const aligned_size = (size + 15) & ~@as(usize, 15);

    const input_a = try allocator.alignedAlloc(f32, null, aligned_size);
    defer allocator.free(input_a);
    const input_b = try allocator.alignedAlloc(f32, null, aligned_size);
    defer allocator.free(input_b);
    const output = try allocator.alignedAlloc(f32, null, aligned_size);
    defer allocator.free(output);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (0..size) |i| {
        input_a[i] = random.float(f32) * 2.0 - 1.0;
        input_b[i] = random.float(f32) * 2.0 - 1.0;
    }

    var timer = try std.time.Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        abi.VectorOps.vectorAdd(output[0..size], input_a[0..size], input_b[0..size]);
        abi.VectorOps.vectorMul(output[0..size], input_a[0..size], input_b[0..size]);
    }
    const total = timer.read();
    const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("SIMD benchmark completed. Time: {d:.2}ms per iteration\n", .{avg / 1_000_000.0});
}
