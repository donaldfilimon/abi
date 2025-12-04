const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const state_mod = @import("../state.zig");
const gpu_detection = @import("../../features/gpu/hardware_detection.zig");

const Size = struct { m: usize, n: usize, p: usize };

fn requireState(ctx: *modern_cli.Context) errors.CommandError!*state_mod.State {
    return ctx.userData(state_mod.State) orelse errors.CommandError.RuntimeFailure;
}

fn parseSize(text: []const u8) errors.CommandError!Size {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return errors.CommandError.InvalidArgument;

    var parts = std.mem.splitScalar(u8, trimmed, 'x');
    var values: [3]usize = .{ 0, 0, 0 };
    var count: usize = 0;
    while (parts.next()) |segment| {
        if (count >= values.len) return errors.CommandError.InvalidArgument;
        const s = std.mem.trim(u8, segment, " \t\r\n");
        if (s.len == 0) return errors.CommandError.InvalidArgument;
        values[count] = std.fmt.parseUnsigned(usize, s, 10) catch return errors.CommandError.InvalidArgument;
        count += 1;
    }
    if (count < 2) return errors.CommandError.InvalidArgument;

    const m = values[0];
    const n = values[1];
    const p = if (count >= 3) values[2] else values[1];
    if (m == 0 or n == 0 or p == 0) return errors.CommandError.InvalidArgument;
    if (m > 512 or n > 512 or p > 512) return errors.CommandError.InvalidArgument;

    return .{ .m = m, .n = n, .p = p };
}

const BenchmarkStats = struct {
    avg_ns: f64,
    min_ns: u64,
    max_ns: u64,
    iterations: usize,
};

fn runCpuBenchmark(allocator: std.mem.Allocator, size: Size, iterations: usize) errors.CommandError!BenchmarkStats {
    if (iterations == 0) return errors.CommandError.InvalidArgument;

    const rows = size.m;
    const shared = size.n;
    const cols = size.p;

    const a = allocator.alloc(f32, rows * shared) catch return errors.CommandError.RuntimeFailure;
    defer allocator.free(a);
    const b = allocator.alloc(f32, shared * cols) catch return errors.CommandError.RuntimeFailure;
    defer allocator.free(b);
    const c = allocator.alloc(f32, rows * cols) catch return errors.CommandError.RuntimeFailure;
    defer allocator.free(c);

    for (a, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i % 97) + 1)) / 97.0;
    for (b, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i % 53) + 1)) / 53.0;

    var total: f128 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    var iter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        std.mem.set(f32, c, 0);
        var timer = std.time.Timer.start() catch return errors.CommandError.RuntimeFailure;
        matmul(rows, cols, shared, a, b, c);
        const elapsed = timer.read();
        if (elapsed < min_ns) min_ns = elapsed;
        if (elapsed > max_ns) max_ns = elapsed;
        total += @as(f128, @floatFromInt(elapsed));
    }

    return .{
        .avg_ns = total / @as(f128, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .iterations = iterations,
    };
}

fn matmul(rows: usize, cols: usize, shared: usize, a: []const f32, b: []const f32, c: []f32) void {
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        var col: usize = 0;
        while (col < cols) : (col += 1) {
            var acc: f32 = 0;
            var k: usize = 0;
            while (k < shared) : (k += 1) {
                acc += a[row * shared + k] * b[k * cols + col];
            }
            c[row * cols + col] = acc;
        }
    }
}

fn benchHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const size_str = args.getString("size", "256x256");
    const size = try parseSize(size_str);

    const iterations_val = args.getInteger("iterations", 8);
    if (iterations_val <= 0) return errors.CommandError.InvalidArgument;
    const iterations = @as(usize, @intCast(iterations_val));

    const stats = try runCpuBenchmark(state.allocator, size, iterations);

    var detector = gpu_detection.GPUDetector.init(state.allocator);
    defer detector.deinit();
    var detection = detector.detectGPUs() catch |err| {
        _ = err;
        return errors.CommandError.RuntimeFailure;
    };
    defer detection.deinit();

    const gpu_available = detection.total_gpus > 0 and detection.available_backends.len > 0;

    const avg_ms = stats.avg_ns / 1_000_000.0;
    const min_ms = @as(f64, @floatFromInt(stats.min_ns)) / 1_000_000.0;
    const max_ms = @as(f64, @floatFromInt(stats.max_ns)) / 1_000_000.0;

    if (args.hasFlag("json")) {
        std.debug.print("{{\"size\":[{d},{d},{d}],\"iterations\":{d},", .{ size.m, size.n, size.p, iterations });
        std.debug.print("\"cpu_ms\":{{\"avg\":{d:.3},\"min\":{d:.3},\"max\":{d:.3}}},", .{ avg_ms, min_ms, max_ms });
        std.debug.print("\"gpu_available\":{s},\"available_backends\":[", .{if (gpu_available) "true" else "false"});
        for (detection.available_backends, 0..) |backend, idx| {
            if (idx != 0) std.debug.print(",", .{});
            std.debug.print("\"{s}\"", .{@tagName(backend)});
        }
        std.debug.print("]}}\n", .{});
    } else {
        std.debug.print(
            "CPU benchmark for {d}x{d}x{d} over {d} iterations\n",
            .{ size.m, size.n, size.p, iterations },
        );
        std.debug.print("  avg: {d:.3} ms  min: {d:.3} ms  max: {d:.3} ms\n", .{ avg_ms, min_ms, max_ms });
        if (gpu_available) {
            std.debug.print("GPU backends detected: ", .{});
            for (detection.available_backends, 0..) |backend, idx| {
                if (idx != 0) std.debug.print(", ", .{});
                std.debug.print("{s}", .{@tagName(backend)});
            }
            std.debug.print("\n", .{});
        } else {
            std.debug.print("No GPU backend detected, CPU fallback used.\n", .{});
        }
    }
}

pub const bench_command = modern_cli.Command{
    .name = "bench",
    .description = "Run dense-layer benchmark with CPU fallback",
    .handler = benchHandler,
    .options = &.{
        .{
            .name = "size",
            .long = "size",
            .description = "Matrix dimensions (MxN or MxNxP, max 512)",
            .arg_type = .string,
        },
        .{
            .name = "iterations",
            .long = "iterations",
            .description = "Number of benchmark iterations",
            .arg_type = .integer,
        },
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const command = modern_cli.Command{
    .name = "gpu",
    .description = "GPU demos and benchmarks",
    .subcommands = &.{&bench_command},
};
