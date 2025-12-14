const std = @import("std");
const gpu = @import("gpu");
const common = @import("common.zig");

pub const command = common.Command{
    .id = .gpu,
    .name = "gpu",
    .summary = "GPU diagnostics and vector utilities",
    .usage = "abi gpu <info|run-examples|dot|search|benchmark> [options]",
    .details = "  info           Inspect available GPU backends\n" ++
        "  run-examples   Execute GPU sample workloads\n" ++
        "  dot            Compute dot product between vectors\n" ++
        "  search         Run GPU-accelerated vector search demo\n" ++
        "  benchmark      Measure GPU performance\n",
    .run = run,
};

fn parseBackend(name: []const u8) ?gpu.Backend {
    if (std.mem.eql(u8, name, "auto")) return .auto;
    if (std.mem.eql(u8, name, "webgpu")) return .webgpu;
    if (std.mem.eql(u8, name, "vulkan")) return .vulkan;
    if (std.mem.eql(u8, name, "metal")) return .metal;
    if (std.mem.eql(u8, name, "dx12")) return .dx12;
    if (std.mem.eql(u8, name, "opengl")) return .opengl;
    if (std.mem.eql(u8, name, "opencl")) return .opencl;
    if (std.mem.eql(u8, name, "cuda")) return .cuda;
    if (std.mem.eql(u8, name, "cpu_fallback")) return .cpu_fallback;
    if (std.mem.eql(u8, name, "cpu")) return .cpu_fallback;
    return null;
}

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3 or common.isHelpToken(args[2])) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "info")) {
        var backend: gpu.Backend = .auto;
        var try_wgpu = true;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
                i += 1;
                if (parseBackend(args[i])) |b| backend = b;
            } else if (std.mem.eql(u8, args[i], "--no-webgpu-first")) {
                try_wgpu = false;
            }
        }

        const cfg = gpu.GPUConfig{
            .backend = backend,
            .try_webgpu_first = try_wgpu,
        };

        var renderer = try gpu.GPURenderer.init(allocator, cfg);
        defer renderer.deinit();
        const stats = renderer.getStats();
        std.debug.print("GPU Backend: {any}\n", .{renderer.backend});
        std.debug.print("Buffers: created={d}, destroyed={d}\n", .{ stats.buffers_created, stats.buffers_destroyed });
        std.debug.print("Memory: current={d}B, peak={d}B\n", .{ stats.bytes_current, stats.bytes_peak });
        return;
    }

    if (std.mem.eql(u8, sub, "run-examples")) {
        std.debug.print("Running GPU examples...\n", .{});

        const cfg = gpu.GPUConfig{ .backend = .auto };
        var renderer = try gpu.GPURenderer.init(allocator, cfg);
        defer renderer.deinit();

        std.debug.print("  ✓ GPU renderer initialized\n", .{});

        const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const buffer = try renderer.createBufferWithData(f32, &test_data, .{ .storage = true });
        defer renderer.destroyBuffer(buffer) catch {};
        std.debug.print("  ✓ Buffer creation and data upload\n", .{});

        if (renderer.backend != .cpu_fallback) {
            std.debug.print("  ✓ GPU compute backend available\n", .{});
        } else {
            std.debug.print("  ✓ CPU fallback mode\n", .{});
        }

        std.debug.print("GPU examples completed successfully!\n", .{});
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
            std.debug.print("gpu dot requires --a and --b CSV vectors\n", .{});
            return;
        }

        const a_vals = try common.parseCsvFloats(allocator, a_str.?);
        defer allocator.free(a_vals);
        const b_vals = try common.parseCsvFloats(allocator, b_str.?);
        defer allocator.free(b_vals);

        const cfg = gpu.GPUConfig{ .backend = .auto };
        var renderer = try gpu.GPURenderer.init(allocator, cfg);
        defer renderer.deinit();

        const ha = try renderer.createBufferWithData(f32, a_vals, .{ .storage = true, .copy_src = true, .copy_dst = true });
        const hb = try renderer.createBufferWithData(f32, b_vals, .{ .storage = true, .copy_src = true, .copy_dst = true });
        const len = if (a_vals.len < b_vals.len) a_vals.len else b_vals.len;
        const dot = try renderer.computeVectorDotBuffers(ha, hb, len);
        std.debug.print("dot({d}): {d:.6}\n", .{ len, dot });
        return;
    }

    if (std.mem.eql(u8, sub, "search")) {
        var db_path: ?[]const u8 = null;
        var vec_str: ?[]const u8 = null;
        var k: usize = 5;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--db") and i + 1 < args.len) {
                i += 1;
                db_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--vector") and i + 1 < args.len) {
                i += 1;
                vec_str = args[i];
            } else if (std.mem.eql(u8, args[i], "--k") and i + 1 < args.len) {
                i += 1;
                k = try std.fmt.parseInt(usize, args[i], 10);
            }
        }
        if (db_path == null or vec_str == null) {
            std.debug.print("gpu search requires --db and --vector\n", .{});
            return;
        }
        const v = try common.parseCsvFloats(allocator, vec_str.?);
        defer allocator.free(v);

        std.debug.print("Performing GPU-accelerated vector search...\n", .{});

        const cfg = gpu.GPUConfig{ .backend = .auto };
        var renderer = try gpu.GPURenderer.init(allocator, cfg);
        defer renderer.deinit();

        const query_buffer = try renderer.createBufferWithData(f32, v, .{ .storage = true, .copy_src = true });
        defer renderer.destroyBuffer(query_buffer) catch {};

        std.debug.print("Query vector uploaded to GPU ({d} dimensions)\n", .{v.len});
        std.debug.print("GPU search completed (k={d}, database={s})\n", .{ k, db_path.? });
        std.debug.print("Note: Full database integration requires additional development\n", .{});
        return;
    }

    if (std.mem.eql(u8, sub, "benchmark")) {
        var backend: gpu.Backend = .auto;
        var size: usize = 1024;
        var iterations: usize = 100;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
                i += 1;
                if (parseBackend(args[i])) |b| backend = b;
            } else if (std.mem.eql(u8, args[i], "--size") and i + 1 < args.len) {
                i += 1;
                size = try std.fmt.parseInt(usize, args[i], 10);
            } else if (std.mem.eql(u8, args[i], "--iterations") and i + 1 < args.len) {
                i += 1;
                iterations = try std.fmt.parseInt(usize, args[i], 10);
            }
        }

        std.debug.print("Running GPU benchmark with backend={any}, size={d}, iterations={d}\n", .{ backend, size, iterations });
        const cfg = gpu.GPUConfig{ .backend = backend };
        var renderer = try gpu.GPURenderer.init(allocator, cfg);
        defer renderer.deinit();

        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        for (a) |*val| val.* = random.float(f32) * 2.0 - 1.0;
        for (b) |*val| val.* = random.float(f32) * 2.0 - 1.0;

        const ha = try renderer.createBufferWithData(f32, a, .{ .storage = true, .copy_src = true, .copy_dst = true });
        const hb = try renderer.createBufferWithData(f32, b, .{ .storage = true, .copy_src = true, .copy_dst = true });

        const start_time = @as(u64, @intCast(std.time.nanoTimestamp));
        for (0..iterations) |_| {
            _ = try renderer.computeVectorDotBuffers(ha, hb, size);
        }
        const end_time = @as(u64, @intCast(std.time.nanoTimestamp));

        const elapsed_ns = @as(f64, @floatFromInt(end_time - start_time));
        const elapsed_ms = elapsed_ns / 1_000_000.0;
        const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
        const flops = @as(f64, @floatFromInt(size * iterations * 2)) / (elapsed_ns / 1_000_000_000.0);

        std.debug.print("Benchmark results:\n", .{});
        std.debug.print("  Total time: {d:.2}ms\n", .{elapsed_ms});
        std.debug.print("  Operations/sec: {d:.0}\n", .{ops_per_sec});
        std.debug.print("  FLOPS: {d:.2}G\n", .{flops / 1_000_000_000.0});
        return;
    }

    std.debug.print("Unknown gpu subcommand: {s}\n", .{sub});
}
