//! Micro-benchmark operations.
//!
//! Provides quick single-operation benchmarks (hash, alloc, parse, noop).

const std = @import("std");
const abi = @import("abi");
const mod = @import("mod.zig");
const utils = @import("../../utils/mod.zig");

/// Run micro-benchmark for specific operation
pub fn runMicroBenchmark(allocator: std.mem.Allocator, args: []const [:0]const u8, config: *mod.BenchConfig) !void {
    if (args.len == 0) {
        utils.output.println("Usage: abi bench micro <operation>", .{});
        utils.output.println("Operations: hash, alloc, parse, noop", .{});
        return;
    }

    const op_name = std.mem.sliceTo(args[0], 0);
    const op = std.meta.stringToEnum(mod.MicroOp, op_name) orelse {
        utils.output.printError("Unknown micro-benchmark: {s}", .{op_name});
        utils.output.println("Available: hash, alloc, parse, noop", .{});
        return;
    };

    // Parse options
    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--iterations")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.iterations = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--json")) {
            config.output_json = true;
        }
    }

    utils.output.println("\nMicro-Benchmark: {s}", .{op.toString()});
    utils.output.println("Iterations: {d}", .{config.iterations});
    utils.output.println("Warmup: {d}\n", .{config.warmup});

    // Warmup
    var warmup_i: u32 = 0;
    while (warmup_i < config.warmup) : (warmup_i += 1) {
        _ = runMicroOp(allocator, op);
    }

    // Benchmark
    const timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("Timer not available.", .{});
        return;
    };

    var iter: u32 = 0;
    while (iter < config.iterations) : (iter += 1) {
        _ = runMicroOp(allocator, op);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(config.iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    if (config.output_json) {
        utils.output.println("{{\"operation\":\"{s}\",\"iterations\":{d},\"mean_ns\":{d:.2},\"ops_per_sec\":{d:.2}}}", .{
            op.toString(),
            config.iterations,
            mean_ns,
            ops_per_sec,
        });
    } else {
        utils.output.println("Results:", .{});
        utils.output.println("  Mean time: {d:.2} ns", .{mean_ns});
        utils.output.println("  Ops/sec: {d:.0}", .{ops_per_sec});
    }
}

pub fn runMicroOp(allocator: std.mem.Allocator, op: mod.MicroOp) usize {
    switch (op) {
        .hash => {
            // Simple hash computation
            const data = "The quick brown fox jumps over the lazy dog";
            var hash: usize = 0;
            for (data) |c| {
                hash = hash *% 31 +% c;
            }
            return hash;
        },
        .alloc => {
            // Allocation pattern
            const buf = allocator.alloc(u8, 4096) catch return 0;
            defer allocator.free(buf);
            return buf.len;
        },
        .parse => {
            // Simple JSON-like parsing
            const json = "{\"key\":\"value\",\"num\":42}";
            var count: usize = 0;
            for (json) |c| {
                if (c == ':' or c == ',') count += 1;
            }
            return count;
        },
        .noop => {
            return 0;
        },
    }
}

test {
    std.testing.refAllDecls(@This());
}
