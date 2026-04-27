//! BenchmarkSuite — the main benchmark runner.
//!
//! Extracted from mod.zig. Provides warm-up loops, statistical computation
//! (min/max/mean/median), report formatting, and JSON formatting.

const std = @import("std");
const types = @import("types.zig");

const Config = types.Config;
const BenchmarkFn = types.BenchmarkFn;
const BenchmarkState = types.BenchmarkState;
const BenchmarkResult = types.BenchmarkResult;

// ---------------------------------------------------------------------------
// High-resolution monotonic clock
// ---------------------------------------------------------------------------

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @intCast(
        @as(u128, @intCast(ts.sec)) * 1_000_000_000 +
            @as(u128, @intCast(ts.nsec)),
    );
}

// ---------------------------------------------------------------------------
// BenchmarkSuite
// ---------------------------------------------------------------------------

pub const BenchmarkSuite = struct {
    name: []const u8,
    benchmarks: std.ArrayListUnmanaged(Entry) = .empty,
    results: std.ArrayListUnmanaged(BenchmarkResult) = .empty,
    config: Config,

    const Entry = struct {
        name: []const u8,
        func: BenchmarkFn,
    };

    pub fn init(name: []const u8, config: Config) BenchmarkSuite {
        return .{
            .name = name,
            .config = config,
        };
    }

    pub fn deinit(self: *BenchmarkSuite, allocator: std.mem.Allocator) void {
        self.benchmarks.deinit(allocator);
        self.results.deinit(allocator);
    }

    /// Register a benchmark function.
    pub fn addBenchmark(
        self: *BenchmarkSuite,
        allocator: std.mem.Allocator,
        name: []const u8,
        func: BenchmarkFn,
    ) !void {
        try self.benchmarks.append(allocator, .{
            .name = name,
            .func = func,
        });
    }

    /// Execute all registered benchmarks.
    ///
    /// For each benchmark:
    ///  1. Run `config.warmup_iterations` warmup iterations (discarded).
    ///  2. Run `config.sample_iterations` timed iterations.
    ///  3. Compute min / max / mean / median from the samples.
    ///  4. Store in `results`.
    pub fn run(self: *BenchmarkSuite, allocator: std.mem.Allocator) !void {
        self.results.clearRetainingCapacity();

        const sample_count: usize = @intCast(self.config.sample_iterations);
        if (sample_count == 0) return;
        const warmup_count: usize = @intCast(self.config.warmup_iterations);

        for (self.benchmarks.items) |entry| {
            // Allocate sample buffer
            const samples = try allocator.alloc(u64, sample_count);
            defer allocator.free(samples);

            var state = BenchmarkState{
                .allocator = allocator,
            };

            // -- warmup (results discarded) --
            state.total_iterations = warmup_count;
            for (0..warmup_count) |i| {
                state.iteration = i;
                entry.func(&state);
            }

            // -- timed iterations --
            state.total_iterations = sample_count;
            var total_ns: u64 = 0;
            for (0..sample_count) |i| {
                state.iteration = i;
                const start = nowNs();
                entry.func(&state);
                const end = nowNs();
                const elapsed = end -| start; // saturating subtract
                samples[i] = elapsed;
                total_ns += elapsed;
            }

            // -- statistics --
            // Sort for median
            std.sort.block(u64, samples, {}, struct {
                fn lessThan(_: void, a: u64, b: u64) bool {
                    return a < b;
                }
            }.lessThan);

            const min_ns = samples[0];
            const max_ns = samples[sample_count - 1];
            const mean_ns: u64 = if (sample_count > 0)
                @intCast(total_ns / sample_count)
            else
                0;
            const median_ns: u64 = if (sample_count % 2 == 1)
                samples[sample_count / 2]
            else
                (samples[sample_count / 2 - 1] + samples[sample_count / 2]) / 2;

            try self.results.append(allocator, .{
                .name = entry.name,
                .iterations = sample_count,
                .total_ns = total_ns,
                .min_ns = min_ns,
                .max_ns = max_ns,
                .mean_ns = mean_ns,
                .median_ns = median_ns,
            });
        }
    }

    /// Format a human-readable report of all results.
    pub fn formatReport(self: *const BenchmarkSuite, allocator: std.mem.Allocator) ![]u8 {
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        const w = &aw.writer;

        try w.print("Benchmark Suite: {s}\n", .{self.name});
        try w.print("{s:<30} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12} {s:>14}\n", .{
            "Name", "Iterations", "Min (ns)", "Max (ns)", "Mean (ns)", "Median (ns)", "Ops/sec",
        });
        // Separator line
        try w.writeAll("-" ** 108 ++ "\n");

        for (self.results.items) |r| {
            const ops = r.opsPerSecond();
            try w.print("{s:<30} {d:>12} {d:>12} {d:>12} {d:>12} {d:>12} {d:>14.2}\n", .{
                r.name,
                r.iterations,
                r.min_ns,
                r.max_ns,
                r.mean_ns,
                r.median_ns,
                ops,
            });
        }

        return aw.toOwnedSlice();
    }

    /// Format results as a JSON array.
    pub fn formatJson(self: *const BenchmarkSuite, allocator: std.mem.Allocator) ![]u8 {
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        const w = &aw.writer;

        try w.print("{{\"suite\":\"{s}\",\"results\":[", .{self.name});

        for (self.results.items, 0..) |r, i| {
            if (i > 0) try w.writeByte(',');
            try w.print(
                "{{\"name\":\"{s}\",\"iterations\":{d},\"total_ns\":{d}," ++
                    "\"min_ns\":{d},\"max_ns\":{d},\"mean_ns\":{d}," ++
                    "\"median_ns\":{d},\"ops_per_second\":{d:.2}}}",
                .{
                    r.name,
                    r.iterations,
                    r.total_ns,
                    r.min_ns,
                    r.max_ns,
                    r.mean_ns,
                    r.median_ns,
                    r.opsPerSecond(),
                },
            );
        }

        try w.writeAll("]}");
        return aw.toOwnedSlice();
    }
};
