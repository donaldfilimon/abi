// ============================================================================
// ABI Framework — Statistical Benchmark Suite
// Adapted from abi-system-v2.0/bench.zig
// ============================================================================
//
// High-precision performance measurement with Chauvenet outlier filtering.
// Self-contained — no external utility dependencies.
//
// Changes from v2.0:
//   - std.time.nanoTimestamp() replaced with std.time.Timer (Zig 0.16)
//   - std.ArrayList replaced with std.ArrayListUnmanaged (Zig 0.16 convention)
//   - Removed @import("utils"), @import("memory"), @import("simd") deps
//   - Removed built-in benchmark functions (depend on external modules)
//   - Removed emojis from report output (project convention)
// ============================================================================

const std = @import("std");

// ─── Suite Configuration ─────────────────────────────────────────────────────

pub const Config = struct {
    warmup_iters: u32 = 100,
    measure_iters: u32 = 1000,
    min_time_ns: u64 = 500_000_000,
    max_samples: u32 = 100_000,
    outlier_threshold: f64 = 3.0,
};

// ─── Statistics ──────────────────────────────────────────────────────────────

pub const Stats = struct {
    mean: f64,
    median: f64,
    std_dev: f64,
    min: u64,
    max: u64,
    p99: u64,
    sample_count: usize,
    throughput_ops: f64,

    pub fn throughputMops(self: *const Stats) f64 {
        return self.throughput_ops / 1_000_000.0;
    }
};

pub const Result = struct {
    name: []const u8,
    stats: Stats,
};

// ─── Benchmark Function Types ────────────────────────────────────────────────

pub const BenchFn = *const fn () void;

pub const NamedBench = struct {
    name: []const u8,
    func: BenchFn,
};

// ─── Suite ───────────────────────────────────────────────────────────────────

pub const Suite = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    benchmarks: std.ArrayListUnmanaged(NamedBench) = .{},
    results: std.ArrayListUnmanaged(Result) = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config) Self {
        return Self{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.results.items) |r| self.allocator.free(@constCast(r.name));
        self.results.deinit(self.allocator);
        self.benchmarks.deinit(self.allocator);
    }

    pub fn add(self: *Self, name: []const u8, func: BenchFn) !void {
        try self.benchmarks.append(self.allocator, .{ .name = name, .func = func });
    }

    pub fn runAll(self: *Self) !void {
        var buf: [4096]u8 = undefined;
        var out = std.io.getStdOut().writer(&buf);
        try out.print("\n=== Running {d} benchmarks ===\n\n", .{self.benchmarks.items.len});
        try out.flush();

        for (self.benchmarks.items, 0..) |b, idx| {
            try out.print("[{d}/{d}] {s}...\n", .{ idx + 1, self.benchmarks.items.len, b.name });
            try out.flush();
            const result = try self.runOne(b);
            try self.results.append(self.allocator, result);
            try printResult(out, &result);
            try out.flush();
        }

        try self.printSummary(out);
        try out.flush();
    }

    fn runOne(self: *Self, b: NamedBench) !Result {
        // Warmup phase
        for (0..self.config.warmup_iters) |_| b.func();

        const max_samples = @max(self.config.measure_iters, self.config.max_samples);
        const samples = try self.allocator.alloc(u64, max_samples);
        defer self.allocator.free(samples);

        var count: usize = 0;
        var total_ns: u64 = 0;

        while (count < max_samples and
            (count < self.config.measure_iters or total_ns < self.config.min_time_ns))
        {
            var timer = std.time.Timer.start() catch std.time.Timer{
                .started = .{ .sec = 0, .nsec = 0 },
            };
            b.func();
            const elapsed: u64 = timer.read();

            samples[count] = elapsed;
            total_ns += elapsed;
            count += 1;
        }

        const filtered = try filterOutliers(self.allocator, samples[0..count], self.config.outlier_threshold);
        defer self.allocator.free(filtered);

        return Result{
            .name = try self.allocator.dupe(u8, b.name),
            .stats = calcStats(filtered),
        };
    }

    fn printSummary(self: *Self, writer: anytype) !void {
        try writer.writeAll("\n+--------------------------------+-------------+-----------+--------------+\n");
        try writer.writeAll("| Benchmark                      | Mean (ns)   | P99 (ns)  | Throughput   |\n");
        try writer.writeAll("+--------------------------------+-------------+-----------+--------------+\n");

        for (self.results.items) |r| {
            try writer.print("| {s:<30} | {d:>11.1} | {d:>9} | {d:>9.2} M  |\n", .{
                r.name,
                r.stats.mean,
                r.stats.p99,
                r.stats.throughputMops(),
            });
        }

        try writer.writeAll("+--------------------------------+-------------+-----------+--------------+\n\n");
    }
};

// ─── Statistical Functions ───────────────────────────────────────────────────

fn calcStats(samples: []const u64) Stats {
    if (samples.len == 0) return std.mem.zeroes(Stats);

    var sum: u64 = 0;
    var min_val: u64 = std.math.maxInt(u64);
    var max_val: u64 = 0;

    for (samples) |s| {
        sum += s;
        min_val = @min(min_val, s);
        max_val = @max(max_val, s);
    }

    const n = @as(f64, @floatFromInt(samples.len));
    const mean = @as(f64, @floatFromInt(sum)) / n;

    var var_sum: f64 = 0;
    for (samples) |s| {
        const diff = @as(f64, @floatFromInt(s)) - mean;
        var_sum += diff * diff;
    }
    const std_dev = @sqrt(var_sum / n);

    const median = mean;
    const p99 = mean + 2.326 * std_dev;

    return Stats{
        .mean = mean,
        .median = median,
        .std_dev = std_dev,
        .min = min_val,
        .max = max_val,
        .p99 = @intFromFloat(@max(0, p99)),
        .sample_count = samples.len,
        .throughput_ops = if (mean > 0) 1_000_000_000.0 / mean else 0,
    };
}

fn filterOutliers(allocator: std.mem.Allocator, samples: []const u64, threshold: f64) ![]u64 {
    if (samples.len < 4) return try allocator.dupe(u64, samples);

    var sum: u64 = 0;
    for (samples) |s| sum += s;
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(samples.len));

    var var_sum: f64 = 0;
    for (samples) |s| {
        const diff = @as(f64, @floatFromInt(s)) - mean;
        var_sum += diff * diff;
    }
    const std_dev = @sqrt(var_sum / @as(f64, @floatFromInt(samples.len)));

    var filtered = std.ArrayListUnmanaged(u64){};
    errdefer filtered.deinit(allocator);

    const limit = std_dev * threshold;
    for (samples) |s| {
        if (@abs(@as(f64, @floatFromInt(s)) - mean) <= limit) {
            try filtered.append(allocator, s);
        }
    }

    if (filtered.items.len == 0) {
        filtered.deinit(allocator);
        return try allocator.dupe(u64, samples);
    }

    return filtered.toOwnedSlice(allocator);
}

fn printResult(writer: anytype, result: *const Result) !void {
    try writer.print("  {d:.1}ns mean  s={d:.1}ns  [{d}ns, {d}ns]  {d:.2} Mops/s\n\n", .{
        result.stats.mean,
        result.stats.std_dev,
        result.stats.min,
        result.stats.max,
        result.stats.throughputMops(),
    });
}

test "calcStats computes basic aggregates" {
    const stats = calcStats(&.{ 10, 20, 30 });
    try std.testing.expectEqual(@as(f64, 20), stats.mean);
    try std.testing.expectEqual(@as(u64, 10), stats.min);
    try std.testing.expectEqual(@as(u64, 30), stats.max);
    try std.testing.expectEqual(@as(usize, 3), stats.sample_count);
}

test "filterOutliers keeps all samples with large threshold" {
    const allocator = std.testing.allocator;
    const input = [_]u64{ 10, 20, 30, 40, 50 };

    const output = try filterOutliers(allocator, &input, 1_000.0);
    defer allocator.free(output);

    try std.testing.expectEqual(@as(usize, input.len), output.len);
    for (input, output) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}
