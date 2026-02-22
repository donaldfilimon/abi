//! Benchmarks Module
//!
//! Performance benchmarking and timing utilities. Provides a Context for
//! running benchmark suites, recording results, and exporting metrics.

const std = @import("std");
const build_options = @import("build_options");
const core_config = @import("../../core/config/benchmarks.zig");

pub const Config = core_config.BenchmarksConfig;
pub const BenchmarksError = error{
    FeatureDisabled,
    OutOfMemory,
    InvalidConfig,
    BenchmarkFailed,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_benchmarks;
}

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

/// Function signature for benchmark bodies.
pub const BenchmarkFn = *const fn (state: *BenchmarkState) void;

/// State passed into every benchmark function.
pub const BenchmarkState = struct {
    iteration: usize = 0,
    total_iterations: usize = 0,
    /// Scratch allocator for benchmark use.
    allocator: std.mem.Allocator,

    /// Prevents the compiler from optimizing away a computed value.
    pub fn doNotOptimize(_: *BenchmarkState, value: anytype) void {
        var v = value;
        _ = &v;
    }
};

/// Timing results for a single benchmark.
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    median_ns: u64,

    /// Compute throughput in operations per second from the mean.
    pub fn opsPerSecond(self: BenchmarkResult) f64 {
        if (self.mean_ns == 0) return 0.0;
        return 1_000_000_000.0 / @as(f64, @floatFromInt(self.mean_ns));
    }
};

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
// BenchmarkSuite — the main runner
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

// ===========================================================================
// Tests
// ===========================================================================

test "basic initialization" {
    const ctx = try Context.init(std.testing.allocator, Config{});
    defer ctx.deinit();
    try std.testing.expect(isEnabled() == build_options.enable_benchmarks);
}

test "Config default values" {
    const config = Config{};
    try std.testing.expectEqual(@as(u32, 3), config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 10), config.sample_iterations);
    try std.testing.expect(!config.export_json);
    try std.testing.expect(config.output_path == null);
}

test "Config.defaults returns same as zero-init" {
    const a = Config{};
    const b = Config.defaults();
    try std.testing.expectEqual(a.warmup_iterations, b.warmup_iterations);
    try std.testing.expectEqual(a.sample_iterations, b.sample_iterations);
    try std.testing.expectEqual(a.export_json, b.export_json);
    try std.testing.expect(b.output_path == null);
}

test "Config custom values" {
    const config = Config{
        .warmup_iterations = 100,
        .sample_iterations = 500,
        .export_json = true,
        .output_path = "/tmp/bench.json",
    };
    try std.testing.expectEqual(@as(u32, 100), config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 500), config.sample_iterations);
    try std.testing.expect(config.export_json);
    try std.testing.expectEqualStrings("/tmp/bench.json", config.output_path.?);
}

test "Context stores config correctly" {
    const config = Config{
        .warmup_iterations = 7,
        .sample_iterations = 20,
        .export_json = true,
    };
    const ctx = try Context.init(std.testing.allocator, config);
    defer ctx.deinit();
    try std.testing.expectEqual(@as(u32, 7), ctx.config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 20), ctx.config.sample_iterations);
    try std.testing.expect(ctx.config.export_json);
}

test "Context multiple init and deinit" {
    const ctx1 = try Context.init(std.testing.allocator, Config{});
    const ctx2 = try Context.init(std.testing.allocator, Config{ .warmup_iterations = 1 });
    ctx1.deinit();
    ctx2.deinit();
}

test "BenchmarksError error set" {
    // Verify all error variants exist in the error set
    const errors = [_]BenchmarksError{
        error.FeatureDisabled,
        error.OutOfMemory,
        error.InvalidConfig,
        error.BenchmarkFailed,
    };
    try std.testing.expectEqual(@as(usize, 4), errors.len);
}

// ---- New benchmarking tests ----

test "Suite init and deinit" {
    var suite = BenchmarkSuite.init("test-suite", Config{});
    defer suite.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("test-suite", suite.name);
    try std.testing.expectEqual(@as(usize, 0), suite.benchmarks.items.len);
    try std.testing.expectEqual(@as(usize, 0), suite.results.items.len);
}

test "addBenchmark increases count" {
    var suite = BenchmarkSuite.init("add-test", Config{});
    defer suite.deinit(std.testing.allocator);

    const noop = struct {
        fn bench(_: *BenchmarkState) void {}
    }.bench;

    try suite.addBenchmark(std.testing.allocator, "noop-1", noop);
    try std.testing.expectEqual(@as(usize, 1), suite.benchmarks.items.len);

    try suite.addBenchmark(std.testing.allocator, "noop-2", noop);
    try std.testing.expectEqual(@as(usize, 2), suite.benchmarks.items.len);
}

test "run trivial benchmark and verify results" {
    const alloc = std.testing.allocator;
    var suite = BenchmarkSuite.init("trivial", Config{
        .warmup_iterations = 2,
        .sample_iterations = 50,
    });
    defer suite.deinit(alloc);

    const sum_bench = struct {
        fn bench(state: *BenchmarkState) void {
            var total: u64 = 0;
            for (0..10_000) |i| {
                total += i;
            }
            state.doNotOptimize(total);
        }
    }.bench;

    try suite.addBenchmark(alloc, "sum-100", sum_bench);
    try suite.run(alloc);

    try std.testing.expectEqual(@as(usize, 1), suite.results.items.len);
    const r = suite.results.items[0];
    try std.testing.expectEqualStrings("sum-100", r.name);
    try std.testing.expectEqual(@as(usize, 50), r.iterations);
    try std.testing.expect(r.total_ns > 0);
}

test "min <= mean <= max invariant" {
    const alloc = std.testing.allocator;
    var suite = BenchmarkSuite.init("invariant", Config{
        .warmup_iterations = 1,
        .sample_iterations = 20,
    });
    defer suite.deinit(alloc);

    const work = struct {
        fn bench(state: *BenchmarkState) void {
            var x: u64 = 1;
            for (0..50) |_| {
                x = x *% 7 +% 3;
            }
            state.doNotOptimize(x);
        }
    }.bench;

    try suite.addBenchmark(alloc, "work", work);
    try suite.run(alloc);

    const r = suite.results.items[0];
    try std.testing.expect(r.min_ns <= r.mean_ns);
    try std.testing.expect(r.mean_ns <= r.max_ns);
}

test "formatReport returns non-empty string" {
    const alloc = std.testing.allocator;
    var suite = BenchmarkSuite.init("report-test", Config{
        .warmup_iterations = 1,
        .sample_iterations = 5,
    });
    defer suite.deinit(alloc);

    const noop = struct {
        fn bench(_: *BenchmarkState) void {}
    }.bench;

    try suite.addBenchmark(alloc, "noop", noop);
    try suite.run(alloc);

    const report = try suite.formatReport(alloc);
    defer alloc.free(report);

    try std.testing.expect(report.len > 0);
    // Must contain the suite name
    try std.testing.expect(std.mem.indexOf(u8, report, "report-test") != null);
    // Must contain the benchmark name
    try std.testing.expect(std.mem.indexOf(u8, report, "noop") != null);
}

test "formatJson returns valid-looking JSON" {
    const alloc = std.testing.allocator;
    var suite = BenchmarkSuite.init("json-test", Config{
        .warmup_iterations = 1,
        .sample_iterations = 5,
    });
    defer suite.deinit(alloc);

    const noop = struct {
        fn bench(_: *BenchmarkState) void {}
    }.bench;

    try suite.addBenchmark(alloc, "noop", noop);
    try suite.run(alloc);

    const json = try suite.formatJson(alloc);
    defer alloc.free(json);

    try std.testing.expect(json.len > 0);
    // Starts with { and ends with }
    try std.testing.expectEqual(@as(u8, '{'), json[0]);
    try std.testing.expectEqual(@as(u8, '}'), json[json.len - 1]);
    // Contains required fields
    try std.testing.expect(std.mem.indexOf(u8, json, "\"suite\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"results\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"min_ns\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ops_per_second\"") != null);
}

test "BenchmarkState.doNotOptimize does not crash" {
    var state = BenchmarkState{
        .allocator = std.testing.allocator,
    };
    state.doNotOptimize(@as(u64, 42));
    state.doNotOptimize(@as(f64, 3.14));
    state.doNotOptimize(true);
    state.doNotOptimize(@as([]const u8, "hello"));
}

test "opsPerSecond calculation" {
    const r = BenchmarkResult{
        .name = "test",
        .iterations = 10,
        .total_ns = 10_000,
        .min_ns = 500,
        .max_ns = 2000,
        .mean_ns = 1000,
        .median_ns = 900,
    };
    // mean_ns = 1000 => ops/sec = 1_000_000_000 / 1000 = 1_000_000
    const ops = r.opsPerSecond();
    try std.testing.expectEqual(@as(f64, 1_000_000.0), ops);

    // Edge case: mean_ns == 0 => 0
    const zero = BenchmarkResult{
        .name = "zero",
        .iterations = 0,
        .total_ns = 0,
        .min_ns = 0,
        .max_ns = 0,
        .mean_ns = 0,
        .median_ns = 0,
    };
    try std.testing.expectEqual(@as(f64, 0.0), zero.opsPerSecond());
}

test {
    std.testing.refAllDecls(@This());
}
