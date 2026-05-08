//! Benchmarks Module
//!
//! Performance benchmarking and timing utilities. Provides a Context for
//! running benchmark suites, recording results, and exporting metrics.

const std = @import("std");
const build_options = @import("build_options");
pub const types = @import("types.zig");
const suite_mod = @import("suite.zig");

pub const Config = types.Config;
pub const BenchmarksError = types.BenchmarksError;
pub const Error = BenchmarksError;

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
    return build_options.feat_benchmarks;
}

pub fn isInitialized() bool {
    return false;
}

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

pub const BenchmarkFn = types.BenchmarkFn;
pub const BenchmarkState = types.BenchmarkState;
pub const BenchmarkResult = types.BenchmarkResult;

// ---------------------------------------------------------------------------
// BenchmarkSuite — re-exported from suite.zig
// ---------------------------------------------------------------------------

pub const BenchmarkSuite = suite_mod.BenchmarkSuite;

// ===========================================================================
// Tests
// ===========================================================================

test "basic initialization" {
    const ctx = try Context.init(std.testing.allocator, Config{});
    defer ctx.deinit();
    try std.testing.expect(isEnabled() == build_options.feat_benchmarks);
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
