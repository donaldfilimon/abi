const std = @import("std");
const foundation_time = @import("foundation/time.zig");
const test_helpers = @import("testing/test_helpers.zig");
const wdbx = @import("features/wdbx/mod.zig");
const constitution = @import("features/ai/constitution.zig");
const gpu_mod = @import("features/gpu/mod.zig");
const router = @import("features/ai/router.zig");

const analyzeSentiment = router.analyzeSentiment;
const selectBestProfile = router.selectBestProfile;

const ITERATIONS = 10;
const WARMUP = 1;

/// Versioned machine-readable artifact path (relative to the build root). The
/// `benchmarks` build step regenerates this; treat checked-in numbers as the
/// reproducible source for any external performance claim.
pub const BENCH_ARTIFACT_PATH = "zig-out/bench/results.json";

const BenchResult = struct {
    label: []const u8,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    iterations: usize,
};

/// Nearest-rank percentile (p in [0,100]) over a sorted copy of `samples`.
/// Returns 0 for empty input. Uses page_allocator for the scratch copy, matching
/// the rest of this file; on alloc failure degrades to 0 rather than failing the
/// benchmark run.
fn percentile(samples: []const f64, p: u8) f64 {
    if (samples.len == 0) return 0.0;
    const copy = std.heap.page_allocator.dupe(f64, samples) catch return 0.0;
    defer std.heap.page_allocator.free(copy);
    std.mem.sort(f64, copy, {}, std.sort.asc(f64));
    const idx_f = @ceil(@as(f64, @floatFromInt(p)) / 100.0 * @as(f64, @floatFromInt(copy.len)));
    const raw_idx = @as(usize, @intFromFloat(idx_f));
    const clamped = if (raw_idx == 0) 0 else @min(raw_idx, copy.len) - 1;
    return copy[clamped];
}

/// Accumulated across all bench tests in this run; serialized to
/// BENCH_ARTIFACT_PATH by the final "write benchmark artifact" test. Labels are
/// comptime string literals (static lifetime), so storing the slice is safe.
var bench_results: std.ArrayListUnmanaged(BenchResult) = .empty;

fn recordBench(label: []const u8, avg_ms: f64, min_ms: f64, max_ms: f64, p50_ms: f64, p95_ms: f64, p99_ms: f64) void {
    bench_results.append(std.heap.page_allocator, .{
        .label = label,
        .avg_ms = avg_ms,
        .min_ms = min_ms,
        .max_ms = max_ms,
        .p50_ms = p50_ms,
        .p95_ms = p95_ms,
        .p99_ms = p99_ms,
        .iterations = ITERATIONS,
    }) catch |err| std.debug.print("bench record failed: {s}\n", .{@errorName(err)});
}

fn measure(comptime label: []const u8, total_ms: f64, min_ms: f64, max_ms: f64, samples: ?[]const f64) void {
    const avg_ms = total_ms / @as(f64, @floatFromInt(ITERATIONS));
    const p50 = if (samples) |s| percentile(s, 50) else 0.0;
    const p95 = if (samples) |s| percentile(s, 95) else 0.0;
    const p99 = if (samples) |s| percentile(s, 99) else 0.0;
    recordBench(label, avg_ms, min_ms, max_ms, p50, p95, p99);
    std.debug.print("bench [{s}]: avg {d:.3}ms p50 {d:.3}ms p95 {d:.3}ms p99 {d:.3}ms min {d:.3}ms max {d:.3}ms ({d} iters)\n", .{ label, avg_ms, p50, p95, p99, min_ms, max_ms, ITERATIONS });
}

fn runBench(comptime label: []const u8, comptime fn_run: anytype) void {
    var i: usize = 0;
    while (i < WARMUP) : (i += 1) {
        fn_run();
    }

    var samples_buf: [ITERATIONS]f64 = undefined;
    var total_ms: f64 = 0;
    var min_ms: f64 = std.math.floatMax(f64);
    var max_ms: f64 = 0;
    i = 0;
    while (i < ITERATIONS) : (i += 1) {
        const start = foundation_time.monotonicNs();
        fn_run();
        const elapsed_ms = @as(f64, @floatFromInt(foundation_time.monotonicNs() - start)) / 1_000_000.0;
        total_ms += elapsed_ms;
        min_ms = @min(min_ms, elapsed_ms);
        max_ms = @max(max_ms, elapsed_ms);
        samples_buf[i] = elapsed_ms;
    }

    measure(label, total_ms, min_ms, max_ms, &samples_buf);
}

fn runBenchWithContext(comptime label: []const u8, context: anytype, comptime fn_run: anytype) void {
    var i: usize = 0;
    while (i < WARMUP) : (i += 1) {
        fn_run(context);
    }

    var samples_buf: [ITERATIONS]f64 = undefined;
    var total_ms: f64 = 0;
    var min_ms: f64 = std.math.floatMax(f64);
    var max_ms: f64 = 0;
    i = 0;
    while (i < ITERATIONS) : (i += 1) {
        const start = foundation_time.monotonicNs();
        fn_run(context);
        const elapsed_ms = @as(f64, @floatFromInt(foundation_time.monotonicNs() - start)) / 1_000_000.0;
        total_ms += elapsed_ms;
        min_ms = @min(min_ms, elapsed_ms);
        max_ms = @max(max_ms, elapsed_ms);
        samples_buf[i] = elapsed_ms;
    }

    measure(label, total_ms, min_ms, max_ms, &samples_buf);
}

/// Serialize all collected benchmark results to BENCH_ARTIFACT_PATH as JSON.
/// Best-effort: failures are logged, not fatal (a missing artifact must not
/// fail the benchmark run).
fn writeBenchArtifact() void {
    const alloc = std.heap.page_allocator;
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(alloc);

    out.print(alloc, "{{\"schema\":\"abi-bench/v2\",\"iterations\":{d},\"benchmarks\":[", .{ITERATIONS}) catch return;
    for (bench_results.items, 0..) |r, idx| {
        out.print(alloc, "{s}{{\"label\":\"{s}\",\"avg_ms\":{d:.4},\"min_ms\":{d:.4},\"max_ms\":{d:.4},\"p50_ms\":{d:.4},\"p95_ms\":{d:.4},\"p99_ms\":{d:.4},\"iterations\":{d}}}", .{
            if (idx == 0) "" else ",",
            r.label,
            r.avg_ms,
            r.min_ms,
            r.max_ms,
            r.p50_ms,
            r.p95_ms,
            r.p99_ms,
            r.iterations,
        }) catch return;
    }
    out.appendSlice(alloc, "]}\n") catch return;

    const dir = std.fs.path.dirname(BENCH_ARTIFACT_PATH) orelse ".";
    std.Io.Dir.cwd().createDirPath(std.testing.io, dir) catch |err| {
        std.debug.print("bench artifact dir create failed ({s}): {s}\n", .{ dir, @errorName(err) });
        return;
    };
    std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = BENCH_ARTIFACT_PATH, .data = out.items }) catch |err| {
        std.debug.print("bench artifact write failed ({s}): {s}\n", .{ BENCH_ARTIFACT_PATH, @errorName(err) });
        return;
    };
    std.debug.print("bench artifact written: {s} ({d} benchmarks)\n", .{ BENCH_ARTIFACT_PATH, bench_results.items.len });
}

test "bench vector dot product" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    const DotBench = struct {
        pub fn run() void {
            const ops = gpu_mod.vectorOps();
            _ = ops.dot(&a, &b) catch |err| {
                std.debug.print("bench vector dot failed: {s}\n", .{@errorName(err)});
                return;
            };
        }
    };

    runBench("vector dot product", DotBench.run);

    const ops = gpu_mod.vectorOps();
    const dot_result = try ops.dot(&a, &b);
    try std.testing.expect(dot_result > 0);
}

test "bench HNSW insert" {
    const InsertBench = struct {
        pub fn run() void {
            var store = wdbx.Store.init(std.testing.allocator);
            defer store.deinit();
            var i: usize = 0;
            while (i < 50) : (i += 1) {
                const vals = [_]f32{
                    @as(f32, @floatFromInt(i)) / 50.0,
                    @as(f32, @floatFromInt(50 - i)) / 50.0,
                    0.0,
                    0.0,
                };
                _ = store.putVector(&vals) catch |err| {
                    std.debug.print("bench HNSW insert putVector failed: {s}\n", .{@errorName(err)});
                    return;
                };
            }
        }
    };

    runBench("HNSW insert", InsertBench.run);
}

test "bench HNSW search" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const vals = [_]f32{
            @as(f32, @floatFromInt(i)) / 100.0,
            @as(f32, @floatFromInt(100 - i)) / 100.0,
            0.0,
            0.0,
        };
        _ = try store.putVector(&vals);
    }

    const SearchBench = struct {
        pub fn run(bench_store: *wdbx.Store) void {
            const query = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
            const results = bench_store.search(&query, 10) catch |err| {
                std.debug.print("bench HNSW search failed: {s}\n", .{@errorName(err)});
                return;
            };
            std.testing.allocator.free(results);
        }
    };

    runBenchWithContext("HNSW search", &store, SearchBench.run);
}

test "bench block chain append" {
    const ChainBench = struct {
        pub fn run() void {
            var store = wdbx.Store.init(std.testing.allocator);
            defer store.deinit();
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                const profile = "test_profile";
                const metadata = "block metadata";
                _ = store.appendBlock(profile, @intCast(i), @intCast(i + 1), metadata) catch |err| {
                    std.debug.print("bench block chain append failed: {s}\n", .{@errorName(err)});
                    return;
                };
            }
        }
    };

    runBench("block chain append", ChainBench.run);
}

test "bench profile routing" {
    const inputs = [_][]const u8{
        "analyze the logical structure",
        "imagine creative possibilities",
        "execute deploy run quickly",
        "design a safe pattern",
        "build and fix the system",
    };

    const RoutingBench = struct {
        pub fn run(inputs_val: *const [5][]const u8) void {
            for (inputs_val.*) |input| {
                const weights = analyzeSentiment(input);
                _ = selectBestProfile(weights);
            }
        }
    };

    runBench("profile routing", struct {
        pub fn run() void {
            RoutingBench.run(&inputs);
        }
    }.run);
}

test "bench constitution check" {
    const responses = [_][]const u8{
        "this is a helpful and safe response for everyone",
        "here is how to do it safely with a safe alternative",
        "this could cause harm and discriminate against users",
        "as an ai, i think i don't know the answer",
        "your password and personal data are at risk",
    };

    const ConstBench = struct {
        pub fn run(responses_val: *const [5][]const u8) void {
            for (responses_val.*) |resp| {
                const result = constitution.Constitution.validate(resp);
                _ = result;
            }
        }
    };

    runBench("constitution check", struct {
        pub fn run() void {
            ConstBench.run(&responses);
        }
    }.run);
}

test "bench wdbx store operations" {
    const StoreBench = struct {
        pub fn run() void {
            var store = wdbx.Store.init(std.testing.allocator);
            defer store.deinit();
            var i: usize = 0;
            while (i < 200) : (i += 1) {
                var key_buf: [32]u8 = undefined;
                const key = std.fmt.bufPrint(&key_buf, "key:{d}", .{i}) catch unreachable;
                var val_buf: [32]u8 = undefined;
                const val = std.fmt.bufPrint(&val_buf, "val:{d}", .{i}) catch unreachable;
                store.store(key, val) catch |err| {
                    std.debug.print("bench wdbx store put failed: {s}\n", .{@errorName(err)});
                    return;
                };
            }
            var j: usize = 0;
            while (j < 200) : (j += 1) {
                var key_buf: [32]u8 = undefined;
                const key = std.fmt.bufPrint(&key_buf, "key:{d}", .{j}) catch unreachable;
                _ = store.get(key);
            }
        }
    };

    runBench("wdbx store put/get", StoreBench.run);
}

// --- Multiway rewriting simulator benchmarks ------------------------------
//
// Small/medium/branching/convergent/cyclic rule sets, each bounded so a bench
// run stays well under a second and cannot consume exponential memory. Every
// bench reports through the same runBench harness; peak unique-state counts
// are asserted after timing so the numbers are grounded in an observed run.

const multiway = wdbx.multiway;

const MW_BRANCHING = [_]multiway.Rule{
    .{ .lhs = "A", .rhs = "AB" },
    .{ .lhs = "A", .rhs = "BA" },
    .{ .lhs = "BB", .rhs = "A" },
};
const MW_CONVERGENT = [_]multiway.Rule{
    .{ .lhs = "A", .rhs = "C" },
    .{ .lhs = "B", .rhs = "C" },
};
const MW_CYCLIC = [_]multiway.Rule{
    .{ .lhs = "A", .rhs = "B" },
    .{ .lhs = "B", .rhs = "A" },
};
const MW_GROWING = [_]multiway.Rule{.{ .lhs = "A", .rhs = "AA" }};

fn mwConfig(rules: []const multiway.Rule, initial: []const []const u8, depth: u32) multiway.Config {
    return .{
        .initial = initial,
        .rules = rules,
        .max_depth = depth,
        .max_states = 4096,
        .max_events = 200_000,
        .max_payload = 64,
    };
}

test "bench multiway successor generation" {
    // Successor generation + full frontier expansion over the branching set.
    const config = mwConfig(&MW_BRANCHING, &.{"A"}, 6);
    const Gen = struct {
        pub fn run() void {
            var result = multiway.run(std.testing.allocator, mwConfig(&MW_BRANCHING, &.{"A"}, 6), null) catch return;
            result.deinit();
        }
    };
    runBench("multiway successor generation (branching, d6)", Gen.run);

    var result = try multiway.run(std.testing.allocator, config, null);
    defer result.deinit();
    try std.testing.expect(result.states.items.len > 0);
    std.debug.print("  peak unique states: {d} (branching d6)\n", .{result.states.items.len});
}

test "bench multiway single-threaded frontier expansion (growing)" {
    const Grow = struct {
        pub fn run() void {
            var result = multiway.run(std.testing.allocator, mwConfig(&MW_GROWING, &.{"A"}, 8), null) catch return;
            result.deinit();
        }
    };
    runBench("multiway frontier expansion (growing, d8)", Grow.run);

    var result = try multiway.run(std.testing.allocator, mwConfig(&MW_GROWING, &.{"A"}, 8), null);
    defer result.deinit();
    std.debug.print("  peak unique states: {d} (growing d8, termination {s})\n", .{ result.states.items.len, result.termination.label() });
}

test "bench multiway canonical hashing + deduplication (convergent)" {
    const Dedup = struct {
        pub fn run() void {
            var result = multiway.run(std.testing.allocator, mwConfig(&MW_CONVERGENT, &.{"ABABAB"}, 6), null) catch return;
            result.deinit();
        }
    };
    runBench("multiway hashing + dedup (convergent, d6)", Dedup.run);
}

test "bench multiway cyclic expansion" {
    const Cyclic = struct {
        pub fn run() void {
            var result = multiway.run(std.testing.allocator, mwConfig(&MW_CYCLIC, &.{"A"}, 32), null) catch return;
            result.deinit();
        }
    };
    runBench("multiway cyclic expansion (d32)", Cyclic.run);

    var result = try multiway.run(std.testing.allocator, mwConfig(&MW_CYCLIC, &.{"A"}, 32), null);
    defer result.deinit();
    // Cyclic system converges to 2 canonical states regardless of depth.
    try std.testing.expectEqual(@as(usize, 2), result.states.items.len);
}

test "bench multiway canonical export" {
    var result = try multiway.run(std.testing.allocator, mwConfig(&MW_BRANCHING, &.{"A"}, 5), null);
    defer result.deinit();
    var metrics = try multiway.computeMetrics(std.testing.allocator, &result);
    defer metrics.deinit();

    const Export = struct {
        result_ptr: *const multiway.Result,
        metrics_ptr: *const multiway.Metrics,
        var ctx: ?@This() = null;
        pub fn run() void {
            const c = ctx.?;
            const bytes = multiway.exportCanonicalJson(std.testing.allocator, mwConfig(&MW_BRANCHING, &.{"A"}, 5), c.result_ptr, c.metrics_ptr) catch return;
            std.testing.allocator.free(bytes);
        }
    };
    Export.ctx = .{ .result_ptr = &result, .metrics_ptr = &metrics };
    runBench("multiway canonical export (branching, d5)", Export.run);
    Export.ctx = null;
}

test "bench multiway wdbx persistence + resume load" {
    const path = "zig-out/bench-multiway.jsonl";
    cleanupBenchStore(path);
    defer cleanupBenchStore(path);

    var result = try multiway.run(std.testing.allocator, mwConfig(&MW_BRANCHING, &.{"A"}, 4), null);
    defer result.deinit();
    var metrics = try multiway.computeMetrics(std.testing.allocator, &result);
    defer metrics.deinit();
    const export_json = try multiway.exportCanonicalJson(std.testing.allocator, mwConfig(&MW_BRANCHING, &.{"A"}, 4), &result, &metrics);
    defer std.testing.allocator.free(export_json);

    const Persist = struct {
        export_ptr: []const u8,
        result_ptr: *const multiway.Result,
        var ctx: ?@This() = null;
        pub fn run() void {
            const c = ctx.?;
            multiway.persistToWdbx(std.testing.io, std.testing.allocator, "zig-out/bench-multiway.jsonl", mwConfig(&MW_BRANCHING, &.{"A"}, 4), c.result_ptr, c.export_ptr) catch return;
        }
    };
    Persist.ctx = .{ .export_ptr = export_json, .result_ptr = &result };
    runBench("multiway wdbx persistence (branching, d4)", Persist.run);
    Persist.ctx = null;

    const ResumeLoad = struct {
        pub fn run() void {
            const bytes = multiway.loadExportFromWdbx(std.testing.io, std.testing.allocator, "zig-out/bench-multiway.jsonl", null) catch return;
            std.testing.allocator.free(bytes);
        }
    };
    runBench("multiway resume load (branching, d4)", ResumeLoad.run);
}

fn cleanupBenchStore(path: []const u8) void {
    var buf: [256]u8 = undefined;
    test_helpers.deleteTestFileIfExists(path);
    if (std.fmt.bufPrint(&buf, "{s}.wal", .{path})) |wp| test_helpers.deleteTestFileIfExists(wp) else |_| {}
    if (std.fmt.bufPrint(&buf, "{s}.manifest", .{path})) |mp| test_helpers.deleteTestFileIfExists(mp) else |_| {}
    var epoch: u64 = 0;
    while (epoch < 8) : (epoch += 1) {
        if (std.fmt.bufPrint(&buf, "{s}.seg.{d}.jsonl", .{ path, epoch })) |sp| test_helpers.deleteTestFileIfExists(sp) else |_| {}
    }
}

// Runs after all bench tests (source order), serializing the collected results.
test "write benchmark artifact" {
    writeBenchArtifact();
    try std.testing.expect(bench_results.items.len >= 7);
}

test {
    std.testing.refAllDecls(@This());
}
