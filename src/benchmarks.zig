const std = @import("std");
const test_helpers = @import("testing/test_helpers.zig");
const wdbx = @import("features/wdbx/mod.zig");
const constitution = @import("features/ai/constitution.zig");
const gpu_mod = @import("features/gpu/mod.zig");
const router = @import("features/ai/router.zig");

const analyzeSentiment = router.analyzeSentiment;
const selectBestProfile = router.selectBestProfile;

const ITERATIONS = 100;
const WARMUP = 3;

fn runBench(comptime label: []const u8, comptime fn_run: anytype) void {
    var i: usize = 0;
    while (i < WARMUP) : (i += 1) {
        fn_run();
    }

    var total_ms: f64 = 0;
    i = 0;
    while (i < ITERATIONS) : (i += 1) {
        const before = std.c.mach_absolute_time();
        fn_run();
        const after = std.c.mach_absolute_time();
        total_ms += @as(f64, @floatFromInt(after - before)) / 1_000_000.0;
    }

    const avg_ms = total_ms / @as(f64, @floatFromInt(ITERATIONS));
    std.debug.print("bench [{s}]: avg {d:.3}ms ({d} iters)\n", .{ label, avg_ms, ITERATIONS });
}

test "bench vector dot product" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    const DotBench = struct {
        pub fn run() void {
            const ops = gpu_mod.vectorOps();
            _ = ops.dot(&a, &b) catch unreachable;
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
                _ = store.putVector(&vals) catch unreachable;
            }
        }
    };

    runBench("HNSW insert", InsertBench.run);
}

test "bench HNSW search" {
    const SearchBench = struct {
        pub fn run() void {
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
                _ = store.putVector(&vals) catch unreachable;
            }
            const query = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
            const results = store.search(&query, 10) catch unreachable;
            std.testing.allocator.free(results);
        }
    };

    runBench("HNSW search", SearchBench.run);
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
                _ = store.appendBlock(profile, @intCast(i), @intCast(i + 1), metadata) catch unreachable;
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
                store.store(key, val) catch unreachable;
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

test {
    std.testing.refAllDecls(@This());
}
