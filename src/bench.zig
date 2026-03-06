//! ABI Framework — Comprehensive Benchmark Suite
//!
//! Measures performance of SIMD operations, HNSW indexing, database
//! operations, persona routing, and inference engine throughput.

const std = @import("std");
const root = @import("root.zig");

const Timer = struct {
    start: i128,

    fn begin() Timer {
        return .{ .start = std.time.nanoTimestamp() };
    }

    fn elapsedNs(self: Timer) u64 {
        const now = std.time.nanoTimestamp();
        return @intCast(now - self.start);
    }

    fn elapsedUs(self: Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsedNs())) / 1000.0;
    }

    fn elapsedMs(self: Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsedNs())) / 1_000_000.0;
    }
};

fn printResult(name: []const u8, ops: u64, elapsed_ns: u64) void {
    const elapsed_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const ops_per_sec = @as(f64, @floatFromInt(ops)) / elapsed_sec;
    const ns_per_op = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(ops));
    std.debug.print("  {s:<35} {d:>12.0} ops/s  ({d:.1} ns/op)\n", .{ name, ops_per_sec, ns_per_op });
}

fn benchSimd(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== SIMD Benchmarks ===\n", .{});

    const dims = [_]u32{ 128, 384, 768, 1536 };

    for (dims) |dim| {
        const a = try allocator.alloc(f32, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, dim);
        defer allocator.free(b);

        // Initialize with random data.
        var rng = std.Random.DefaultPrng.init(42);
        for (a) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        for (b) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        const iterations: u64 = 100_000;

        // Cosine similarity.
        {
            const t = Timer.begin();
            for (0..iterations) |_| {
                _ = root.Distance.cosine(a, b);
            }
            var buf: [64]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "cosine({d}-dim)", .{dim}) catch "cosine";
            printResult(label, iterations, t.elapsedNs());
        }

        // L2 squared.
        {
            const t = Timer.begin();
            for (0..iterations) |_| {
                _ = root.Distance.l2Squared(a, b);
            }
            var buf: [64]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "l2squared({d}-dim)", .{dim}) catch "l2squared";
            printResult(label, iterations, t.elapsedNs());
        }

        // Inner product.
        {
            const t = Timer.begin();
            for (0..iterations) |_| {
                _ = root.Distance.innerProduct(a, b);
            }
            var buf: [64]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "innerProduct({d}-dim)", .{dim}) catch "innerProduct";
            printResult(label, iterations, t.elapsedNs());
        }
    }
}

fn benchHnsw(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== HNSW Benchmarks ===\n", .{});

    const dim: u32 = 128;
    const counts = [_]u32{ 1000, 5000, 10000 };

    for (counts) |count| {
        var index = root.HnswIndex.init(allocator, .{
            .dimension = dim,
            .M = 16,
            .M0 = 32,
            .ef_construction = 100,
            .ef_search = 50,
        });
        defer index.deinit();

        var rng = std.Random.DefaultPrng.init(42);

        // Insert.
        {
            const t = Timer.begin();
            for (0..count) |i| {
                const vec = try allocator.alloc(f32, dim);
                defer allocator.free(vec);
                for (vec) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
                try index.insert(@intCast(i), vec);
            }
            var buf: [64]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "hnsw insert (n={d})", .{count}) catch "hnsw insert";
            printResult(label, count, t.elapsedNs());
        }

        // Search.
        {
            const search_iters: u64 = 1000;
            const t = Timer.begin();
            for (0..search_iters) |_| {
                const query = try allocator.alloc(f32, dim);
                defer allocator.free(query);
                for (query) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
                const results = try index.search(query, 10);
                allocator.free(results);
            }
            var buf: [64]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "hnsw search k=10 (n={d})", .{count}) catch "hnsw search";
            printResult(label, search_iters, t.elapsedNs());
        }
    }
}

fn benchPersonas() void {
    std.debug.print("\n=== Persona Routing Benchmarks ===\n", .{});

    const inputs = [_][]const u8{
        "How do I optimize the database API performance?",
        "I feel really confused and frustrated with this problem",
        "Write me a creative story about space exploration",
        "What is the capital of France?",
        "pub fn main() !void { return error.NotImplemented; }",
    };

    const iterations: u64 = 10_000;
    var moderator = root.AbiModerator.init(std.heap.page_allocator);
    defer moderator.deinit();

    const t = Timer.begin();
    for (0..iterations) |i| {
        const input = inputs[i % inputs.len];
        _ = moderator.route(input, null) catch {};
    }
    printResult("persona routing", iterations, t.elapsedNs());
}

fn benchSampler() void {
    std.debug.print("\n=== Sampler Benchmarks ===\n", .{});

    var logits: [1024]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(42);
    for (&logits) |*l| l.* = rng.random().float(f32) * 10.0 - 5.0;

    const iterations: u64 = 100_000;
    var sampler = root.sampler.Sampler.initWithSeed(.{}, 42);

    const t = Timer.begin();
    for (0..iterations) |_| {
        var logits_copy = logits;
        _ = sampler.sample(&logits_copy);
    }
    printResult("token sampling (vocab=1024)", iterations, t.elapsedNs());
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print(
        \\
        \\╔══════════════════════════════════════╗
        \\║   ABI Framework v{s} Benchmarks  ║
        \\╚══════════════════════════════════════╝
        \\
    , .{root.version()});

    try benchSimd(allocator);
    try benchHnsw(allocator);
    benchPersonas();
    benchSampler();

    std.debug.print("\nBenchmark complete.\n", .{});
}
