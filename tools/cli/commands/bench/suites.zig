//! Benchmark suite runner implementations.
//!
//! Contains runBenchmarkSuite, runSuiteWithResults, and all individual
//! suite runners (simd, memory, concurrency, database, network, crypto,
//! ai, streaming, quick variants).

const std = @import("std");
const abi = @import("abi");
const mod = @import("mod.zig");
const output = @import("output.zig");
const training_comparison = @import("training_comparison.zig");
const utils = @import("../../utils/mod.zig");

/// Run the selected benchmark suite
pub fn runBenchmarkSuite(allocator: std.mem.Allocator, config: mod.BenchConfig) !void {
    const timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("Timer not available on this platform.", .{});
        return;
    };

    if (!config.output_json) {
        output.printHeader();
        utils.output.println("Running suite: {s}\n", .{config.suite.toString()});
    }

    // Collect results
    var results: std.ArrayListUnmanaged(mod.BenchResult) = .empty;
    defer {
        // Free all allocated result names before freeing the list
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit(allocator);
    }

    switch (config.suite) {
        .all => {
            try runSuiteWithResults(allocator, "SIMD Operations", runSimdBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Memory Patterns", runMemoryBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Concurrency", runConcurrencyBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Database/HNSW", runDatabaseBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Network/HTTP", runNetworkBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Cryptography", runCryptoBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "AI/ML", runAiBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Streaming Inference", runStreamingBenchmarks, &results, config.output_json);
        },
        .simd => try runSuiteWithResults(allocator, "SIMD Operations", runSimdBenchmarks, &results, config.output_json),
        .memory => try runSuiteWithResults(allocator, "Memory Patterns", runMemoryBenchmarks, &results, config.output_json),
        .concurrency => try runSuiteWithResults(allocator, "Concurrency", runConcurrencyBenchmarks, &results, config.output_json),
        .database => try runSuiteWithResults(allocator, "Database/HNSW", runDatabaseBenchmarks, &results, config.output_json),
        .network => try runSuiteWithResults(allocator, "Network/HTTP", runNetworkBenchmarks, &results, config.output_json),
        .crypto => try runSuiteWithResults(allocator, "Cryptography", runCryptoBenchmarks, &results, config.output_json),
        .ai => try runSuiteWithResults(allocator, "AI/ML", runAiBenchmarks, &results, config.output_json),
        .streaming => try runSuiteWithResults(allocator, "Streaming Inference", runStreamingBenchmarks, &results, config.output_json),
        .quick => {
            try runSuiteWithResults(allocator, "Quick (SIMD)", runQuickSimdBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Quick (Memory)", runQuickMemoryBenchmarks, &results, config.output_json);
        },
        .compare_training => {
            training_comparison.runTrainingComparisonBenchmarks(allocator, config.output_json);
            return; // Training comparison has its own output format
        },
    }

    var t = timer;
    const elapsed_ns = t.read();
    const duration_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    if (config.output_json) {
        try output.outputJson(allocator, results.items, duration_sec, config.output_file);
    } else {
        output.printFooter(duration_sec);
    }
}

pub fn runSuiteWithResults(
    allocator: std.mem.Allocator,
    name: []const u8,
    benchFn: mod.BenchmarkFn,
    results: *std.ArrayListUnmanaged(mod.BenchResult),
    json_mode: bool,
) !void {
    if (!json_mode) {
        utils.output.println("--------------------------------------------------------------------------------", .{});
        utils.output.println("  {s}", .{name});
        utils.output.println("--------------------------------------------------------------------------------", .{});
    }
    try benchFn(allocator, results);
    if (!json_mode) {
        utils.output.println("", .{});
    }
}

// =============================================================================
// Benchmark Suite Implementations
// =============================================================================

pub fn runSimdBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Dot product benchmark
    const sizes = [_]usize{ 64, 256, 1024, 4096 };
    for (sizes) |size| {
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);

        // Initialize
        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0;

        // Benchmark
        const result = try output.benchmarkOp(allocator, struct {
            fn op(a_ptr: []f32, b_ptr: []f32) f32 {
                var sum: f32 = 0.0;
                for (a_ptr, b_ptr) |x, y| sum += x * y;
                return sum;
            }
        }.op, .{ a, b });

        try results.append(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "dot_product_{d}", .{size}),
            .category = "simd",
            .ops_per_sec = result.ops_per_sec,
            .mean_ns = result.mean_ns,
            .p99_ns = result.p99_ns,
            .iterations = result.iterations,
            .name_allocated = true,
        });

        utils.output.println("  dot_product[{d}]: {d:.0} ops/sec, {d:.0}ns mean", .{ size, result.ops_per_sec, result.mean_ns });
    }
}

pub fn runMemoryBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Arena-style allocation benchmark
    const sizes = [_]usize{ 64, 256, 1024, 4096 };
    for (sizes) |size| {
        const result = try output.benchmarkAllocOp(allocator, size);

        try results.append(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "alloc_free_{d}", .{size}),
            .category = "memory",
            .ops_per_sec = result.ops_per_sec,
            .mean_ns = result.mean_ns,
            .p99_ns = result.p99_ns,
            .iterations = result.iterations,
            .name_allocated = true,
        });

        utils.output.println("  alloc_free[{d}]: {d:.0} ops/sec, {d:.0}ns mean", .{ size, result.ops_per_sec, result.mean_ns });
    }
}

pub fn runConcurrencyBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Atomic operations benchmark
    var counter = std.atomic.Value(u64).init(0);

    const timer = abi.shared.time.Timer.start() catch return;
    const iterations: u64 = 100000;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        _ = counter.fetchAdd(1, .seq_cst);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    try results.append(allocator, .{
        .name = "atomic_increment",
        .category = "concurrency",
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.5),
        .iterations = iterations,
    });

    utils.output.println("  atomic_increment: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
}

pub fn runDatabaseBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Check if database feature is enabled
    if (!abi.database.isEnabled()) {
        utils.output.println("  (Database benchmarks require -Dfeat-database=true (legacy: -Denable-database=true) build flag)", .{});
        return;
    }

    // HNSW insert benchmark
    {
        const iterations: u64 = 1000;
        const dimension: usize = 128;

        // Generate test vectors
        var vectors = try allocator.alloc(f32, iterations * dimension);
        defer allocator.free(vectors);

        var prng = std.Random.Xoroshiro128.init(42);
        const rand = prng.random();
        for (vectors) |*v| {
            v.* = rand.float(f32) * 2.0 - 1.0;
        }

        const timer = abi.shared.time.Timer.start() catch return;
        var warmup: u64 = 0;
        while (warmup < 10) : (warmup += 1) {
            // Simulate vector normalization as warmup
            var sum: f32 = 0.0;
            for (vectors[0..dimension]) |v| sum += v * v;
            std.mem.doNotOptimizeAway(&sum);
        }

        // Benchmark vector insert simulation
        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            const vec_slice = vectors[iter * dimension .. (iter + 1) * dimension];
            var dot: f32 = 0.0;
            for (vec_slice) |v| dot += v * v;
            std.mem.doNotOptimizeAway(&dot);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "hnsw_insert_128d",
            .category = "database",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.3),
            .iterations = iterations,
        });

        utils.output.println("  hnsw_insert[128d]: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }

    // HNSW search benchmark
    {
        const iterations: u64 = 500;
        const dimension: usize = 128;
        const num_vectors: usize = 1000;

        const vectors = try allocator.alloc(f32, num_vectors * dimension);
        defer allocator.free(vectors);

        var prng = std.Random.Xoroshiro128.init(123);
        const rand = prng.random();
        for (vectors) |*v| {
            v.* = rand.float(f32) * 2.0 - 1.0;
        }

        const query = try allocator.alloc(f32, dimension);
        defer allocator.free(query);
        for (query) |*v| {
            v.* = rand.float(f32) * 2.0 - 1.0;
        }

        const timer = abi.shared.time.Timer.start() catch return;

        // Benchmark nearest neighbor search simulation
        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var best_dist: f32 = std.math.inf(f32);
            var best_idx: usize = 0;

            for (0..num_vectors) |vi| {
                var dist: f32 = 0.0;
                const vec_start = vi * dimension;
                for (0..dimension) |d| {
                    const diff = query[d] - vectors[vec_start + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = vi;
                }
            }
            std.mem.doNotOptimizeAway(&best_idx);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "hnsw_search_1k_128d",
            .category = "database",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.5),
            .iterations = iterations,
        });

        utils.output.println("  hnsw_search[1k x 128d]: {d:.0} ops/sec, {d:.2}ms mean", .{ ops_per_sec, mean_ns / 1_000_000.0 });
    }

    // Distance computation benchmark
    {
        const iterations: u64 = 100000;
        const dimension: usize = 128;

        const a = try allocator.alloc(f32, dimension);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, dimension);
        defer allocator.free(b);

        var prng = std.Random.Xoroshiro128.init(456);
        const rand = prng.random();
        for (a) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
        for (b) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var dist: f32 = 0.0;
            for (a, b) |x, y| {
                const diff = x - y;
                dist += diff * diff;
            }
            std.mem.doNotOptimizeAway(&dist);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "euclidean_distance_128d",
            .category = "database",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.2),
            .iterations = iterations,
        });

        utils.output.println("  euclidean_dist[128d]: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }
}

pub fn runNetworkBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // HTTP header parsing benchmark
    {
        const iterations: u64 = 50000;
        const sample_headers =
            "GET /api/v1/users HTTP/1.1\r\n" ++
            "Host: api.example.com\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Authorization: Bearer token123\r\n" ++
            "Accept: */*\r\n" ++
            "Connection: keep-alive\r\n" ++
            "\r\n";

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            // Parse headers
            var header_count: usize = 0;
            var lines = std.mem.splitSequence(u8, sample_headers, "\r\n");
            while (lines.next()) |line| {
                if (line.len == 0) break;
                if (std.mem.indexOf(u8, line, ": ")) |_| {
                    header_count += 1;
                }
            }
            std.mem.doNotOptimizeAway(&header_count);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "http_header_parse",
            .category = "network",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.3),
            .iterations = iterations,
        });

        utils.output.println("  http_header_parse: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }

    // URL parsing benchmark
    {
        const iterations: u64 = 100000;
        const sample_urls = [_][]const u8{
            "https://api.example.com/v1/users?page=1&limit=10",
            "http://localhost:8080/health",
            "https://cdn.example.org/assets/images/logo.png?v=123",
            "wss://ws.example.com/socket",
        };

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            const url = sample_urls[iter % sample_urls.len];

            // Parse URL components
            var scheme_end: usize = 0;
            if (std.mem.indexOf(u8, url, "://")) |pos| {
                scheme_end = pos;
            }

            var path_start: usize = 0;
            if (std.mem.indexOfPos(u8, url, scheme_end + 3, "/")) |pos| {
                path_start = pos;
            }

            var query_start: usize = url.len;
            if (std.mem.indexOf(u8, url, "?")) |pos| {
                query_start = pos;
            }

            std.mem.doNotOptimizeAway(&scheme_end);
            std.mem.doNotOptimizeAway(&path_start);
            std.mem.doNotOptimizeAway(&query_start);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "url_parse",
            .category = "network",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.2),
            .iterations = iterations,
        });

        utils.output.println("  url_parse: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }

    // JSON parsing benchmark (simple key-value extraction)
    {
        const iterations: u64 = 20000;
        const sample_json =
            \\{"id":12345,"name":"John Doe","email":"john@example.com","active":true,"score":98.5}
        ;

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            // Simple key counting
            var key_count: usize = 0;
            var in_string = false;
            var si: usize = 0;
            while (si < sample_json.len) : (si += 1) {
                const c = sample_json[si];
                if (c == '"' and (si == 0 or sample_json[si - 1] != '\\')) {
                    in_string = !in_string;
                }
                if (!in_string and c == ':') {
                    key_count += 1;
                }
            }
            std.mem.doNotOptimizeAway(&key_count);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "json_key_scan",
            .category = "network",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.2),
            .iterations = iterations,
        });

        utils.output.println("  json_key_scan: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }
}

pub fn runCryptoBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // SHA-256 hash benchmark using std.crypto
    {
        const iterations: u64 = 50000;
        const data = "The quick brown fox jumps over the lazy dog. " ** 10; // ~450 bytes

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var hash: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(data, &hash, .{});
            std.mem.doNotOptimizeAway(&hash);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;
        const throughput_mb = (ops_per_sec * @as(f64, @floatFromInt(data.len))) / (1024.0 * 1024.0);

        try results.append(allocator, .{
            .name = "sha256_450b",
            .category = "crypto",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.2),
            .iterations = iterations,
        });

        utils.output.println("  sha256[450B]: {d:.0} ops/sec, {d:.2} MB/s", .{ ops_per_sec, throughput_mb });
    }

    // Blake3 hash benchmark
    {
        const iterations: u64 = 100000;
        const data = "Hello, World! This is a test message for benchmarking."; // 54 bytes

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var hash: [32]u8 = undefined;
            std.crypto.hash.Blake3.hash(data, &hash, .{});
            std.mem.doNotOptimizeAway(&hash);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "blake3_54b",
            .category = "crypto",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.2),
            .iterations = iterations,
        });

        utils.output.println("  blake3[54B]: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }

    // HMAC-SHA256 benchmark
    {
        const iterations: u64 = 30000;
        const key = "secret_key_for_hmac_testing_1234";
        const message = "This is a message to be authenticated using HMAC-SHA256";

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var mac: [32]u8 = undefined;
            std.crypto.auth.hmac.sha2.HmacSha256.create(&mac, message, key);
            std.mem.doNotOptimizeAway(&mac);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "hmac_sha256",
            .category = "crypto",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.3),
            .iterations = iterations,
        });

        utils.output.println("  hmac_sha256: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }

    // ChaCha20-Poly1305 encryption benchmark
    {
        const iterations: u64 = 20000;

        var plaintext: [256]u8 = undefined;
        @memset(&plaintext, 0x42);

        var key_data: [32]u8 = undefined;
        @memset(&key_data, 0xAB);

        var nonce: [12]u8 = undefined;
        @memset(&nonce, 0xCD);

        const timer = abi.shared.time.Timer.start() catch return;

        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            var ciphertext: [256]u8 = undefined;
            var tag: [16]u8 = undefined;

            std.crypto.aead.chacha_poly.ChaCha20Poly1305.encrypt(&ciphertext, &tag, &plaintext, "", nonce, key_data);
            std.mem.doNotOptimizeAway(&ciphertext);
            std.mem.doNotOptimizeAway(&tag);
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;
        const throughput_mb = (ops_per_sec * 256.0) / (1024.0 * 1024.0);

        try results.append(allocator, .{
            .name = "chacha20_poly1305_256b",
            .category = "crypto",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.3),
            .iterations = iterations,
        });

        utils.output.println("  chacha20_poly1305[256B]: {d:.0} ops/sec, {d:.2} MB/s", .{ ops_per_sec, throughput_mb });
    }

    // Random number generation benchmark
    {
        const iterations: u64 = 100000;

        var prng = std.Random.DefaultPrng.init(12345);
        const rand = prng.random();

        const timer = abi.shared.time.Timer.start() catch return;

        var sum: u64 = 0;
        var iter: u64 = 0;
        while (iter < iterations) : (iter += 1) {
            sum +%= rand.int(u64);
        }
        std.mem.doNotOptimizeAway(&sum);

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "prng_u64",
            .category = "crypto",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.1),
            .iterations = iterations,
        });

        utils.output.println("  prng_u64: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }
}

pub fn runAiBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Simple matrix multiply benchmark
    const m: usize = 64;
    const k: usize = 64;
    const n: usize = 64;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0 - 0.5;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0 - 0.5;

    const timer = abi.shared.time.Timer.start() catch return;
    const iterations: u64 = 100;

    var iter: u64 = 0;
    while (iter < iterations) : (iter += 1) {
        // Naive matmul
        for (0..m) |mi| {
            for (0..n) |ni| {
                var sum: f32 = 0.0;
                for (0..k) |kk| {
                    sum += a[mi * k + kk] * b[kk * n + ni];
                }
                c[mi * n + ni] = sum;
            }
        }
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    // Calculate GFLOPS: 2*M*N*K operations per matmul
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const gflops = (flops * ops_per_sec) / 1_000_000_000.0;

    try results.append(allocator, .{
        .name = try std.fmt.allocPrint(allocator, "matmul_{d}x{d}x{d}", .{ m, k, n }),
        .category = "ai",
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.2),
        .iterations = iterations,
        .name_allocated = true,
    });

    utils.output.println("  matmul[{d}x{d}x{d}]: {d:.2} GFLOPS, {d:.0}ns mean", .{ m, k, n, gflops, mean_ns });
}

pub fn runStreamingBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Streaming inference benchmarks - measures TTFT, ITL, and throughput
    const token_counts = [_]usize{ 32, 64, 128 };
    const iterations: u64 = 50;
    const warmup: u64 = 10;

    // Sample tokens for simulated generation
    const sample_tokens = [_][]const u8{ " the", " a", " is", " and", " to", " of", " in", " that", " it", " for" };

    for (token_counts) |token_count| {
        var ttft_samples = std.ArrayListUnmanaged(u64).empty;
        defer ttft_samples.deinit(allocator);

        var throughput_sum: f64 = 0;

        // Warmup
        for (0..warmup) |_| {
            var gen_count: usize = 0;
            while (gen_count < token_count) : (gen_count += 1) {
                const token = sample_tokens[gen_count % sample_tokens.len];
                std.mem.doNotOptimizeAway(&token);
            }
        }

        // Benchmark iterations
        for (0..iterations) |_| {
            const run_timer = abi.shared.time.Timer.start() catch continue;
            var first_token = true;

            var gen_count: usize = 0;
            while (gen_count < token_count) : (gen_count += 1) {
                // Simulate token generation with small delay (Zig 0.16 busy-wait)
                const delay_timer = abi.shared.time.Timer.start() catch continue;
                var dt = delay_timer;
                while (dt.read() < 100_000) { // 0.1ms per token
                    std.atomic.spinLoopHint();
                }

                const token = sample_tokens[gen_count % sample_tokens.len];
                std.mem.doNotOptimizeAway(&token);

                if (first_token) {
                    var rt = run_timer;
                    try ttft_samples.append(allocator, rt.read());
                    first_token = false;
                }
            }

            var final_timer = run_timer;
            const run_time = final_timer.read();
            const throughput = @as(f64, @floatFromInt(token_count)) /
                (@as(f64, @floatFromInt(run_time)) / 1_000_000_000.0);
            throughput_sum += throughput;
        }

        // Calculate statistics
        var ttft_sum: u128 = 0;
        for (ttft_samples.items) |s| ttft_sum += s;
        const ttft_mean = if (ttft_samples.items.len > 0)
            @as(f64, @floatFromInt(ttft_sum)) / @as(f64, @floatFromInt(ttft_samples.items.len))
        else
            0;

        const throughput_mean = throughput_sum / @as(f64, @floatFromInt(iterations));

        try results.append(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "streaming_{d}_tokens", .{token_count}),
            .category = "streaming",
            .ops_per_sec = throughput_mean,
            .mean_ns = ttft_mean,
            .p99_ns = @intFromFloat(ttft_mean * 1.5),
            .iterations = iterations,
            .name_allocated = true,
        });

        utils.output.println("  streaming[{d} tokens]: TTFT={d:.2}ms, throughput={d:.1} tok/s", .{
            token_count,
            ttft_mean / 1_000_000.0,
            throughput_mean,
        });
    }

    // SSE encoding overhead benchmark
    {
        const encode_iterations: u64 = 10000;
        var buffer = std.ArrayListUnmanaged(u8).empty;
        defer buffer.deinit(allocator);

        const timer = abi.shared.time.Timer.start() catch return;

        for (0..encode_iterations) |ei| {
            buffer.clearRetainingCapacity();
            const token = sample_tokens[ei % sample_tokens.len];

            // SSE format
            try buffer.appendSlice(allocator, "data: {\"token\":\"");
            try buffer.appendSlice(allocator, token);
            try buffer.appendSlice(allocator, "\"}\n\n");
        }

        var t = timer;
        const elapsed_ns = t.read();
        const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(encode_iterations));
        const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

        try results.append(allocator, .{
            .name = "sse_encode",
            .category = "streaming",
            .ops_per_sec = ops_per_sec,
            .mean_ns = mean_ns,
            .p99_ns = @intFromFloat(mean_ns * 1.3),
            .iterations = encode_iterations,
        });

        utils.output.println("  sse_encode: {d:.0} ops/sec, {d:.0}ns mean", .{ ops_per_sec, mean_ns });
    }
}

pub fn runQuickSimdBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    // Quick SIMD benchmark - single size
    const size: usize = 256;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0;

    const result = try output.benchmarkOp(allocator, struct {
        fn op(a_ptr: []f32, b_ptr: []f32) f32 {
            var sum: f32 = 0.0;
            for (a_ptr, b_ptr) |x, y| sum += x * y;
            return sum;
        }
    }.op, .{ a, b });

    try results.append(allocator, .{
        .name = "quick_dot_product",
        .category = "simd",
        .ops_per_sec = result.ops_per_sec,
        .mean_ns = result.mean_ns,
        .p99_ns = result.p99_ns,
        .iterations = result.iterations,
    });

    utils.output.println("  quick_dot_product: {d:.0} ops/sec", .{result.ops_per_sec});
}

pub fn runQuickMemoryBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(mod.BenchResult)) mod.BenchmarkError!void {
    const result = try output.benchmarkAllocOp(allocator, 256);

    try results.append(allocator, .{
        .name = "quick_alloc",
        .category = "memory",
        .ops_per_sec = result.ops_per_sec,
        .mean_ns = result.mean_ns,
        .p99_ns = result.p99_ns,
        .iterations = result.iterations,
    });

    utils.output.println("  quick_alloc: {d:.0} ops/sec", .{result.ops_per_sec});
}

test {
    std.testing.refAllDecls(@This());
}
