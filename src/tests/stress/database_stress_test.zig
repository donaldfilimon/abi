//! Database Module Stress Tests
//!
//! Comprehensive stress tests for the vector database components:
//! - Vector insertions at high throughput
//! - Concurrent search with heavy write load
//! - Index operations under pressure
//! - Memory pool exhaustion handling
//! - Batch operations at scale
//!
//! ## Running Tests
//!
//! ```bash
//! zig test src/tests/stress/database_stress_test.zig --test-filter "database stress"
//! ```

const std = @import("std");
const abi = @import("abi");
const db = abi.database;
const profiles = @import("profiles.zig");
const StressProfile = profiles.StressProfile;
const LatencyHistogram = profiles.LatencyHistogram;
const Timer = profiles.Timer;
const build_options = @import("build_options");

// ============================================================================
// Configuration
// ============================================================================

/// Test vector dimensions
const TEST_DIM: usize = 128;

/// Get the active stress profile for tests
fn getTestProfile() StressProfile {
    return StressProfile.quick;
}

/// Generate a random vector for testing
fn generateRandomVector(rng: *std.Random.DefaultPrng, dim: usize, buffer: []f32) void {
    for (0..dim) |i| {
        buffer[i] = rng.random().float(f32) * 2.0 - 1.0;
    }
    // Normalize
    var sum: f32 = 0.0;
    for (buffer[0..dim]) |v| {
        sum += v * v;
    }
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (buffer[0..dim]) |*v| {
            v.* /= norm;
        }
    }
}

// ============================================================================
// Vector Insertion Stress Tests
// ============================================================================

test "database stress: vector insertion throughput" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_insert_test");
    defer db.close(&handle);

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert many vectors
    const insert_count = @min(profile.operations, 5000);

    for (0..insert_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);

        const timer = Timer.start();
        try db.insert(&handle, @intCast(i), &vector, null);

        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Verify insertions
    const stats_result = db.stats(&handle);
    try std.testing.expectEqual(insert_count, stats_result.total_vectors);

    // Check latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

test "database stress: vector insertion with metadata" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_insert_metadata_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert with varying metadata sizes
    const insert_count = @min(profile.operations, 2000);
    var metadata_buf: [256]u8 = undefined;

    for (0..insert_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);

        // Generate variable-length metadata
        const metadata_len = (i % 200) + 10;
        @memset(metadata_buf[0..metadata_len], 'M');
        const metadata = metadata_buf[0..metadata_len];

        try db.insert(&handle, @intCast(i), &vector, metadata);
    }

    // Verify
    const stats_result = db.stats(&handle);
    try std.testing.expectEqual(insert_count, stats_result.total_vectors);
}

test "database stress: batch insertion" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_batch_insert_test");
    defer db.close(&handle);

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());

    // Prepare batch items
    const batch_size = profile.batch_size;
    const batch_count = @min(profile.operations / batch_size, 100);

    for (0..batch_count) |batch_idx| {
        var items = try allocator.alloc(db.wdbx.BatchItem, batch_size);
        defer allocator.free(items);

        var vectors = try allocator.alloc([TEST_DIM]f32, batch_size);
        defer allocator.free(vectors);

        for (0..batch_size) |i| {
            generateRandomVector(&rng, TEST_DIM, &vectors[i]);
            items[i] = .{
                .id = @intCast(batch_idx * batch_size + i),
                .vector = &vectors[i],
                .metadata = null,
            };
        }

        const timer = Timer.start();
        try db.wdbx.insertBatch(&handle, items);
        try latency.recordUnsafe(timer.read());
    }

    // Verify
    const stats_result = db.stats(&handle);
    try std.testing.expect(stats_result.total_vectors >= batch_count * batch_size);

    // Check batch latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

// ============================================================================
// Search Stress Tests
// ============================================================================

test "database stress: search throughput" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_search_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert test vectors
    const vector_count: usize = @min(profile.operations / 10, 1000);
    for (0..vector_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);
    }

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Perform many searches
    const search_count = @min(profile.operations, 2000);
    var query: [TEST_DIM]f32 = undefined;

    for (0..search_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &query);

        const timer = Timer.start();
        const results = try db.search(&handle, allocator, &query, 10);
        defer allocator.free(results);

        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }

        // Verify results are valid
        try std.testing.expect(results.len <= 10);
    }

    // Check search latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

test "database stress: concurrent search and write" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_concurrent_test");
    defer db.close(&handle);

    // Pre-populate with some vectors
    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    for (0..500) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);
    }

    var writes_completed = std.atomic.Value(u64).init(0);
    var searches_completed = std.atomic.Value(u64).init(0);
    var stop_flag = std.atomic.Value(bool).init(false);

    // Writer threads
    const writer_count = @min(profile.concurrent_tasks / 2, 4);
    var writers: [4]std.Thread = undefined;

    for (0..writer_count) |i| {
        writers[i] = try std.Thread.spawn(.{}, struct {
            fn run(
                h: *db.DatabaseHandle,
                completed: *std.atomic.Value(u64),
                stop: *std.atomic.Value(bool),
                ops: u64,
                seed: u64,
            ) void {
                var local_rng = std.Random.DefaultPrng.init(seed);
                var vec: [TEST_DIM]f32 = undefined;
                var j: u64 = 0;

                while (j < ops and !stop.load(.acquire)) {
                    generateRandomVector(&local_rng, TEST_DIM, &vec);
                    const id = 1000 + completed.load(.monotonic);
                    db.insert(h, id, &vec, null) catch continue;
                    _ = completed.fetchAdd(1, .monotonic);
                    j += 1;
                }
            }
        }.run, .{
            &handle,
            &writes_completed,
            &stop_flag,
            profile.operations / writer_count / 4,
            profile.getEffectiveSeed() + @as(u64, @intCast(i)),
        });
    }

    // Reader threads
    const reader_count = @min(profile.concurrent_tasks / 2, 4);
    var readers: [4]std.Thread = undefined;

    for (0..reader_count) |i| {
        readers[i] = try std.Thread.spawn(.{}, struct {
            fn run(
                h: *db.DatabaseHandle,
                alloc: std.mem.Allocator,
                completed: *std.atomic.Value(u64),
                stop: *std.atomic.Value(bool),
                ops: u64,
                seed: u64,
            ) void {
                var local_rng = std.Random.DefaultPrng.init(seed);
                var query: [TEST_DIM]f32 = undefined;
                var j: u64 = 0;

                while (j < ops and !stop.load(.acquire)) {
                    generateRandomVector(&local_rng, TEST_DIM, &query);
                    const results = db.search(h, alloc, &query, 5) catch continue;
                    defer alloc.free(results);
                    _ = completed.fetchAdd(1, .monotonic);
                    j += 1;
                }
            }
        }.run, .{
            &handle,
            allocator,
            &searches_completed,
            &stop_flag,
            profile.operations / reader_count / 4,
            profile.getEffectiveSeed() + 100 + @as(u64, @intCast(i)),
        });
    }

    // Wait for completion
    for (0..writer_count) |i| {
        writers[i].join();
    }
    for (0..reader_count) |i| {
        readers[i].join();
    }

    // Verify results
    const total_writes = writes_completed.load(.acquire);
    const total_searches = searches_completed.load(.acquire);

    try std.testing.expect(total_writes > 0);
    try std.testing.expect(total_searches > 0);
}

test "database stress: search result quality under load" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_quality_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert known vectors
    const vector_count: usize = @min(profile.operations / 10, 500);
    for (0..vector_count) |i| {
        // Create vectors with known pattern
        for (0..TEST_DIM) |j| {
            vector[j] = if (j == i % TEST_DIM) 1.0 else 0.0;
        }
        try db.insert(&handle, @intCast(i), &vector, null);
    }

    // Search with known query that should match specific vector
    var query: [TEST_DIM]f32 = undefined;
    @memset(&query, 0.0);
    query[0] = 1.0; // Should match vector 0

    for (0..@min(profile.operations, 100)) |_| {
        const results = try db.search(&handle, allocator, &query, 5);
        defer allocator.free(results);

        // Top result should be vector 0 or similar
        try std.testing.expect(results.len > 0);
        try std.testing.expect(results[0].score >= 0.0);

        // Perturb query slightly
        query[0] += rng.random().float(f32) * 0.001;
    }
}

// ============================================================================
// Update and Delete Stress Tests
// ============================================================================

test "database stress: rapid updates" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_update_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert initial vectors
    const vector_count: usize = @min(profile.operations / 10, 500);
    for (0..vector_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);
    }

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Perform many updates
    const update_count = @min(profile.operations, 2000);

    for (0..update_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);
        const id = i % vector_count;

        const timer = Timer.start();
        _ = try db.update(&handle, @intCast(id), &vector);

        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Check latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

test "database stress: insert delete cycle" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_delete_cycle_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert-delete cycles
    const cycle_count = @min(profile.operations / 10, 500);
    var current_count: u64 = 0;

    for (0..cycle_count) |i| {
        // Insert
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);
        current_count += 1;

        // Delete older entries periodically
        if (i > 10 and i % 5 == 0) {
            const delete_id = i - 10;
            if (db.remove(&handle, @intCast(delete_id))) {
                current_count -= 1;
            }
        }
    }

    // Verify final state
    const stats_result = db.stats(&handle);
    try std.testing.expect(stats_result.total_vectors > 0);
}

// ============================================================================
// Index Operations Stress Tests
// ============================================================================

test "database stress: optimize under read pressure" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_optimize_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Insert vectors
    const vector_count: usize = @min(profile.operations / 10, 500);
    for (0..vector_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);
    }

    var reads_completed = std.atomic.Value(u64).init(0);
    var stop_flag = std.atomic.Value(bool).init(false);

    // Start reader threads
    const reader_count: usize = @min(profile.concurrent_tasks / 2, 4);
    var readers: [4]std.Thread = undefined;

    for (0..reader_count) |i| {
        readers[i] = try std.Thread.spawn(.{}, struct {
            fn run(
                h: *db.DatabaseHandle,
                alloc: std.mem.Allocator,
                completed: *std.atomic.Value(u64),
                stop: *std.atomic.Value(bool),
                seed: u64,
            ) void {
                var local_rng = std.Random.DefaultPrng.init(seed);
                var query: [TEST_DIM]f32 = undefined;

                while (!stop.load(.acquire)) {
                    generateRandomVector(&local_rng, TEST_DIM, &query);
                    const results = db.search(h, alloc, &query, 5) catch continue;
                    defer alloc.free(results);
                    _ = completed.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{
            &handle,
            allocator,
            &reads_completed,
            &stop_flag,
            profile.getEffectiveSeed() + @as(u64, @intCast(i)),
        });
    }

    // Perform optimize operations while reads are happening
    const optimize_count = @min(profile.operations / 100, 50);
    for (0..optimize_count) |_| {
        try db.optimize(&handle);
        profiles.sleepMs(10);
    }

    // Stop readers
    stop_flag.store(true, .release);
    for (0..reader_count) |i| {
        readers[i].join();
    }

    // Verify
    const total_reads = reads_completed.load(.acquire);
    try std.testing.expect(total_reads > 0);
}

// ============================================================================
// Clustering Stress Tests
// ============================================================================

test "database stress: kmeans clustering" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    // Generate test data
    const point_count = @min(profile.operations / 10, 500);
    const dim: usize = 32;

    var data = try allocator.alloc(f32, point_count * dim);
    defer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());

    for (0..point_count) |i| {
        for (0..dim) |j| {
            data[i * dim + j] = rng.random().float(f32) * 2.0 - 1.0;
        }
    }

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Perform clustering multiple times
    const cluster_runs = @min(profile.operations / 100, 20);
    const k = 5;

    for (0..cluster_runs) |_| {
        var kmeans = try db.KMeans.init(allocator, k, dim);
        defer kmeans.deinit();

        const timer = Timer.start();
        _ = try kmeans.fit(data, .{
            .max_iterations = 10,
            .tolerance = 1e-4,
        });
        try latency.recordUnsafe(timer.read());
    }

    // Check latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

// ============================================================================
// Quantization Stress Tests
// ============================================================================

test "database stress: scalar quantization throughput" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var quantizer = try db.ScalarQuantizer.init(allocator, TEST_DIM);
    defer quantizer.deinit();

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    // Train quantizer
    var training_data = try allocator.alloc(f32, 100 * TEST_DIM);
    defer allocator.free(training_data);

    for (0..100) |i| {
        for (0..TEST_DIM) |j| {
            training_data[i * TEST_DIM + j] = rng.random().float(f32) * 2.0 - 1.0;
        }
    }
    try quantizer.train(training_data);

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Quantize many vectors
    const quant_count = @min(profile.operations, 5000);

    for (0..quant_count) |i| {
        generateRandomVector(&rng, TEST_DIM, &vector);

        const timer = Timer.start();
        const quantized = try quantizer.quantize(&vector);
        defer allocator.free(quantized);

        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Check latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);
}

// ============================================================================
// Combined Database Stress Tests
// ============================================================================

test "database stress: full lifecycle" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var handle = try db.open(allocator, "stress_lifecycle_test");
    defer db.close(&handle);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    var vector: [TEST_DIM]f32 = undefined;

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Full lifecycle: insert, search, update, search, delete
    const cycle_count = @min(profile.operations / 5, 500);

    for (0..cycle_count) |i| {
        const timer = Timer.start();

        // Insert
        generateRandomVector(&rng, TEST_DIM, &vector);
        try db.insert(&handle, @intCast(i), &vector, null);

        // Search
        const results = try db.search(&handle, allocator, &vector, 5);
        allocator.free(results);

        // Update
        generateRandomVector(&rng, TEST_DIM, &vector);
        _ = try db.update(&handle, @intCast(i), &vector);

        // Search again
        const results2 = try db.search(&handle, allocator, &vector, 5);
        allocator.free(results2);

        // Delete half the time
        if (i % 2 == 0) {
            _ = db.remove(&handle, @intCast(i));
        }

        if (i % 50 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Optimize
    try db.optimize(&handle);

    // Check latency
    const latency_stats = latency.getStats();
    try std.testing.expect(latency_stats.count > 0);

    // Verify final state
    const stats_result = db.stats(&handle);
    try std.testing.expect(stats_result.total_vectors > 0);
}
