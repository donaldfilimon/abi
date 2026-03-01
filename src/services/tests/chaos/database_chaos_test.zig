//! Database Chaos Tests
//!
//! Tests for database module components under failure conditions:
//! - Index integrity after allocation failures
//! - Transaction rollback on random failures
//! - Search correctness after partial writes
//! - Vector store recovery after crashes
//!
//! These tests verify that the database system:
//! 1. Maintains data integrity under adverse conditions
//! 2. Handles failures gracefully without corruption
//! 3. Recovers correctly after chaos ends

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const database = abi.features.database;
const simd = abi.services.shared.simd;
const chaos = @import("mod.zig");
const helpers = @import("../helpers.zig");

// ============================================================================
// Test Helpers
// ============================================================================

/// Generate a random vector for testing using shared helper
fn generateRandomVector(rng: *std.Random.DefaultPrng, dims: usize, allocator: std.mem.Allocator) ![]f32 {
    return helpers.generateRandomVectorAlloc(allocator, rng, dims);
}

/// Compute cosine similarity using shared SIMD implementation
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
}

// ============================================================================
// Vector Database Chaos Tests
// ============================================================================

test "database chaos: vector insertion survives allocation failures" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 12345);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.05, // 5% failure rate
        .warmup_ops = 20,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    // Create database handle
    var handle = database.open(allocator, "chaos_test_db") catch |err| {
        switch (err) {
            error.OutOfMemory => return, // Expected under chaos
            else => return err,
        }
    };
    defer database.close(&handle);

    // Try to insert vectors under chaos
    var rng = std.Random.DefaultPrng.init(54321);
    var successful_inserts: u32 = 0;
    var failed_inserts: u32 = 0;

    var i: u64 = 0;
    while (i < 100) : (i += 1) {
        const vec = generateRandomVector(&rng, 128, allocator) catch |err| {
            switch (err) {
                error.OutOfMemory => {
                    failed_inserts += 1;
                    continue;
                },
                else => return err,
            }
        };
        defer allocator.free(vec);

        database.insert(&handle, i, vec, null) catch |err| {
            switch (err) {
                error.OutOfMemory => {
                    failed_inserts += 1;
                    continue;
                },
                else => return err,
            }
        };
        successful_inserts += 1;
    }

    // Verify some insertions succeeded
    try std.testing.expect(successful_inserts > 0);

    // Verify database is in consistent state
    const db_stats = database.stats(&handle);
    try std.testing.expect(db_stats.vector_count == successful_inserts);

    const chaos_stats = chaos_ctx.getStats();
    std.log.info("Insert chaos: success={d}, failed={d}, faults={d}", .{
        successful_inserts,
        failed_inserts,
        chaos_stats.faults_injected,
    });
}

test "database chaos: search returns correct results after partial writes" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 23456);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.03,
        .warmup_ops = 50, // Let initial inserts complete
    });

    // Create database and insert baseline vectors WITHOUT chaos
    var handle = try database.open(allocator, "chaos_search_db");
    defer database.close(&handle);

    var rng = std.Random.DefaultPrng.init(67890);
    const dims: usize = 64;

    // Insert some vectors without chaos to establish baseline
    var baseline_vecs: std.ArrayListUnmanaged([]f32) = .empty;
    defer {
        for (baseline_vecs.items) |v| allocator.free(v);
        baseline_vecs.deinit(allocator);
    }

    var i: u64 = 0;
    while (i < 20) : (i += 1) {
        const vec = try generateRandomVector(&rng, dims, allocator);
        try baseline_vecs.append(allocator, vec);
        try database.insert(&handle, i, vec, null);
    }

    // Now enable chaos and insert more vectors
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var chaos_inserts: u32 = 0;
    while (i < 50) : (i += 1) {
        const vec = generateRandomVector(&rng, dims, allocator) catch continue;
        defer allocator.free(vec);

        database.insert(&handle, i, vec, null) catch continue;
        chaos_inserts += 1;
    }

    // Search should still find baseline vectors
    const query = try generateRandomVector(&rng, dims, allocator);
    defer allocator.free(query);

    const results = database.search(&handle, allocator, query, 10) catch |err| {
        switch (err) {
            error.OutOfMemory => {
                // Search failed but database should still be consistent
                const final_stats = database.stats(&handle);
                try std.testing.expect(final_stats.vector_count >= 20); // At least baseline
                return;
            },
            else => return err,
        }
    };
    defer allocator.free(results);

    // Results should be valid (non-empty if we have vectors)
    if (results.len > 0) {
        // Verify result IDs are valid
        for (results) |result| {
            try std.testing.expect(result.id < i);
            try std.testing.expect(result.score >= 0.0);
            try std.testing.expect(result.score <= 1.0);
        }
    }

    const stats = chaos_ctx.getStats();
    std.log.info("Search chaos: baseline=20, chaos_inserts={d}, results={d}, faults={d}", .{
        chaos_inserts,
        results.len,
        stats.faults_injected,
    });
}

test "database chaos: update operations maintain consistency" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 34567);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.08,
        .warmup_ops = 30,
    });

    var handle = try database.open(allocator, "chaos_update_db");
    defer database.close(&handle);

    var rng = std.Random.DefaultPrng.init(11111);
    const dims: usize = 32;

    // Insert initial vectors
    var j: u64 = 0;
    while (j < 20) : (j += 1) {
        const vec = try generateRandomVector(&rng, dims, allocator);
        defer allocator.free(vec);
        try database.insert(&handle, j, vec, null);
    }

    const initial_count = database.stats(&handle).vector_count;

    // Enable chaos and perform updates
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var successful_updates: u32 = 0;
    var failed_updates: u32 = 0;

    var i: u64 = 0;
    while (i < 20) : (i += 1) {
        const new_vec = generateRandomVector(&rng, dims, allocator) catch {
            failed_updates += 1;
            continue;
        };
        defer allocator.free(new_vec);

        const updated = database.update(&handle, i, new_vec) catch |err| {
            switch (err) {
                error.OutOfMemory => {
                    failed_updates += 1;
                    continue;
                },
                else => return err,
            }
        };

        if (updated) {
            successful_updates += 1;
        }
    }

    // Vector count should remain the same (updates don't change count)
    const final_count = database.stats(&handle).vector_count;
    try std.testing.expectEqual(initial_count, final_count);

    const stats = chaos_ctx.getStats();
    std.log.info("Update chaos: success={d}, failed={d}, faults={d}", .{
        successful_updates,
        failed_updates,
        stats.faults_injected,
    });
}

test "database chaos: delete operations are atomic" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 45678);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.1,
        .warmup_ops = 25,
    });

    var handle = try database.open(allocator, "chaos_delete_db");
    defer database.close(&handle);

    var rng = std.Random.DefaultPrng.init(22222);
    const dims: usize = 32;

    // Insert vectors
    var k: u64 = 0;
    while (k < 30) : (k += 1) {
        const vec = try generateRandomVector(&rng, dims, allocator);
        defer allocator.free(vec);
        try database.insert(&handle, k, vec, null);
    }

    const initial_count = database.stats(&handle).vector_count;
    try std.testing.expectEqual(@as(u64, 30), initial_count);

    // Enable chaos and perform deletes
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var successful_deletes: u32 = 0;
    var i: u64 = 0;
    while (i < 30) : (i += 1) {
        // Deletes should either fully succeed or fully fail
        const removed = database.remove(&handle, i);
        if (removed) {
            successful_deletes += 1;

            // Verify the vector is actually gone
            const view = database.get(&handle, i);
            try std.testing.expect(view == null);
        }
    }

    // Final count should match
    const final_count = database.stats(&handle).vector_count;
    try std.testing.expectEqual(initial_count - successful_deletes, final_count);

    const stats = chaos_ctx.getStats();
    std.log.info("Delete chaos: success={d}, remaining={d}, faults={d}", .{
        successful_deletes,
        final_count,
        stats.faults_injected,
    });
}

// ============================================================================
// Index Structure Chaos Tests
// ============================================================================

test "database chaos: HNSW index survives allocation failures during construction" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 56789);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.04,
        .warmup_ops = 100, // Allow index initialization
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    var handle = database.open(allocator, "chaos_hnsw_db") catch |err| {
        switch (err) {
            error.OutOfMemory => return, // Expected
            else => return err,
        }
    };
    defer database.close(&handle);

    var rng = std.Random.DefaultPrng.init(33333);
    const dims: usize = 64;

    // Insert many vectors to build index
    var inserted: u32 = 0;
    var i: u64 = 0;
    while (i < 200) : (i += 1) {
        const vec = generateRandomVector(&rng, dims, allocator) catch continue;
        defer allocator.free(vec);

        database.insert(&handle, i, vec, null) catch continue;
        inserted += 1;
    }

    // Optimize index (may fail under chaos)
    database.optimize(&handle) catch |err| {
        switch (err) {
            error.OutOfMemory => {
                // Expected - index may be partially optimized
            },
            else => return err,
        }
    };

    // Verify database is usable after chaos
    const query = generateRandomVector(&rng, dims, allocator) catch return;
    defer allocator.free(query);

    const results = database.search(&handle, allocator, query, 5) catch |err| {
        switch (err) {
            error.OutOfMemory => return, // Search allocation failed
            else => return err,
        }
    };
    defer allocator.free(results);

    // If search succeeded, results should be valid
    for (results) |r| {
        try std.testing.expect(r.id < i);
    }

    const stats = chaos_ctx.getStats();
    std.log.info("HNSW chaos: inserted={d}, results={d}, faults={d}", .{
        inserted,
        results.len,
        stats.faults_injected,
    });
}

// ============================================================================
// Batch Operations Chaos Tests
// ============================================================================

test "database chaos: batch processor handles failures gracefully" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 67890);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.06,
        .warmup_ops = 10,
    });
    chaos_ctx.enable();
    defer chaos_ctx.disable();

    // Test batch processor initialization
    var processor = database.BatchProcessor.init(allocator, .{
        .batch_size = 100,
        .parallel_writes = false, // Disable parallel to simplify chaos testing
    }) catch |err| {
        switch (err) {
            error.OutOfMemory => return, // Expected
            else => return err,
        }
    };
    defer processor.deinit();

    var rng = std.Random.DefaultPrng.init(44444);
    const dims: usize = 32;

    // Queue batch records
    var queued: u32 = 0;
    var i: u64 = 0;
    while (i < 50) : (i += 1) {
        const vec = generateRandomVector(&rng, dims, allocator) catch continue;
        defer allocator.free(vec);

        processor.queueInsert(i, vec, null) catch continue;
        queued += 1;
    }

    // Flush may partially succeed under chaos
    const result = processor.flush() catch |err| {
        switch (err) {
            error.OutOfMemory => {
                // Partial flush - some records may have been processed
                std.log.info("Batch flush failed under chaos (expected)", .{});
                return;
            },
            else => return err,
        }
    };

    // If flush succeeded, verify results
    try std.testing.expect(result.successful + result.failed == queued);

    const stats = chaos_ctx.getStats();
    std.log.info("Batch chaos: queued={d}, success={d}, failed={d}, faults={d}", .{
        queued,
        result.successful,
        result.failed,
        stats.faults_injected,
    });
}

// ============================================================================
// Concurrent Access Chaos Tests
// ============================================================================

test "database chaos: concurrent access survives random failures" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 78901);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.02,
        .warmup_ops = 100,
    });

    var handle = try database.open(allocator, "chaos_concurrent_db");
    defer database.close(&handle);

    var rng_seed = std.Random.DefaultPrng.init(55555);

    // Insert baseline data
    var base_id: u64 = 0;
    while (base_id < 50) : (base_id += 1) {
        var vec: [32]f32 = undefined;
        for (&vec) |*v| {
            v.* = rng_seed.random().float(f32);
        }
        try database.insert(&handle, base_id, &vec, null);
    }

    chaos_ctx.enable();
    defer chaos_ctx.disable();

    // Simulate concurrent operations
    var ops_completed = std.atomic.Value(u32).init(0);
    var ops_failed = std.atomic.Value(u32).init(0);

    const thread_count = 4;
    var threads: [thread_count]std.Thread = undefined;

    for (&threads, 0..) |*t, tid| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(
                h: *database.DatabaseHandle,
                alloc: std.mem.Allocator,
                completed: *std.atomic.Value(u32),
                failed: *std.atomic.Value(u32),
                thread_id: usize,
            ) void {
                var rng = std.Random.DefaultPrng.init(@intCast(thread_id * 12345));
                const dims: usize = 32;

                var i: u32 = 0;
                while (i < 25) : (i += 1) {
                    // Randomly choose operation
                    const op = rng.random().intRangeAtMost(u8, 0, 2);

                    switch (op) {
                        0 => {
                            // Insert
                            const vec = generateRandomVector(&rng, dims, alloc) catch {
                                _ = failed.fetchAdd(1, .monotonic);
                                continue;
                            };
                            defer alloc.free(vec);

                            const id = @as(u64, thread_id * 1000 + i);
                            database.insert(h, id, vec, null) catch {
                                _ = failed.fetchAdd(1, .monotonic);
                                continue;
                            };
                            _ = completed.fetchAdd(1, .monotonic);
                        },
                        1 => {
                            // Search
                            const query = generateRandomVector(&rng, dims, alloc) catch {
                                _ = failed.fetchAdd(1, .monotonic);
                                continue;
                            };
                            defer alloc.free(query);

                            const results = database.search(h, alloc, query, 5) catch {
                                _ = failed.fetchAdd(1, .monotonic);
                                continue;
                            };
                            defer alloc.free(results);
                            _ = completed.fetchAdd(1, .monotonic);
                        },
                        else => {
                            // Get
                            const id = rng.random().intRangeAtMost(u64, 0, 50);
                            _ = database.get(h, id);
                            _ = completed.fetchAdd(1, .monotonic);
                        },
                    }
                }
            }
        }.run, .{ &handle, allocator, &ops_completed, &ops_failed, tid });
    }

    for (&threads) |*t| {
        t.join();
    }

    // Verify database is still consistent
    const final_stats = database.stats(&handle);
    try std.testing.expect(final_stats.vector_count >= 50); // At least baseline

    const chaos_stats = chaos_ctx.getStats();
    std.log.info("Concurrent chaos: completed={d}, failed={d}, final_vectors={d}, faults={d}", .{
        ops_completed.load(.acquire),
        ops_failed.load(.acquire),
        final_stats.vector_count,
        chaos_stats.faults_injected,
    });
}

// ============================================================================
// Recovery Tests
// ============================================================================

test "database chaos: full recovery after chaos period" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var chaos_ctx = chaos.ChaosContext.init(allocator, 89012);
    defer chaos_ctx.deinit();

    try chaos_ctx.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.15, // High failure rate
        .max_faults = 20, // Limit total faults
    });

    var handle = try database.open(allocator, "chaos_recovery_db");
    defer database.close(&handle);

    var rng = std.Random.DefaultPrng.init(66666);
    const dims: usize = 32;

    // Phase 1: Operations under high chaos
    chaos_ctx.enable();

    var chaos_phase_inserts: u32 = 0;
    var i: u64 = 0;
    while (i < 50) : (i += 1) {
        const vec = generateRandomVector(&rng, dims, allocator) catch continue;
        defer allocator.free(vec);

        database.insert(&handle, i, vec, null) catch continue;
        chaos_phase_inserts += 1;
    }

    chaos_ctx.disable();

    const chaos_phase_count = database.stats(&handle).vector_count;

    // Phase 2: All operations after chaos should succeed
    var recovery_phase_inserts: u32 = 0;
    while (i < 100) : (i += 1) {
        const vec = try generateRandomVector(&rng, dims, allocator);
        defer allocator.free(vec);

        try database.insert(&handle, i, vec, null);
        recovery_phase_inserts += 1;
    }

    // All recovery phase inserts should succeed
    try std.testing.expectEqual(@as(u32, 50), recovery_phase_inserts);

    // Final count should be chaos_phase + recovery_phase
    const final_count = database.stats(&handle).vector_count;
    try std.testing.expectEqual(chaos_phase_count + recovery_phase_inserts, final_count);

    // Search should work correctly
    const query = try generateRandomVector(&rng, dims, allocator);
    defer allocator.free(query);

    const results = try database.search(&handle, allocator, query, 10);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);

    const stats = chaos_ctx.getStats();
    std.log.info("Recovery test: chaos_inserts={d}, recovery_inserts={d}, final={d}, faults={d}", .{
        chaos_phase_inserts,
        recovery_phase_inserts,
        final_count,
        stats.faults_injected,
    });
}
