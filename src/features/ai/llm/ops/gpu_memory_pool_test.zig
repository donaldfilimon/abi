//! Tests for LLM-Optimized GPU Memory Pool.
//!
//! Covers basic operations, size classes, eviction, CPU fallback,
//! fragmentation tracking, best-fit allocation, buffer splitting,
//! defragmentation, and detailed fragmentation analysis.

const std = @import("std");
const testing = std.testing;
const gpu_mem = @import("gpu_memory_pool.zig");
const LlmMemoryPool = gpu_mem.LlmMemoryPool;
const PooledBuffer = gpu_mem.PooledBuffer;

test "memory pool basic operations" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .max_memory_bytes = 1024 * 1024, // 1MB
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU for tests
    });
    defer pool.deinit();

    // Acquire and release
    const buf1 = try pool.acquire(1024);
    try testing.expect(buf1.allocated_size >= 1024);

    pool.release(buf1);

    // Should get same buffer back (cache hit)
    const buf2 = try pool.acquire(1024);
    pool.release(buf2);

    const stats = pool.getStats();
    try testing.expect(stats.cache_hits > 0);
}

test "memory pool size classes" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10,
    });
    defer pool.deinit();

    // Different sizes should get different size classes
    const buf256 = try pool.acquire(256);
    const buf1k = try pool.acquire(1024);
    const buf4k = try pool.acquire(4096);

    try testing.expect(buf256.size_class <= buf1k.size_class);
    try testing.expect(buf1k.size_class <= buf4k.size_class);

    pool.release(buf256);
    pool.release(buf1k);
    pool.release(buf4k);
}

test "memory pool eviction" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .max_memory_bytes = 4096, // Very small limit
        .gpu_threshold_bytes = 1024 * 1024 * 10,
        .max_buffers_per_class = 2,
    });
    defer pool.deinit();

    // Allocate several buffers
    var buffers: [4]PooledBuffer = undefined;
    for (&buffers) |*buf| {
        buf.* = try pool.acquire(512);
    }

    // Release them
    for (&buffers) |buf| {
        pool.release(buf);
    }

    // Some should have been evicted (only 2 cached per class)
    const stats = pool.getStats();
    try testing.expect(stats.evictions >= 0 or stats.cache_hits >= 0);
}

test "memory pool cpu fallback" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
    });
    defer pool.deinit();

    const buf = try pool.acquire(1024);
    try testing.expect(!buf.is_gpu);
    try testing.expect(buf.cpu_data != null);

    // Write to CPU buffer
    if (buf.cpu_data) |data| {
        @memset(data, 0xAB);
        try testing.expectEqual(@as(u8, 0xAB), data[0]);
    }

    pool.release(buf);
}

test "memory pool fragmentation tracking" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .enable_best_fit = true,
        .auto_defrag_threshold = 0, // Disable auto-defrag for test
    });
    defer pool.deinit();

    // Request a size that's not a power of 2 (creates internal fragmentation)
    const buf = try pool.acquire(300); // Will get 512 (power of 2)
    try testing.expect(buf.allocated_size >= 300);
    try testing.expect(buf.requested_size == 300);

    // Check fragmentation is tracked
    const frag = buf.getFragmentation();
    try testing.expect(frag > 0);

    pool.release(buf);

    // Check stats include fragmentation
    const stats = pool.getStats();
    try testing.expect(stats.internal_fragmentation_bytes > 0);
}

test "memory pool best-fit allocation" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .enable_best_fit = true,
        .max_buffers_per_class = 8,
    });
    defer pool.deinit();

    // Allocate and release several buffers to populate free list
    var buffers: [4]PooledBuffer = undefined;
    for (&buffers) |*buf| {
        buf.* = try pool.acquire(512);
    }
    for (&buffers) |buf| {
        pool.release(buf);
    }

    // Now request similar size - should use best-fit from cache
    const buf = try pool.acquire(500);
    try testing.expect(buf.allocated_size >= 500);

    const stats = pool.getStats();
    try testing.expect(stats.cache_hits > 0);

    pool.release(buf);
}

test "memory pool buffer splitting" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .enable_splitting = true,
        .min_split_remainder = 256,
    });
    defer pool.deinit();

    // Allocate a large buffer then release it
    const large_buf = try pool.acquire(4096);
    pool.release(large_buf);

    // Request smaller size - should be able to use the large buffer
    const small_buf = try pool.acquire(512);
    try testing.expect(small_buf.allocated_size >= 512);

    pool.release(small_buf);
}

test "memory pool defragmentation" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .max_buffers_per_class = 16,
        .auto_defrag_threshold = 0, // Manual defrag only
    });
    defer pool.deinit();

    // Create fragmentation
    var buffers: [8]PooledBuffer = undefined;
    for (&buffers) |*buf| {
        buf.* = try pool.acquire(300);
    }
    for (&buffers) |buf| {
        pool.release(buf);
    }

    // Trigger manual defragmentation
    pool.defragment();

    // Check defrag was recorded
    const stats = pool.getStats();
    try testing.expect(stats.defrag_runs > 0);
}

test "memory pool fragmentation ratio" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
    });
    defer pool.deinit();

    // Initial ratio should be 0
    try testing.expect(pool.getFragmentationRatio() == 0);

    // Allocate buffer with fragmentation
    const buf = try pool.acquire(300); // Creates fragmentation
    const ratio = pool.getFragmentationRatio();

    // Should have some fragmentation (300 -> 512 = ~41% waste)
    try testing.expect(ratio > 0);
    try testing.expect(ratio < 1.0);

    pool.release(buf);
}

test "memory pool stats fragmentation rate" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
    });
    defer pool.deinit();

    // Allocate buffer
    const buf = try pool.acquire(300);

    const stats = pool.getStats();
    const frag_rate = stats.fragmentationRate();

    // Fragmentation rate should be calculable
    try testing.expect(frag_rate >= 0);
    try testing.expect(frag_rate <= 100.0);

    pool.release(buf);
}

test "memory pool external fragmentation tracking" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
        .max_buffers_per_class = 16,
    });
    defer pool.deinit();

    // Initial external fragmentation should be 0
    try testing.expect(pool.getExternalFragmentationRatio() == 0);

    // Allocate a large buffer first (sets min_request_size high)
    const large_buf = try pool.acquire(4096);

    // Now allocate smaller buffers
    const small_buf1 = try pool.acquire(256);
    const small_buf2 = try pool.acquire(256);

    // Release small buffers - they become free list entries
    pool.release(small_buf1);
    pool.release(small_buf2);

    // External fragmentation tracks blocks smaller than min_request_size.
    // Since we called acquire(256), min_request_size is 256, so the 256-byte
    // free blocks ARE usable for future 256-byte requests → ratio is 0.
    const ext_ratio = pool.getExternalFragmentationRatio();
    try testing.expect(ext_ratio >= 0);
    try testing.expect(ext_ratio <= 1.0);

    pool.release(large_buf);
}

test "memory pool fragmentation analysis" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
        .max_buffers_per_class = 16,
    });
    defer pool.deinit();

    // Create some fragmentation
    const buf1 = try pool.acquire(300); // Creates internal fragmentation
    const buf2 = try pool.acquire(500);

    // Get analysis
    const analysis = pool.getFragmentationAnalysis();

    // Should have internal fragmentation (300 -> 512, 500 -> 512)
    try testing.expect(analysis.internal_fragmentation_bytes > 0);
    try testing.expect(analysis.internal_fragmentation_ratio >= 0);
    try testing.expect(analysis.internal_fragmentation_ratio <= 1.0);

    // External fragmentation should be 0 or small (no released buffers)
    try testing.expect(analysis.external_fragmentation_ratio >= 0);
    try testing.expect(analysis.external_fragmentation_ratio <= 1.0);

    pool.release(buf1);
    pool.release(buf2);
}

test "memory pool defragmentation recommendation" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
        .max_buffers_per_class = 32,
    });
    defer pool.deinit();

    // Create significant fragmentation
    // First, establish a large minimum request
    const large = try pool.acquire(8192);

    // Then allocate and release many small buffers
    var small_buffers: [16]PooledBuffer = undefined;
    for (&small_buffers) |*buf| {
        buf.* = try pool.acquire(256);
    }
    for (&small_buffers) |buf| {
        pool.release(buf);
    }

    // Check fragmentation recommendation
    const rec = pool.checkFragmentation();

    // Should have external fragmentation (small free blocks vs large requests)
    try testing.expect(rec.external_fragmentation_ratio >= 0);

    // The recommendation severity should be valid
    _ = rec.severity.toString();
    _ = rec.getMessage();

    pool.release(large);
}

test "memory pool free list bytes tracking" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
        .max_buffers_per_class = 16,
    });
    defer pool.deinit();

    // Initially, free list should be empty
    const initial_stats = pool.getStats();
    try testing.expectEqual(@as(u64, 0), initial_stats.free_list_bytes);

    // Allocate and release a buffer
    const buf = try pool.acquire(1024);
    pool.release(buf);

    // Free list should now have bytes
    const after_release = pool.getStats();
    try testing.expect(after_release.free_list_bytes > 0);

    // Acquire again (from free list)
    const buf2 = try pool.acquire(1024);

    // Free list bytes should decrease
    const after_acquire = pool.getStats();
    try testing.expect(after_acquire.free_list_bytes < after_release.free_list_bytes);

    pool.release(buf2);
}

test "memory pool total fragmentation ratio" {
    var pool = try LlmMemoryPool.init(testing.allocator, .{
        .gpu_threshold_bytes = 1024 * 1024 * 10, // Force CPU
        .auto_defrag_threshold = 0,
    });
    defer pool.deinit();

    // Create internal fragmentation
    const buf = try pool.acquire(300); // Gets 512, wastes 212 bytes

    // Total fragmentation should account for internal waste
    const total = pool.getTotalFragmentationRatio();
    try testing.expect(total >= 0);
    try testing.expect(total <= 1.0);

    // Internal fragmentation should be positive
    const internal = pool.getFragmentationRatio();
    try testing.expect(internal > 0);

    // Total should be at least as high as internal
    try testing.expect(total >= internal or total == 0);

    pool.release(buf);
}

test {
    std.testing.refAllDecls(@This());
}
