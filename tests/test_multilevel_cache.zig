const std = @import("std");
const testing = std.testing;
const core = @import("core");

test "Multi-level cache performance test" {
    const allocator = testing.allocator;
    
    var cache = try core.storage.cache.MultiLevelCache.init(allocator, .{
        .l1_size_mb = 10,
        .l2_size_mb = 50,
        .l3_size_mb = 200,
        .l2_compression = true,
        .eviction_policy = .lru,
        .enable_prefetch = true,
    });
    defer cache.deinit();
    
    // Simulate vector database cache usage
    const num_vectors = 10000;
    const vector_size = 384 * @sizeOf(f32); // 384-dimensional vectors
    
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    // Insert vectors
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < num_vectors) : (i += 1) {
        var vector = try allocator.alloc(u8, vector_size);
        defer allocator.free(vector);
        
        // Fill with random data
        for (vector) |*byte| {
            byte.* = random.int(u8);
        }
        
        try cache.put(i, vector);
    }
    const insert_time = timer.read();
    
    // Access pattern: 80% of accesses to 20% of vectors (Pareto principle)
    const hot_vectors = num_vectors / 5;
    const num_accesses = 50000;
    
    timer.reset();
    var hits: u64 = 0;
    var access: u64 = 0;
    while (access < num_accesses) : (access += 1) {
        const key = if (random.float(f32) < 0.8)
            random.intRangeAtMost(u64, 0, hot_vectors - 1)
        else
            random.intRangeAtMost(u64, hot_vectors, num_vectors - 1);
        
        if (try cache.get(key, allocator)) |value| {
            allocator.free(value);
            hits += 1;
        }
    }
    const access_time = timer.read();
    
    const stats = cache.getStats();
    
    std.debug.print("\n=== Multi-level Cache Performance ===\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  L1: 10 MB, L2: 50 MB (compressed), L3: 200 MB\n", .{});
    std.debug.print("  Vectors: {}, Size: {} bytes each\n", .{ num_vectors, vector_size });
    std.debug.print("\nPerformance:\n", .{});
    std.debug.print("  Insert time: {d:.3} ms ({d:.0} vectors/sec)\n", .{
        @as(f64, @floatFromInt(insert_time)) / 1e6,
        @as(f64, @floatFromInt(num_vectors)) * 1e9 / @as(f64, @floatFromInt(insert_time)),
    });
    std.debug.print("  Access time: {d:.3} ms ({d:.0} accesses/sec)\n", .{
        @as(f64, @floatFromInt(access_time)) / 1e6,
        @as(f64, @floatFromInt(num_accesses)) * 1e9 / @as(f64, @floatFromInt(access_time)),
    });
    std.debug.print("\nCache Statistics:\n", .{});
    std.debug.print("  L1 hits: {} ({d:.1}%)\n", .{ stats.l1_hits, stats.getHitRate(.l1) * 100 });
    std.debug.print("  L2 hits: {} ({d:.1}%)\n", .{ stats.l2_hits, stats.getHitRate(.l2) * 100 });
    std.debug.print("  L3 hits: {} ({d:.1}%)\n", .{ stats.l3_hits, stats.getHitRate(.l3) * 100 });
    std.debug.print("  Total hit rate: {d:.1}%\n", .{stats.getTotalHitRate() * 100});
    std.debug.print("  Total evictions: {}\n", .{stats.total_evictions});
    std.debug.print("  Memory used: {d:.2} MB\n", .{@as(f64, @floatFromInt(stats.bytes_used)) / (1024 * 1024)});
    
    // Verify good hit rate for hot data
    try testing.expect(stats.getTotalHitRate() > 0.7);
}

test "Cache with different eviction policies" {
    const allocator = testing.allocator;
    
    const policies = [_]core.storage.cache.EvictionPolicy{
        .lru,
        .lfu,
        .fifo,
        .arc,
    };
    
    for (policies) |policy| {
        std.debug.print("\nTesting eviction policy: {s}\n", .{@tagName(policy)});
        
        var cache = try core.storage.cache.MultiLevelCache.init(allocator, .{
            .l1_size_mb = 1,
            .l2_size_mb = 2,
            .l3_size_mb = 4,
            .eviction_policy = policy,
        });
        defer cache.deinit();
        
        // Small cache, many items to force eviction
        const item_size = 1024; // 1KB
        const num_items = 2048; // 2MB worth of data
        
        // Insert items
        var i: u64 = 0;
        while (i < num_items) : (i += 1) {
            var data = try allocator.alloc(u8, item_size);
            defer allocator.free(data);
            @memset(data, @intCast(i % 256));
            
            try cache.put(i, data);
        }
        
        // Access pattern varies by policy effectiveness
        var hits: u64 = 0;
        var access: u64 = 0;
        const num_accesses = 1000;
        
        while (access < num_accesses) : (access += 1) {
            const key = switch (policy) {
                .lru => num_items - 1 - (access % 100), // Recent items
                .lfu => access % 50, // Frequent items
                .fifo => access % num_items, // Random
                .arc => if (access % 2 == 0) access % 50 else num_items - 1 - (access % 50), // Mixed
            };
            
            if (try cache.get(key, allocator)) |value| {
                allocator.free(value);
                hits += 1;
            }
        }
        
        const hit_rate = @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(num_accesses));
        std.debug.print("  Hit rate: {d:.1}%\n", .{hit_rate * 100});
        
        // Different policies should have different effectiveness
        switch (policy) {
            .lru => try testing.expect(hit_rate > 0.5), // Good for recency
            .lfu => try testing.expect(hit_rate > 0.6), // Good for frequency
            .fifo => try testing.expect(hit_rate > 0.1), // Basic
            .arc => try testing.expect(hit_rate > 0.4), // Adaptive
        }
    }
}

test "Cache promotion and demotion" {
    const allocator = testing.allocator;
    
    var cache = try core.storage.cache.MultiLevelCache.init(allocator, .{
        .l1_size_mb = 1,
        .l2_size_mb = 2,
        .l3_size_mb = 4,
    });
    defer cache.deinit();
    
    const key: u64 = 123;
    const value = "test value for promotion/demotion";
    
    // Insert into cache (goes to L1)
    try cache.put(key, value);
    
    // Verify in L1
    const from_l1 = try cache.get(key, allocator);
    try testing.expect(from_l1 != null);
    if (from_l1) |v| allocator.free(v);
    
    var stats = cache.getStats();
    try testing.expectEqual(@as(u64, 1), stats.l1_hits);
    
    // Demote to L2
    try testing.expect(try cache.demote(key, .l1));
    
    // Should now hit L2
    const from_l2 = try cache.get(key, allocator);
    try testing.expect(from_l2 != null);
    if (from_l2) |v| allocator.free(v);
    
    stats = cache.getStats();
    try testing.expectEqual(@as(u64, 1), stats.l2_hits);
    
    // Demote to L3
    try testing.expect(try cache.demote(key, .l2));
    
    // Should now hit L3
    const from_l3 = try cache.get(key, allocator);
    try testing.expect(from_l3 != null);
    if (from_l3) |v| allocator.free(v);
    
    stats = cache.getStats();
    try testing.expectEqual(@as(u64, 1), stats.l3_hits);
    
    // Promote back to L1
    try testing.expect(try cache.promote(key, .l1));
    
    // Should hit L1 again
    const back_in_l1 = try cache.get(key, allocator);
    try testing.expect(back_in_l1 != null);
    if (back_in_l1) |v| allocator.free(v);
    
    stats = cache.getStats();
    try testing.expectEqual(@as(u64, 2), stats.l1_hits);
}

test "Cache prefetching" {
    const allocator = testing.allocator;
    
    var cache = try core.storage.cache.MultiLevelCache.init(allocator, .{
        .l1_size_mb = 1,
        .l2_size_mb = 2,
        .l3_size_mb = 10,
        .enable_prefetch = true,
        .prefetch_count = 8,
    });
    defer cache.deinit();
    
    // Insert sequential data into L3
    var i: u64 = 0;
    while (i < 100) : (i += 1) {
        const value = try std.fmt.allocPrint(allocator, "value_{}", .{i});
        defer allocator.free(value);
        
        try cache.put(i, value);
        // Force to L3
        _ = try cache.demote(i, .l1);
        _ = try cache.demote(i, .l2);
    }
    
    // Access one item - should prefetch adjacent
    const key: u64 = 50;
    if (try cache.get(key, allocator)) |value| {
        allocator.free(value);
    }
    
    // Check if adjacent items were prefetched to L2
    var prefetched: u32 = 0;
    var j: u64 = key - 4;
    while (j < key + 4) : (j += 1) {
        if (j == key) continue;
        
        // Manually check L2 (not through get which would promote to L1)
        cache.mutex.lock();
        const in_l2 = cache.l2_cache.get(j) catch null;
        cache.mutex.unlock();
        
        if (in_l2 != null) {
            prefetched += 1;
        }
    }
    
    std.debug.print("\nPrefetching test: {} adjacent items prefetched\n", .{prefetched});
    try testing.expect(prefetched > 0);
}