const std = @import("std");
const testing = std.testing;
const core = @import("core");

test "LSH index integration test" {
    const allocator = testing.allocator;
    
    // Create database with LSH index
    var db = try core.Database.open(allocator, ":memory:", true);
    defer db.close();
    
    try db.init(.{
        .dimensions = 128,
        .index_type = .lsh,
        .distance_metric = .euclidean,
    });
    
    // Generate test vectors
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    // Add 1000 random vectors
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        var vec = try allocator.alloc(f32, 128);
        defer allocator.free(vec);
        
        for (vec) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0; // Range [-1, 1]
        }
        
        _ = try db.addVector(vec, null);
    }
    
    // Create a query vector
    var query = try allocator.alloc(f32, 128);
    defer allocator.free(query);
    
    for (query) |*v| {
        v.* = random.float(f32) * 2.0 - 1.0;
    }
    
    // Search for similar vectors
    const results = try db.search(query, 10, allocator);
    defer allocator.free(results);
    
    // Verify results
    try testing.expect(results.len > 0);
    try testing.expect(results.len <= 10);
    
    // Results should be sorted by distance
    for (results[0..results.len-1], results[1..]) |a, b| {
        try testing.expect(a.distance <= b.distance);
    }
}

test "LSH index performance comparison" {
    const allocator = testing.allocator;
    const dimensions = 256;
    const num_vectors = 10000;
    const num_queries = 100;
    const k = 50;
    
    // Generate test data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    var vectors = try allocator.alloc([]f32, num_vectors);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }
    
    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dimensions);
        for (vec.*) |*v| {
            v.* = random.float(f32);
        }
    }
    
    // Test LSH index
    {
        var lsh_index = try core.index.lsh.LSHIndex.init(allocator, dimensions, .euclidean, .{
            .num_tables = 10,
            .hash_functions_per_table = 12,
            .enable_multi_probe = true,
        });
        defer lsh_index.deinit();
        
        // Build index
        var timer = try std.time.Timer.start();
        for (vectors, 0..) |vec, i| {
            try lsh_index.add(i, vec);
        }
        const build_time = timer.read();
        
        // Search benchmark
        timer.reset();
        var total_results: usize = 0;
        var q: usize = 0;
        while (q < num_queries) : (q += 1) {
            const query_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
            const results = try lsh_index.search(vectors[query_idx], k, allocator);
            defer allocator.free(results);
            total_results += results.len;
        }
        const search_time = timer.read();
        
        std.debug.print("\nLSH Index Performance:\n", .{});
        std.debug.print("  Build time: {d:.3} ms ({d:.0} vectors/sec)\n", .{
            @as(f64, @floatFromInt(build_time)) / 1e6,
            @as(f64, @floatFromInt(num_vectors)) * 1e9 / @as(f64, @floatFromInt(build_time)),
        });
        std.debug.print("  Search time: {d:.3} ms/query\n", .{
            @as(f64, @floatFromInt(search_time)) / @as(f64, @floatFromInt(num_queries)) / 1e6,
        });
        std.debug.print("  Average recall: {d:.1}%\n", .{
            @as(f64, @floatFromInt(total_results)) / @as(f64, @floatFromInt(num_queries * k)) * 100,
        });
        
        const stats = lsh_index.getStats();
        std.debug.print("  Memory usage: {d:.2} MB\n", .{
            @as(f64, @floatFromInt(stats.memory_usage)) / (1024 * 1024),
        });
    }
    
    // Compare with flat index for accuracy
    {
        var flat_index = try core.index.flat.FlatIndex.init(allocator, dimensions, .euclidean);
        defer flat_index.deinit();
        
        // Build index
        var timer = try std.time.Timer.start();
        for (vectors, 0..) |vec, i| {
            try flat_index.add(i, vec);
        }
        const build_time = timer.read();
        
        // Search benchmark
        timer.reset();
        var q: usize = 0;
        while (q < @min(10, num_queries)) : (q += 1) { // Fewer queries for flat index
            const query_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
            const results = try flat_index.search(vectors[query_idx], k, allocator);
            allocator.free(results);
        }
        const search_time = timer.read();
        
        std.debug.print("\nFlat Index Performance (baseline):\n", .{});
        std.debug.print("  Build time: {d:.3} ms\n", .{
            @as(f64, @floatFromInt(build_time)) / 1e6,
        });
        std.debug.print("  Search time: {d:.3} ms/query\n", .{
            @as(f64, @floatFromInt(search_time)) / @as(f64, @floatFromInt(@min(10, num_queries))) / 1e6,
        });
    }
}

test "LSH multi-probe effectiveness" {
    const allocator = testing.allocator;
    const dimensions = 64;
    const num_vectors = 5000;
    
    // Generate clustered data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    var vectors = try allocator.alloc([]f32, num_vectors);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }
    
    // Create 5 clusters
    const num_clusters = 5;
    const vectors_per_cluster = num_vectors / num_clusters;
    
    for (0..num_clusters) |cluster| {
        // Generate cluster center
        var center: [64]f32 = undefined;
        for (&center) |*c| {
            c.* = random.float(f32) * 10.0;
        }
        
        // Generate vectors around center
        for (0..vectors_per_cluster) |i| {
            const idx = cluster * vectors_per_cluster + i;
            vectors[idx] = try allocator.alloc(f32, dimensions);
            for (vectors[idx], center) |*v, c| {
                v.* = c + (random.float(f32) - 0.5) * 0.5; // Small variance
            }
        }
    }
    
    // Test with different probe settings
    const probe_settings = [_]struct { probes: u32, name: []const u8 }{
        .{ .probes = 1, .name = "Single probe" },
        .{ .probes = 3, .name = "3 probes" },
        .{ .probes = 5, .name = "5 probes" },
    };
    
    for (probe_settings) |setting| {
        var index = try core.index.lsh.LSHIndex.init(allocator, dimensions, .euclidean, .{
            .num_tables = 8,
            .hash_functions_per_table = 10,
            .enable_multi_probe = setting.probes > 1,
            .num_probes = setting.probes,
        });
        defer index.deinit();
        
        // Build index
        for (vectors, 0..) |vec, i| {
            try index.add(i, vec);
        }
        
        // Test recall within clusters
        var total_recall: f32 = 0;
        const test_queries = 20;
        
        for (0..test_queries) |_| {
            const query_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
            const query_cluster = query_idx / vectors_per_cluster;
            
            const results = try index.search(vectors[query_idx], 20, allocator);
            defer allocator.free(results);
            
            // Count how many results are from the same cluster
            var same_cluster: usize = 0;
            for (results) |result| {
                const result_cluster = result.id / vectors_per_cluster;
                if (result_cluster == query_cluster) {
                    same_cluster += 1;
                }
            }
            
            total_recall += @as(f32, @floatFromInt(same_cluster)) / @as(f32, @floatFromInt(results.len));
        }
        
        const avg_recall = total_recall / @as(f32, @floatFromInt(test_queries));
        std.debug.print("\n{s}: {d:.1}% cluster recall\n", .{ setting.name, avg_recall * 100 });
        
        // Multi-probe should improve recall
        if (setting.probes > 1) {
            try testing.expect(avg_recall > 0.7); // Should find most vectors from same cluster
        }
    }
}