const std = @import("std");
const testing = std.testing;
const core = @import("core");

test "Vector compression integration test" {
    const allocator = testing.allocator;
    
    // Create test vectors
    const num_vectors = 1000;
    const dimensions = 384; // Common embedding size
    
    var vectors = try allocator.alloc([]f32, num_vectors);
    defer allocator.free(vectors);
    
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    // Generate random vectors
    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dimensions);
        for (vec.*) |*val| {
            val.* = random.float(f32) * 2.0 - 1.0; // Range [-1, 1]
        }
    }
    defer {
        for (vectors) |vec| allocator.free(vec);
    }
    
    // Test different compression methods
    const methods = [_]core.vector.CompressionMethod{
        .scalar_quantization_8bit,
        .scalar_quantization_4bit,
        .binary,
    };
    
    for (methods) |method| {
        std.debug.print("\nTesting compression method: {s}\n", .{@tagName(method)});
        
        var compressor = core.vector.Compressor.init(allocator, method);
        
        // Compress
        var timer = try std.time.Timer.start();
        var compressed = try compressor.compress(vectors);
        const compress_time = timer.read();
        defer compressed.deinit();
        
        // Decompress
        timer.reset();
        const decompressed = try compressor.decompress(compressed, allocator);
        const decompress_time = timer.read();
        defer {
            for (decompressed) |vec| allocator.free(vec);
            allocator.free(decompressed);
        }
        
        // Calculate metrics
        const original_size = num_vectors * dimensions * @sizeOf(f32);
        const compressed_size = compressed.data.len;
        const compression_ratio = @as(f32, @floatFromInt(original_size)) / @as(f32, @floatFromInt(compressed_size));
        
        // Calculate reconstruction error
        var total_error: f64 = 0;
        var max_error: f32 = 0;
        
        for (vectors, decompressed) |orig, decomp| {
            var vec_error: f32 = 0;
            for (orig, decomp) |o, d| {
                const diff = o - d;
                vec_error += diff * diff;
                max_error = @max(max_error, @abs(diff));
            }
            total_error += @sqrt(vec_error / @as(f32, @floatFromInt(dimensions)));
        }
        
        const avg_error = total_error / @as(f64, @floatFromInt(num_vectors));
        
        std.debug.print("  Compression ratio: {d:.2}x\n", .{compression_ratio});
        std.debug.print("  Compress time: {d:.3} ms\n", .{@as(f64, @floatFromInt(compress_time)) / 1e6});
        std.debug.print("  Decompress time: {d:.3} ms\n", .{@as(f64, @floatFromInt(decompress_time)) / 1e6});
        std.debug.print("  Average reconstruction error: {d:.6}\n", .{avg_error});
        std.debug.print("  Max element error: {d:.6}\n", .{max_error});
        std.debug.print("  Memory saved: {d:.2} MB\n", .{
            @as(f32, @floatFromInt(original_size - compressed_size)) / (1024 * 1024)
        });
        
        // Verify compression ratios
        switch (method) {
            .scalar_quantization_8bit => try testing.expect(compression_ratio > 3.9 and compression_ratio < 4.1),
            .scalar_quantization_4bit => try testing.expect(compression_ratio > 7.5 and compression_ratio < 8.5),
            .binary => try testing.expect(compression_ratio > 30),
            else => {},
        }
    }
}

test "Compression preserves similarity search accuracy" {
    const allocator = testing.allocator;
    const dimensions = 128;
    const num_vectors = 500;
    const k = 10;
    
    // Generate clustered data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    var vectors = try allocator.alloc([]f32, num_vectors);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }
    
    // Create 5 clusters
    for (0..5) |cluster| {
        // Generate cluster center
        var center = try allocator.alloc(f32, dimensions);
        defer allocator.free(center);
        for (center) |*c| {
            c.* = random.float(f32) * 10.0;
        }
        
        // Generate vectors around center
        for (0..100) |i| {
            const idx = cluster * 100 + i;
            vectors[idx] = try allocator.alloc(f32, dimensions);
            for (vectors[idx], center) |*v, c| {
                v.* = c + (random.float(f32) - 0.5) * 0.5;
            }
        }
    }
    
    // Create a simple index for testing
    var index = try core.index.flat.FlatIndex.init(allocator, dimensions, .euclidean);
    defer index.deinit();
    
    // Add original vectors
    for (vectors, 0..) |vec, i| {
        try index.add(i, vec);
    }
    
    // Test search with compressed vectors
    var compressor = core.vector.Compressor.init(allocator, .scalar_quantization_8bit);
    var compressed = try compressor.compress(vectors);
    defer compressed.deinit();
    
    const decompressed = try compressor.decompress(compressed, allocator);
    defer {
        for (decompressed) |vec| allocator.free(vec);
        allocator.free(decompressed);
    }
    
    // Compare search results
    var total_overlap: f32 = 0;
    const num_queries = 20;
    
    for (0..num_queries) |_| {
        const query_idx = random.intRangeAtMost(usize, 0, num_vectors - 1);
        
        // Search with original
        const orig_results = try index.search(vectors[query_idx], k, allocator);
        defer allocator.free(orig_results);
        
        // Search with decompressed
        const decomp_results = try index.search(decompressed[query_idx], k, allocator);
        defer allocator.free(decomp_results);
        
        // Calculate overlap
        var overlap: usize = 0;
        for (orig_results) |orig| {
            for (decomp_results) |decomp| {
                if (orig.id == decomp.id) {
                    overlap += 1;
                    break;
                }
            }
        }
        
        total_overlap += @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(k));
    }
    
    const avg_overlap = total_overlap / @as(f32, @floatFromInt(num_queries));
    std.debug.print("\nSearch accuracy with compressed vectors: {d:.1}% overlap\n", .{avg_overlap * 100});
    
    // Should maintain good search accuracy
    try testing.expect(avg_overlap > 0.8); // At least 80% overlap
}

test "Compression with database integration" {
    const allocator = testing.allocator;
    
    // Create database with compression enabled
    var db = try core.Database.open(allocator, ":memory:", true);
    defer db.close();
    
    try db.init(.{
        .dimensions = 256,
        .index_type = .flat,
        .distance_metric = .euclidean,
        .compression_method = .scalar_quantization_8bit,
    });
    
    // Add vectors
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    const num_vectors = 100;
    for (0..num_vectors) |_| {
        var vec = try allocator.alloc(f32, 256);
        defer allocator.free(vec);
        
        for (vec) |*v| {
            v.* = random.float(f32);
        }
        
        _ = try db.addVector(vec, null);
    }
    
    // Get database stats
    const stats = db.getStats();
    std.debug.print("\nDatabase with compression:\n", .{});
    std.debug.print("  Vectors: {}\n", .{stats.total_vectors});
    std.debug.print("  Memory usage: {d:.2} MB\n", .{@as(f32, @floatFromInt(stats.memory_usage_bytes)) / (1024 * 1024)});
    std.debug.print("  Compression enabled: {}\n", .{stats.compression_enabled});
    
    // Search should still work
    var query = try allocator.alloc(f32, 256);
    defer allocator.free(query);
    for (query) |*v| {
        v.* = random.float(f32);
    }
    
    const results = try db.search(query, 10, allocator);
    defer allocator.free(results);
    
    try testing.expect(results.len > 0);
}