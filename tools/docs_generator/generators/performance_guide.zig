const std = @import("std");

/// Generate performance guide
pub fn generatePerformanceGuide(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/PERFORMANCE_GUIDE.md", .{ .truncate = true });
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Performance Guide"
        \\description: "Comprehensive performance optimization guide with benchmarks and best practices"
        \\---
        \\
        \\# ABI Performance Guide
        \\
        \\## 🚀 Performance Characteristics
        \\
        \\### Database Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Single Insert | ~2.5ms | ~512 bytes | 128-dim vectors |
        \\| Batch Insert (100) | ~40ms | ~51KB | 100 vectors |
        \\| Batch Insert (1000) | ~400ms | ~512KB | 1000 vectors |
        \\| Search (k=10) | ~13ms | ~1.6KB | 10k vectors |
        \\| Search (k=100) | ~14ms | ~16KB | 10k vectors |
        \\| Update | ~1ms | ~512 bytes | Single vector |
        \\| Delete | ~0.5ms | ~0 bytes | Single vector |
        \\
        \\### AI/ML Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Network Creation | ~1ms | ~1MB | 128→64→32→10 |
        \\| Training Iteration | ~30μs | ~1MB | Batch size 32 |
        \\| Prediction | ~10μs | ~1KB | Single input |
        \\| Batch Prediction | ~100μs | ~10KB | 100 inputs |
        \\
        \\### SIMD Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Vector Add (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Multiply (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Normalize (2048) | ~5μs | ~16KB | Includes sqrt |
        \\| Matrix Multiply (64x64) | ~50μs | ~32KB | SIMD optimized |
        \\
        \\## ⚡ Optimization Strategies
        \\
        \\### 1. Memory Management
        \\```zig
        \\// Use arena allocators for bulk operations
        \\var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        \\defer arena.deinit();
        \\const allocator = arena.allocator();
        \\
        \\// Pre-allocate buffers for repeated operations
        \\const buffer_size = 1024 * 1024; // 1MB
        \\const buffer = try allocator.alloc(u8, buffer_size);
        \\defer allocator.free(buffer);
        \\```
        \\
        \\### 2. Batch Processing
        \\```zig
        \\// Process vectors in batches for better performance
        \\const BATCH_SIZE = 100;
        \\for (0..total_vectors / BATCH_SIZE) |batch| {
        \\    const start = batch * BATCH_SIZE;
        \\    const end = @min(start + BATCH_SIZE, total_vectors);
        \\    
        \\    // Process batch
        \\    for (vectors[start..end]) |vector| {
        \\        _ = try db.insert(vector, null);
        \\    }
        \\}
        \\```
        \\
        \\### 3. SIMD Optimization
        \\```zig
        \\// Use SIMD operations for vector processing
        \\const VECTOR_SIZE = 128;
        \\const SIMD_SIZE = 4; // Process 4 elements at once
        \\
        \\var i: usize = 0;
        \\while (i + SIMD_SIZE <= VECTOR_SIZE) : (i += SIMD_SIZE) {
        \\    const va = @as(@Vector(4, f32), a[i..][0..4].*);
        \\    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
        \\    const result = va + vb;
        \\    @memcpy(output[i..][0..4], @as([4]f32, result)[0..]);
        \\}
        \\```
        \\
        \\### 4. Caching Strategy
        \\```zig
        \\// Enable database caching for repeated queries
        \\const config = abi.DatabaseConfig{
        \\    .enable_caching = true,
        \\    .cache_size = 1024 * 1024, // 1MB cache
        \\};
        \\
        \\// Use LRU cache for frequently accessed data
        \\var cache = std.HashMap(u64, []f32, std.hash_map.default_hash_fn(u64), std.hash_map.default_eql_fn(u64)).init(allocator);
        \\defer {
        \\    var iterator = cache.iterator();
        \\    while (iterator.next()) |entry| {
        \\        allocator.free(entry.value_ptr.*);
        \\    }
        \\    cache.deinit();
        \\}
        \\```
        \\
        \\## 📊 Benchmarking
        \\
        \\### Running Benchmarks
        \\```bash
        \\# Run all benchmarks
        \\zig build benchmark
        \\
        \\# Run specific benchmark types
        \\zig build benchmark-db      # Database performance
        \\zig build benchmark-neural  # AI/ML performance
        \\zig build benchmark-simple  # General performance
        \\
        \\# Run with profiling
        \\zig build profile
        \\```
        \\
        \\### Custom Benchmarking
        \\```zig
        \\pub fn benchmarkOperation() !void {
        \\    const iterations = 1000;
        \\    var times = try allocator.alloc(u64, iterations);
        \\    defer allocator.free(times);
        \\
        \\    // Warm up
        \\    for (0..10) |_| {
        \\        // Perform operation
        \\    }
        \\
        \\    // Benchmark
        \\    for (times, 0..) |*time, i| {
        \\        const start = std.time.nanoTimestamp();
        \\        
        \\        // Perform operation
        \\        
        \\        const end = std.time.nanoTimestamp();
        \\        time.* = end - start;
        \\    }
        \\
        \\    // Calculate statistics
        \\    std.sort.heap(u64, times, {}, comptime std.sort.asc(u64));
        \\    const p50 = times[iterations / 2];
        \\    const p95 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.95))];
        \\    const p99 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.99))];
        \\
        \\    std.log.info("P50: {}ns, P95: {}ns, P99: {}ns", .{ p50, p95, p99 });
        \\}
        \\```
        \\
        \\## 🔍 Profiling Tools
        \\
        \\### Memory Profiling
        \\```zig
        \\// Enable memory tracking
        \\const memory_tracker = abi.memory_tracker.init(allocator);
        \\defer memory_tracker.deinit();
        \\
        \\// Track allocations
        \\memory_tracker.startTracking();
        \\
        \\// Perform operations
        \\
        \\// Get memory statistics
        \\const stats = memory_tracker.getStats();
        \\std.log.info("Peak memory: {} bytes", .{stats.peak_memory});
        \\std.log.info("Total allocations: {}", .{stats.total_allocations});
        \\```
        \\
        \\### Performance Profiling
        \\```zig
        \\// Use performance profiler
        \\const profiler = abi.performance_profiler.init(allocator);
        \\defer profiler.deinit();
        \\
        \\// Start profiling
        \\profiler.startProfiling("operation_name");
        \\
        \\// Perform operation
        \\
        \\// Stop profiling
        \\profiler.stopProfiling("operation_name");
        \\
        \\// Get results
        \\const results = profiler.getResults();
        \\for (results) |result| {
        \\    std.log.info("{}: {}ms", .{ result.name, result.duration_ms });
        \\}
        \\```
        \\
        \\## 🎯 Performance Tips
        \\
        \\### 1. Vector Dimensions
        \\- **Optimal range**: 128-512 dimensions
        \\- **Too small**: Poor representation quality
        \\- **Too large**: Increased memory and computation
        \\
        \\### 2. Batch Sizes
        \\- **Database inserts**: 100-1000 vectors per batch
        \\- **Neural training**: 32-128 samples per batch
        \\- **SIMD operations**: 1024-4096 elements per batch
        \\
        \\### 3. Memory Allocation
        \\- **Use arena allocators** for bulk operations
        \\- **Pre-allocate buffers** for repeated operations
        \\- **Enable caching** for frequently accessed data
        \\
        \\### 4. SIMD Usage
        \\- **Automatic optimization** for supported operations
        \\- **Vector size alignment** for best performance
        \\- **Batch processing** for maximum throughput
        \\
        \\## 📈 Performance Monitoring
        \\
        \\### Real-time Metrics
        \\```zig
        \\// Monitor performance in real-time
        \\const monitor = abi.performance_monitor.init(allocator);
        \\defer monitor.deinit();
        \\
        \\// Start monitoring
        \\monitor.startMonitoring();
        \\
        \\// Perform operations
        \\
        \\// Get metrics
        \\const metrics = monitor.getMetrics();
        \\std.log.info("Operations/sec: {}", .{metrics.operations_per_second});
        \\std.log.info("Average latency: {}ms", .{metrics.average_latency_ms});
        \\std.log.info("Memory usage: {}MB", .{metrics.memory_usage_mb});
        \\```
        \\
        \\### Performance Regression Detection
        \\```zig
        \\// Compare with baseline performance
        \\const baseline = try loadBaselinePerformance("baseline.json");
        \\const current = try measureCurrentPerformance();
        \\
        \\const regression_threshold = 0.05; // 5% regression
        \\if (current.avg_latency > baseline.avg_latency * (1.0 + regression_threshold)) {
        \\    std.log.warn("Performance regression detected!");
        \\    std.log.warn("Baseline: {}ms, Current: {}ms", .{ baseline.avg_latency, current.avg_latency });
        \\}
        \\```
        \\
    ;

    try file.writeAll(content);
}
