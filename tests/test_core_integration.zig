//! Comprehensive integration tests for the core system

const std = @import("std");
const testing = std.testing;
const core = @import("../src/core/mod.zig");

test "Core system initialization and cleanup" {
    // Test basic initialization
    try core.init(testing.allocator);
    try testing.expect(core.isInitialized());
    try testing.expect(core.getAllocator() != null);
    
    // Test cleanup
    core.deinit();
    try testing.expect(!core.isInitialized());
    try testing.expect(core.getAllocator() == null);
}

test "Core system with configuration" {
    const config = core.CoreConfig{
        .log_level = .debug,
        .enable_performance_monitoring = true,
        .memory_pool_size = 512 * 1024,
        .thread_pool_size = 2,
    };
    
    try core.initWithConfig(testing.allocator, config);
    defer core.deinit();
    
    try testing.expect(core.isInitialized());
    try testing.expect(core.log.getLevel() == .debug);
}

test "String utilities integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test string operations
    const trimmed = core.string.trim("  hello world  ");
    try testing.expectEqualStrings("hello world", trimmed);
    
    const parts = try core.string.split(testing.allocator, "a,b,c", ',');
    defer testing.allocator.free(parts);
    try testing.expect(parts.len == 3);
    
    const joined = try core.string.join(testing.allocator, parts, "-");
    defer testing.allocator.free(joined);
    try testing.expectEqualStrings("a-b-c", joined);
}

test "Time utilities integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    const start_time = core.time.now();
    core.time.sleep(1); // Sleep 1ms
    const end_time = core.time.now();
    
    try testing.expect(end_time > start_time);
    
    // Test timer
    var timer = core.time.Timer.start();
    core.time.sleep(1);
    const elapsed = timer.elapsedMillis();
    try testing.expect(elapsed >= 0);
    
    // Test stopwatch
    var stopwatch = core.time.Stopwatch.init(testing.allocator);
    defer stopwatch.deinit();
    
    try stopwatch.startLap();
    core.time.sleep(1);
    try stopwatch.startLap();
    
    try testing.expect(stopwatch.lapCount() == 2);
    try testing.expect(stopwatch.totalTime() > 0);
}

test "Random utilities integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test deterministic random with seed
    core.random.initWithSeed(12345);
    
    const val1 = core.random.int(u32, 1, 100);
    const val2 = core.random.int(u32, 1, 100);
    try testing.expect(val1 >= 1 and val1 <= 100);
    try testing.expect(val2 >= 1 and val2 <= 100);
    
    // Test float generation
    const f = core.random.float(f32);
    try testing.expect(f >= 0.0 and f < 1.0);
    
    // Test vector generation
    const vec = try core.random.vector(f32, testing.allocator, 5, 0.0, 1.0);
    defer testing.allocator.free(vec);
    try testing.expect(vec.len == 5);
    
    const norm_vec = try core.random.normalizedVector(f32, testing.allocator, 3);
    defer testing.allocator.free(norm_vec);
    
    // Check normalization
    var magnitude: f32 = 0;
    for (norm_vec) |v| {
        magnitude += v * v;
    }
    magnitude = std.math.sqrt(magnitude);
    try testing.expect(std.math.approxEqAbs(f32, magnitude, 1.0, 0.001));
}

test "Performance monitoring integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test performance timing
    var timer = try core.performance.startTimer("test_operation");
    core.time.sleep(1);
    core.performance.endTimer("test_operation", timer);
    
    // Test multiple measurements
    for (0..3) |_| {
        timer = try core.performance.startTimer("repeated_op");
        core.time.sleep(1);
        core.performance.endTimer("repeated_op", timer);
    }
    
    const stats = core.performance.getStats("repeated_op");
    try testing.expect(stats != null);
    try testing.expect(stats.?.count == 3);
    try testing.expect(stats.?.average() > 0);
}

test "Memory management integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test memory pool
    var pool = core.memory.MemoryPool.init(testing.allocator);
    defer pool.deinit();
    
    const pool_alloc = pool.allocator();
    const data = try pool_alloc.alloc(u8, 100);
    try testing.expect(data.len == 100);
    
    // Test memory tracking
    if (core.memory.getTracker()) |tracker| {
        const initial_stats = tracker.getStats();
        
        // Simulate allocation tracking
        try tracker.trackAllocation(0x1000, 256, "test_location");
        
        const updated_stats = tracker.getStats();
        try testing.expect(updated_stats.total_allocated > initial_stats.total_allocated);
        try testing.expect(updated_stats.active_allocations > initial_stats.active_allocations);
        
        // Simulate free
        tracker.trackFree(0x1000);
        
        const final_stats = tracker.getStats();
        try testing.expect(final_stats.total_freed > initial_stats.total_freed);
    }
}

test "Error handling integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test error creation and tracking
    const error_info = core.errors.systemError(1001, "Test system error")
        .withContext("During testing")
        .withLocation("test_core_integration.zig", 123, "test function");
    
    try core.errors.recordError(error_info);
    
    const stats = core.errors.getGlobalErrorStats();
    try testing.expect(stats != null);
    try testing.expect(stats.?.total_errors >= 1);
    
    // Test error formatting
    const formatted = try error_info.format(testing.allocator);
    defer testing.allocator.free(formatted);
    try testing.expect(std.mem.indexOf(u8, formatted, "SYSTEM") != null);
    try testing.expect(std.mem.indexOf(u8, formatted, "Test system error") != null);
}

test "Threading integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    // Test thread pool
    var pool = try core.threading.ThreadPool.init(testing.allocator, 2);
    defer pool.deinit();
    
    var counter = std.atomic.Value(u32).init(0);
    
    const incrementTask = struct {
        fn call(data: ?*anyopaque) void {
            const c: *std.atomic.Value(u32) = @ptrCast(@alignCast(data.?));
            _ = c.fetchAdd(1, .monotonic);
        }
    }.call;
    
    // Submit tasks
    for (0..5) |_| {
        try pool.submit(incrementTask, &counter);
    }
    
    // Wait for completion
    core.time.sleep(50); // 50ms should be enough
    
    try testing.expect(counter.load(.monotonic) == 5);
}

test "Allocator integration" {
    try core.init(testing.allocator);
    defer core.deinit();
    
    try core.allocators.init(testing.allocator);
    defer core.allocators.deinit();
    
    // Test smart allocator
    if (core.allocators.getSmartAllocator()) |smart_alloc| {
        const small_data = try smart_alloc.alloc(u8, 64);
        defer smart_alloc.free(small_data);
        try testing.expect(small_data.len == 64);
        
        const large_data = try smart_alloc.alloc(u8, 2048);
        defer smart_alloc.free(large_data);
        try testing.expect(large_data.len == 2048);
    }
    
    // Test string interning
    const str1 = try core.allocators.internString("test string");
    const str2 = try core.allocators.internString("test string");
    try testing.expect(str1.ptr == str2.ptr); // Should be interned
    
    const stats = core.allocators.getAllocatorStats();
    try testing.expect(stats != null);
}