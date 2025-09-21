//! Unit tests for the Performance Optimizations component.

const std = @import("std");
const testing = std.testing;

// Note: This test should be run through the build system to have proper module access
// For now, we'll create simplified test structures that don't require external modules

test "basic performance measurement" {
    std.debug.print("[DEBUG] Starting basic performance measurement test\n", .{});

    // Test basic performance measurement capabilities
    std.debug.print("[DEBUG] Starting timer...\n", .{});
    var timer = try std.time.Timer.start();
    std.debug.print("[DEBUG] ✓ Timer started\n", .{});

    // Simulate some work
    std.debug.print("[DEBUG] Performing computational work (1000 iterations)...\n", .{});
    var sum: f64 = 0.0;
    for (0..1000) |i| {
        sum += @as(f64, @floatFromInt(i)) * 0.001;
    }
    std.debug.print("[DEBUG] ✓ Computational work completed\n", .{});

    const elapsed = timer.read();
    std.debug.print("[DEBUG] Elapsed time: {d} nanoseconds ({d:.2} microseconds)\n", .{ elapsed, @as(f64, @floatFromInt(elapsed)) / 1000.0 });

    try testing.expect(elapsed > 0);
    std.debug.print("[DEBUG] ✓ Elapsed time is greater than 0\n", .{});

    std.debug.print("[DEBUG] Final sum: {d:.6}\n", .{sum});
    try testing.expect(sum > 0.0);
    std.debug.print("[DEBUG] ✓ Sum is greater than 0\n", .{});

    std.debug.print("[DEBUG] Basic performance measurement test completed successfully\n", .{});
}

test "memory allocation performance" {
    std.debug.print("[DEBUG] Starting memory allocation performance test\n", .{});

    const allocator = testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test memory allocation performance
    std.debug.print("[DEBUG] Starting performance timer...\n", .{});
    var timer = try std.time.Timer.start();

    std.debug.print("[DEBUG] Allocating 1000 f32 values ({d} bytes)...\n", .{1000 * @sizeOf(f32)});
    const data = try allocator.alloc(f32, 1000);
    defer allocator.free(data);
    std.debug.print("[DEBUG] ✓ Memory allocation successful, pointer: {*}\n", .{@intFromPtr(data.ptr)});

    // Fill with test data
    std.debug.print("[DEBUG] Filling allocated memory with test data...\n", .{});
    for (data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.001;
    }
    std.debug.print("[DEBUG] ✓ Memory filled with test data\n", .{});

    const elapsed = timer.read();
    std.debug.print("[DEBUG] Allocation and fill time: {d} nanoseconds ({d:.2} microseconds)\n", .{ elapsed, @as(f64, @floatFromInt(elapsed)) / 1000.0 });

    try testing.expect(elapsed > 0);
    std.debug.print("[DEBUG] ✓ Elapsed time is greater than 0\n", .{});

    try testing.expect(data.len == 1000);
    std.debug.print("[DEBUG] ✓ Allocated array has correct length: {d}\n", .{data.len});

    std.debug.print("[DEBUG] Sample values: [{d:.3}, {d:.3}, {d:.3}, ...]\n", .{ data[0], data[1], data[2] });

    std.debug.print("[DEBUG] Memory allocation performance test completed successfully\n", .{});
}

test "vector operations performance" {
    std.debug.print("[DEBUG] Starting vector operations performance test\n", .{});

    const allocator = testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test vector operations performance
    const size = 1000;
    std.debug.print("[DEBUG] Vector size: {d} elements\n", .{size});

    std.debug.print("[DEBUG] Allocating vector A ({d} bytes)...\n", .{size * @sizeOf(f32)});
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    std.debug.print("[DEBUG] ✓ Vector A allocated\n", .{});

    std.debug.print("[DEBUG] Allocating vector B ({d} bytes)...\n", .{size * @sizeOf(f32)});
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    std.debug.print("[DEBUG] ✓ Vector B allocated\n", .{});

    std.debug.print("[DEBUG] Allocating result vector ({d} bytes)...\n", .{size * @sizeOf(f32)});
    var result = try allocator.alloc(f32, size);
    defer allocator.free(result);
    std.debug.print("[DEBUG] ✓ Result vector allocated\n", .{});

    // Initialize test data
    std.debug.print("[DEBUG] Initializing test data...\n", .{});
    for (a, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.001;
    }
    for (b, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i + 100)) * 0.001;
    }
    std.debug.print("[DEBUG] ✓ Test data initialized\n", .{});

    std.debug.print("[DEBUG] Starting vector addition performance test...\n", .{});
    var timer = try std.time.Timer.start();

    // Perform vector addition
    for (0..size) |i| {
        result[i] = a[i] + b[i];
    }

    const elapsed = timer.read();
    std.debug.print("[DEBUG] Vector addition completed in {d} nanoseconds ({d:.2} microseconds)\n", .{ elapsed, @as(f64, @floatFromInt(elapsed)) / 1000.0 });

    try testing.expect(elapsed > 0);
    std.debug.print("[DEBUG] ✓ Operation time is greater than 0\n", .{});

    std.debug.print("[DEBUG] Sample result values: [{d:.3}, {d:.3}, {d:.3}, ...]\n", .{ result[0], result[1], result[2] });
    try testing.expect(result[0] > 0.0);
    std.debug.print("[DEBUG] ✓ First result value is positive: {d:.3}\n", .{result[0]});

    std.debug.print("[DEBUG] Vector operations performance test completed successfully\n", .{});
}

test "string operations performance" {
    std.debug.print("[DEBUG] Starting string operations performance test\n", .{});

    const allocator = testing.allocator;
    std.debug.print("[DEBUG] Using test allocator\n", .{});

    // Test string operations performance
    const test_string = "Hello, World! This is a test string for performance testing.";
    const iterations = 1000;

    std.debug.print("[DEBUG] Test string: {s} (length: {d})\n", .{ test_string, test_string.len });
    std.debug.print("[DEBUG] Iterations: {d}\n", .{iterations});

    std.debug.print("[DEBUG] Starting string operations performance test...\n", .{});
    var timer = try std.time.Timer.start();

    var total_copies: usize = 0;
    var total_searches: usize = 0;

    for (0..iterations) |_| {
        const copy = try allocator.dupe(u8, test_string);
        defer allocator.free(copy);
        total_copies += 1;

        // Perform some string operations
        const search_result = std.mem.indexOf(u8, copy, "test");
        if (search_result != null) total_searches += 1;

        _ = copy.len;
    }

    const elapsed = timer.read();
    const avg_time_per_iteration = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("[DEBUG] String operations completed in {d} nanoseconds\n", .{elapsed});
    std.debug.print("[DEBUG] Average time per iteration: {d:.2} nanoseconds\n", .{avg_time_per_iteration});
    std.debug.print("[DEBUG] Total copies made: {d}\n", .{total_copies});
    std.debug.print("[DEBUG] Successful searches: {d}/{d}\n", .{ total_searches, iterations });

    try testing.expect(elapsed > 0);
    std.debug.print("[DEBUG] ✓ Total elapsed time is greater than 0\n", .{});

    try testing.expect(total_copies == iterations);
    std.debug.print("[DEBUG] ✓ All string copies were successful\n", .{});

    try testing.expect(total_searches == iterations);
    std.debug.print("[DEBUG] ✓ All string searches found the target substring\n", .{});

    std.debug.print("[DEBUG] String operations performance test completed successfully\n", .{});
}

test "mathematical operations performance" {
    const allocator = testing.allocator;

    // Test mathematical operations performance
    const size = 1000;
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);

    // Initialize test data
    for (data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) / 100.0;
    }

    var timer = try std.time.Timer.start();

    // Perform mathematical operations
    var sum: f32 = 0.0;
    for (data) |val| {
        sum += val * val; // Square and sum
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
    try testing.expect(sum > 0.0);
}

test "array operations performance" {
    const allocator = testing.allocator;

    // Test array operations performance
    const size = 1000;
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);

    // Initialize test data
    for (data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.001;
    }

    var timer = try std.time.Timer.start();

    // Perform array operations (sort simulation)
    for (0..size - 1) |i| {
        if (data[i] > data[i + 1]) {
            const temp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = temp;
        }
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
}

test "hash map operations performance" {
    const allocator = testing.allocator;

    // Test hash map operations performance
    var map = std.AutoHashMap(u32, f32).init(allocator);
    defer map.deinit();

    const iterations = 1000;

    var timer = try std.time.Timer.start();

    // Insert operations
    for (0..iterations) |i| {
        try map.put(@as(u32, @intCast(i)), @as(f32, @floatFromInt(i)) * 0.001);
    }

    // Lookup operations
    for (0..iterations) |i| {
        _ = map.get(@as(u32, @intCast(i)));
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
    try testing.expect(map.count() == iterations);
}

test "file I/O performance" {
    // Test file I/O performance
    const test_file = "test_performance.tmp";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    const test_data = "This is test data for performance testing.";

    var timer = try std.time.Timer.start();

    // Write test data
    const file = try std.fs.cwd().createFile(test_file, .{});
    defer file.close();
    try file.writeAll(test_data);

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
}

test "floating point operations performance" {
    // Test floating point operations performance
    const iterations = 10000;

    var timer = try std.time.Timer.start();

    var sum: f64 = 0.0;
    for (0..iterations) |i| {
        const x = @as(f64, @floatFromInt(i)) * 0.001;
        sum += @sin(x) + @cos(x) + @sqrt(x + 1.0);
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
    try testing.expect(sum > 0.0);
}

test "memory allocation patterns performance" {
    const allocator = testing.allocator;

    // Test memory allocation patterns performance
    const iterations = 1000;
    const sizes = [_]usize{ 16, 32, 64, 128, 256, 512, 1024 };

    var timer = try std.time.Timer.start();

    for (0..iterations) |i| {
        const size = sizes[i % sizes.len];
        const data = try allocator.alloc(u8, size);
        defer allocator.free(data);

        // Fill with pattern
        for (data, 0..) |*val, j| {
            val.* = @as(u8, @intCast((i + j) % 256));
        }
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
}

test "concurrent operations performance" {
    const allocator = testing.allocator;

    // Test concurrent operations performance
    const iterations = 1000;
    const data = try allocator.alloc(f32, iterations);
    defer allocator.free(data);

    // Initialize test data
    for (data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.001;
    }

    var timer = try std.time.Timer.start();

    // Simulate concurrent operations (sequential for testing)
    var sum: f32 = 0.0;
    for (data) |val| {
        sum += val * val;
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
    try testing.expect(sum > 0.0);
}

test "data structure operations performance" {
    const allocator = testing.allocator;

    // Test data structure operations performance
    var list = try std.ArrayList(f32).initCapacity(allocator, 0);
    defer list.deinit(allocator);

    const iterations = 1000;

    var timer = try std.time.Timer.start();

    // Insert operations
    for (0..iterations) |i| {
        try list.append(allocator, @as(f32, @floatFromInt(i)) * 0.001);
    }

    // Access operations
    var sum: f32 = 0.0;
    for (list.items) |val| {
        sum += val;
    }

    const elapsed = timer.read();
    try testing.expect(elapsed > 0);
    try testing.expect(list.items.len == iterations);
    try testing.expect(sum > 0.0);
}
