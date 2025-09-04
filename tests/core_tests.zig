//! Core module tests

const std = @import("std");
const testing = std.testing;
const wdbx = @import("wdbx");
const test_framework = @import("test_framework.zig");

test "allocators: memory pool" {
    const allocator = testing.allocator;
    
    var pool = try wdbx.core.allocators.MemoryPool.init(allocator, 64, 10);
    defer pool.deinit();
    
    // Test allocation
    var allocations: [10]?[]u8 = undefined;
    for (&allocations) |*alloc| {
        alloc.* = pool.alloc();
        try testing.expect(alloc.* != null);
    }
    
    // Pool should be exhausted
    try testing.expect(pool.alloc() == null);
    
    // Test deallocation and reallocation
    pool.free(allocations[0].?);
    const new_alloc = pool.alloc();
    try testing.expect(new_alloc != null);
}

test "allocators: stack allocator" {
    var buffer: [1024]u8 = undefined;
    var stack = wdbx.core.allocators.StackAllocator.init(&buffer);
    const allocator = stack.allocator();
    
    // Test sequential allocations
    const ptr1 = try allocator.alloc(u8, 100);
    const ptr2 = try allocator.alloc(u8, 200);
    const ptr3 = try allocator.alloc(u8, 300);
    
    // Verify sequential layout
    try testing.expect(@intFromPtr(ptr2.ptr) > @intFromPtr(ptr1.ptr));
    try testing.expect(@intFromPtr(ptr3.ptr) > @intFromPtr(ptr2.ptr));
    
    // Test reset
    stack.reset();
    const ptr4 = try allocator.alloc(u8, 50);
    try testing.expectEqual(@intFromPtr(ptr1.ptr), @intFromPtr(ptr4.ptr));
}

test "collections: ring buffer" {
    const allocator = testing.allocator;
    
    var ring = try wdbx.core.collections.RingBuffer.init(allocator, 256);
    defer ring.deinit();
    
    // Test write and read
    const data = "Hello, World!";
    try ring.write(data);
    
    var buf: [32]u8 = undefined;
    const read = try ring.read(&buf);
    try testing.expectEqual(data.len, read);
    try testing.expectEqualStrings(data, buf[0..read]);
}

test "collections: bloom filter" {
    const allocator = testing.allocator;
    
    var filter = try wdbx.core.collections.BloomFilter.init(allocator, 1000, 0.01);
    defer filter.deinit();
    
    // Add items
    filter.add("apple");
    filter.add("banana");
    filter.add("cherry");
    
    // Test membership
    try testing.expect(filter.contains("apple"));
    try testing.expect(filter.contains("banana"));
    try testing.expect(filter.contains("cherry"));
    try testing.expect(!filter.contains("durian"));
}

test "collections: trie" {
    const allocator = testing.allocator;
    
    var trie = try wdbx.core.collections.Trie.init(allocator);
    defer trie.deinit();
    
    // Insert items
    try trie.insert("hello", "world");
    try trie.insert("help", "me");
    try trie.insert("hero", "batman");
    
    // Test retrieval
    try testing.expectEqualStrings("world", trie.get("hello").?);
    try testing.expectEqualStrings("me", trie.get("help").?);
    try testing.expect(trie.get("hell") == null);
    
    // Test prefix search
    var results = std.ArrayList([]const u8).init(allocator);
    defer results.deinit();
    
    try trie.prefixSearch("hel", &results);
    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "error handling: error info" {
    const info = wdbx.core.error_handling.ErrorInfo{
        .code = wdbx.core.error_handling.Error.NotFound,
        .message = "Resource not found",
        .source = "test.zig",
        .line = 42,
    };
    
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try info.format(stream.writer());
    
    const output = stream.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "NotFound") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Resource not found") != null);
}

test "error handling: result type" {
    const ResultI32 = wdbx.core.error_handling.Result(i32);
    
    // Test success case
    const ok = ResultI32{ .ok = 42 };
    try testing.expect(ok.isOk());
    try testing.expectEqual(@as(i32, 42), ok.unwrap());
    
    // Test error case
    const err = ResultI32{ 
        .err = .{
            .code = wdbx.core.error_handling.Error.InvalidArgument,
            .message = "Invalid value",
        }
    };
    try testing.expect(err.isErr());
    try testing.expectEqual(@as(i32, 0), err.unwrapOr(0));
}

test "logging: logger basic functionality" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const temp_path = try ctx.getTempPath("test.log");
    defer allocator.free(temp_path);
    
    var logger = try wdbx.core.logging.Logger.init(allocator, .{
        .level = .debug,
        .output = .{ .file = temp_path },
        .use_color = false,
    });
    defer logger.deinit();
    
    // Log messages
    logger.debug(@src(), "Debug message", .{});
    logger.info(@src(), "Info message", .{});
    logger.warn(@src(), "Warning message", .{});
    logger.err(@src(), "Error message", .{});
    
    // Verify log file exists
    const file = try std.fs.cwd().openFile(temp_path, .{});
    defer file.close();
    
    const content = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(content);
    
    try testing.expect(std.mem.indexOf(u8, content, "Debug message") != null);
    try testing.expect(std.mem.indexOf(u8, content, "Info message") != null);
}

test "config: basic operations" {
    const allocator = testing.allocator;
    
    var config = wdbx.core.config.Config.init(allocator);
    defer config.deinit();
    
    // Set values
    try config.setString("app.name", "TestApp");
    try config.setInt("app.port", 8080);
    try config.setBool("app.debug", true);
    try config.setFloat("app.timeout", 30.5);
    
    // Get values
    try testing.expectEqualStrings("TestApp", config.getString("app.name").?);
    try testing.expectEqual(@as(i64, 8080), config.getInt("app.port").?);
    try testing.expectEqual(true, config.getBool("app.debug").?);
    try testing.expectApproxEqAbs(@as(f64, 30.5), config.getFloat("app.timeout").?, 0.001);
}

test "threading: thread pool" {
    const allocator = testing.allocator;
    
    var pool = try wdbx.core.threading.ThreadPool.init(allocator, 4);
    defer pool.deinit();
    
    var counter = std.atomic.Value(i32).init(0);
    
    const Task = struct {
        fn increment(c: *std.atomic.Value(i32)) void {
            _ = c.fetchAdd(1, .monotonic);
        }
    };
    
    // Submit multiple tasks
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try pool.submit(Task.increment, &counter);
    }
    
    pool.wait();
    try testing.expectEqual(@as(i32, 100), counter.load(.acquire));
}

test "threading: channel" {
    const allocator = testing.allocator;
    
    var channel = try wdbx.core.threading.Channel(i32).init(allocator, 10);
    defer channel.deinit();
    
    // Test send and receive
    try channel.send(42);
    try channel.send(43);
    try channel.send(44);
    
    try testing.expectEqual(@as(i32, 42), try channel.receive());
    try testing.expectEqual(@as(i32, 43), try channel.receive());
    try testing.expectEqual(@as(i32, 44), try channel.receive());
    
    // Test non-blocking receive
    try testing.expect(channel.tryReceive() == null);
}

test "time: timer and duration" {
    var timer = wdbx.core.time.Timer.start();
    
    // Simulate work
    std.time.sleep(10 * std.time.ns_per_ms);
    
    const elapsed = wdbx.core.time.Duration.fromNanos(timer.read());
    try testing.expect(elapsed.toMillis() >= 10);
    
    // Test duration operations
    const d1 = wdbx.core.time.Duration.fromSeconds(1);
    const d2 = wdbx.core.time.Duration.fromMillis(500);
    const d3 = d1.add(d2);
    
    try testing.expectEqual(@as(u64, 1500), d3.toMillis());
}

test "time: rate limiter" {
    var limiter = wdbx.core.time.RateLimiter.init(10, 5);
    
    // Should allow up to burst
    var allowed: usize = 0;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        if (limiter.allow(1)) {
            allowed += 1;
        }
    }
    
    try testing.expectEqual(@as(usize, 5), allowed);
    
    // Wait for refill
    std.time.sleep(150 * std.time.ns_per_ms);
    
    // Should allow more
    try testing.expect(limiter.allow(1));
}

test "utils: UUID generation" {
    const uuid1 = wdbx.core.utils.generateUuid();
    const uuid2 = wdbx.core.utils.generateUuid();
    
    // UUIDs should be unique
    try testing.expect(!std.mem.eql(u8, &uuid1, &uuid2));
    
    // Test formatting
    const formatted = wdbx.core.utils.formatUuid(uuid1);
    try testing.expectEqual(@as(usize, 36), formatted.len);
    try testing.expect(formatted[8] == '-');
    try testing.expect(formatted[13] == '-');
    try testing.expect(formatted[18] == '-');
    try testing.expect(formatted[23] == '-');
}

test "utils: byte formatting" {
    const cases = .{
        .{ 0, 0.0, "B" },
        .{ 1023, 1023.0, "B" },
        .{ 1024, 1.0, "KB" },
        .{ 1536, 1.5, "KB" },
        .{ 1048576, 1.0, "MB" },
        .{ 1073741824, 1.0, "GB" },
    };
    
    inline for (cases) |case| {
        const result = wdbx.core.utils.formatBytes(case[0]);
        try testing.expectApproxEqAbs(case[1], result.value, 0.01);
        try testing.expectEqualStrings(case[2], result.unit);
    }
}

test "debug: hex dump" {
    const data = "Hello, World! 1234567890";
    
    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    
    try wdbx.core.debug.hexDump(data, stream.writer());
    
    const output = stream.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "48 65 6c 6c 6f") != null); // "Hello"
    try testing.expect(std.mem.indexOf(u8, output, "|Hello, World!") != null);
}