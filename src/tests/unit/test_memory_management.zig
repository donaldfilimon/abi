//! Unit tests for the Memory Management component.

const std = @import("std");
const testing = std.testing;

// Enhanced memory management tests with comprehensive coverage
test "memory management - basic allocation patterns" {
    const allocator = testing.allocator;

    // Test various allocation sizes
    {
        // Small allocation
        const small = try allocator.alloc(u8, 8);
        defer allocator.free(small);
        try testing.expectEqual(@as(usize, 8), small.len);

        // Medium allocation
        const medium = try allocator.alloc(u32, 1024);
        defer allocator.free(medium);
        try testing.expectEqual(@as(usize, 1024), medium.len);

        // Large allocation
        const large = try allocator.alloc(f64, 100_000);
        defer allocator.free(large);
        try testing.expectEqual(@as(usize, 100_000), large.len);
    }
}

test "memory management - zero-sized allocations" {
    const allocator = testing.allocator;

    // Test edge case: zero-sized allocation
    {
        const empty = try allocator.alloc(u8, 0);
        defer allocator.free(empty);
        try testing.expectEqual(@as(usize, 0), empty.len);
    }
}

test "memory management - memory alignment" {
    const allocator = testing.allocator;

    // Test memory alignment for different types
    {
        // Test alignment for SIMD-friendly types
        const aligned_f32 = try allocator.alignedAlloc(f32, std.mem.Alignment.fromByteUnits(16), 64);
        defer allocator.free(aligned_f32);

        // Check alignment
        const addr = @intFromPtr(aligned_f32.ptr);
        try testing.expectEqual(@as(usize, 0), addr % 16);

        // Test with different alignment
        const aligned_u64 = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(32), 8);
        defer allocator.free(aligned_u64);

        const addr_u64 = @intFromPtr(aligned_u64.ptr);
        try testing.expectEqual(@as(usize, 0), addr_u64 % 32);
    }
}

test "memory management - realloc behavior" {
    const allocator = testing.allocator;

    // Test memory reallocation
    {
        var data = try allocator.alloc(u8, 16);
        defer allocator.free(data);

        // Fill initial data
        for (data, 0..) |*val, i| {
            val.* = @as(u8, @intCast(i));
        }

        // Realloc to larger size
        data = try allocator.realloc(data, 32);
        try testing.expectEqual(@as(usize, 32), data.len);

        // Verify original data is preserved
        for (0..16) |i| {
            try testing.expectEqual(@as(u8, @intCast(i)), data[i]);
        }
    }
}

test "memory management - ArrayList capacity management" {
    const allocator = testing.allocator;

    // Test ArrayList capacity growth and memory efficiency
    {
        var list = try std.ArrayList(u32).initCapacity(allocator, 4);
        defer list.deinit(allocator);

        // Add elements to trigger capacity growth
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            try list.append(allocator, i);
        }

        try testing.expectEqual(@as(usize, 100), list.items.len);
        try testing.expect(list.capacity >= list.items.len);

        // Test shrink operation
        list.shrinkAndFree(allocator, 50);
        try testing.expectEqual(@as(usize, 50), list.items.len);
    }
}

test "memory management - HashMap memory efficiency" {
    const allocator = testing.allocator;

    // Test HashMap memory usage patterns
    {
        var map = std.AutoHashMap(u32, u32).init(allocator);
        defer map.deinit();

        // Add many entries
        for (0..1000) |i| {
            try map.put(@as(u32, @intCast(i)), @as(u32, @intCast(i * 2)));
        }

        try testing.expectEqual(@as(usize, 1000), map.count());

        // Test memory usage after removals
        for (0..500) |i| {
            _ = map.remove(@as(u32, @intCast(i)));
        }

        try testing.expectEqual(@as(usize, 500), map.count());
    }
}

test "memory management - string duplication and formatting" {
    const allocator = testing.allocator;

    // Test string memory management patterns
    {
        // Test string duplication
        const original = "Hello, World!";
        const copy = try allocator.dupe(u8, original);
        defer allocator.free(copy);

        try testing.expectEqualStrings(original, copy);

        // Test formatted string allocation
        const formatted = try std.fmt.allocPrint(allocator, "Value: {d}", .{42});
        defer allocator.free(formatted);

        try testing.expectEqualStrings("Value: 42", formatted);
    }
}

test "memory management - arena allocation" {
    // Test arena allocator for temporary allocations
    {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        // Allocate multiple items in arena
        const str1 = try arena_allocator.dupe(u8, "First string");
        const str2 = try arena_allocator.dupe(u8, "Second string");
        const data = try arena_allocator.alloc(u32, 100);

        // All should be valid
        try testing.expectEqualStrings("First string", str1);
        try testing.expectEqualStrings("Second string", str2);
        try testing.expectEqual(@as(usize, 100), data.len);

        // All memory freed at once when arena deinit is called
    }
}

test "memory management - memory leak detection pattern" {
    const allocator = testing.allocator;

    // Test pattern for detecting potential memory leaks
    {
        var allocations = try std.ArrayList([]u8).initCapacity(allocator, 0);
        defer {
            // Clean up all allocations
            for (allocations.items) |alloc| {
                allocator.free(alloc);
            }
            allocations.deinit(allocator);
        }

        // Simulate some work with allocations
        for (0..10) |i| {
            const size = (i + 1) * 64;
            const data = try allocator.alloc(u8, size);
            try allocations.append(allocator, data);

            // Fill with pattern
            for (data) |*val| {
                val.* = @as(u8, @intCast(i));
            }
        }

        try testing.expectEqual(@as(usize, 10), allocations.items.len);

        // Verify all allocations are still valid
        for (allocations.items, 0..) |data, i| {
            const expected_size = (i + 1) * 64;
            try testing.expectEqual(expected_size, data.len);
        }
    }
}

test "memory management - stack fallback allocation" {
    // Test stack allocation for small, temporary data
    {
        var buffer: [1024]u8 = undefined;

        // Use FixedBufferAllocator for small allocations
        var fba = std.heap.FixedBufferAllocator.init(&buffer);
        const stack_allocator = fba.allocator();

        // Allocate from stack buffer
        const data = try stack_allocator.alloc(u32, 100);
        try testing.expectEqual(@as(usize, 100), data.len);

        // Fill data
        for (data, 0..) |*val, i| {
            val.* = @as(u32, @intCast(i));
        }

        // Verify data
        for (data, 0..) |val, i| {
            try testing.expectEqual(@as(u32, @intCast(i)), val);
        }
    }
}
