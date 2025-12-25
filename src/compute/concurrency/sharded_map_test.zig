//! Sharded map tests
//!
//! Tests for the lock-based sharded hash map.

const std = @import("std");
const ShardedMap = @import("sharded_map.zig").ShardedMap;

test "sharded map init and deinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const map = try ShardedMap.init(allocator, 4);
    defer map.deinit(allocator);

    try std.testing.expect(map.shards.items.len == 4);
}

test "sharded map put and get" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const map = try ShardedMap.init(allocator, 2);
    defer map.deinit(allocator);

    try map.put(42, 100);
    try map.put(43, 101);

    const value1 = map.get(42);
    try std.testing.expect(value1.? == 100);
    try std.testing.expect(value1.? == 100);

    const value2 = map.get(43);
    try std.testing.expect(value2.? == 101);
    try std.testing.expect(value2.? == 101);
}

test "sharded map remove" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const map = try ShardedMap.init(allocator, 2);
    defer map.deinit(allocator);

    try map.put(42, 100);
    try map.put(43, 101);

    try std.testing.expect(map.contains(42));
    try std.testing.expect(map.contains(43));

    const removed1 = map.remove(42);
    try std.testing.expect(removed1.? == 100);

    const removed2 = map.remove(43);
    try std.testing.expect(removed2.? == 101);

    try std.testing.expect(!map.contains(42));
    try std.testing.expect(!map.contains(43));
}

test "sharded map contains" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const map = try ShardedMap.init(allocator, 2);
    defer map.deinit(allocator);

    try std.testing.expect(!map.contains(1));
    try std.testing.expect(!map.contains(2));

    try map.put(1, 100);
    try map.put(2, 200);

    try std.testing.expect(map.contains(1));
    try std.testing.expect(map.contains(2));
}
