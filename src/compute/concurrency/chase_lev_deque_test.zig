//! Chase-Lev deque tests
//!
//! Tests for the lock-free work-stealing deque.

const std = @import("std");
const ChaseLevDeque = @import("chase_lev_deque.zig").ChaseLevDeque;

test "deque init and deinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deque = try ChaseLevDeque.init(allocator, 16);
    defer deque.deinit(allocator);

    try std.testing.expect(deque.capacity == 16);
}

test "deque push and pop from bottom" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deque = try ChaseLevDeque.init(allocator, 16);
    defer deque.deinit(allocator);

    try deque.pushBottom(allocator, 42);
    try deque.pushBottom(allocator, 43);

    const value1 = deque.popBottom();
    try std.testing.expect(value1.? == 43);
    try std.testing.expect(value1.? == 43);
}

test "deque steal from top" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deque = try ChaseLevDeque.init(allocator, 16);
    defer deque.deinit(allocator);

    try deque.pushBottom(allocator, 42);

    const stolen = deque.steal();
    try std.testing.expect(stolen.? == 42);
}

test "deque handles empty correctly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deque = try ChaseLevDeque.init(allocator, 16);
    defer deque.deinit(allocator);

    const popped = deque.popBottom();
    try std.testing.expect(popped == null);
}

test "deque handles overflow" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deque = try ChaseLevDeque.init(allocator, 4);
    defer deque.deinit(allocator);

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try deque.pushBottom(allocator, @as(u64, @intCast(i)));
    }

    const value = deque.popBottom();
    try std.testing.expect(value != null);
}
