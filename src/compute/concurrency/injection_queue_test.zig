//! Injection queue tests
//!
//! Tests for the global task injection queue.

const std = @import("std");
const ChaseLevDeque = @import("chase_lev_deque.zig").ChaseLevDeque;
const InjectionQueue = @import("injection_queue.zig").InjectionQueue;

test "injection queue init and deinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const queue = try InjectionQueue.init(allocator, 16);
    defer queue.deinit(allocator);

    try std.testing.expect(queue.capacity == 16);
}

test "injection queue push and pop" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const queue = try InjectionQueue.init(allocator, 16);
    defer queue.deinit(allocator);

    try queue.push(42);
    try queue.push(43);
    try queue.push(44);

    const value1 = queue.pop();
    try std.testing.expect(value1.? == 42);
    try std.testing.expect(value1.? == 42);

    const value2 = queue.pop();
    try std.testing.expect(value2.? == 43);
    try std.testing.expect(value2.? == 43);

    const value3 = queue.pop();
    try std.testing.expect(value3.? == 44);
    try std.testing.expect(value3.? == 44);
}

test "injection queue handles empty" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const queue = try InjectionQueue.init(allocator, 16);
    defer queue.deinit(allocator);

    const popped = queue.pop();
    try std.testing.expect(popped == null);
}
