//! Concurrency Example
//!
//! Demonstrates lock-free concurrency primitives:
//! - MPMC Queue: Multi-producer multi-consumer bounded queue
//! - Chase-Lev Deque: Work-stealing deque for task scheduling
//!
//! Run with: zig build run-concurrency

const std = @import("std");
const abi = @import("abi");
const primitives = abi.services.shared.utils.primitives;

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== ABI Concurrency Primitives Demo ===\n\n", .{});
    std.debug.print("Platform: {s}\n\n", .{primitives.Platform.description()});

    // Demo 1: MPMC Queue
    try demoMpmcQueue(allocator);

    // Demo 2: Chase-Lev Work-Stealing Deque
    try demoChaseLevDeque(allocator);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

/// Demonstrate Multi-Producer Multi-Consumer Queue
fn demoMpmcQueue(allocator: std.mem.Allocator) !void {
    std.debug.print("--- MPMC Queue Demo ---\n", .{});

    const MpmcQueue = abi.services.runtime.concurrency.MpmcQueue(u64);
    var queue = try MpmcQueue.init(allocator, 256);
    defer queue.deinit();

    // Push some values
    std.debug.print("Pushing values: ", .{});
    var i: u64 = 0;
    while (i < 5) : (i += 1) {
        queue.push(i * 10) catch |err| {
            std.debug.print("(push failed: {t}) ", .{err});
            continue;
        };
        std.debug.print("{d} ", .{i * 10});
    }
    std.debug.print("\n", .{});

    // Pop values
    std.debug.print("Popping values: ", .{});
    while (queue.pop()) |value| {
        std.debug.print("{d} ", .{value});
    }
    std.debug.print("\n\n", .{});
}

/// Demonstrate Chase-Lev Work-Stealing Deque
fn demoChaseLevDeque(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Chase-Lev Deque Demo ---\n", .{});

    const ChaseLevDeque = abi.services.runtime.concurrency.ChaseLevDeque(u32);
    var deque = try ChaseLevDeque.init(allocator);
    defer deque.deinit();

    // Owner pushes work items (from bottom)
    std.debug.print("Owner pushing tasks: ", .{});
    var task_id: u32 = 100;
    while (task_id < 105) : (task_id += 1) {
        deque.push(task_id) catch |err| {
            std.debug.print("(push failed: {t}) ", .{err});
            continue;
        };
        std.debug.print("{d} ", .{task_id});
    }
    std.debug.print("\n", .{});

    // Owner pops own work (LIFO - from bottom)
    std.debug.print("Owner taking tasks (LIFO): ", .{});
    for (0..2) |_| {
        if (deque.pop()) |task| {
            std.debug.print("{d} ", .{task});
        }
    }
    std.debug.print("\n", .{});

    // Thieves steal work (from top)
    std.debug.print("Thief stealing tasks: ", .{});
    for (0..3) |_| {
        if (deque.steal()) |task| {
            std.debug.print("{d} ", .{task});
        }
    }
    std.debug.print("\n\n", .{});
}
