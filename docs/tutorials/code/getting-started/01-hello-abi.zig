//! Getting Started Tutorial - Example 1: Hello ABI
//!
//! This example demonstrates basic ABI framework initialization.
//! Run with: zig run docs/tutorials/code/getting-started/01-hello-abi.zig

const std = @import("std");

const abi = @import("abi");

pub fn main() !void {
    // Get an allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize ABI
    std.debug.print("Initializing ABI framework...\n", .{});
    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    // Print version
    const version = abi.version();
    std.debug.print("ABI Version: {d}.{d}.{d}\n", .{
        version.major,
        version.minor,
        version.patch,
    });

    std.debug.print("ABI framework initialized successfully!\n", .{});
}
