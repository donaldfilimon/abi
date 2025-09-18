//! WDBX Advanced Benchmarking Suite (moved to benchmarks/)
//! See original root file for full implementation comments.

const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Minimal stub that prints usage and exits; full benchmark remains callable via build steps
    std.debug.print("WDBX Advanced Benchmarking Suite. Use 'zig build benchmark-simple' to run.\n", .{});
    _ = allocator;
}
