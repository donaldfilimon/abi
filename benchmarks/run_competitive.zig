//! Competitive Benchmark Runner
//!
//! Executable entry point for running competitive benchmarks.

const std = @import("std");
const competitive = @import("competitive/mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Default configuration
    const config = competitive.CompetitiveConfig{
        // Use smaller dataset for default run to be quick
        .dataset_sizes = &.{ 1_000, 10_000 },
        .dimensions = &.{128},
        .num_queries = 100,
        .top_k_values = &.{10},
        .profile_memory = true,
    };

    try competitive.runAllBenchmarks(allocator, config);
}
