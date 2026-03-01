//! Competitive Benchmark Runner
//!
//! Executable entry point for running competitive benchmarks.

const std = @import("std");
const competitive = @import("competitive/mod.zig");
const framework = @import("system/framework.zig");

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const args = try init.args.toSlice(arena.allocator());

    var json_output = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
        }
    }

    // Default configuration
    const config = competitive.CompetitiveConfig{
        // Use smaller dataset for default run to be quick
        .dataset_sizes = &.{ 1_000, 10_000 },
        .dimensions = &.{128},
        .num_queries = 100,
        .top_k_values = &.{10},
        .profile_memory = true,
        .output_format = if (json_output) .json else .markdown,
    };

    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    try competitive.runAllBenchmarks(allocator, config, &runner);

    if (json_output) {
        runner.exportJson();
    } else {
        runner.printSummaryDebug();
    }
}
