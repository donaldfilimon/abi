//! Memory Allocator Benchmarks
//!
//! Measures performance of various allocator strategies in the framework.
//! Strategies: GPA (General Purpose), Arena, and Threaded benchmarks.

const std = @import("std");
const abi = @import("abi");
const framework = @import("../system/framework.zig");

pub fn runMemoryBenchmarks(allocator: std.mem.Allocator, config: anytype) !void {
    _ = config;
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    MEMORY ALLOCATOR BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    try benchmarkStrategy(allocator, &runner, .gpa);
    try benchmarkStrategy(allocator, &runner, .arena);

    runner.printSummaryDebug();
}

const Strategy = enum { gpa, arena };

fn benchmarkStrategy(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, strategy: Strategy) !void {
    const sizes = [_]usize{ 16, 256, 1024, 4096 };
    const count = 1000;

    for (sizes) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "{s}_alloc_{d}B", .{ @tagName(strategy), size }) catch "alloc";

        _ = try runner.run(
            .{ .name = name, .category = "memory", .warmup_iterations = 10 },
            struct {
                fn bench(s: Strategy, sz: usize, c: usize, a: std.mem.Allocator) !void {
                    switch (s) {
                        .gpa => {
                            var gpa = std.heap.DebugAllocator(.{}){};
                            defer _ = gpa.deinit();
                            const gpa_alloc = gpa.allocator();
                            for (0..c) |_| {
                                const ptr = try gpa_alloc.alloc(u8, sz);
                                gpa_alloc.free(ptr);
                            }
                        },
                        .arena => {
                            var arena = std.heap.ArenaAllocator.init(a);
                            defer arena.deinit();
                            const arena_alloc = arena.allocator();
                            for (0..c) |_| {
                                _ = try arena_alloc.alloc(u8, sz);
                            }
                        },
                    }
                }
            }.bench,
            .{ strategy, size, count, allocator },
        );
    }
}
