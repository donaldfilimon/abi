//! Core Infrastructure Performance Benchmarks
//!
//! This benchmark suite measures the performance of the new core infrastructure
//! including I/O abstraction, error handling, and diagnostics.

const std = @import("std");
const abi = @import("../src/mod.zig");

const Benchmark = struct {
    name: []const u8,
    iterations: u32,
    setup: ?*const fn (std.mem.Allocator) anyerror!void = null,
    teardown: ?*const fn (std.mem.Allocator) anyerror!void = null,
    run: *const fn (std.mem.Allocator) anyerror!void,
};

// I/O Abstraction Benchmarks

fn benchmark_stdout_writer(allocator: std.mem.Allocator) !void {
    const writer = abi.core.Writer.stdout();
    for (0..1000) |i| {
        try writer.print("Iteration {d}: Hello, world!\n", .{i});
    }
}

fn benchmark_test_writer(allocator: std.mem.Allocator) !void {
    var test_writer = abi.core.TestWriter.init(allocator);
    defer test_writer.deinit();
    
    const writer = test_writer.writer();
    for (0..1000) |i| {
        try writer.print("Iteration {d}: Hello, world!\n", .{i});
    }
}

fn benchmark_buffered_writer(allocator: std.mem.Allocator) !void {
    var buffered = abi.core.BufferedWriter.init(allocator, 4096);
    defer buffered.deinit();
    
    const writer = buffered.writer();
    for (0..1000) |i| {
        try writer.print("Iteration {d}: Hello, world!\n", .{i});
    }
    try buffered.flush();
}

// Error Handling Benchmarks

fn benchmark_error_context_creation(allocator: std.mem.Allocator) !void {
    for (0..10000) |i| {
        const ctx = abi.core.ErrorContext.init(error.TestError, "Test error {d}", .{i})
            .withLocation(abi.core.here())
            .withContext("Additional context {d}", .{i});
        
        // Prevent optimization
        _ = ctx;
    }
}

fn benchmark_error_recovery(allocator: std.mem.Allocator) !void {
    for (0..10000) |_| {
        const result = riskyOperation() catch |err| {
            const ctx = abi.core.ErrorContext.init(err, "Operation failed")
                .withLocation(abi.core.here());
            return ctx;
        };
        _ = result;
    }
}

fn riskyOperation() !u32 {
    return error.TestError;
}

// Diagnostics Benchmarks

fn benchmark_diagnostic_collection(allocator: std.mem.Allocator) !void {
    var diagnostics = abi.core.DiagnosticCollector.init(allocator);
    defer diagnostics.deinit();
    
    for (0..1000) |i| {
        try diagnostics.add(.{
            .severity = if (i % 2 == 0) .info else .warning,
            .message = try std.fmt.allocPrint(allocator, "Message {d}", .{i}),
            .location = abi.core.here(),
        });
    }
}

fn benchmark_diagnostic_processing(allocator: std.mem.Allocator) !void {
    var diagnostics = abi.core.DiagnosticCollector.init(allocator);
    defer diagnostics.deinit();
    
    // Add some diagnostics
    for (0..100) |i| {
        try diagnostics.add(.{
            .severity = .info,
            .message = try std.fmt.allocPrint(allocator, "Message {d}", .{i}),
            .location = abi.core.here(),
        });
    }
    
    // Process them
    for (diagnostics.items) |diag| {
        _ = diag;
    }
}

// Memory Management Benchmarks

fn benchmark_allocator_usage(allocator: std.mem.Allocator) !void {
    for (0..1000) |i| {
        const data = try allocator.alloc(u8, 1024);
        defer allocator.free(data);
        
        // Use the data
        @memset(data, @intCast(i % 256));
    }
}

fn benchmark_collection_operations(allocator: std.mem.Allocator) !void {
    var list = std.ArrayList(u32).init(allocator);
    defer list.deinit();
    
    // Add elements
    for (0..10000) |i| {
        try list.append(@intCast(i));
    }
    
    // Access elements
    for (list.items) |item| {
        _ = item;
    }
}

// Benchmark Suite

const benchmarks = [_]Benchmark{
    .{
        .name = "stdout_writer",
        .iterations = 10,
        .run = benchmark_stdout_writer,
    },
    .{
        .name = "test_writer",
        .iterations = 10,
        .run = benchmark_test_writer,
    },
    .{
        .name = "buffered_writer",
        .iterations = 10,
        .run = benchmark_buffered_writer,
    },
    .{
        .name = "error_context_creation",
        .iterations = 5,
        .run = benchmark_error_context_creation,
    },
    .{
        .name = "error_recovery",
        .iterations = 5,
        .run = benchmark_error_recovery,
    },
    .{
        .name = "diagnostic_collection",
        .iterations = 5,
        .run = benchmark_diagnostic_collection,
    },
    .{
        .name = "diagnostic_processing",
        .iterations = 5,
        .run = benchmark_diagnostic_processing,
    },
    .{
        .name = "allocator_usage",
        .iterations = 10,
        .run = benchmark_allocator_usage,
    },
    .{
        .name = "collection_operations",
        .iterations = 10,
        .run = benchmark_collection_operations,
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const writer = abi.core.Writer.stdout();
    try writer.print("ABI Framework Core Infrastructure Benchmarks\n");
    try writer.print("============================================\n\n");
    
    for (benchmarks) |benchmark| {
        try writer.print("Running benchmark: {s}\n", .{benchmark.name});
        
        var total_time: u64 = 0;
        var min_time: u64 = std.math.maxInt(u64);
        var max_time: u64 = 0;
        
        for (0..benchmark.iterations) |_| {
            const start = std.time.nanoTimestamp();
            
            if (benchmark.setup) |setup_fn| {
                try setup_fn(allocator);
            }
            
            try benchmark.run(allocator);
            
            if (benchmark.teardown) |teardown_fn| {
                try teardown_fn(allocator);
            }
            
            const end = std.time.nanoTimestamp();
            const duration = @as(u64, @intCast(end - start));
            
            total_time += duration;
            min_time = @min(min_time, duration);
            max_time = @max(max_time, duration);
        }
        
        const avg_time = total_time / benchmark.iterations;
        const avg_ms = avg_time / 1_000_000;
        const min_ms = min_time / 1_000_000;
        const max_ms = max_time / 1_000_000;
        
        try writer.print("  Average: {d}ms\n", .{avg_ms});
        try writer.print("  Min: {d}ms\n", .{min_ms});
        try writer.print("  Max: {d}ms\n", .{max_ms});
        try writer.print("  Iterations: {d}\n\n", .{benchmark.iterations});
    }
    
    try writer.print("Benchmark suite completed!\n");
}