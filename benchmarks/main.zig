//! ABI Framework Comprehensive Benchmark Suite
//!
//! Industry-standard and custom benchmarks for:
//! - SIMD/Vector operations
//! - Memory allocation patterns
//! - Concurrency primitives
//! - Database/HNSW vector search
//! - HTTP/Network operations
//! - Cryptography
//! - AI/ML inference
//!
//! Usage:
//!   zig build benchmarks                    # Run all benchmarks
//!   zig build benchmarks -- --suite=simd    # Run specific suite
//!   zig build benchmarks -- --help          # Show help

const std = @import("std");
const framework = @import("framework.zig");
const simd = @import("simd.zig");
const memory = @import("memory.zig");
const concurrency = @import("concurrency.zig");
const network = @import("network.zig");
const crypto = @import("crypto.zig");

// New consolidated modules
const core = @import("core/mod.zig");
const database = @import("database/mod.zig");
const ai = @import("ai/mod.zig");

const BenchmarkSuite = enum {
    all,
    simd,
    memory,
    concurrency,
    database,
    network,
    crypto,
    ai,
    quick,
};

const Args = struct {
    suite: BenchmarkSuite = .all,
    output_json: ?[]const u8 = null,
    verbose: bool = false,
    quick: bool = false,
};

fn parseArgs(allocator: std.mem.Allocator) Args {
    var args_val = Args{};
    const args = std.process.argsAlloc(allocator) catch return args_val;
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--suite=")) {
            const suite_name = arg["--suite=".len..];
            args_val.suite = std.meta.stringToEnum(BenchmarkSuite, suite_name) orelse .all;
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            args_val.output_json = arg["--output=".len..];
        } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            args_val.verbose = true;
        } else if (std.mem.eql(u8, arg, "--quick") or std.mem.eql(u8, arg, "-q")) {
            args_val.quick = true;
            args_val.suite = .quick;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }

    return args_val;
}

fn printHelp() void {
    const help =
        \\ABI Framework Benchmark Suite
        \\
        \\Usage: benchmarks [OPTIONS]
        \\
        \\Options:
        \\  --suite=<name>    Run specific benchmark suite
        \\                    Available: all, simd, memory, concurrency,
        \\                               database, network, crypto, ai, quick
        \\  --output=<file>   Output results to JSON file
        \\  --verbose, -v     Show verbose output
        \\  --quick, -q       Run quick benchmark subset (for CI)
        \\  --help, -h        Show this help message
        \\
        \\Benchmark Suites:
        \\  simd          SIMD/Vector operations (dot product, matmul, distances)
        \\  memory        Memory allocator patterns (arena, pool, fragmentation)
        \\  concurrency   Concurrency primitives (locks, queues, work stealing)
        \\  database      Vector database operations (HNSW, k-NN search, ANN-benchmarks)
        \\  network       HTTP/Network operations (parsing, JSON, WebSocket)
        \\  crypto        Cryptographic operations (hashing, encryption, KDF)
        \\  ai            AI/ML inference (GEMM, attention, activations, LLM metrics)
        \\  quick         Fast subset for continuous integration
        \\  all           Run all benchmark suites (default)
        \\
        \\Examples:
        \\  benchmarks                         # Run all benchmarks
        \\  benchmarks --suite=simd            # Run only SIMD benchmarks
        \\  benchmarks --quick                 # Run quick CI benchmarks
        \\  benchmarks --suite=ai --verbose    # AI benchmarks with details
        \\
    ;
    std.debug.print("{s}", .{help});
}

fn printHeader() void {
    const header =
        \\
        \\================================================================================
        \\
        \\              ABI FRAMEWORK COMPREHENSIVE BENCHMARK SUITE
        \\
        \\  Industry-standard benchmarks with statistical analysis
        \\  - Warm-up phases for CPU cache stabilization
        \\  - Outlier detection and removal
        \\  - Percentile reporting (p50, p90, p95, p99)
        \\
        \\================================================================================
        \\
    ;
    std.debug.print("{s}", .{header});
}

fn printSuiteHeader(name: []const u8) void {
    std.debug.print("\n", .{});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
    std.debug.print("  {s}\n", .{name});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = parseArgs(allocator);

    printHeader();

    var timer = std.time.Timer.start() catch {
        std.debug.print("Timer not supported on this platform\n", .{});
        return;
    };

    switch (args.suite) {
        .all => {
            printSuiteHeader("SIMD/Vector Operations");
            try simd.runSIMDBenchmarks(allocator, .{});

            printSuiteHeader("Memory Allocator Patterns");
            try memory.runMemoryBenchmarks(allocator, .{});

            printSuiteHeader("Concurrency Primitives");
            try concurrency.runConcurrencyBenchmarks(allocator, .{});

            printSuiteHeader("Database/HNSW Vector Search");
            try database.runAllBenchmarks(allocator, .standard);

            printSuiteHeader("HTTP/Network Operations");
            try network.runNetworkBenchmarks(allocator, .{});

            printSuiteHeader("Cryptography");
            try crypto.runCryptoBenchmarks(allocator, .{});

            printSuiteHeader("AI/ML Inference");
            try ai.runAllBenchmarks(allocator, .standard);
        },
        .simd => {
            printSuiteHeader("SIMD/Vector Operations");
            try simd.runSIMDBenchmarks(allocator, .{});
        },
        .memory => {
            printSuiteHeader("Memory Allocator Patterns");
            try memory.runMemoryBenchmarks(allocator, .{});
        },
        .concurrency => {
            printSuiteHeader("Concurrency Primitives");
            try concurrency.runConcurrencyBenchmarks(allocator, .{});
        },
        .database => {
            printSuiteHeader("Database/HNSW Vector Search");
            try database.runAllBenchmarks(allocator, .standard);
        },
        .network => {
            printSuiteHeader("HTTP/Network Operations");
            try network.runNetworkBenchmarks(allocator, .{});
        },
        .crypto => {
            printSuiteHeader("Cryptography");
            try crypto.runCryptoBenchmarks(allocator, .{});
        },
        .ai => {
            printSuiteHeader("AI/ML Inference");
            try ai.runAllBenchmarks(allocator, .standard);
        },
        .quick => {
            printSuiteHeader("Quick Benchmark Suite (CI Mode)");
            std.debug.print("\nRunning reduced benchmark set for CI...\n\n", .{});

            // SIMD - quick config
            try simd.runSIMDBenchmarks(allocator, .{
                .dimensions = &.{ 128, 512 },
                .include_scalar_comparison = false,
            });

            // Memory - quick config
            try memory.runMemoryBenchmarks(allocator, .{
                .small_sizes = &.{64},
                .medium_sizes = &.{ 256, 1024 },
                .alloc_count = 100,
                .thread_counts = &.{2},
                .test_fragmentation = false,
            });

            // Database - quick config
            try database.runAllBenchmarks(allocator, .quick);

            // Crypto - quick config
            try crypto.runCryptoBenchmarks(allocator, .{
                .data_sizes = &.{1024},
                .pbkdf_iterations = &.{1000},
            });

            // AI - quick config
            try ai.runAllBenchmarks(allocator, .quick);
        },
    }

    const elapsed_ns = timer.read();
    const duration_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print(" BENCHMARK COMPLETE\n", .{});
    std.debug.print(" Total runtime: {d:.2}s\n", .{duration_sec});
    std.debug.print("================================================================================\n", .{});
}

test "benchmark imports" {
    _ = framework;
    _ = simd;
    _ = memory;
    _ = concurrency;
    _ = database;
    _ = network;
    _ = crypto;
    _ = ai;
    _ = core;
}
