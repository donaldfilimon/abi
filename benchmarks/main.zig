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
const framework = @import("system/framework.zig");
const infrastructure = @import("infrastructure/mod.zig");
const domains = @import("mod.zig").domains;

// Individual infrastructure modules
const simd = @import("infrastructure/simd.zig");
const memory = @import("infrastructure/memory.zig");
const concurrency = @import("infrastructure/concurrency.zig");
const network = @import("infrastructure/network.zig");
const crypto = @import("infrastructure/crypto.zig");

// Consolidated domain modules
const database = @import("domain/database/mod.zig");
const ai = @import("domain/ai/mod.zig");
const gpu_bench = @import("domain/gpu/mod.zig");

// Core utilities
const core = @import("core/mod.zig");

const BenchmarkSuite = enum {
    all,
    simd,
    memory,
    concurrency,
    database,
    network,
    crypto,
    ai,
    gpu,
    quick,
};

const Args = struct {
    suite: BenchmarkSuite = .all,
    output_json: ?[]const u8 = null,
    verbose: bool = false,
    quick: bool = false,
    json: bool = false,
};

fn parseArgs(allocator: std.mem.Allocator, init_args: std.process.Args) Args {
    var args_result = Args{};

    // Get command line arguments using Zig 0.16 API
    var argv_iter = init_args.iterateAllocator(allocator) catch return args_result;
    defer argv_iter.deinit();

    // Skip program name (first arg)
    _ = argv_iter.next();

    while (argv_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            args_result.verbose = true;
        } else if (std.mem.eql(u8, arg, "--quick") or std.mem.eql(u8, arg, "-q")) {
            args_result.quick = true;
            args_result.suite = .quick;
        } else if (std.mem.eql(u8, arg, "--json")) {
            args_result.json = true;
        } else if (std.mem.startsWith(u8, arg, "--suite=")) {
            const suite_name = arg["--suite=".len..];
            args_result.suite = parseSuite(suite_name);
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            args_result.output_json = allocator.dupe(u8, arg["--output=".len..]) catch null;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional argument - treat as suite name
            args_result.suite = parseSuite(arg);
        }
    }

    return args_result;
}

fn parseSuite(name: []const u8) BenchmarkSuite {
    if (std.mem.eql(u8, name, "simd")) return .simd;
    if (std.mem.eql(u8, name, "memory")) return .memory;
    if (std.mem.eql(u8, name, "concurrency")) return .concurrency;
    if (std.mem.eql(u8, name, "database")) return .database;
    if (std.mem.eql(u8, name, "network")) return .network;
    if (std.mem.eql(u8, name, "crypto")) return .crypto;
    if (std.mem.eql(u8, name, "ai")) return .ai;
    if (std.mem.eql(u8, name, "gpu")) return .gpu;
    if (std.mem.eql(u8, name, "quick")) return .quick;
    if (std.mem.eql(u8, name, "all")) return .all;
    return .all; // Default to all if unknown
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
        \\                               database, network, crypto, ai, gpu, quick
        \\  --output=<file>   Output results to JSON file
        \\  --json            Output results as JSON to stdout
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
        \\  gpu           GPU kernel operations (matmul, vector ops, reductions, memory)
        \\  quick         Fast subset for continuous integration
        \\  all           Run all benchmark suites (default)
        \\
        \\Examples:
        \\  benchmarks                         # Run all benchmarks
        \\  benchmarks --suite=simd            # Run only SIMD benchmarks
        \\  benchmarks --suite=gpu             # Run only GPU benchmarks
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

const BenchJsonMeta = struct {
    suite: []const u8,
    quick: bool,
    duration_ns: u64,
    duration_sec: f64,
};

fn writeJsonReport(
    writer: *std.Io.Writer,
    results: []const framework.BenchResult,
    meta: BenchJsonMeta,
) !void {
    try writer.writeAll("{\n  \"metadata\": {\n    \"suite\": ");
    try std.json.Stringify.encodeJsonString(meta.suite, .{}, writer);
    try std.fmt.format(
        writer,
        ",\n    \"quick\": {s},\n    \"duration_ns\": {d},\n    \"duration_sec\": {d:.4},\n    \"benchmarks\": {d}\n  },\n  \"benchmarks\": [\n",
        .{
            if (meta.quick) "true" else "false",
            meta.duration_ns,
            meta.duration_sec,
            results.len,
        },
    );

    for (results, 0..) |result, idx| {
        if (idx > 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\"name\":");
        try std.json.Stringify.encodeJsonString(result.config.name, .{}, writer);
        try writer.writeAll(",\"category\":");
        try std.json.Stringify.encodeJsonString(result.config.category, .{}, writer);
        try std.fmt.format(
            writer,
            ",\"iterations\":{d},\"mean_ns\":{d:.2},\"median_ns\":{d:.2},\"std_dev_ns\":{d:.2},\"min_ns\":{d},\"max_ns\":{d},\"p50_ns\":{d},\"p90_ns\":{d},\"p95_ns\":{d},\"p99_ns\":{d},\"ops_per_sec\":{d:.2},\"bytes_per_op\":{d},\"throughput_mb_s\":{d:.2},\"memory_allocated\":{d},\"memory_freed\":{d}}",
            .{
                result.stats.iterations,
                result.stats.mean_ns,
                result.stats.median_ns,
                result.stats.std_dev_ns,
                result.stats.min_ns,
                result.stats.max_ns,
                result.stats.p50_ns,
                result.stats.p90_ns,
                result.stats.p95_ns,
                result.stats.p99_ns,
                result.stats.opsPerSecond(),
                result.config.bytes_per_op,
                result.stats.throughputMBps(result.config.bytes_per_op),
                result.memory_allocated,
                result.memory_freed,
            },
        );
    }

    try writer.writeAll("\n  ]\n}\n");
}

fn initThreadedIo(allocator: std.mem.Allocator, options: std.Io.Threaded.InitOptions) !std.Io.Threaded {
    const Result = @TypeOf(std.Io.Threaded.init(allocator, options));
    if (@typeInfo(Result) == .error_union) {
        return try std.Io.Threaded.init(allocator, options);
    }
    return std.Io.Threaded.init(allocator, options);
}

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = parseArgs(allocator, init.args);
    defer if (args.output_json) |path| allocator.free(path);

    var collector_storage: ?framework.BenchCollector = null;
    if (args.json or args.output_json != null) {
        collector_storage = framework.BenchCollector.init(allocator);
        framework.setGlobalCollector(&collector_storage.?);
    }
    defer if (collector_storage) |*collector| {
        framework.setGlobalCollector(null);
        collector.deinit();
    };

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

            printSuiteHeader("GPU Kernel Operations");
            try gpu_bench.runAllBenchmarks(allocator, .standard);
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
        .gpu => {
            printSuiteHeader("GPU Kernel Operations");
            try gpu_bench.runAllBenchmarks(allocator, .standard);
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

            // GPU - quick config
            try gpu_bench.runAllBenchmarks(allocator, .quick);
        },
    }

    const elapsed_ns = timer.read();
    const duration_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    if (collector_storage) |*collector| {
        const meta = BenchJsonMeta{
            .suite = @tagName(args.suite),
            .quick = args.suite == .quick,
            .duration_ns = elapsed_ns,
            .duration_sec = duration_sec,
        };

        if (args.json or args.output_json != null) {
            var io_backend = try initThreadedIo(allocator, .{
                .environ = std.process.Environ.empty,
            });
            defer io_backend.deinit();
            const io = io_backend.io();

            if (args.json) {
                var stdout_buffer: [4096]u8 = undefined;
                var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
                try writeJsonReport(&stdout_writer.interface, collector.results.items, meta);
                try stdout_writer.flush();
            }

            if (args.output_json) |path| {
                var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
                defer file.close(io);
                var file_buffer: [4096]u8 = undefined;
                var file_writer = file.writer(io, &file_buffer);
                try writeJsonReport(&file_writer.interface, collector.results.items, meta);
                try file_writer.flush();
            }
        }
    }

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
    _ = gpu_bench;
    _ = core;
}
