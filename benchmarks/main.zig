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
const abi = @import("abi");
const framework = @import("system/framework.zig");
const infrastructure = @import("infrastructure/mod.zig");
const domains = @import("mod.zig").domains;

// Individual infrastructure modules
const simd = @import("infrastructure/simd.zig");
const memory = @import("infrastructure/memory.zig");
const concurrency = @import("infrastructure/concurrency.zig");
const network = @import("infrastructure/network/mod.zig");
const crypto = @import("infrastructure/crypto.zig");
const v2_modules = @import("infrastructure/v2_modules.zig");

// Consolidated domain modules
const database = @import("domain/database/mod.zig");
const ai = @import("domain/ai/mod.zig");
const gpu_bench = @import("domain/gpu/mod.zig");
const services = @import("domain/services/mod.zig");

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
    v2,
    services,
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
    if (std.mem.eql(u8, name, "v2")) return .v2;
    if (std.mem.eql(u8, name, "services")) return .services;
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
        \\                    Available: all, simd, memory, concurrency, database,
        \\                    network, crypto, ai, gpu, v2, services, quick
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
        \\  v2            v2 modules (SIMD activations, matrix, SwissMap, primitives)
        \\  services      Service modules (cache, search, gateway, messaging, storage)
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
    writer: anytype,
    results: []const framework.BenchResult,
    meta: BenchJsonMeta,
) !void {
    try writer.writeAll("{\n  \"metadata\": {\n    \"suite\": ");
    try writer.print("{}", .{std.json.fmt(meta.suite, .{})});
    try writer.writeAll(",\n    \"quick\": ");
    try writer.writeAll(if (meta.quick) "true" else "false");
    try writer.writeAll(",\n    \"duration_ns\": ");
    try writer.print("{d}", .{meta.duration_ns});
    try writer.writeAll(",\n    \"duration_sec\": ");
    try writer.print("{d:.4}", .{meta.duration_sec});
    try writer.writeAll(",\n    \"benchmarks\": ");
    try writer.print("{d}", .{results.len});
    try writer.writeAll("\n  },\n  \"benchmarks\": [\n");

    for (results, 0..) |result, idx| {
        if (idx > 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\"name\":");
        try writer.print("{}", .{std.json.fmt(result.config.name, .{})});
        try writer.writeAll(",\"category\":");
        try writer.print("{}", .{std.json.fmt(result.config.category, .{})});
        try writer.writeAll(",\"iterations\":");
        try writer.print("{d}", .{result.stats.iterations});
        try writer.writeAll(",\"mean_ns\":");
        try writer.print("{d:.2}", .{result.stats.mean_ns});
        try writer.writeAll(",\"median_ns\":");
        try writer.print("{d:.2}", .{result.stats.median_ns});
        try writer.writeAll(",\"std_dev_ns\":");
        try writer.print("{d:.2}", .{result.stats.std_dev_ns});
        try writer.writeAll(",\"min_ns\":");
        try writer.print("{d}", .{result.stats.min_ns});
        try writer.writeAll(",\"max_ns\":");
        try writer.print("{d}", .{result.stats.max_ns});
        try writer.writeAll(",\"p50_ns\":");
        try writer.print("{d}", .{result.stats.p50_ns});
        try writer.writeAll(",\"p90_ns\":");
        try writer.print("{d}", .{result.stats.p90_ns});
        try writer.writeAll(",\"p95_ns\":");
        try writer.print("{d}", .{result.stats.p95_ns});
        try writer.writeAll(",\"p99_ns\":");
        try writer.print("{d}", .{result.stats.p99_ns});
        try writer.writeAll(",\"ops_per_sec\":");
        try writer.print("{d:.2}", .{result.stats.opsPerSecond()});
        try writer.writeAll(",\"bytes_per_op\":");
        try writer.print("{d}", .{result.config.bytes_per_op});
        try writer.writeAll(",\"throughput_mb_s\":");
        try writer.print("{d:.2}", .{result.stats.throughputMBps(result.config.bytes_per_op)});
        try writer.writeAll(",\"memory_allocated\":");
        try writer.print("{d}", .{result.memory_allocated});
        try writer.writeAll(",\"memory_freed\":");
        try writer.print("{d}", .{result.memory_freed});
        try writer.writeAll("}");
    }

    try writer.writeAll("\n  ]\n}\n");
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

    var timer = abi.shared.time.Timer.start() catch {
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

            printSuiteHeader("v2 Module Benchmarks");
            try v2_modules.runV2Benchmarks(allocator, .{});

            printSuiteHeader("Service Module Benchmarks (Cache, Search, Gateway, Messaging, Storage)");
            try services.runAllBenchmarks(allocator);
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
        .v2 => {
            printSuiteHeader("v2 Module Benchmarks");
            try v2_modules.runV2Benchmarks(allocator, .{});
        },
        .services => {
            printSuiteHeader("Service Module Benchmarks (Cache, Search, Gateway, Messaging, Storage)");
            try services.runAllBenchmarks(allocator);
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

            // v2 modules - quick config
            try v2_modules.runV2Benchmarks(allocator, .{
                .vector_sizes = &.{ 64, 256 },
                .matrix_sizes = &.{ 32, 64 },
                .map_sizes = &.{100},
            });

            // Service modules
            try services.runAllBenchmarks(allocator);
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
            var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.environ });
            defer io_backend.deinit();
            const io = io_backend.io();

            if (args.json) {
                var stdout_buffer: [4096]u8 = undefined;
                var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
                const stdout = &stdout_writer.interface;
                try writeJsonReport(stdout, collector.results.items, meta);
                try stdout.flush();
            }

            if (args.output_json) |path| {
                var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
                defer file.close(io);
                var buffer: [4096]u8 = undefined;
                var writer = file.writer(io, &buffer);
                const out = &writer.interface;
                try writeJsonReport(out, collector.results.items, meta);
                try out.flush();
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
    _ = v2_modules;
    _ = services;
    _ = core;
}
