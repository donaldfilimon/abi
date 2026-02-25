//! Benchmark command for running performance benchmarks.
//!
//! Commands:
//! - bench all - Run all benchmark suites
//! - bench <suite> - Run specific suite (simd, memory, concurrency, database, network, crypto, ai)
//! - bench quick - Run quick benchmarks for CI
//! - bench micro <operation> - Run quick micro-benchmark (hash, alloc, parse)
//! - bench list - List available benchmark suites
//!
//! Options:
//! - --json - Output results in JSON format
//! - --output <file> - Write results to file

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const suites = @import("suites.zig");
const micro = @import("micro.zig");

pub const meta: command_mod.Meta = .{
    .name = "bench",
    .description = "Run performance benchmarks (all, simd, memory, ai, quick)",
    .aliases = &.{"run"},
    .subcommands = &.{ "all", "simd", "memory", "ai", "quick", "compare-training", "list", "micro" },
};

/// Benchmark suite selection
pub const BenchmarkSuite = enum {
    all,
    simd,
    memory,
    concurrency,
    database,
    network,
    crypto,
    ai,
    streaming,
    quick,
    compare_training,

    pub fn toString(self: BenchmarkSuite) []const u8 {
        return switch (self) {
            .all => "all",
            .simd => "simd",
            .memory => "memory",
            .concurrency => "concurrency",
            .database => "database",
            .network => "network",
            .crypto => "crypto",
            .ai => "ai",
            .streaming => "streaming",
            .quick => "quick",
            .compare_training => "compare-training",
        };
    }
};

/// Micro-benchmark operation
pub const MicroOp = enum {
    hash,
    alloc,
    parse,
    noop,

    pub fn toString(self: MicroOp) []const u8 {
        return switch (self) {
            .hash => "hash",
            .alloc => "alloc",
            .parse => "parse",
            .noop => "noop",
        };
    }
};

/// Benchmark configuration
pub const BenchConfig = struct {
    suite: BenchmarkSuite = .all,
    output_json: bool = false,
    output_file: ?[]const u8 = null,
    micro_op: ?MicroOp = null,
    iterations: u32 = 1000,
    warmup: u32 = 100,
};

/// Benchmark result structure
pub const BenchResult = struct {
    name: []const u8,
    category: []const u8,
    ops_per_sec: f64,
    mean_ns: f64,
    p99_ns: u64,
    iterations: u64,
    name_allocated: bool = false, // Track if name was dynamically allocated

    /// Free allocated name if it was dynamically allocated
    pub fn deinit(self: *BenchResult, allocator: std.mem.Allocator) void {
        if (self.name_allocated) {
            allocator.free(self.name);
        }
    }
};

/// Errors that can occur during benchmark execution.
/// Used for benchmark function signatures instead of anyerror.
pub const BenchmarkError = std.mem.Allocator.Error || error{
    TimerUnavailable,
    BenchmarkFailed,
};

/// Benchmark function type for suite runners.
pub const BenchmarkFn = *const fn (std.mem.Allocator, *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void;

/// Run the benchmark command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);

    if (!parser.hasMore() or parser.wantsHelp()) {
        printHelp();
        return;
    }

    var config = BenchConfig{};
    const command = parser.next().?;

    // Handle explicit subcommands
    if (std.mem.eql(u8, command, "list")) {
        printAvailableSuites();
        return;
    }

    if (std.mem.eql(u8, command, "micro")) {
        return micro.runMicroBenchmark(allocator, parser.remaining(), &config);
    }

    // Parse suite name
    if (std.mem.eql(u8, command, "compare-training")) {
        config.suite = .compare_training;
    } else if (std.meta.stringToEnum(BenchmarkSuite, command)) |suite| {
        config.suite = suite;
    } else {
        utils.output.printError("Unknown benchmark suite: {s}", .{command});
        utils.output.printInfo("Use 'abi bench list' to see available suites.", .{});
        return;
    }

    // Parse remaining options
    while (parser.hasMore()) {
        if (parser.consumeFlag(&[_][]const u8{"--json"})) {
            config.output_json = true;
        } else if (parser.consumeOption(&[_][]const u8{ "--output", "-o" })) |path| {
            config.output_file = path;
        } else if (parser.consumeOption(&[_][]const u8{"--iterations"})) |val| {
            config.iterations = std.fmt.parseInt(u32, val, 10) catch 1000;
        } else {
            _ = parser.next();
        }
    }

    try suites.runBenchmarkSuite(allocator, config);
}

pub fn printAvailableSuites() void {
    utils.output.println("Available benchmark suites:\n", .{});
    utils.output.println("  all          Run all benchmark suites", .{});
    utils.output.println("  simd         SIMD/Vector operations (dot product, matmul)", .{});
    utils.output.println("  memory       Memory allocator patterns (arena, pool)", .{});
    utils.output.println("  concurrency  Concurrency primitives (atomics, locks)", .{});
    utils.output.println("  database     Database/HNSW vector search", .{});
    utils.output.println("  network      HTTP/Network operations", .{});
    utils.output.println("  crypto       Cryptographic operations", .{});
    utils.output.println("  ai           AI/ML inference (GEMM, attention)", .{});
    utils.output.println("  streaming           Streaming inference (TTFT, ITL, throughput)", .{});
    utils.output.println("  quick               Quick benchmarks for CI", .{});
    utils.output.println("  compare-training    Compare training optimizers (AdamW vs Adam vs SGD)", .{});
    utils.output.println("\nMicro-benchmarks:", .{});
    utils.output.println("  abi bench micro hash   - Simple hash computation", .{});
    utils.output.println("  abi bench micro alloc  - Memory allocation pattern", .{});
    utils.output.println("  abi bench micro parse  - Simple parsing operation", .{});
    utils.output.println("  abi bench micro noop   - Empty operation (baseline)", .{});
}

pub fn printHelp() void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi bench", "<suite> [options]")
        .description("Run performance benchmarks.")
        .section("Suites")
        .subcommand(.{ .name = "all", .description = "Run all benchmark suites" })
        .subcommand(.{ .name = "simd", .description = "SIMD/Vector operations" })
        .subcommand(.{ .name = "memory", .description = "Memory allocator patterns" })
        .subcommand(.{ .name = "concurrency", .description = "Concurrency primitives" })
        .subcommand(.{ .name = "database", .description = "Database/HNSW vector search" })
        .subcommand(.{ .name = "network", .description = "HTTP/Network operations" })
        .subcommand(.{ .name = "crypto", .description = "Cryptographic operations" })
        .subcommand(.{ .name = "ai", .description = "AI/ML inference" })
        .subcommand(.{ .name = "streaming", .description = "Streaming inference (TTFT, ITL)" })
        .subcommand(.{ .name = "quick", .description = "Quick benchmarks for CI" })
        .subcommand(.{ .name = "compare-training", .description = "Compare training optimizers" })
        .subcommand(.{ .name = "list", .description = "List available suites" })
        .subcommand(.{ .name = "micro <op>", .description = "Run micro-benchmark (hash, alloc, parse, noop)" })
        .newline()
        .section("Options")
        .option(.{ .long = "--json", .description = "Output results in JSON format" })
        .option(.{ .short = "-o", .long = "--output", .arg = "file", .description = "Write results to file" })
        .option(.{ .long = "--iterations", .arg = "n", .description = "Number of iterations for micro-benchmarks" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi bench all", "Run all benchmarks")
        .example("abi bench simd --json", "SIMD benchmarks with JSON output")
        .example("abi bench quick", "Quick CI benchmarks")
        .example("abi bench micro hash", "Run hash micro-benchmark")
        .example("abi bench ai --output results.json", "");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
