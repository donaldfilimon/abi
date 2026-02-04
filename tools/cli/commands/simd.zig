//! SIMD performance demonstration command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the SIMD performance demonstration.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // Check if SIMD is available
    const has_simd = abi.hasSimdSupport();
    utils.output.printKeyValue("SIMD Support", if (has_simd) "available" else "unavailable");

    if (!has_simd) {
        utils.output.printWarning("SIMD not available on this platform.", .{});
        return;
    }

    // Parse size option
    const size = parser.consumeInt(usize, &[_][]const u8{ "--size", "-s" }, 1000);
    const total_size = size * 3;

    // Allocate memory for vectors
    const data = try allocator.alloc(f32, total_size);
    defer allocator.free(data);

    // Initialize vectors
    const a = data[0..size];
    const b = data[size .. size * 2];
    const result = data[size * 2 .. total_size];

    // Initialize with test data
    for (0..size) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    // Warmup cache
    var timer = try std.time.Timer.start();
    _ = timer.lap();

    // Run vector addition benchmark
    abi.simd.vectorAdd(a, b, result);
    const add_time = timer.lap();

    // Run dot product benchmark
    const dot_result = abi.simd.vectorDot(a, b);
    const dot_time = timer.lap();

    // Run L2 norm benchmark
    const norm_result = abi.simd.vectorL2Norm(a);
    const norm_time = timer.lap();

    // Run cosine similarity benchmark
    const cos_result = abi.simd.cosineSimilarity(a, b);
    const cos_time = timer.lap();

    // Print results
    utils.output.printHeader("SIMD Performance Results");

    utils.output.printKeyValueFmt("Vector Add", "{d}ns", .{add_time});
    utils.output.printKeyValueFmt("Dot Product", "{d}ns (result: {d:.6})", .{ dot_time, dot_result });
    utils.output.printKeyValueFmt("L2 Norm", "{d}ns (result: {d:.6})", .{ norm_time, norm_result });
    utils.output.printKeyValueFmt("Cosine Similarity", "{d}ns (result: {d:.6})", .{ cos_time, cos_result });
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi simd", "[options]")
        .description("Run SIMD performance benchmarks.")
        .option(.{ .short = "-s", .long = "--size", .arg = "N", .description = "Number of elements (default: 1000)" })
        .section("Examples")
        .example("abi simd", "Run with default size")
        .example("abi simd --size 1000000", "Run with 1M elements")
        .option(utils.help.common_options.help);

    builder.print();
}
