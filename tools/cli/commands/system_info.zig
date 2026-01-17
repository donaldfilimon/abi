//! System information command.
//!
//! Provides detailed information about the host system, platform capabilities,
//! and the current ABI framework configuration.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const gpu = @import("gpu.zig");
const network = @import("network.zig");

/// Run the system-info command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // Load framework instance for feature matrix (required for runtime checks)
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const platform = abi.platform.platform;
    const info = platform.PlatformInfo.detect();

    utils.output.printHeader("System Information");

    // Core Platform Info
    utils.output.printKeyValue("OS", try std.fmt.allocPrint(allocator, "{t}", .{info.os}));
    utils.output.printKeyValue("Architecture", try std.fmt.allocPrint(allocator, "{t}", .{info.arch}));
    utils.output.printKeyValue("CPU Threads", try std.fmt.allocPrint(allocator, "{d}", .{info.max_threads}));
    utils.output.printKeyValue("ABI Version", abi.version());

    // Hardware Capabilities
    utils.output.printKeyValue("SIMD Support", if (abi.hasSimdSupport()) "available" else "unavailable");

    // GPU and Network Summaries (using modernized summary functions)
    try gpu.printSummary(allocator);
    network.printSummary();

    // Feature Matrix
    utils.output.printHeader("Feature Matrix");
    const features = std.enums.values(abi.Feature);
    for (features) |tag| {
        const enabled = framework.isFeatureEnabled(tag);
        utils.output.printKeyValue(@tagName(tag), utils.output.boolLabel(enabled));
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi system-info", "")
        .description("Display detailed information about the host environment and framework features.")
        .section("Options")
        .option(utils.help.common_options.help)
        .newline()
        .section("Details")
        .text("  This command performs hardware detection to report CPU capabilities, GPU\n")
        .text("  availability, and network cluster status. It also shows which compile-time\n")
        .text("  features are currently enabled in your build.\n");

    builder.print();
}
