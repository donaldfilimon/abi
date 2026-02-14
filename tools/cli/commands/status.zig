//! CLI command: abi status
//!
//! Shows framework health and component status with color-coded output.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    var fw = abi.initDefault(allocator) catch |err| {
        utils.output.printError("Framework initialization failed: {t}", .{err});
        std.debug.print("\nStatus: UNHEALTHY\n", .{});
        return;
    };
    defer fw.deinit();

    utils.output.printHeader("ABI Framework Status");

    utils.output.printKeyValue("Version", abi.version());
    utils.output.printKeyValue("State", @tagName(fw.state));

    // Feature status using the registry for complete coverage
    std.debug.print("\n", .{});
    const features = std.enums.values(abi.Feature);
    var enabled_count: usize = 0;
    for (features) |tag| {
        const enabled = fw.isEnabled(tag);
        const icon = if (enabled) utils.output.Color.get(utils.output.Color.green) else utils.output.Color.get(utils.output.Color.dim);
        const marker = if (enabled) "[ok]" else "[--]";
        const reset = utils.output.Color.get(utils.output.Color.reset);
        std.debug.print("  {s}{s}{s} {s}\n", .{ icon, marker, reset, @tagName(tag) });
        if (enabled) enabled_count += 1;
    }

    std.debug.print("\n  {d}/{d} features active\n", .{ enabled_count, features.len });

    // Connector status
    std.debug.print("\n", .{});
    const connectors = [_]struct { name: []const u8, available: bool }{
        .{ .name = "openai", .available = abi.connectors.openai.isAvailable() },
        .{ .name = "anthropic", .available = abi.connectors.anthropic.isAvailable() },
        .{ .name = "ollama", .available = abi.connectors.ollama.isAvailable() },
        .{ .name = "huggingface", .available = abi.connectors.huggingface.isAvailable() },
        .{ .name = "mistral", .available = abi.connectors.mistral.isAvailable() },
        .{ .name = "cohere", .available = abi.connectors.cohere.isAvailable() },
    };
    var conn_count: usize = 0;
    for (connectors) |c| {
        if (c.available) conn_count += 1;
    }
    std.debug.print("  {d}/{d} connectors configured\n", .{ conn_count, connectors.len });

    // Overall health
    const healthy = fw.state == .running or fw.state == .initializing;
    std.debug.print("\n  Status: {s}{s}{s}\n\n", .{
        if (healthy) utils.output.Color.get(utils.output.Color.green) else utils.output.Color.get(utils.output.Color.red),
        if (healthy) "HEALTHY" else "UNHEALTHY",
        utils.output.Color.get(utils.output.Color.reset),
    });
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi status", "")
        .description("Show framework health, feature status, and connector availability.")
        .section("Options")
        .option(utils.help.common_options.help);

    builder.print();
}
