//! CLI command: abi status
//!
//! Shows framework health and component status with color-coded output.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "status",
    .description = "Show framework health and component status",
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    var fw = abi.initDefault(allocator) catch |err| {
        utils.output.printError("Framework initialization failed: {t}", .{err});
        utils.output.println("", .{});
        utils.output.println("Status: UNHEALTHY", .{});
        return;
    };
    defer fw.deinit();

    utils.output.printHeader("ABI Framework Status");

    utils.output.printKeyValue("Version", abi.version());
    var state_buf: [32]u8 = undefined;
    const state_str = std.fmt.bufPrint(&state_buf, "{t}", .{fw.state}) catch "unknown";
    utils.output.printKeyValue("State", state_str);

    // Feature status using the registry for complete coverage
    utils.output.println("", .{});
    const features = std.enums.values(abi.Feature);
    var enabled_count: usize = 0;
    for (features) |tag| {
        const enabled = fw.isEnabled(tag);
        utils.output.printStatusLineFmt("{t}", .{tag}, enabled);
        if (enabled) enabled_count += 1;
    }

    utils.output.printCountSummary(enabled_count, features.len, "features active");

    // Connector status (all 15 providers)
    utils.output.println("", .{});
    const connectors = [_]struct { name: []const u8, available: bool }{
        .{ .name = "openai", .available = abi.connectors.openai.isAvailable() },
        .{ .name = "anthropic", .available = abi.connectors.anthropic.isAvailable() },
        .{ .name = "claude", .available = abi.connectors.claude.isAvailable() },
        .{ .name = "ollama", .available = abi.connectors.ollama.isAvailable() },
        .{ .name = "ollama_passthrough", .available = abi.connectors.ollama_passthrough.isAvailable() },
        .{ .name = "huggingface", .available = abi.connectors.huggingface.isAvailable() },
        .{ .name = "mistral", .available = abi.connectors.mistral.isAvailable() },
        .{ .name = "cohere", .available = abi.connectors.cohere.isAvailable() },
        .{ .name = "gemini", .available = abi.connectors.gemini.isAvailable() },
        .{ .name = "codex", .available = abi.connectors.codex.isAvailable() },
        .{ .name = "opencode", .available = abi.connectors.opencode.isAvailable() },
        .{ .name = "lm_studio", .available = abi.connectors.lm_studio.isAvailable() },
        .{ .name = "vllm", .available = abi.connectors.vllm.isAvailable() },
        .{ .name = "mlx", .available = abi.connectors.mlx.isAvailable() },
        .{ .name = "llama_cpp", .available = abi.connectors.llama_cpp.isAvailable() },
    };
    var conn_count: usize = 0;
    for (connectors) |c| {
        if (c.available) conn_count += 1;
    }
    utils.output.printCountSummary(conn_count, connectors.len, "connectors configured");

    // Overall health
    const healthy = fw.state == .running or fw.state == .initializing;
    utils.output.println("", .{});
    utils.output.println("  Status: {s}{s}{s}", .{
        if (healthy) utils.output.Color.green() else utils.output.Color.red(),
        if (healthy) "HEALTHY" else "UNHEALTHY",
        utils.output.Color.reset(),
    });
    utils.output.println("", .{});
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
