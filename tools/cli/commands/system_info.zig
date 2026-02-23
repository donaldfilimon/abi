//! System information command.
//!
//! Provides detailed information about the host system, platform capabilities,
//! and the current ABI framework configuration.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const gpu = @import("gpu.zig");
const network = @import("network.zig");

pub const meta: command_mod.Meta = .{
    .name = "system-info",
    .description = "Show system and framework status",
    .aliases = &.{ "info", "sysinfo" },
};

/// Run the system-info command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // Load framework instance for feature matrix (required for runtime checks)
    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    const info = abi.platform.getPlatformInfo();

    utils.output.printHeader("System Information");

    // Core Platform Info
    utils.output.printKeyValueFmt("OS", "{t}", .{info.os});
    utils.output.printKeyValueFmt("Architecture", "{t}", .{info.arch});
    utils.output.printKeyValueFmt("CPU Threads", "{d}", .{info.max_threads});

    utils.output.printKeyValue("ABI Version", abi.version());

    // Hardware Capabilities
    utils.output.printKeyValue("SIMD Support", if (abi.simd.hasSimdSupport()) "available" else "unavailable");

    // GPU and Network Summaries (using modernized summary functions)
    try gpu.printSummary(allocator);
    network.printSummary();

    // AI Connector Availability (all 15 providers)
    utils.output.printHeader("AI Connectors");
    const connector_status = [_]struct { name: []const u8, available: bool }{
        .{ .name = "OpenAI", .available = abi.connectors.openai.isAvailable() },
        .{ .name = "Anthropic", .available = abi.connectors.anthropic.isAvailable() },
        .{ .name = "Claude", .available = abi.connectors.claude.isAvailable() },
        .{ .name = "Ollama", .available = abi.connectors.ollama.isAvailable() },
        .{ .name = "Ollama Passthrough", .available = abi.connectors.ollama_passthrough.isAvailable() },
        .{ .name = "HuggingFace", .available = abi.connectors.huggingface.isAvailable() },
        .{ .name = "Mistral", .available = abi.connectors.mistral.isAvailable() },
        .{ .name = "Cohere", .available = abi.connectors.cohere.isAvailable() },
        .{ .name = "Gemini", .available = abi.connectors.gemini.isAvailable() },
        .{ .name = "Codex", .available = abi.connectors.codex.isAvailable() },
        .{ .name = "OpenCode", .available = abi.connectors.opencode.isAvailable() },
        .{ .name = "LM Studio", .available = abi.connectors.lm_studio.isAvailable() },
        .{ .name = "vLLM", .available = abi.connectors.vllm.isAvailable() },
        .{ .name = "MLX", .available = abi.connectors.mlx.isAvailable() },
        .{ .name = "llama.cpp", .available = abi.connectors.llama_cpp.isAvailable() },
    };
    for (connector_status) |conn| {
        utils.output.printKeyValue(conn.name, if (conn.available) "configured" else "not configured");
    }

    // Feature Matrix
    utils.output.printHeader("Feature Matrix");
    const features = std.enums.values(abi.Feature);
    for (features) |tag| {
        const enabled = framework.isEnabled(tag);
        var tag_buf: [32]u8 = undefined;
        const tag_str = std.fmt.bufPrint(&tag_buf, "{t}", .{tag}) catch "unknown";
        utils.output.printKeyValue(tag_str, utils.output.boolLabel(enabled));
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
