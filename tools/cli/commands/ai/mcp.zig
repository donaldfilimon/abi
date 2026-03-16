//! MCP Server CLI command.
//!
//! Starts a Model Context Protocol (JSON-RPC 2.0 over stdio) server that
//! exposes the combined WDBX and ZLS tool surface to MCP-compatible AI clients.
//!
//! Usage:
//!   abi mcp serve        Start the MCP server (reads stdin, writes stdout)
//!   abi mcp tools        List available MCP tools
//!   abi mcp help         Show help

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = @import("../../utils/io_backend.zig");
const mcp = abi.services.mcp;

// Wrapper functions for comptime children dispatch
fn wrapServe(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runServeSubcommand(allocator, &parser);
}
fn wrapTools(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runToolsSubcommand(allocator, &parser);
}

pub const meta: command_mod.Meta = .{
    .name = "mcp",
    .description = "MCP server for combined WDBX + ZLS tools (serve, tools)",
    .kind = .group,
    .subcommands = &.{ "serve", "tools", "help" },
    .children = &.{
        .{ .name = "serve", .description = "Start MCP server (JSON-RPC over stdio)", .handler = wrapServe },
        .{ .name = "tools", .description = "List available MCP tools", .handler = wrapTools },
    },
};

const mcp_subcommands = [_][]const u8{ "serve", "tools", "help" };

/// Run the mcp command with the provided arguments.
/// Only reached when no child matches (help / unknown).
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        printHelp(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown mcp command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &mcp_subcommands)) |suggestion| {
        utils.output.println("Did you mean: {s}", .{suggestion});
    }
}

fn runServeSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    var used_compat_flag = false;
    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (parser.consumeFlag(&[_][]const u8{"--zls"})) {
            used_compat_flag = true;
            continue;
        }
        const arg = parser.next().?;
        utils.output.printError("Unexpected argument for 'serve': {s}", .{arg});
        utils.output.printInfo("Usage: abi mcp serve [--zls]", .{});
        return;
    }
    try runServe(allocator, used_compat_flag);
}

fn runToolsSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    var used_compat_flag = false;
    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (parser.consumeFlag(&[_][]const u8{"--zls"})) {
            used_compat_flag = true;
            continue;
        }
        const arg = parser.next().?;
        utils.output.printError("Unexpected argument for 'tools': {s}", .{arg});
        utils.output.printInfo("Usage: abi mcp tools [--zls]", .{});
        return;
    }
    try runTools(allocator, used_compat_flag);
}

fn runServe(allocator: std.mem.Allocator, used_compat_flag: bool) !void {
    // Write startup message to stderr (stdout is reserved for JSON-RPC)
    std.log.info("ABI MCP Server v{s} starting (database + ZLS)", .{abi.version()});
    if (used_compat_flag) {
        std.log.warn("`abi mcp serve --zls` is deprecated; the default server already includes ZLS tools.", .{});
    }

    var server = try mcp.createCombinedServer(allocator, abi.version());
    defer server.deinit();

    // Initialize I/O backend for stdio access
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    // Run the server loop (blocks until stdin closes / EOF)
    try server.run(io_backend.io());
}

fn runTools(allocator: std.mem.Allocator, used_compat_flag: bool) !void {
    if (used_compat_flag) {
        utils.output.printWarning("`abi mcp tools --zls` is deprecated; the default listing already includes ZLS tools.", .{});
    }

    var server = try mcp.createCombinedServer(allocator, abi.version());
    defer server.deinit();

    utils.output.printHeader("MCP Tools (database + ZLS)");

    for (server.tools.items) |tool| {
        utils.output.println("", .{});
        utils.output.println("  {s}", .{tool.def.name});
        utils.output.println("    {s}", .{tool.def.description});
    }

    utils.output.println("", .{});
    utils.output.println("Total: {d} tools available", .{server.tools.items.len});
    utils.output.println("", .{});
    utils.output.println("Usage with Claude Desktop:", .{});
    utils.output.println("  Add to claude_desktop_config.json:", .{});
    utils.output.println("  {{\"mcpServers\":{{\"abi\":{{\"command\":\"abi\",\"args\":[\"mcp\",\"serve\"]}}}}}}", .{});
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi mcp <subcommand>", "")
        .description("Model Context Protocol server for combined database and ZLS tools.")
        .section("Subcommands")
        .subcommand(.{ .name = "serve", .description = "Start MCP server (JSON-RPC over stdio)" })
        .subcommand(.{ .name = "tools", .description = "List available MCP tools" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .option(.{ .long = "--zls", .description = "Deprecated compatibility alias; default server already includes ZLS tools" })
        .newline()
        .section("MCP Tools Exposed")
        .text("  db_*         Database tools\n")
        .text("  zls_*        ZLS LSP tools (hover, completion, definition, ...)\n")
        .newline()
        .section("Examples")
        .example("abi mcp serve", "Start the combined MCP server")
        .example("abi mcp tools", "List database and ZLS tools")
        .example("abi mcp serve --zls", "Deprecated alias for the combined MCP server");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
