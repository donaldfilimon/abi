//! MCP Server CLI command.
//!
//! Starts a Model Context Protocol (JSON-RPC 2.0 over stdio) server that
//! exposes the WDBX vector database to MCP-compatible AI clients.
//!
//! Usage:
//!   abi mcp serve        Start the MCP server (reads stdin, writes stdout)
//!   abi mcp tools        List available MCP tools
//!   abi mcp help         Show help

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = @import("../utils/io_backend.zig");
const subcommand = @import("../utils/subcommand.zig");
const mcp = abi.mcp;

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp(allocator);
        return;
    }

    var parser = utils.args.ArgParser.init(allocator, args);
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    const commands = [_]subcommand.Command{
        .{ .names = &.{"serve"}, .run = runServeSubcommand },
        .{ .names = &.{"tools"}, .run = runToolsSubcommand },
    };

    try subcommand.runSubcommand(
        allocator,
        &parser,
        &commands,
        null,
        printHelp,
        onUnknownCommand,
    );
}

fn runServeSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    if (parser.hasMore()) {
        const arg = parser.next().?;
        utils.output.printError("Unexpected argument for 'serve': {s}", .{arg});
        utils.output.printInfo("Usage: abi mcp serve", .{});
        return;
    }
    try runServe(allocator);
}

fn runToolsSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    if (parser.hasMore()) {
        const arg = parser.next().?;
        utils.output.printError("Unexpected argument for 'tools': {s}", .{arg});
        utils.output.printInfo("Usage: abi mcp tools", .{});
        return;
    }
    try runTools(allocator);
}

fn onUnknownCommand(command: []const u8) void {
    utils.output.printError("Unknown subcommand: {s}", .{command});
}

fn runServe(allocator: std.mem.Allocator) !void {
    // Write startup message to stderr (stdout is reserved for JSON-RPC)
    std.log.info("ABI MCP Server v{s} starting (WDBX database)", .{abi.version()});

    var server = try mcp.createWdbxServer(allocator, abi.version());
    defer server.deinit();

    // Initialize I/O backend for stdio access
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    // Run the server loop (blocks until stdin closes / EOF)
    try server.run(io_backend.io());
}

fn runTools(allocator: std.mem.Allocator) !void {
    var server = try mcp.createWdbxServer(allocator, abi.version());
    defer server.deinit();

    utils.output.printHeader("MCP Tools");

    for (server.tools.items) |tool| {
        std.debug.print("\n  {s}\n", .{tool.def.name});
        std.debug.print("    {s}\n", .{tool.def.description});
    }

    std.debug.print("\nTotal: {d} tools available\n", .{server.tools.items.len});
    std.debug.print("\nUsage with Claude Desktop:\n", .{});
    std.debug.print("  Add to claude_desktop_config.json:\n", .{});
    std.debug.print("  {{\"mcpServers\":{{\"abi-wdbx\":{{\"command\":\"abi\",\"args\":[\"mcp\",\"serve\"]}}}}}}\n", .{});
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi mcp <subcommand>", "")
        .description("Model Context Protocol server for WDBX vector database.")
        .section("Subcommands")
        .subcommand(.{ .name = "serve", .description = "Start MCP server (JSON-RPC over stdio)" })
        .subcommand(.{ .name = "tools", .description = "List available MCP tools" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .newline()
        .section("MCP Tools Exposed")
        .text("  db_query     Vector similarity search\n")
        .text("  db_insert    Insert vectors with metadata\n")
        .text("  db_stats     Database statistics\n")
        .text("  db_list      List stored vectors\n")
        .text("  db_delete    Delete vector by ID\n")
        .newline()
        .section("Examples")
        .example("abi mcp serve", "Start MCP server")
        .example("abi mcp tools", "List available tools");

    builder.print();
}
