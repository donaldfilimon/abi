//! ACP Server CLI command.
//!
//! Provides Agent Communication Protocol functionality for agent-to-agent
//! communication. Exposes an agent card and task management endpoints.
//!
//! Usage:
//!   abi acp card         Print agent card JSON to stdout
//!   abi acp serve        Start ACP HTTP server (default 127.0.0.1:8080)
//!   abi acp help         Show help

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const acp = abi.acp;

// Wrapper functions for comptime children dispatch
fn wrapCard(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runCardSubcommand(allocator, &parser);
}
fn wrapServe(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runServeSubcommand(allocator, &parser);
}

pub const meta: command_mod.Meta = .{
    .name = "acp",
    .description = "Agent Communication Protocol (card, serve)",
    .subcommands = &.{ "card", "serve", "help" },
    .children = &.{
        .{ .name = "card", .description = "Print agent card JSON to stdout", .handler = wrapCard },
        .{ .name = "serve", .description = "Start ACP HTTP server (default 127.0.0.1:8080)", .handler = wrapServe },
    },
};

const acp_subcommands = [_][]const u8{ "card", "serve", "help" };

/// Run the acp command with the provided arguments.
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
    utils.output.printError("Unknown acp command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &acp_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
}

fn runCardSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    if (parser.hasMore()) {
        const arg = parser.next().?;
        utils.output.printError("Unexpected argument for 'card': {s}", .{arg});
        utils.output.printInfo("Usage: abi acp card", .{});
        return;
    }
    try runCard(allocator);
}

fn runServeSubcommand(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    try runServe(allocator, parser);
}

fn runCard(allocator: std.mem.Allocator) !void {
    const card = acp.AgentCard{
        .name = "abi-agent",
        .description = "ABI Framework AI Agent with vector database, training, and inference",
        .version = abi.version(),
        .url = "http://localhost:8080",
        .capabilities = .{
            .streaming = true,
            .pushNotifications = false,
        },
    };

    const json = try card.toJson(allocator);
    defer allocator.free(json);

    std.debug.print("{s}\n", .{json});
}

fn runServe(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    var address: []const u8 = "127.0.0.1:8080";
    var address_allocated = false;
    while (parser.hasMore()) {
        const arg = parser.next().?;
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--address", "-a" })) {
            const value = parser.next() orelse {
                utils.output.printError("Missing value for --address", .{});
                utils.output.printInfo("Usage: abi acp serve [--address <addr>] [--port <port>]", .{});
                return;
            };
            if (address_allocated) allocator.free(address);
            address = try allocator.dupe(u8, value);
            address_allocated = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--port", "-p" })) {
            const port_text = parser.next() orelse {
                utils.output.printError("Missing value for --port", .{});
                utils.output.printInfo("Usage: abi acp serve [--address <addr>] [--port <port>]", .{});
                return;
            };
            const port = std.fmt.parseInt(u16, port_text, 10) catch {
                utils.output.printError("Invalid port: {s}", .{port_text});
                utils.output.printInfo("Port must be an integer between 1 and 65535", .{});
                return;
            };
            if (port == 0) {
                utils.output.printError("Invalid port: 0", .{});
                utils.output.printInfo("Port must be an integer between 1 and 65535", .{});
                return;
            }
            if (address_allocated) allocator.free(address);
            address = try std.fmt.allocPrint(allocator, "127.0.0.1:{d}", .{port});
            address_allocated = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "help", "--help", "-h" })) {
            printHelp(allocator);
            return;
        } else {
            utils.output.printError("Unknown option for 'serve': {s}", .{arg});
            utils.output.printInfo("Usage: abi acp serve [--address <addr>] [--port <port>]", .{});
            return;
        }
    }
    defer if (address_allocated) allocator.free(address);

    const card_url = try std.fmt.allocPrint(allocator, "http://{s}", .{address});
    defer allocator.free(card_url);

    const card = acp.AgentCard{
        .name = "abi-agent",
        .description = "ABI Framework AI Agent with vector database, training, and inference",
        .version = abi.version(),
        .url = card_url,
        .capabilities = .{
            .streaming = true,
            .pushNotifications = false,
        },
    };

    std.debug.print("ACP HTTP server starting on http://{s}\n", .{address});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    try acp.serveHttp(allocator, io_backend.io(), address, card);
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi acp <subcommand>", "")
        .description("Agent Communication Protocol for agent-to-agent communication.")
        .section("Subcommands")
        .subcommand(.{ .name = "card", .description = "Print agent card JSON to stdout" })
        .subcommand(.{ .name = "serve", .description = "Start ACP HTTP server (default 127.0.0.1:8080)" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .text("  --address, -a <addr>   Bind address (e.g. 0.0.0.0:8080)\n")
        .text("  --port, -p <port>       Bind port (implies 127.0.0.1:port)\n")
        .newline()
        .section("Agent Skills")
        .text("  db_query      Vector similarity search\n")
        .text("  db_insert     Insert vectors with metadata\n")
        .text("  agent_chat    Conversational interaction\n")
        .newline()
        .section("Examples")
        .example("abi acp card", "Print agent card as JSON")
        .example("abi acp card | jq .", "Pretty-print agent card")
        .example("abi acp serve", "Start ACP HTTP server on 127.0.0.1:8080")
        .example("abi acp serve -p 9090", "Start server on port 9090");

    builder.print();
}
