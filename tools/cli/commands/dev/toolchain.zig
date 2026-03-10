//! Zig bootstrap management for the CEL transition.
//!
//! Usage:
//!   abi bootstrap-zig install       Build bootstrap Zig and ZLS into .zig-bootstrap/bin
//!   abi bootstrap-zig zig           Build only bootstrap Zig
//!   abi bootstrap-zig zls           Build only ZLS using bootstrap Zig
//!   abi bootstrap-zig status        Show bootstrap Zig/ZLS status
//!   abi bootstrap-zig update        Rebuild the repo-local bootstrap Zig bridge
//!   abi bootstrap-zig path          Print the repo-local .zig-bootstrap/bin path
//!   abi bootstrap-zig bootstrap     Run the emergency bootstrap prerequisite

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");

const output = utils.output;
const ArgParser = utils.args.ArgParser;
const HelpBuilder = utils.help.HelpBuilder;
const common_options = utils.help.common_options;

const BuildTarget = enum {
    both,
    zig_only,
    zls_only,
};

pub const meta: command_mod.Meta = .{
    .name = "bootstrap-zig",
    .description = "Manage the repo-local Zig bootstrap bridge that backs ABI's CEL transition",
    .aliases = &.{"toolchain"},
    .kind = .group,
    .subcommands = &.{ "install", "zig", "zls", "status", "update", "path", "bootstrap", "help" },
    .children = &.{
        .{ .name = "install", .description = "Build bootstrap Zig and ZLS into .zig-bootstrap/bin", .handler = command_mod.parserHandler(runInstallBoth) },
        .{ .name = "zig", .description = "Build only bootstrap Zig into .zig-bootstrap/bin", .handler = command_mod.parserHandler(runInstallZig) },
        .{ .name = "zls", .description = "Build only ZLS using bootstrap Zig", .handler = command_mod.parserHandler(runInstallZls) },
        .{ .name = "status", .description = "Show bootstrap Zig/ZLS status", .handler = command_mod.parserHandler(runStatusSubcommand) },
        .{ .name = "update", .description = "Rebuild the repo-local bootstrap Zig bridge", .handler = command_mod.parserHandler(runUpdateSubcommand) },
        .{ .name = "path", .description = "Print the repo-local .zig-bootstrap/bin path", .handler = command_mod.parserHandler(runPathSubcommand) },
        .{ .name = "bootstrap", .description = "Run the emergency bootstrap prerequisite path", .handler = command_mod.parserHandler(runBootstrapSubcommand) },
    },
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp(ctx.allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(ctx.allocator);
        return;
    }
    output.printError("Unknown bootstrap-zig command: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        output.println("Did you mean: {s}", .{suggestion});
    }
}

fn runInstallBoth(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runBuildSubcommand(allocator, parser, .both, false);
}

fn runInstallZig(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runBuildSubcommand(allocator, parser, .zig_only, false);
}

fn runInstallZls(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runBuildSubcommand(allocator, parser, .zls_only, false);
}

fn runStatusSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    try expectNoTrailingArgs(parser, "status");
    try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--status" });
}

fn runUpdateSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    var target: BuildTarget = .both;
    var clean = true;

    while (parser.hasMore()) {
        if (parser.consumeFlag(&[_][]const u8{"--zig"})) {
            target = .zig_only;
            continue;
        }
        if (parser.consumeFlag(&[_][]const u8{"--zls"})) {
            target = .zls_only;
            continue;
        }
        if (parser.consumeFlag(&[_][]const u8{"--clean"})) {
            clean = true;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'update': {s}", .{arg});
        output.printInfo("Usage: abi bootstrap-zig update [--zig|--zls] [--clean]", .{});
        return;
    }

    try runBuildCommand(allocator, target, clean);
}

fn runPathSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    try expectNoTrailingArgs(parser, "path");
    output.println(".zig-bootstrap/bin", .{});
}

fn runBootstrapSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    try expectNoTrailingArgs(parser, "bootstrap");

    output.printHeader("Bootstrap Zig Emergency Bootstrap");
    output.printInfo("Compiling and running tools/scripts/emergency_bootstrap.c", .{});
    try runCommand(allocator, &.{
        "sh",
        "-c",
        \\set -e
        \\tmp="${TMPDIR:-/tmp}/abi-emergency-bootstrap.$$"
        \\cc tools/scripts/emergency_bootstrap.c -o "$tmp"
        \\"$tmp"
        \\rm -f "$tmp"
    });
}

fn runBuildSubcommand(
    allocator: std.mem.Allocator,
    parser: *ArgParser,
    target: BuildTarget,
    default_clean: bool,
) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    var clean = default_clean;
    while (parser.hasMore()) {
        if (parser.consumeFlag(&[_][]const u8{"--clean"})) {
            clean = true;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument: {s}", .{arg});
        output.printInfo("Usage: abi bootstrap-zig install|zig|zls [--clean]", .{});
        return;
    }

    try runBuildCommand(allocator, target, clean);
}

fn runBuildCommand(allocator: std.mem.Allocator, target: BuildTarget, clean: bool) !void {
    output.printHeader("Bootstrap Zig");
    output.printKeyValue("Target", switch (target) {
        .both => "zig + zls",
        .zig_only => "zig",
        .zls_only => "zls",
    });
    output.printKeyValue("Location", ".zig-bootstrap/bin");
    if (clean) output.printInfo("Clean rebuild requested", .{});

    switch (target) {
        .both => {
            if (clean) {
                try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--clean" });
            } else {
                try runCommand(allocator, &.{"./.zig-bootstrap/build.sh"});
            }
        },
        .zig_only => {
            if (clean) {
                try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--clean", "--zig-only" });
            } else {
                try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--zig-only" });
            }
        },
        .zls_only => {
            if (clean) {
                try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--clean", "--zls-only" });
            } else {
                try runCommand(allocator, &.{ "./.zig-bootstrap/build.sh", "--zls-only" });
            }
        },
    }
}

fn expectNoTrailingArgs(parser: *ArgParser, name: []const u8) !void {
    if (!parser.hasMore()) return;
    const arg = parser.next().?;
    output.printError("Unexpected argument for '{s}': {s}", .{ name, arg });
    return error.InvalidArgument;
}

fn runCommand(allocator: std.mem.Allocator, argv: []const []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var child = try std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return error.CommandFailed,
        else => return error.CommandFailed,
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi bootstrap-zig", "<command> [options]")
        .description("Manage ABI's repo-local Zig bootstrap bridge during the CEL transition.")
        .section("Commands")
        .subcommand(.{ .name = "install", .description = "Build bootstrap Zig and ZLS into .zig-bootstrap/bin" })
        .subcommand(.{ .name = "zig", .description = "Build only bootstrap Zig into .zig-bootstrap/bin" })
        .subcommand(.{ .name = "zls", .description = "Build only ZLS using bootstrap Zig" })
        .subcommand(.{ .name = "status", .description = "Show bootstrap Zig/ZLS status" })
        .subcommand(.{ .name = "update", .description = "Rebuild the repo-local bootstrap Zig bridge" })
        .subcommand(.{ .name = "path", .description = "Print the repo-local .zig-bootstrap/bin path" })
        .subcommand(.{ .name = "bootstrap", .description = "Run the emergency bootstrap prerequisite path" })
        .newline()
        .section("Options")
        .option(.{ .short = "-c", .long = "--clean", .description = "Remove generated bootstrap Zig source/build state first" })
        .option(.{ .long = "--zig", .description = "Update only bootstrap Zig" })
        .option(.{ .long = "--zls", .description = "Update only ZLS" })
        .option(common_options.help)
        .newline()
        .section("Examples")
        .example("abi bootstrap-zig install", "Build bootstrap Zig and ZLS")
        .example("abi bootstrap-zig zig --clean", "Clean-rebuild only bootstrap Zig")
        .example("abi bootstrap-zig zls", "Build only ZLS using bootstrap Zig")
        .example("abi bootstrap-zig status", "Inspect the repo-local bootstrap Zig bridge")
        .example("abi bootstrap-zig path", "Print .zig-bootstrap/bin")
        .example("abi bootstrap-zig bootstrap", "Run the emergency bootstrap prerequisite");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
