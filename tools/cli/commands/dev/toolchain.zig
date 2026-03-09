//! CEL-first Zig toolchain management.
//!
//! Usage:
//!   abi toolchain install           Build CEL Zig and ZLS into .cel/bin
//!   abi toolchain zig               Build only CEL Zig
//!   abi toolchain zls               Build only ZLS with CEL Zig
//!   abi toolchain status            Show CEL Zig/ZLS status
//!   abi toolchain update            Rebuild the repo-local CEL toolchain
//!   abi toolchain path              Print the repo-local .cel/bin path
//!   abi toolchain bootstrap         Run the emergency bootstrap prerequisite

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
    .name = "toolchain",
    .description = "Manage the repo-local CEL Zig/ZLS toolchain (install, update, status, bootstrap)",
    .kind = .group,
    .subcommands = &.{ "install", "zig", "zls", "status", "update", "path", "bootstrap", "help" },
    .children = &.{
        .{ .name = "install", .description = "Build CEL Zig and ZLS into .cel/bin", .handler = command_mod.parserHandler(runInstallBoth) },
        .{ .name = "zig", .description = "Build only CEL Zig into .cel/bin", .handler = command_mod.parserHandler(runInstallZig) },
        .{ .name = "zls", .description = "Build only ZLS using .cel/bin/zig", .handler = command_mod.parserHandler(runInstallZls) },
        .{ .name = "status", .description = "Show CEL Zig/ZLS status", .handler = command_mod.parserHandler(runStatusSubcommand) },
        .{ .name = "update", .description = "Rebuild the repo-local CEL toolchain", .handler = command_mod.parserHandler(runUpdateSubcommand) },
        .{ .name = "path", .description = "Print the repo-local .cel/bin path", .handler = command_mod.parserHandler(runPathSubcommand) },
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
    output.printError("Unknown toolchain command: {s}", .{cmd});
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
    try runCommand(allocator, &.{ "./.cel/build.sh", "--status" });
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
        output.printInfo("Usage: abi toolchain update [--zig|--zls] [--clean]", .{});
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
    output.println(".cel/bin", .{});
}

fn runBootstrapSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }
    try expectNoTrailingArgs(parser, "bootstrap");

    output.printHeader("CEL Emergency Bootstrap");
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
        output.printInfo("Usage: abi toolchain install|zig|zls [--clean]", .{});
        return;
    }

    try runBuildCommand(allocator, target, clean);
}

fn runBuildCommand(allocator: std.mem.Allocator, target: BuildTarget, clean: bool) !void {
    output.printHeader("CEL Toolchain");
    output.printKeyValue("Target", switch (target) {
        .both => "zig + zls",
        .zig_only => "zig",
        .zls_only => "zls",
    });
    output.printKeyValue("Location", ".cel/bin");
    if (clean) output.printInfo("Clean rebuild requested", .{});

    switch (target) {
        .both => {
            if (clean) {
                try runCommand(allocator, &.{ "./.cel/build.sh", "--clean" });
            } else {
                try runCommand(allocator, &.{"./.cel/build.sh"});
            }
        },
        .zig_only => {
            if (clean) {
                try runCommand(allocator, &.{ "./.cel/build.sh", "--clean", "--zig-only" });
            } else {
                try runCommand(allocator, &.{ "./.cel/build.sh", "--zig-only" });
            }
        },
        .zls_only => {
            if (clean) {
                try runCommand(allocator, &.{ "./.cel/build.sh", "--clean", "--zls-only" });
            } else {
                try runCommand(allocator, &.{ "./.cel/build.sh", "--zls-only" });
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
    var child = std.process.Child.init(argv, allocator);
    const term = try child.spawnAndWait();
    switch (term) {
        .Exited => |code| if (code != 0) return error.CommandFailed,
        else => return error.CommandFailed,
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi toolchain", "<command> [options]")
        .description("Manage ABI's repo-local CEL Zig/ZLS toolchain.")
        .section("Commands")
        .subcommand(.{ .name = "install", .description = "Build CEL Zig and ZLS into .cel/bin" })
        .subcommand(.{ .name = "zig", .description = "Build only CEL Zig into .cel/bin" })
        .subcommand(.{ .name = "zls", .description = "Build only ZLS using .cel/bin/zig" })
        .subcommand(.{ .name = "status", .description = "Show CEL Zig/ZLS status" })
        .subcommand(.{ .name = "update", .description = "Rebuild the repo-local CEL toolchain" })
        .subcommand(.{ .name = "path", .description = "Print the repo-local .cel/bin path" })
        .subcommand(.{ .name = "bootstrap", .description = "Run the emergency bootstrap prerequisite path" })
        .newline()
        .section("Options")
        .option(.{ .short = "-c", .long = "--clean", .description = "Remove generated CEL source/build state first" })
        .option(.{ .long = "--zig", .description = "Update only CEL Zig" })
        .option(.{ .long = "--zls", .description = "Update only ZLS" })
        .option(common_options.help)
        .newline()
        .section("Examples")
        .example("abi toolchain install", "Build CEL Zig and ZLS")
        .example("abi toolchain zig --clean", "Clean-rebuild only CEL Zig")
        .example("abi toolchain zls", "Build only ZLS using .cel/bin/zig")
        .example("abi toolchain status", "Inspect the repo-local CEL toolchain")
        .example("abi toolchain path", "Print .cel/bin")
        .example("abi toolchain bootstrap", "Run the emergency bootstrap prerequisite");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
