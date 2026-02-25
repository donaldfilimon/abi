//! CLI command: abi clean
//!
//! Remove build artifacts, state files, and cached data.

const std = @import("std");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

pub const meta: command_mod.Meta = .{
    .name = "clean",
    .description = "Remove build cache, state files, and downloaded models",
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    var clean_state = false;
    var clean_all = false;
    var force = false;

    for (args) |arg| {
        const a = std.mem.sliceTo(arg, 0);
        if (std.mem.eql(u8, a, "--state")) clean_state = true;
        if (std.mem.eql(u8, a, "--all")) clean_all = true;
        if (std.mem.eql(u8, a, "--force")) force = true;
    }

    // --all implies --state
    if (clean_all) clean_state = true;

    // Safety check for destructive --all
    if (clean_all and !force) {
        utils.output.printError("--all removes downloaded models and is destructive.", .{});
        utils.output.printInfo("Add --force to confirm: abi clean --all --force", .{});
        return;
    }

    utils.output.printHeader("ABI Clean");

    // Always: remove .zig-cache/
    utils.output.println("", .{});
    removeDir(allocator, ".zig-cache", "Build cache");

    // --state: remove .ralph/state.json, .abi/
    if (clean_state) {
        removeFile(allocator, ".ralph/state.json", "Ralph state");
        removeDir(allocator, ".abi", "ABI local state");
    }

    // --all: remove model cache
    if (clean_all) {
        removeDir(allocator, ".cache/abi/models", "Downloaded models");
    }

    utils.output.println("", .{});
    utils.output.printSuccess("Clean complete.", .{});
}

fn removeDir(allocator: std.mem.Allocator, path: []const u8, label: []const u8) void {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    const dir = std.Io.Dir.cwd();

    // Check if directory exists by trying to open it
    var sub_dir = dir.openDir(io, path, .{}) catch {
        utils.output.println("  {s}\u{2013}{s} {s} ({s}) — not found, skipping", .{
            utils.output.Color.dim(),
            utils.output.Color.reset(),
            label,
            path,
        });
        return;
    };
    sub_dir.close(io);

    utils.output.println("  {s}\u{2717}{s} {s} ({s}) — found", .{
        utils.output.Color.yellow(),
        utils.output.Color.reset(),
        label,
        path,
    });

    // Recursive delete via deleteTree if available, otherwise instruct user
    dir.deleteTree(io, path) catch {
        utils.output.printInfo("  Could not auto-remove. Run: rm -rf {s}", .{path});
        return;
    };
    utils.output.printSuccess("  Removed {s}", .{path});
}

fn removeFile(allocator: std.mem.Allocator, path: []const u8, label: []const u8) void {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    const dir = std.Io.Dir.cwd();

    // Try to read file to check existence
    if (dir.readFileAlloc(io, path, allocator, .limited(1))) |content| {
        allocator.free(content);
        utils.output.println("  {s}\u{2717}{s} {s} ({s}) — removing...", .{
            utils.output.Color.yellow(),
            utils.output.Color.reset(),
            label,
            path,
        });
        dir.deleteFile(io, path) catch |err| {
            utils.output.printWarning("Could not delete {s}: {t}", .{ path, err });
        };
    } else |_| {
        utils.output.println("  {s}{s}{s} {s} ({s}) — not found, skipping", .{
            utils.output.Color.dim(),
            "\u{2013}",
            utils.output.Color.reset(),
            label,
            path,
        });
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi clean", "[options]")
        .description("Remove build artifacts, state files, and cached data.")
        .section("Options")
        .option(.{ .long = "--state", .description = "Also remove .ralph/state.json and .abi/" })
        .option(.{ .long = "--all", .description = "Also remove downloaded models (destructive)" })
        .option(.{ .long = "--force", .description = "Required with --all to confirm destructive action" })
        .option(utils.help.common_options.help)
        .newline()
        .section("What Gets Removed")
        .text("  abi clean            .zig-cache/\n")
        .text("  abi clean --state    + .ralph/state.json, .abi/\n")
        .text("  abi clean --all      + downloaded models (requires --force)\n")
        .newline()
        .section("Examples")
        .example("abi clean", "Remove build cache")
        .example("abi clean --state", "Reset project state")
        .example("abi clean --all --force", "Full cleanup including models");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
