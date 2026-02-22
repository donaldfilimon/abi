//! ABI Framework Command-Line Interface
//!
//! Descriptor-driven CLI dispatcher shared by command help/completion/routing.

const std = @import("std");
const abi = @import("abi");
const commands = @import("commands/mod.zig");
const framework = @import("framework/mod.zig");
const utils = @import("utils/mod.zig");
const cli_io = utils.io_backend;
const spec = @import("spec.zig");

/// Main entry point with args from Zig 0.16 Init.Minimal
pub fn mainWithArgs(proc_args: std.process.Args, environ: std.process.Environ) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const raw_args = try proc_args.toSlice(arena.allocator());

    var global_flags = try utils.global_flags.parseGlobalFlags(allocator, raw_args);
    defer global_flags.deinit();
    defer allocator.free(global_flags.remaining_args);

    if (global_flags.show_features) {
        utils.global_flags.printFeaturesToStderr(utils.global_flags.ComptimeStatus);
        return;
    }

    if (global_flags.validate(utils.global_flags.ComptimeStatus)) |validation_error| {
        validation_error.print();
        std.process.exit(1);
    }

    var io_backend = cli_io.initIoBackendWithEnv(allocator, environ);
    defer io_backend.deinit();
    const io = io_backend.io();

    const fw_config = abi.Config.defaults();
    var framework_runtime = try abi.Framework.initWithIo(allocator, fw_config, io);
    defer framework_runtime.deinit();

    const registry = framework_runtime.getRegistry();
    global_flags.applyToRegistry(registry) catch |err| {
        std.debug.print("Warning: Could not apply feature override: {t}\n", .{err});
    };

    const args = global_flags.remaining_args;

    if (args.len <= 1) {
        framework.help.printTopLevel(&commands.descriptors);
        return;
    }

    const command = std.mem.sliceTo(args[1], 0);
    if (framework.types.isHelpToken(command)) {
        if (args.len >= 3) {
            const help_target = std.mem.sliceTo(args[2], 0);
            if (framework.types.isHelpToken(help_target)) {
                framework.help.printTopLevel(&commands.descriptors);
                return;
            }
            try runHelpTarget(allocator, arena.allocator(), io, help_target, args[3..]);
            return;
        }

        framework.help.printTopLevel(&commands.descriptors);
        return;
    }

    if (utils.args.matchesAny(command, &[_][]const u8{ "version", "--version", "-v" })) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    const ctx = framework.context.CommandContext{
        .allocator = allocator,
        .io = io,
    };

    if (try framework.router.runCommand(ctx, &commands.descriptors, command, args[2..])) {
        return;
    }

    printUnknownCommand(command);
    std.process.exit(1);
}

fn runHelpTarget(
    allocator: std.mem.Allocator,
    arena_allocator: std.mem.Allocator,
    io: std.Io,
    raw_command: []const u8,
    extra_args: []const [:0]const u8,
) !void {
    const command = framework.completion.resolveAlias(&commands.descriptors, raw_command);
    var forwarded = std.ArrayListUnmanaged([:0]const u8).empty;

    const help_arg: [:0]const u8 = "help";

    // Force help mode first so nested help requests don't accidentally execute
    // command logic that expects positional arguments (for example:
    // `abi help plugins enable` should not run `plugins enable help`).
    if (extra_args.len == 0) {
        try forwarded.append(arena_allocator, help_arg);
    } else if (framework.types.isHelpToken(std.mem.sliceTo(extra_args[0], 0))) {
        for (extra_args) |arg| {
            try forwarded.append(arena_allocator, arg);
        }
    } else {
        try forwarded.append(arena_allocator, help_arg);
        for (extra_args) |arg| {
            try forwarded.append(arena_allocator, arg);
        }
    }

    const ctx = framework.context.CommandContext{
        .allocator = allocator,
        .io = io,
    };

    if (try framework.router.runCommand(ctx, &commands.descriptors, command, forwarded.items)) {
        return;
    }

    printUnknownCommand(command);
    std.process.exit(1);
}

fn printUnknownCommand(command: []const u8) void {
    std.debug.print("Unknown command: {s}\n", .{command});
    if (utils.args.suggestCommand(command, spec.command_names_with_aliases)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
    std.debug.print("Use 'help' for usage.\n", .{});
}
