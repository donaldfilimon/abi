//! CLI command: abi gendocs

const std = @import("std");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "gendocs",
    .description = "Generate API docs (runs zig build gendocs)",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var file = std.Io.Dir.cwd().openFile(io, "build.zig", .{}) catch {
        std.debug.print("Error: build.zig not found in current directory. Run from repo root.\n", .{});
        return error.ExecutionFailed;
    };
    defer file.close(io);

    var flags = std.ArrayListUnmanaged([]const u8).empty;
    defer flags.deinit(allocator);

    for (args) |arg_z| {
        const arg = std.mem.sliceTo(arg_z, 0);
        if (std.mem.eql(u8, arg, "--")) continue;
        if (std.mem.eql(u8, arg, "--check") or
            std.mem.eql(u8, arg, "--api-only") or
            std.mem.eql(u8, arg, "--no-wasm"))
        {
            try flags.append(allocator, arg);
            continue;
        }

        std.debug.print("Error: unknown argument for gendocs: {s}\n", .{arg});
        printHelp();
        return error.ExecutionFailed;
    }

    var argv = std.ArrayListUnmanaged([]const u8).empty;
    defer argv.deinit(allocator);

    try argv.append(allocator, "zig");
    try argv.append(allocator, "build");
    try argv.append(allocator, "gendocs");
    if (flags.items.len > 0) {
        try argv.append(allocator, "--");
        try argv.appendSlice(allocator, flags.items);
    }

    var child = try std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });

    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return error.ExecutionFailed,
        else => return error.ExecutionFailed,
    }
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi gendocs [--check] [--api-only] [--no-wasm]
        \\
        \\Generate ABI docs outputs via `zig build gendocs`.
        \\Outputs: docs/api, docs/_docs, docs/plans, docs/api-app.
        \\
        \\Flags:
        \\  --check      Verify generated docs are up to date (no writes)
        \\  --api-only   Generate only docs/api markdown files
        \\  --no-wasm    Skip docs api-app wasm runtime generation
        \\
        \\Examples:
        \\  abi gendocs
        \\  abi gendocs --check
        \\  abi gendocs --api-only --no-wasm
        \\
    , .{});
}
