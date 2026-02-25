//! Brain management command.
//!
//! Dual export (.wdbx + .gguf) and brain file inspection.
//!
//! Usage:
//!   abi brain export --wdbx out.wdbx [--gguf out.gguf] [--model model.bin]
//!   abi brain info <path.wdbx>

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");

fn wrapExport(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runExport(ctx, args);
}
fn wrapInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runInfo(ctx, args);
}

pub const meta: command_mod.Meta = .{
    .name = "brain",
    .description = "Brain file management (export, info)",
    .kind = .group,
    .subcommands = &.{ "export", "info", "help" },
    .children = &.{
        .{ .name = "export", .description = "Export model to .wdbx (+ optional .gguf)", .handler = wrapExport },
        .{ .name = "info", .description = "Show brain file metadata", .handler = wrapInfo },
    },
};

pub fn run(_: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    utils.output.printError("Unknown brain command: {s}", .{cmd});
    utils.output.printInfo("Run 'abi brain help' for usage.", .{});
}

fn runExport(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = ctx;
    var wdbx_path: ?[]const u8 = null;
    var gguf_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &.{"--wdbx"})) {
            if (i < args.len) {
                wdbx_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (utils.args.matchesAny(arg, &.{"--gguf"})) {
            if (i < args.len) {
                gguf_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    if (wdbx_path == null) {
        utils.output.printError("--wdbx <path> is required", .{});
        utils.output.println("", .{});
        utils.output.println("Usage: abi brain export --wdbx <output.wdbx> [--gguf <output.gguf>]", .{});
        return;
    }

    utils.output.println("Brain export configuration:", .{});
    utils.output.printKeyValue("WDBX output", wdbx_path.?);
    if (gguf_path) |gp| {
        utils.output.printKeyValue("GGUF output", gp);
    }
    utils.output.println("", .{});
    utils.output.println("Note: To export a trained model, use:", .{});
    utils.output.print("  abi train run --wdbx-output {s}", .{wdbx_path.?});
    if (gguf_path) |gp| {
        utils.output.print(" --output {s}", .{gp});
    }
    utils.output.println("", .{});
}

fn runInfo(_: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        utils.output.printError("Path to .wdbx file required", .{});
        utils.output.println("Usage: abi brain info <path.wdbx>", .{});
        return;
    }

    const path = std.mem.sliceTo(args[0], 0);
    utils.output.println("Brain file: {s}", .{path});

    // Check file extension
    if (!std.mem.endsWith(u8, path, ".wdbx")) {
        utils.output.printWarning("File does not have .wdbx extension", .{});
    }

    utils.output.printKeyValue("Format", "WDBX (native brain)");
    utils.output.printKeyValue("Status", "File inspection requires WDBX database module");
    utils.output.println("", .{});
    utils.output.println("Use 'abi db stats' for detailed database statistics.", .{});
}

fn printHelp() void {
    const help_text =
        \\Usage: abi brain <command> [options]
        \\
        \\Manage brain files (native .wdbx format with dual .gguf export).
        \\
        \\Commands:
        \\  export             Export model to .wdbx (+ optional .gguf)
        \\  info <path.wdbx>   Show brain file metadata
        \\  help               Show this help message
        \\
        \\Export options:
        \\  --wdbx <path>      Output .wdbx brain file (required)
        \\  --gguf <path>      Also export .gguf for Ollama serving
        \\
        \\Examples:
        \\  abi brain export --wdbx abbey.wdbx
        \\  abi brain export --wdbx abbey.wdbx --gguf abbey.gguf
        \\  abi brain info abbey.wdbx
        \\
    ;
    utils.output.print("{s}", .{help_text});
}
