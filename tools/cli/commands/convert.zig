//! Dataset and Model Conversion CLI command.
//!
//! Commands:
//! - convert dataset --input <file> --output <file> --format <to-wdbx|to-tokenbin>
//! - convert model --input <gguf> --output <wdbx> --name <name> (Not fully implemented, uses train export)
//! - convert help

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "dataset")) {
        try runDataset(allocator, args[1..]);
        return;
    }

    // if (std.mem.eql(u8, command, "model")) { ... }

    std.debug.print("Unknown convert command: {s}\n", .{command});
    printHelp();
}

fn runDataset(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: ?[]const u8 = null;
    var block_size: usize = 2048;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--input")) {
            if (i < args.len) {
                input_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--format")) {
            if (i < args.len) {
                format = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--block-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                block_size = std.fmt.parseInt(usize, val, 10) catch 2048;
                i += 1;
            }
            continue;
        }
    }

    if (input_path == null or output_path == null or format == null) {
        std.debug.print("Error: --input, --output, and --format are required.\n", .{});
        return;
    }

    if (std.mem.eql(u8, format.?, "to-wdbx")) {
        std.debug.print("Converting TokenBin {s} -> WDBX {s}...\n", .{ input_path.?, output_path.? });
        abi.ai.tokenBinToWdbx(allocator, input_path.?, output_path.?, block_size) catch |err| {
            std.debug.print("Conversion failed: {t}\n", .{err});
            return;
        };
        std.debug.print("Success.\n", .{});
    } else if (std.mem.eql(u8, format.?, "to-tokenbin")) {
        std.debug.print("Converting WDBX {s} -> TokenBin {s}...\n", .{ input_path.?, output_path.? });
        abi.ai.wdbxToTokenBin(allocator, input_path.?, output_path.?) catch |err| {
            std.debug.print("Conversion failed: {t}\n", .{err});
            return;
        };
        std.debug.print("Success.\n", .{});
    } else {
        std.debug.print("Unknown format: {s}\n", .{format.?});
    }
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi convert <command> [options]
        \\
        \\Commands:
        \\  dataset --input <file> --output <file> --format <fmt>
        \\
        \\Formats:
        \\  to-wdbx      Convert TokenBin (.bin) to WDBX
        \\  to-tokenbin  Convert WDBX to TokenBin (.bin)
        \\
    , .{});
}
