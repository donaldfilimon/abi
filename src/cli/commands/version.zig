const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const build_options = @import("../../build_options");
const builtin = @import("builtin");

fn versionHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    _ = ctx;
    const stdout = std.io.getStdOut().writer();
    if (args.hasFlag("json")) {
        try std.json.stringify(
            .{
                .version = build_options.package_version,
                .zig = builtin.zig_version_string,
                .target = builtin.target,
            },
            .{},
            stdout,
        );
        try stdout.writeByte('\n');
    } else {
        try stdout.print("ABI {s}\n", .{build_options.package_version});
        try stdout.print("Zig {s}\n", .{builtin.zig_version_string});
        try stdout.print("Target {s}\n", .{@tagName(builtin.target.cpu.arch)});
    }
}

pub const command = modern_cli.Command{
    .name = "version",
    .description = "Display CLI and toolchain versions",
    .handler = versionHandler,
    .options = &.{
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};