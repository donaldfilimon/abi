const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const build_options = @import("../../build_options");
const builtin = @import("builtin");

fn versionHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    _ = ctx;
    if (args.hasFlag("json")) {
        std.debug.print("{{\"version\":\"{s}\",\"zig\":\"{s}\",\"target\":\"{s}\"}}\n", .{
            build_options.package_version,
            builtin.zig_version_string,
            @tagName(builtin.target.cpu.arch),
        });
    } else {
        std.debug.print("ABI {s}\n", .{build_options.package_version});
        std.debug.print("Zig {s}\n", .{builtin.zig_version_string});
        std.debug.print("Target {s}\n", .{@tagName(builtin.target.cpu.arch)});
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
