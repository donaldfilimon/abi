const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const state_mod = @import("../state.zig");
const config = @import("../../framework/config.zig");

fn requireState(ctx: *modern_cli.Context) errors.CommandError!*state_mod.State {
    return ctx.userData(state_mod.State) orelse errors.CommandError.RuntimeFailure;
}

fn parseFeature(name: []const u8) errors.CommandError!config.Feature {
    inline for (std.meta.tags(config.Feature)) |tag| {
        if (std.ascii.eqlIgnoreCase(name, @tagName(tag))) {
            return tag;
        }
    }
    return errors.CommandError.InvalidArgument;
}

fn formatStatus(enabled: bool) []const u8 {
    return if (enabled) "enabled" else "disabled";
}

fn listHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const stdout = std.io.getStdOut().writer();
    const all = std.meta.tags(config.Feature);

    var enabled_count: usize = 0;
    for (all) |feature| {
        if (state.framework.isFeatureEnabled(feature)) enabled_count += 1;
    }

    if (args.hasFlag("json")) {
        try stdout.writeAll("{\"features\":{");
        for (all, 0..) |feature, idx| {
            if (idx != 0) try stdout.writeAll(",");
            const status = if (state.framework.isFeatureEnabled(feature)) "true" else "false";
            try stdout.print("\"{s}\":{s}", .{ @tagName(feature), status });
        }
        try stdout.writeAll("}}\n");
        return;
    }

    try stdout.print("Features ({d} enabled / {d} total)\n", .{ enabled_count, all.len });
    for (all) |feature| {
        const name = @tagName(feature);
        const label = config.featureLabel(feature);
        const description = config.featureDescription(feature);
        const enabled = state.framework.isFeatureEnabled(feature);
        try stdout.print("  {s:<12} {s:<22} [{s}]\n", .{ name, label, formatStatus(enabled) });
        try stdout.print("      {s}\n", .{description});
    }
}

fn toggleHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs, enable: bool) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const arg = args.getArgument(0) orelse return errors.CommandError.MissingArgument;
    const feature_name = switch (arg) {
        .string => |value| value,
        .path => |value| value,
        else => return errors.CommandError.InvalidArgument,
    };

    const feature = try parseFeature(feature_name);
    const changed = if (enable) state.framework.enableFeature(feature) else state.framework.disableFeature(feature);
    const stdout = std.io.getStdOut().writer();
    const current = state.framework.isFeatureEnabled(feature);

    if (args.hasFlag("json")) {
        try stdout.print(
            "{\"feature\":\"{s}\",\"status\":\"{s}\",\"changed\":{s}}\n",
            .{
                @tagName(feature),
                formatStatus(current),
                if (changed) "true" else "false",
            },
        );
        return;
    }

    if (changed) {
        try stdout.print("Feature '{s}' {s}.\n", .{ @tagName(feature), formatStatus(current) });
    } else {
        try stdout.print("Feature '{s}' already {s}.\n", .{ @tagName(feature), formatStatus(current) });
    }
}

fn enableHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    try toggleHandler(ctx, args, true);
}

fn disableHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    try toggleHandler(ctx, args, false);
}

pub const list_command = modern_cli.Command{
    .name = "list",
    .description = "List runtime feature toggles",
    .handler = listHandler,
    .options = &.{
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const enable_command = modern_cli.Command{
    .name = "enable",
    .description = "Enable a feature",
    .handler = enableHandler,
    .arguments = &.{
        .{
            .name = "feature",
            .description = "Feature name (e.g. ai, database)",
            .arg_type = .string,
        },
    },
    .options = &.{
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const disable_command = modern_cli.Command{
    .name = "disable",
    .description = "Disable a feature",
    .handler = disableHandler,
    .arguments = &.{
        .{
            .name = "feature",
            .description = "Feature name (e.g. ai, database)",
            .arg_type = .string,
        },
    },
    .options = &.{
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const command = modern_cli.Command{
    .name = "features",
    .description = "Inspect or toggle framework features",
    .subcommands = &.{ &list_command, &enable_command, &disable_command },
};