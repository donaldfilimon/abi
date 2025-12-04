const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const state_mod = @import("../state.zig");

fn requireState(ctx: *modern_cli.Context) errors.CommandError!*state_mod.State {
    return ctx.userData(state_mod.State) orelse errors.CommandError.RuntimeFailure;
}

fn runHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const agent_name = args.getString("name", "");
    if (agent_name.len == 0) return errors.CommandError.MissingArgument;

    // For now, use a simple placeholder message instead of stdin
    const prompt = "Hello ABI";

    const reply = std.fmt.allocPrint(
        state.allocator,
        "Agent {s} processed input ({d} bytes) and suggests continuing the workflow.",
        .{ agent_name, prompt.len },
    ) catch return errors.CommandError.RuntimeFailure;
    defer state.allocator.free(reply);

    if (args.hasFlag("json")) {
        std.debug.print("{{\"agent\":\"{s}\",\"input\":\"{s}\",\"reply\":\"{s}\"}}\n", .{ agent_name, prompt, reply });
    } else {
        std.debug.print("Agent {s} received: {s}\n", .{ agent_name, prompt });
        std.debug.print("Response: {s}\n", .{reply});
    }
}

pub const run_command = modern_cli.Command{
    .name = "run",
    .description = "Run a demo agent using stdin as input",
    .handler = runHandler,
    .options = &.{
        .{
            .name = "name",
            .long = "name",
            .description = "Agent persona to execute",
            .arg_type = .string,
            .required = true,
        },
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const command = modern_cli.Command{
    .name = "agent",
    .description = "Agent runtime entry points",
    .subcommands = &.{&run_command},
};
