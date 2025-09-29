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

    const stdin_reader = std.io.getStdIn().reader();
    var message = stdin_reader.readAllAlloc(state.allocator, 16 * 1024) catch |err| {
        return switch (err) {
            error.StreamTooLong => errors.CommandError.InvalidArgument,
            error.OutOfMemory => errors.CommandError.RuntimeFailure,
            else => errors.CommandError.RuntimeFailure,
        };
    };
    defer state.allocator.free(message);

    const trimmed = std.mem.trim(u8, message, " \t\r\n");
    const prompt = if (trimmed.len > 0) trimmed else "Hello ABI";

    const reply = std.fmt.allocPrint(
        state.allocator,
        "Agent {s} processed input ({d} bytes) and suggests continuing the workflow.",
        .{ agent_name, trimmed.len },
    ) catch return errors.CommandError.RuntimeFailure;
    defer state.allocator.free(reply);

    const stdout = std.io.getStdOut().writer();
    if (args.hasFlag("json")) {
        var buffer = std.ArrayList(u8).init(state.allocator);
        defer buffer.deinit();
        try std.json.stringify(
            .{
                .agent = agent_name,
                .input = prompt,
                .reply = reply,
            },
            .{},
            buffer.writer(),
        );
        try stdout.writeAll(buffer.items);
        try stdout.writeByte('\n');
    } else {
        try stdout.print("Agent {s} received: {s}\n", .{ agent_name, prompt });
        try stdout.print("Response: {s}\n", .{reply});
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
    .subcommands = &.{ &run_command },
};