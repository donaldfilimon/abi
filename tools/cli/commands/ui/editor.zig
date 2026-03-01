//! `abi ui editor` subcommand wrapper.
//!
//! The editor engine lives under `tools/cli/ui/editor` so command wiring stays thin.

const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const engine = @import("../../ui/editor/engine.zig");

pub const meta: command_mod.Meta = .{
    .name = "editor",
    .description = "Open an inline Cursor-like terminal text editor",
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try engine.run(ctx, args);
}
