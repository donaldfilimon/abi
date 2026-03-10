//! `abi ui editor` subcommand wrapper.
//!
//! The editor engine lives under `tools/cli/ui/editor` so command wiring stays thin.

const command_mod = @import("../../../command");
const context_mod = @import("../../../framework/context");
const engine = @import("../../../terminal/editor/engine");

pub const meta: command_mod.Meta = .{
    .name = "editor",
    .description = "Open an inline Cursor-like terminal text editor",
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try engine.run(ctx, args);
}
