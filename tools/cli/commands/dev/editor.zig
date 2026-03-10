//! Top-level `abi editor` wrapper.
//!
//! The shared editor runtime lives under `tools/cli/terminal/editor` so both
//! `abi editor` and `abi ui editor` stay on the same implementation path.

const command_mod = @import("../../command");
const context_mod = @import("../../framework/context");
const engine = @import("../../terminal/editor/engine");

pub const meta: command_mod.Meta = .{
    .name = "editor",
    .description = "Open the shared inline terminal text editor",
    .aliases = &.{"edit"},
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try engine.run(ctx, args);
}
