const modern_cli = @import("../../tools/cli/modern_cli.zig");
const features = @import("features.zig");
const agent = @import("agent.zig");
const db = @import("db.zig");
const gpu = @import("gpu.zig");
const version = @import("version.zig");

pub const root_command = modern_cli.Command{
    .name = "abi",
    .description = "ABI AI/ML framework CLI",
    .usage = "abi [COMMAND] [OPTIONS]",
    .subcommands = &.{
        &features.command,
        &agent.command,
        &db.command,
        &gpu.command,
        &version.command,
    },
    .examples = &.{
        "abi features list",
        "abi agent run --name quickstart",
        "abi db search --vec \"[0.1,0.2,0.3]\" --k 5",
        "abi gpu bench --size 256x256",
    },
};

pub fn getRootCommand() *const modern_cli.Command {
    return &root_command;
}

pub fn findCommand(root: *const modern_cli.Command, path: []const []const u8) ?*const modern_cli.Command {
    var current = root;
    for (path) |segment| {
        current = current.findSubcommand(segment) orelse return null;
    }
    return current;
}
