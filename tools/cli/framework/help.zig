const std = @import("std");
const types = @import("types.zig");
const utils = @import("../utils/mod.zig");

pub fn printTopLevel(descriptors: []const types.CommandDescriptor) void {
    utils.output.print(
        \\Usage: abi [global-flags] <command> [options]
        \\
        \\Global Flags:
        \\  --list-features       List available features and their status
        \\  --enable-<feature>    Enable a feature at runtime
        \\  --disable-<feature>   Disable a feature at runtime
        \\  --no-color            Disable colored output (also respects NO_COLOR env var)
        \\
        \\Features: gpu, ai, llm, embeddings, agents, training, database, network, observability, web
        \\
        \\Commands:
        \\
    , .{});

    for (descriptors) |descriptor| {
        const padding = if (descriptor.name.len < 14) 14 - descriptor.name.len else 2;
        utils.output.print("  {s}", .{descriptor.name});
        for (0..padding) |_| utils.output.print(" ", .{});
        utils.output.print("{s}\n", .{descriptor.description});
    }

    utils.output.print("  version       Show framework version\n", .{});
    utils.output.print("  help          Show help (use: abi help <command>)\n", .{});

    utils.output.print(
        \\
        \\Examples:
        \\  abi --list-features              # Show available features
        \\  abi --disable-gpu db stats       # Run db stats with GPU disabled
        \\  abi --enable-ai llm run          # Run LLM inference with AI enabled
        \\  abi help llm run                 # Show help for nested subcommand
        \\
        \\Aliases:
        \\
        \\Run 'abi <command> help' or 'abi help <command>' for command-specific help.
        \\
    , .{});

    for (descriptors) |descriptor| {
        for (descriptor.aliases) |alias| {
            utils.output.print("  {s} -> {s}\n", .{ alias, descriptor.name });
        }
    }
}
