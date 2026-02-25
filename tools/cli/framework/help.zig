const std = @import("std");
const types = @import("types.zig");
const utils = @import("../utils/mod.zig");

/// Global flags shown in top-level help.
const global_flags = [_]help_utils.Option{
    .{
        .long = "--list-features",
        .description = "List available features and their status",
    },
    .{
        .long = "--enable-<feature>",
        .description = "Enable a feature at runtime",
    },
    .{
        .long = "--disable-<feature>",
        .description = "Disable a feature at runtime",
    },
    .{
        .long = "--no-color",
        .description = "Disable colored output (also respects NO_COLOR env var)",
    },
};

/// Feature list for the top-level help text.
const features_text =
    \\Features: gpu, ai, llm, embeddings, agents, training, reasoning, database, network,
    \\          observability, web, cloud, analytics, auth, messaging, cache, storage,
    \\          search, mobile, gateway, pages, benchmarks, personas, constitution
;

/// Print the top-level help screen using the unified HelpBuilder.
/// Uses a stack-backed fixed buffer allocator since help output is bounded.
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

    var builder = HelpBuilder.init(allocator);
    // No deinit needed — backed by stack buffer.

    _ = builder
        .usage("abi", "[global-flags] <command> [options]")
        .section("Global Flags")
        .options(&global_flags)
        .newline()
        .text(features_text)
        .newline()
        .newline()
        .section("Commands");

    // Add all registered commands
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

    // Add alias mappings
    for (descriptors) |descriptor| {
        for (descriptor.aliases) |alias| {
            utils.output.print("  {s} -> {s}\n", .{ alias, descriptor.name });
        }
    }

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
