const std = @import("std");
const types = @import("types.zig");
const utils = @import("../utils/mod.zig");

const help_utils = utils.help;
const HelpBuilder = help_utils.HelpBuilder;
const Option = help_utils.Option;
const Subcommand = help_utils.Subcommand;

/// Global flags shown in top-level help.
const global_flags = [_]Option{
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
    var stack_buffer: [32 * 1024]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&stack_buffer);
    var builder = HelpBuilder.init(fixed.allocator());
    defer builder.deinit();

    _ = builder
        .usage("abi", "[global-flags] <command> [options]")
        .section("Global Flags")
        .options(&global_flags)
        .newline()
        .text(features_text)
        .newline()
        .newline()
        .section("Commands");

    for (descriptors) |descriptor| {
        _ = builder.subcommand(Subcommand{
            .name = descriptor.name,
            .description = descriptor.description,
        });
    }

    _ = builder
        .subcommand(.{ .name = "version", .description = "Show framework version" })
        .subcommand(.{ .name = "help", .description = "Show help (use: abi help <command>)" })
        .newline()
        .section("Examples")
        .example("abi --list-features", "Show available features")
        .example("abi --disable-gpu db stats", "Run db stats with GPU disabled")
        .example("abi --enable-ai llm run", "Run LLM inference with AI enabled")
        .example("abi help llm run", "Show help for nested subcommand")
        .newline()
        .section("Aliases");

    for (descriptors) |descriptor| {
        for (descriptor.aliases) |alias| {
            _ = builder
                .text("  ")
                .text(alias)
                .text(" -> ")
                .text(descriptor.name)
                .newline();
        }
    }

    _ = builder
        .newline()
        .text("Run 'abi <command> help' or 'abi help <command>' for command-specific help.\n");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
