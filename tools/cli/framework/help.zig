const std = @import("std");
const types = @import("types.zig");
const help_utils = @import("../utils/help.zig");
const HelpBuilder = help_utils.HelpBuilder;
const Subcommand = help_utils.Subcommand;

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
    // 16 KB is plenty for the full top-level help screen.
    var buf: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const allocator = fba.allocator();

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
        _ = builder.subcommand(.{
            .name = descriptor.name,
            .description = descriptor.description,
        });
    }

    // Built-in pseudo-commands
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
        .section("Aliases")
        .text("Run 'abi <command> help' or 'abi help <command>' for command-specific help.\n");

    // Add alias mappings
    for (descriptors) |descriptor| {
        for (descriptor.aliases) |alias| {
            // Format "  alias -> target\n" via raw text append
            var alias_buf: [128]u8 = undefined;
            const alias_line = std.fmt.bufPrint(&alias_buf, "  {s} -> {s}\n", .{ alias, descriptor.name }) catch continue;
            _ = builder.text(alias_line);
        }
    }

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
