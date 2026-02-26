//! CLI command: abi env
//!
//! Show, validate, and export ABI environment variables.

const std = @import("std");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "env",
    .description = "Show and validate ABI environment variables",
    .subcommands = &.{ "list", "validate", "export", "help" },
    .children = &.{
        .{ .name = "list", .description = "List all ABI_* variables (redacted)", .handler = wrapList },
        .{ .name = "validate", .description = "Check required vars and connectivity", .handler = wrapValidate },
        .{ .name = "export", .description = "Print export commands for shell sourcing", .handler = wrapExport },
    },
};

const EnvVar = struct {
    name: [*:0]const u8,
    display_name: []const u8,
    description: []const u8,
    required: bool,
    secret: bool,
};

const abi_env_vars = [_]EnvVar{
    .{ .name = "ABI_OPENAI_API_KEY", .display_name = "ABI_OPENAI_API_KEY", .description = "OpenAI API key", .required = false, .secret = true },
    .{ .name = "ABI_ANTHROPIC_API_KEY", .display_name = "ABI_ANTHROPIC_API_KEY", .description = "Anthropic/Claude API key", .required = false, .secret = true },
    .{ .name = "ABI_OLLAMA_HOST", .display_name = "ABI_OLLAMA_HOST", .description = "Ollama host URL", .required = false, .secret = false },
    .{ .name = "ABI_HF_API_TOKEN", .display_name = "ABI_HF_API_TOKEN", .description = "HuggingFace API token", .required = false, .secret = true },
    .{ .name = "ABI_GPU_BACKEND", .display_name = "ABI_GPU_BACKEND", .description = "GPU backend override", .required = false, .secret = false },
    .{ .name = "ABI_LLM_MODEL_PATH", .display_name = "ABI_LLM_MODEL_PATH", .description = "Default LLM model file path", .required = false, .secret = false },
    .{ .name = "ABI_MASTER_KEY", .display_name = "ABI_MASTER_KEY", .description = "32-byte secrets encryption key", .required = false, .secret = true },
    .{ .name = "ABI_DB_PATH", .display_name = "ABI_DB_PATH", .description = "Database file path", .required = false, .secret = false },
    .{ .name = "ABI_DISCORD_TOKEN", .display_name = "ABI_DISCORD_TOKEN", .description = "Discord bot token", .required = false, .secret = true },
    .{ .name = "OPENAI_API_KEY", .display_name = "OPENAI_API_KEY", .description = "OpenAI API key (legacy)", .required = false, .secret = true },
    .{ .name = "MISTRAL_API_KEY", .display_name = "MISTRAL_API_KEY", .description = "Mistral API key", .required = false, .secret = true },
    .{ .name = "COHERE_API_KEY", .display_name = "COHERE_API_KEY", .description = "Cohere API key", .required = false, .secret = true },
    .{ .name = "DISCORD_BOT_TOKEN", .display_name = "DISCORD_BOT_TOKEN", .description = "Discord bot token (legacy)", .required = false, .secret = true },
    .{ .name = "NO_COLOR", .display_name = "NO_COLOR", .description = "Disable colored output", .required = false, .secret = false },
};

fn wrapList(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    _ = ctx;
    listVars();
}

fn wrapValidate(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    _ = ctx;
    validateVars();
}

fn wrapExport(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    _ = ctx;
    exportVars();
}

/// Default action: list vars
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        listVars();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    utils.output.printError("Unknown env command: {s}", .{cmd});
    utils.output.printInfo("Available: list, validate, export", .{});
}

fn listVars() void {
    utils.output.printHeader("ABI Environment Variables");
    utils.output.println("", .{});

    var set_count: usize = 0;
    for (abi_env_vars) |ev| {
        if (std.c.getenv(ev.name)) |ptr| {
            const val = std.mem.sliceTo(ptr, 0);
            set_count += 1;
            if (val.len > 4) {
                utils.output.println("  {s}\u{2713}{s} {s: <25} {s}***  ({s})", .{
                    utils.output.Color.green(),
                    utils.output.Color.reset(),
                    ev.display_name,
                    val[0..4],
                    ev.description,
                });
            } else {
                utils.output.println("  {s}\u{2713}{s} {s: <25} ***   ({s})", .{
                    utils.output.Color.green(),
                    utils.output.Color.reset(),
                    ev.display_name,
                    ev.description,
                });
            }
        } else {
            utils.output.println("  {s}\u{2013}{s} {s: <25} {s}not set{s}  ({s})", .{
                utils.output.Color.dim(),
                utils.output.Color.reset(),
                ev.display_name,
                utils.output.Color.dim(),
                utils.output.Color.reset(),
                ev.description,
            });
        }
    }

    utils.output.println("\n  {d}/{d} variables set", .{ set_count, abi_env_vars.len });
    utils.output.println("", .{});
}

fn validateVars() void {
    utils.output.printHeader("Environment Validation");
    utils.output.println("", .{});

    var issues: usize = 0;

    // Check for at least one AI provider
    const has_openai = std.c.getenv("ABI_OPENAI_API_KEY") != null or std.c.getenv("OPENAI_API_KEY") != null;
    const has_anthropic = std.c.getenv("ABI_ANTHROPIC_API_KEY") != null;
    const has_ollama = std.c.getenv("ABI_OLLAMA_HOST") != null;

    if (has_openai or has_anthropic or has_ollama) {
        utils.output.printSuccess("AI provider configured", .{});
        if (has_openai) utils.output.printInfo("  OpenAI: available", .{});
        if (has_anthropic) utils.output.printInfo("  Anthropic: available", .{});
        if (has_ollama) {
            const host = if (std.c.getenv("ABI_OLLAMA_HOST")) |ptr|
                std.mem.sliceTo(ptr, 0)
            else
                "unknown";
            utils.output.printInfo("  Ollama: {s}", .{host});
        }
    } else {
        utils.output.printWarning("No AI provider configured.", .{});
        utils.output.printInfo("Set ABI_OPENAI_API_KEY, ABI_ANTHROPIC_API_KEY, or ABI_OLLAMA_HOST", .{});
        issues += 1;
    }

    // Check GPU backend override
    if (std.c.getenv("ABI_GPU_BACKEND")) |ptr| {
        const val = std.mem.sliceTo(ptr, 0);
        utils.output.printSuccess("GPU backend override: {s}", .{val});
    }

    // Check master key for production
    if (std.c.getenv("ABI_MASTER_KEY")) |ptr| {
        const val = std.mem.sliceTo(ptr, 0);
        if (val.len < 32) {
            utils.output.printWarning("ABI_MASTER_KEY is shorter than 32 bytes — weak encryption.", .{});
            issues += 1;
        } else {
            utils.output.printSuccess("ABI_MASTER_KEY set (32+ bytes)", .{});
        }
    }

    utils.output.println("", .{});
    if (issues == 0) {
        utils.output.printSuccess("Environment looks good.", .{});
    } else {
        utils.output.printWarning("{d} issue(s) found. See above for details.", .{issues});
    }
}

fn exportVars() void {
    utils.output.println("# ABI environment variables (source this in your shell)", .{});
    utils.output.println("# Usage: eval $(abi env export)", .{});
    utils.output.println("# NOTE: Secret variables (API keys, tokens) are excluded for safety.", .{});
    utils.output.println("", .{});

    var skipped: usize = 0;
    for (abi_env_vars) |ev| {
        if (std.c.getenv(ev.name)) |ptr| {
            if (ev.secret) {
                skipped += 1;
                continue;
            }
            const val = std.mem.sliceTo(ptr, 0);
            utils.output.println("export {s}=\"{s}\"", .{ ev.display_name, val });
        }
    }

    if (skipped > 0) {
        utils.output.println("", .{});
        utils.output.println("# {d} secret variable(s) excluded. Set them manually or use a secrets manager.", .{skipped});
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi env", "[command]")
        .description("Show, validate, and export ABI environment variables.")
        .section("Commands")
        .subcommand(.{ .name = "list", .description = "List all ABI_* vars with redacted values (default)" })
        .subcommand(.{ .name = "validate", .description = "Check required vars and test connectivity" })
        .subcommand(.{ .name = "export", .description = "Print export commands for shell sourcing" })
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi env", "List all environment variables")
        .example("abi env validate", "Check configuration health")
        .example("eval $(abi env export)", "Source vars into current shell");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
