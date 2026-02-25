//! LLM command family.
//!
//! Canonical surface:
//! - `abi llm run`
//! - `abi llm session`
//! - `abi llm serve`
//! - `abi llm providers`
//! - `abi llm plugins`

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");

const run_cmd = @import("run.zig");
const session_cmd = @import("session.zig");
const providers_cmd = @import("providers.zig");
const plugins_cmd = @import("plugins.zig");
const serve_cmd = @import("serve.zig");
const discover_cmd = @import("discover.zig");

pub const meta: command_mod.Meta = .{
    .name = "llm",
    .description = "LLM inference (run, session, serve, providers, plugins, discover)",
    .aliases = &.{ "chat", "reasoning" },
    .subcommands = &.{ "run", "session", "serve", "providers", "plugins", "discover", "help" },
    .kind = .group,
    .children = &.{
        .{ .name = "run", .description = "One-shot generation through provider router", .handler = run_cmd.runRun },
        .{ .name = "session", .description = "Interactive session through provider router", .handler = session_cmd.runSession },
        .{ .name = "serve", .description = "Start streaming HTTP server", .handler = serve_cmd.runServe },
        .{ .name = "providers", .description = "Show provider availability and routing order", .handler = providers_cmd.runProviders },
        .{ .name = "plugins", .description = "Manage HTTP/native provider plugins", .handler = plugins_cmd.runPlugins },
        .{ .name = "discover", .description = "Auto-discover available LLM providers", .handler = discover_cmd.runDiscover },
    },
};

/// Run the LLM command with the provided arguments.
/// Known subcommands (run, session, serve, providers, plugins) are dispatched
/// by the framework router via meta.children. This function handles:
/// - No args: print help
/// - Unknown subcommands: print error + suggestion
/// - Explicit help request
pub fn run(_: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }
    const sub = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(sub, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }
    utils.output.printError("Unknown llm command: {s}", .{sub});
    if (utils.args.suggestCommand(sub, &.{ "run", "session", "serve", "providers", "plugins", "discover" })) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    return error.ExecutionFailed;
}

pub fn printHelp() void {
    utils.output.println(
        "Usage: abi llm <subcommand> [options]\\n\\n" ++
            "Canonical commands:\\n" ++
            "  run         One-shot generation through provider router\\n" ++
            "  session     Interactive session through provider router\\n" ++
            "  serve       Start streaming HTTP server\\n" ++
            "  providers   Show provider availability and routing order\\n" ++
            "  plugins     Manage HTTP/native provider plugins\\n" ++
            "  discover    Auto-discover available LLM providers\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm run --model ./model.gguf --prompt \"hello\"\\n" ++
            "  abi llm run --model llama3 --prompt \"status\" --fallback mlx,ollama\\n" ++
            "  abi llm session --model llama3 --backend ollama\\n" ++
            "  abi llm discover\\n" ++
            "  abi llm providers\\n" ++
            "  abi llm plugins list\\n" ++
            "  abi llm serve --help",
        .{},
    );
}

test {
    std.testing.refAllDecls(@This());
}
