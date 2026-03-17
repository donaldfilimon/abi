//! LLM command family.
//!
//! Canonical surface:
//! - `abi llm run`
//! - `abi llm session`
//! - `abi llm serve`
//! - `abi llm providers`
//! - `abi llm plugins`

const std = @import("std");
const command_mod = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const utils = @import("../../../utils/mod.zig");

const run_cmd = @import("run.zig");
const session_cmd = @import("session.zig");
const providers_cmd = @import("providers.zig");
const plugins_cmd = @import("plugins.zig");
const serve_cmd = @import("serve.zig");
const discover_cmd = @import("discover.zig");
const bench_cmd = @import("bench.zig");
const chat_cmd = @import("chat.zig");
const demo_cmd = @import("demo.zig");
const download_cmd = @import("download.zig");
const generate_cmd = @import("generate.zig");
const info_cmd = @import("info.zig");
const list_cmd = @import("list.zig");

pub const meta: command_mod.Meta = .{
    .name = "llm",
    .description = "LLM inference (run, chat, session, serve, bench, generate, info, download, list, discover, providers, plugins)",
    .aliases = &.{ "chat", "reasoning" },
    .subcommands = &.{ "run", "chat", "session", "serve", "bench", "generate", "info", "download", "list", "demo", "discover", "providers", "plugins", "help" },
    .kind = .group,
    .children = &.{
        .{ .name = "run", .description = "One-shot generation through provider router", .handler = run_cmd.runRun },
        .{ .name = "chat", .description = "Interactive chat session with a model", .handler = chat_cmd.runChat },
        .{ .name = "session", .description = "Interactive session through provider router", .handler = session_cmd.runSession },
        .{ .name = "serve", .description = "Start streaming HTTP server", .handler = serve_cmd.runServe },
        .{ .name = "bench", .description = "Benchmark model performance", .handler = bench_cmd.runBench },
        .{ .name = "generate", .description = "Generate text from a prompt", .handler = generate_cmd.runGenerate },
        .{ .name = "info", .description = "Show model information (GGUF)", .handler = info_cmd.runInfo },
        .{ .name = "download", .description = "Download a GGUF model from a URL", .handler = download_cmd.runDownload },
        .{ .name = "demo", .description = "Demo mode with simulated output", .handler = demo_cmd.runDemo },
        .{ .name = "list", .description = "List supported models and local GGUF files", .handler = command_mod.allocatorHandler(list_cmd.runListLocal) },
        .{ .name = "discover", .description = "Auto-discover available LLM providers", .handler = discover_cmd.runDiscover },
        .{ .name = "providers", .description = "Show provider availability and routing order", .handler = providers_cmd.runProviders },
        .{ .name = "plugins", .description = "Manage HTTP/native provider plugins", .handler = plugins_cmd.runPlugins },
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

    // If it starts with a dash, they forgot a subcommand or are trying to pass global flags
    if (std.mem.startsWith(u8, sub, "-")) {
        utils.output.printError("Options must be passed to a subcommand for 'llm'", .{});
        printHelp();
        return error.ExecutionFailed;
    }

    // Since 'chat' is an alias for 'llm', they might just want to start a session.
    // If they provided something else, suggest known subcommands.
    utils.output.printError("Unknown llm command: {s}", .{sub});
    if (utils.args.suggestCommand(sub, &.{ "run", "chat", "session", "serve", "bench", "generate", "info", "download", "list", "demo", "discover", "providers", "plugins" })) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    return error.ExecutionFailed;
}

pub fn printHelp() void {
    utils.output.println(
        "Usage: abi llm <subcommand> [options]\\n\\n" ++
            "Inference & Model Management:\\n" ++
            "  run         One-shot generation through provider router\\n" ++
            "  chat        Interactive chat session with a model\\n" ++
            "  session     Interactive session through provider router\\n" ++
            "  generate    Generate text from a prompt with advanced sampling\\n" ++
            "  serve       Start streaming HTTP server\\n" ++
            "  bench       Benchmark model performance\\n\\n" ++
            "Model Management:\\n" ++
            "  info        Show GGUF model information\\n" ++
            "  download    Download a GGUF model from a URL\\n" ++
            "  list        List supported models and local GGUF files\\n" ++
            "  demo        Demo mode with simulated output\\n\\n" ++
            "Provider Management:\\n" ++
            "  discover    Auto-discover available LLM providers\\n" ++
            "  providers   Show provider availability and routing order\\n" ++
            "  plugins     Manage HTTP/native provider plugins\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm run --model ./model.gguf --prompt \"hello\"\\n" ++
            "  abi llm chat ./model.gguf\\n" ++
            "  abi llm bench ./model.gguf --compare-ollama\\n" ++
            "  abi llm info ./model.gguf\\n" ++
            "  abi llm download https://huggingface.co/.../model.gguf\\n" ++
            "  abi llm generate ./model.gguf --prompt \"once upon a time\"\\n" ++
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
