//! LLM command family.
//!
//! Canonical surface:
//! - `abi llm run`
//! - `abi llm session`
//! - `abi llm serve`
//! - `abi llm providers`
//! - `abi llm plugins`

const std = @import("std");
const utils = @import("../../utils/mod.zig");

const info = @import("info.zig");
const bench_cmd = @import("bench.zig");
const list = @import("list.zig");
const download = @import("download.zig");
const serve = @import("serve.zig");
const run_cmd = @import("run.zig");
const session_cmd = @import("session.zig");
const providers_cmd = @import("providers.zig");
const plugins_cmd = @import("plugins.zig");

const llm_subcommands = [_][]const u8{
    "run",
    "session",
    "serve",
    "providers",
    "plugins",
    "info",
    "bench",
    "list",
    "download",
};

fn lRun(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try run_cmd.runRun(a, p.remaining());
}

fn lSession(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try session_cmd.runSession(a, p.remaining());
}

fn lServe(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try serve.runServe(a, p.remaining());
}

fn lProviders(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try providers_cmd.runProviders(a, p.remaining());
}

fn lPlugins(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try plugins_cmd.runPlugins(a, p.remaining());
}

fn lInfo(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try info.runInfo(a, p.remaining());
}

fn lBench(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try bench_cmd.runBench(a, p.remaining());
}

fn lList(_: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    if (p.containsHelp()) {
        printHelp();
        return;
    }
    list.runList();
}

fn lDownload(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try download.runDownload(a, p.remaining());
}

fn lUnknown(cmd: []const u8) void {
    std.debug.print("Unknown llm command: {s}\n", .{cmd});
    if (utils.args.suggestCommand(cmd, &llm_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
}

fn lHelp(_: std.mem.Allocator) void {
    printHelp();
}

const llm_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"run"}, .run = lRun },
    .{ .names = &.{"session"}, .run = lSession },
    .{ .names = &.{"serve"}, .run = lServe },
    .{ .names = &.{"providers"}, .run = lProviders },
    .{ .names = &.{"plugins"}, .run = lPlugins },
    .{ .names = &.{"info"}, .run = lInfo },
    .{ .names = &.{"bench"}, .run = lBench },
    .{ .names = &.{"list"}, .run = lList },
    .{ .names = &.{"download"}, .run = lDownload },
};

/// Run the LLM command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(allocator, &parser, &llm_commands, null, lHelp, lUnknown);
}

pub fn printHelp() void {
    std.debug.print(
        "Usage: abi llm <subcommand> [options]\\n\\n" ++
            "Canonical commands:\\n" ++
            "  run         One-shot generation through provider router\\n" ++
            "  session     Interactive session through provider router\\n" ++
            "  serve       Start streaming HTTP server\\n" ++
            "  providers   Show provider availability and routing order\\n" ++
            "  plugins     Manage HTTP/native provider plugins\\n\\n" ++
            "Additional commands:\\n" ++
            "  info        Inspect local GGUF model metadata\\n" ++
            "  bench       Benchmark local and connector backends\\n" ++
            "  list        List supported local model formats\\n" ++
            "  download    Download model artifacts\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm run --model ./model.gguf --prompt \"hello\"\\n" ++
            "  abi llm run --model llama3 --prompt \"status\" --fallback mlx,ollama\\n" ++
            "  abi llm session --model llama3 --backend ollama\\n" ++
            "  abi llm providers\\n" ++
            "  abi llm plugins list\\n" ++
            "  abi llm serve --help\\n",
        .{},
    );
}
