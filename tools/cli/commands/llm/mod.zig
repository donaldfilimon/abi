//! LLM command for local model inference.
//!
//! Commands:
//! - llm info <model> - Show model information
//! - llm generate <model> --prompt <text> - Generate text
//! - llm chat <model> - Interactive chat mode
//! - llm bench <model> - Benchmark model performance

const std = @import("std");
const utils = @import("../../utils/mod.zig");

const info = @import("info.zig");
const generate = @import("generate.zig");
const chat = @import("chat.zig");
const bench_cmd = @import("bench.zig");
const list = @import("list.zig");
const demo = @import("demo.zig");
const download = @import("download.zig");
const serve = @import("serve.zig");

const llm_subcommands = [_][]const u8{
    "info",
    "generate",
    "chat",
    "bench",
    "list",
    "list-local",
    "demo",
    "download",
    "serve",
};

fn lInfo(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try info.runInfo(a, p.remaining());
}
fn lGenerate(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try generate.runGenerate(a, p.remaining());
}
fn lChat(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try chat.runChat(a, p.remaining());
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
fn lListLocal(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    if (p.containsHelp()) {
        printHelp();
        return;
    }
    list.runListLocal(a, p.remaining());
}
fn lDemo(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try demo.runDemo(a, p.remaining());
}
fn lDownload(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try download.runDownload(a, p.remaining());
}
fn lServe(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try serve.runServe(a, p.remaining());
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
    .{ .names = &.{"info"}, .run = lInfo },
    .{ .names = &.{"generate"}, .run = lGenerate },
    .{ .names = &.{"chat"}, .run = lChat },
    .{ .names = &.{"bench"}, .run = lBench },
    .{ .names = &.{"list"}, .run = lList },
    .{ .names = &.{"list-local"}, .run = lListLocal },
    .{ .names = &.{"demo"}, .run = lDemo },
    .{ .names = &.{"download"}, .run = lDownload },
    .{ .names = &.{"serve"}, .run = lServe },
};

/// Run the LLM command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(allocator, &parser, &llm_commands, null, lHelp, lUnknown);
}

pub fn printHelp() void {
    const help_text =
        "Usage: abi llm <command> [options]\n\n" ++
        "Run local LLM inference with GGUF models (llama.cpp compatible).\n\n" ++
        "Commands:\n" ++
        "  info <model>       Show model information\n" ++
        "  generate <model>   Generate text from a prompt\n" ++
        "  chat <model>       Interactive chat mode\n" ++
        "  bench <model>      Benchmark model performance\n" ++
        "  serve              Start streaming inference HTTP server\n" ++
        "  list               List supported models and formats\n" ++
        "  list-local [dir]   List GGUF models in directory\n" ++
        "  download <url>     Download a model from URL\n" ++
        "  help               Show this help message\n\n" ++
        "Generate options:\n" ++
        "  -m, --model <path>      Path to GGUF model file\n" ++
        "  -p, --prompt <text>     Text prompt for generation\n" ++
        "  -n, --max-tokens <n>    Maximum tokens to generate (default: 256)\n" ++
        "  -t, --temperature <f>   Temperature for sampling (default: 0.7)\n" ++
        "  --top-p <f>             Top-p nucleus sampling (default: 0.9)\n" ++
        "  --top-k <n>             Top-k filtering (default: 40)\n" ++
        "  --repeat-penalty <f>    Repetition penalty (default: 1.1)\n" ++
        "  --seed <n>              Random seed for reproducibility\n" ++
        "  --stop <text>           Stop sequence (can specify multiple)\n" ++
        "  --stream                Enable streaming output\n" ++
        "  --ollama-model <name>   Override Ollama model for fallback execution\n" ++
        "  --no-ollama-fallback    Disable Ollama fallback for unsupported GGUF\n\n" ++
        "Advanced sampling (llama.cpp parity):\n" ++
        "  --tfs <f>               Tail-free sampling parameter (default: 1.0 = disabled)\n" ++
        "  --mirostat <n>          Mirostat mode (0=off, 1=v1, 2=v2, default: 0)\n" ++
        "  --mirostat-tau <f>      Mirostat target entropy (default: 5.0)\n" ++
        "  --mirostat-eta <f>      Mirostat learning rate (default: 0.1)\n\n" ++
        "Benchmark options:\n" ++
        "  --prompt-tokens <n>     Number of prompt tokens (default: 128)\n" ++
        "  --gen-tokens <n>        Number of tokens to generate (default: 64)\n" ++
        "  --runs <n>              Number of runtime runs for percentile stats (default: 3)\n" ++
        "  --prompt <text>         Prompt text for runtime benchmark\n" ++
        "  --compare-ollama        Run Ollama benchmark and compare decode speed\n" ++
        "  --ollama-model <name>   Override Ollama model for compare run\n" ++
        "  --compare-mlx           Run MLX benchmark and compare decode speed\n" ++
        "  --mlx-model <name>      Override MLX model for compare run\n" ++
        "  --compare-vllm          Run vLLM benchmark and compare decode speed\n" ++
        "  --vllm-model <name>     Override vLLM model for compare run\n" ++
        "  --compare-lmstudio      Run LM Studio benchmark and compare decode speed\n" ++
        "  --lmstudio-model <name> Override LM Studio model for compare run\n" ++
        "  --compare-all           Compare all available backends\n" ++
        "  --json                  Output results in JSON format\n" ++
        "  --wdbx-out <path>       Append benchmark record to WDBX database\n\n" ++
        "Serve options:\n" ++
        "  -m, --model <path>      Path to GGUF model file (required for local inference)\n" ++
        "  -a, --address <addr>    Listen address (default: 127.0.0.1:8080)\n" ++
        "  --auth-token <token>    Bearer token for authentication (optional)\n" ++
        "  --preload               Pre-load model on startup (reduces first-request latency)\n\n" ++
        "Examples:\n" ++
        "  abi llm info ./llama-7b.gguf\n" ++
        "  abi llm generate ./llama-7b.gguf -p \"Hello, how are you?\"\n" ++
        "  abi llm generate ./mistral-7b.gguf -p \"Write a poem\" -n 100 -t 0.8 --stream\n" ++
        "  abi llm generate ./model.gguf -p \"Complete this:\" --seed 42 --repeat-penalty 1.2\n" ++
        "  abi llm generate ./model.gguf -p \"Story:\" --mirostat 2 --mirostat-tau 3.0\n" ++
        "  abi llm generate ./model.gguf -p \"Code:\" --tfs 0.95 --top-k 50\n" ++
        "  abi llm generate ./gpt-oss-20b.gguf -p \"Hi\" --ollama-model gpt-oss\n" ++
        "  abi llm bench ./llama-7b.gguf --gen-tokens 128 --runs 5 --compare-ollama\n" ++
        "  abi llm bench --compare-ollama --compare-mlx --runs 5 --gen-tokens 128\n" ++
        "  abi llm bench --compare-all --runs 5 --json\n" ++
        "  abi llm bench --compare-vllm --compare-lmstudio --runs 3\n" ++
        "  abi llm bench ./llama-7b.gguf --compare-ollama --runs 7 --wdbx-out ./bench.wdbx\n" ++
        "  abi llm serve -m ./llama-7b.gguf --preload\n" ++
        "  abi llm serve -m ./model.gguf -a 0.0.0.0:8000 --auth-token secret\n" ++
        "  abi llm list-local ./models\n" ++
        "  abi llm download https://huggingface.co/.../model.gguf -o my-model.gguf\n";
    std.debug.print("{s}", .{help_text});
}
