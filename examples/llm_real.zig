//! Real LLM inference via ABI connectors.
//!
//! Tries, in order: Ollama → LM Studio → vLLM. Uses the first available
//! backend to run one chat completion. No GGUF file required.
//!
//! ## Setup
//!
//! - **Ollama**: Install from https://ollama.com, run `ollama serve` and
//!   `ollama run <model>`. Optional: `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL`.
//! - **LM Studio**: Start a local server; optional `ABI_LM_STUDIO_HOST`, `ABI_LM_STUDIO_MODEL`.
//! - **vLLM**: Start a server; optional `ABI_VLLM_HOST`, `ABI_VLLM_MODEL`.
//!
//! ## Run
//!
//!   zig build run-llm-real
//!   zig build run-llm-real -- "Your prompt here"
//!
//! Start one backend (e.g. `ollama serve` and `ollama run llama2`) then run the example.

const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const prompt = blk: {
        var args_it = init.args.iterateAllocator(allocator) catch break :blk "What is 2 + 2? Reply in one short sentence.";
        defer args_it.deinit();
        _ = args_it.next(); // skip exe name
        if (args_it.next()) |arg| break :blk arg[0..arg.len];
        break :blk "What is 2 + 2? Reply in one short sentence.";
    };

    std.debug.print("=== ABI Real LLM (Ollama / LM Studio / vLLM) ===\n\n", .{});
    std.debug.print("Prompt: {s}\n\n", .{prompt});

    // 1) Try Ollama
    if (abi.connectors.ollama.loadFromEnv(allocator) catch null) |config| {
        var client = abi.connectors.ollama.Client.init(allocator, config) catch |err| {
            std.debug.print("Ollama client init failed: {t}\n", .{err});
            printSetupHelp();
            return;
        };
        defer client.deinit();

        var response = client.chatSimple(prompt) catch |err| {
            std.debug.print("Ollama request failed: {t}\n", .{err});
            std.debug.print("Is Ollama running? Try: ollama serve && ollama run llama2\n\n", .{});
            printSetupHelp();
            return;
        };
        defer response.deinit(allocator);

        std.debug.print("Backend: Ollama\n", .{});
        std.debug.print("Model:  {s}\n", .{response.model});
        std.debug.print("Reply:  {s}\n", .{response.message.content});
        std.debug.print("\n=== Done ===\n", .{});
        return;
    }

    // 2) Try LM Studio
    if (abi.connectors.tryLoadLMStudio(allocator) catch null) |config| {
        var client = abi.connectors.lm_studio.Client.init(allocator, config) catch |err| {
            std.debug.print("LM Studio client init failed: {t}\n", .{err});
            printSetupHelp();
            return;
        };
        defer client.deinit();

        var response = client.chatSimple(prompt) catch |err| {
            std.debug.print("LM Studio request failed: {t}\n", .{err});
            std.debug.print("Is LM Studio server running on ABI_LM_STUDIO_HOST?\n\n", .{});
            printSetupHelp();
            return;
        };
        // Response owns id, model, choices[].message content - skip explicit deinit for this example

        std.debug.print("Backend: LM Studio\n", .{});
        std.debug.print("Model:  {s}\n", .{response.model});
        if (response.choices.len > 0) {
            std.debug.print("Reply:  {s}\n", .{response.choices[0].message.content});
        }
        std.debug.print("\n=== Done ===\n", .{});
        return;
    }

    // 3) Try vLLM
    if (abi.connectors.tryLoadVLLM(allocator) catch null) |config| {
        var client = abi.connectors.vllm.Client.init(allocator, config) catch |err| {
            std.debug.print("vLLM client init failed: {t}\n", .{err});
            printSetupHelp();
            return;
        };
        defer client.deinit();

        var response = client.chatSimple(prompt) catch |err| {
            std.debug.print("vLLM request failed: {t}\n", .{err});
            std.debug.print("Is vLLM server running on ABI_VLLM_HOST?\n\n", .{});
            printSetupHelp();
            return;
        };
        // Response owns allocated fields - skip explicit deinit for this example

        std.debug.print("Backend: vLLM\n", .{});
        std.debug.print("Model:  {s}\n", .{response.model});
        if (response.choices.len > 0) {
            std.debug.print("Reply:  {s}\n", .{response.choices[0].message.content});
        }
        std.debug.print("\n=== Done ===\n", .{});
        return;
    }

    std.debug.print("No local LLM backend available.\n\n", .{});
    printSetupHelp();
}

fn printSetupHelp() void {
    std.debug.print(
        \\Setup one of:
        \\  Ollama:     install from https://ollama.com, then: ollama serve && ollama run llama2
        \\  LM Studio:  start local server, optional ABI_LM_STUDIO_HOST (default http://localhost:1234)
        \\  vLLM:       start server, optional ABI_VLLM_HOST (default http://localhost:8000)
        \\
        \\Then run: zig build run-llm-real
        \\
    , .{});
}
