//! LLM demo subcommand - Demo mode with simulated output.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const mod = @import("mod.zig");

pub fn runDemo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    _ = allocator; // Not used in demo mode
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;

    // Parse arguments
    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--prompt", "-p" })) {
            if (i < args.len) {
                prompt = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--max-tokens", "-n" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_tokens = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        // Positional: prompt
        if (prompt == null) {
            prompt = std.mem.sliceTo(arg, 0);
        }
    }

    if (prompt == null) {
        prompt = "Hello, can you tell me about the ABI framework?";
    }

    std.debug.print("\xf0\x9f\xa4\x96 ABI LLM Demo Mode\n", .{});
    std.debug.print("==================\n", .{});
    std.debug.print("Prompt: {s}\n", .{prompt.?});
    std.debug.print("Max tokens: {d}\n\n", .{max_tokens});

    std.debug.print("Generating response...\n\n", .{});

    // Simulate generation with a demo response
    const response =
        "Hello! I'm the ABI framework's demo LLM assistant. While I don't have a real language model loaded right now, I can still help you understand how the system works!\n\n" ++
        "The ABI framework is designed for modular AI services, GPU compute, and vector databases. It supports:\n\n" ++
        "\xe2\x80\xa2 Multiple LLM formats (GGUF, PyTorch, etc.)\n" ++
        "\xe2\x80\xa2 GPU acceleration (CUDA, Vulkan, Metal)\n" ++
        "\xe2\x80\xa2 Vector databases with HNSW indexing\n" ++
        "\xe2\x80\xa2 Distributed computing with Raft consensus\n" ++
        "\xe2\x80\xa2 Real-time observability and monitoring\n\n" ++
        "To use real models, download GGUF files from https://huggingface.co/TheBloke\n" ++
        "For example: abi llm download https://huggingface.co/.../gpt2.gguf\n\n" ++
        "This demo shows the interface works correctly - the actual model loading happens when you provide a real GGUF file path.";

    const truncated_response = if (response.len > max_tokens * 4) response[0 .. max_tokens * 4] else response;

    std.debug.print("{s}\n", .{truncated_response});

    std.debug.print("\n---\n", .{});
    std.debug.print("Demo Stats: 25.0 tok/s prefill, 15.0 tok/s decode\n", .{});
    std.debug.print("\xf0\x9f\x92\xa1 Tip: Use 'abi llm list' to see supported model formats\n", .{});
}
