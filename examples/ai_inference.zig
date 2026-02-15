//! AI Inference Example
//!
//! Demonstrates the inference module: LLM engine configuration,
//! embeddings, streaming generation, and model loading.
//!
//! Run with: `zig build run-ai-inference`

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI AI Inference Example ===\n\n", .{});

    if (!abi.inference.isEnabled()) {
        std.debug.print("Inference feature is disabled. Enable with -Denable-llm=true\n", .{});
        return;
    }

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withAiDefaults()
        .build();
    defer framework.deinit();

    // --- LLM Configuration ---
    std.debug.print("--- LLM Configuration ---\n", .{});
    const llm_config = abi.ai.LlmConfig{
        .max_context_length = 2048,
        .max_new_tokens = 512,
        .temperature = 0.7,
    };
    std.debug.print("Max context: {d} tokens\n", .{llm_config.max_context_length});
    std.debug.print("Max new tokens: {d}\n", .{llm_config.max_new_tokens});

    // --- Streaming Types ---
    std.debug.print("\n--- Streaming ---\n", .{});
    const StreamToken = abi.ai.StreamToken;
    _ = StreamToken;
    std.debug.print("StreamToken type available for token-by-token generation\n", .{});

    // --- Embeddings ---
    std.debug.print("\n--- Embeddings ---\n", .{});
    std.debug.print("Embeddings module: {s}\n", .{
        if (@hasDecl(abi.ai.embeddings, "EmbeddingConfig")) "EmbeddingConfig available" else "basic mode",
    });

    std.debug.print("\nInference example complete.\n", .{});
}
