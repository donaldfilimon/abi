//! AI Reasoning Example
//!
//! Demonstrates the reasoning module: Abbey advanced reasoning,
//! RAG pipelines, evaluation templates, and orchestration.
//!
//! Run with: `zig build run-ai-reasoning`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI AI Reasoning Example ===\n\n", .{});

    if (!abi.features.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    var builder = abi.App.builder(allocator);

    var framework = try builder
        .withDefault(.ai)
        .build();
    defer framework.deinit();

    // --- Sub-modules ---
    std.debug.print("--- Reasoning Sub-modules ---\n", .{});
    std.debug.print("Abbey: advanced reasoning with meta-learning\n", .{});
    std.debug.print("Explore: search and discovery\n", .{});
    std.debug.print("Orchestration: multi-model coordination\n", .{});
    std.debug.print("Documents: document processing pipelines\n", .{});

    // Verify sub-module accessibility (canonical paths under abi.features.ai.*)
    _ = abi.features.ai.abbey;
    _ = abi.features.ai.explore;
    _ = abi.features.ai.orchestration;
    _ = abi.features.ai.documents;
    std.debug.print("\nAll sub-modules accessible.\n", .{});

    // --- RAG Pipeline ---
    std.debug.print("\n--- RAG (Retrieval Augmented Generation) ---\n", .{});
    if (@hasDecl(abi.features.ai, "rag")) {
        std.debug.print("RAG module available via abi.features.ai.rag\n", .{});
    }

    // --- Eval Templates ---
    std.debug.print("\n--- Evaluation ---\n", .{});
    if (@hasDecl(abi.features.ai, "eval")) {
        std.debug.print("Eval module available for model evaluation\n", .{});
    }
    if (@hasDecl(abi.features.ai, "templates")) {
        std.debug.print("Templates module available for prompt templates\n", .{});
    }

    std.debug.print("\nReasoning example complete.\n", .{});
}
