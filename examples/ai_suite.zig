//! ABI AI Suite Example
//!
//! A consolidated example suite for ABI's AI features: core, inference,
//! reasoning, training, and multimodal agents.
//!
//! Usage: `zig build run-ai-suite -- [core|inference|reasoning|training|multimodal|connectors]`

const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const args = init.minimal.args.vector;

    if (args.len < 2) {
        usage();
        return;
    }

    const mode = std.mem.sliceTo(args[1], 0);
    std.debug.print("=== ABI AI Suite: {s} ===\n\n", .{mode});

    if (std.mem.eql(u8, mode, "core")) {
        try runCore(allocator);
    } else if (std.mem.eql(u8, mode, "inference")) {
        try runInference(allocator);
    } else if (std.mem.eql(u8, mode, "reasoning")) {
        try runReasoning(allocator);
    } else if (std.mem.eql(u8, mode, "training")) {
        try runTraining(allocator);
    } else if (std.mem.eql(u8, mode, "multimodal")) {
        try runMultiModal(allocator);
    } else {
        usage();
    }
}

fn usage() void {
    std.debug.print("Usage: abi-ai-suite [core|inference|reasoning|training|multimodal]\n", .{});
}

fn runCore(allocator: std.mem.Allocator) !void {
    var builder = abi.App.builder(allocator);
    var framework = try builder.withDefault(.ai).build();
    defer framework.deinit();

    var registry = abi.features.ai.tools.ToolRegistry.init(allocator);
    defer registry.deinit();
    try abi.features.ai.tools.registerDiscordTools(&registry);

    std.debug.print("AI Core active. Tools registered.\n", .{});
}

fn runInference(allocator: std.mem.Allocator) !void {
    var builder = abi.App.builder(allocator);
    var framework = try builder.with(.llm, abi.config.LlmConfig{}).build();
    defer framework.deinit();

    const cfg = abi.features.ai.llm.InferenceConfig{ .max_new_tokens = 512 };
    std.debug.print("Inference engine ready. Max tokens: {d}\n", .{cfg.max_new_tokens});
}

fn runReasoning(allocator: std.mem.Allocator) !void {
    var builder = abi.App.builder(allocator);
    var framework = try builder.withDefault(.ai).build();
    defer framework.deinit();

    std.debug.print("Reasoning modules accessible: abbey, explore, orchestration.\n", .{});
}

fn runTraining(allocator: std.mem.Allocator) !void {
    var builder = abi.App.builder(allocator);
    var framework = try builder.with(.ai, abi.config.AiConfig{ .training = .{} }).build();
    defer framework.deinit();

    const config = abi.features.ai.training.TrainingConfig{ .optimizer = .adamw };
    std.debug.print("Training pipeline ready. Optimizer: {s}\n", .{@tagName(config.optimizer)});
}

fn runMultiModal(allocator: std.mem.Allocator) !void {
    var builder = abi.App.builder(allocator);
    var framework = try builder.withDefault(.ai).build();
    defer framework.deinit();

    std.debug.print("Initializing Multi-Modal Agent...\n", .{});
    var agent = try abi.features.ai.agent.Agent.init(allocator, .{
        .name = "multimodal-agent",
    });
    defer agent.deinit();

    std.debug.print("Agent '{s}' (Multi-Modal logic active) online.\n", .{agent.config.name});
}
