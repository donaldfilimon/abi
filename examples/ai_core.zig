//! AI Core Example
//!
//! Demonstrates the ai_core module: agents, tool registries,
//! prompt builders, and model discovery.
//!
//! Run with: `zig build run-ai-core`

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI AI Core Example ===\n\n", .{});

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withAiDefaults()
        .build();
    defer framework.deinit();

    // --- Tool Registry ---
    std.debug.print("--- Tool Registry ---\n", .{});
    var registry = abi.ai.ToolRegistry.init(allocator);
    defer registry.deinit();

    // Register Discord tools as an example of bulk tool registration
    try abi.ai.registerDiscordTools(&registry);
    std.debug.print("Registered Discord tools in registry\n", .{});

    // Look up a tool by name
    if (registry.get("discord_send_message")) |tool| {
        std.debug.print("Found tool: {s} ({d} params)\n", .{ tool.name, tool.parameters.len });
    }

    // --- Agent Creation ---
    std.debug.print("\n--- Agent ---\n", .{});
    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "demo-agent",
        .temperature = 0.7,
    }) catch |err| {
        std.debug.print("Agent init: {t} (expected without backend)\n", .{err});
        return;
    };
    defer agent.deinit();

    std.debug.print("Agent '{s}' created\n", .{agent.config.name});

    // --- Model Registry ---
    std.debug.print("\n--- Model Registry ---\n", .{});
    const ModelInfo = abi.ai_core.ModelInfo;
    _ = ModelInfo;
    std.debug.print("ModelRegistry type available\n", .{});

    std.debug.print("\nAI Core example complete.\n", .{});
}
