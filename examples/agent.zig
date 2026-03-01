//! AI Agent Example
//!
//! Demonstrates creating and using an AI agent with the ABI framework.
//! Shows agent initialization, tool registration, and query processing.
//!
//! Run with: `zig build run-agent`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (!abi.features.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    var builder = abi.App.builder(allocator);
    _ = builder.withDefault(.ai);
    var framework = try builder.build();
    defer framework.deinit();

    var agent = abi.features.ai.agent.Agent.init(allocator, .{
        .name = "example-agent",
        .temperature = 0.7,
    }) catch |err| {
        std.debug.print("Failed to create AI agent: {t}\n", .{err});
        std.debug.print("AI feature may not be properly configured\n", .{});
        return err;
    };
    defer agent.deinit();

    const user_input = "Hello, how are you today?";
    // Using the chat() method (alias for process())
    const response = agent.chat(user_input, allocator) catch |err| {
        std.debug.print("Failed to chat with agent: {t}\n", .{err});
        return err;
    };
    defer allocator.free(response);

    std.debug.print("User: {s}\n", .{user_input});
    std.debug.print("Agent: {s}\n", .{response});
}
