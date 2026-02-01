const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withAiDefaults()
        .build();
    defer framework.deinit();

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "example-agent",
        .temperature = 0.7,
    }) catch |err| {
        std.debug.print("Failed to create AI agent: {t}\n", .{err});
        std.debug.print("AI feature may not be properly configured\n", .{});
        return err;
    };
    defer agent.deinit();

    const user_input = "Hello, how are you today?";
    // Using the chat() method (alias for process()) as documented in docs/content/ai.html
    const response = agent.chat(user_input, allocator) catch |err| {
        std.debug.print("Failed to chat with agent: {t}\n", .{err});
        return err;
    };
    defer allocator.free(response);

    std.debug.print("User: {s}\n", .{user_input});
    std.debug.print("Agent: {s}\n", .{response});
}
