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

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
        .enable_gpu = false,
    });
    defer abi.shutdown(&framework);

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "example-agent",
        .temperature = 0.7,
    }) catch |err| {
        std.debug.print("Failed to create AI agent: {}\n", .{err});
        std.debug.print("AI feature may not be properly configured\n", .{});
        return err;
    };
    defer agent.deinit();

    const user_input = "Hello, how are you today?";
    const response = agent.process(user_input, allocator) catch |err| {
        std.debug.print("Failed to process user input: {}\n", .{err});
        return err;
    };
    defer allocator.free(response);

    std.debug.print("User: {s}\n", .{user_input});
    std.debug.print("Agent: {s}\n", .{response});
}
