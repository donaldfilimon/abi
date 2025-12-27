const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
    });
    defer abi.shutdown(&framework);

    var agent = try abi.ai.agent.Agent.init(allocator, .{
        .name = "example-agent",
        .temperature = 0.7,
    });
    defer agent.deinit();

    const user_input = "Hello, how are you today?";
    const response = try agent.process(user_input, allocator);
    defer allocator.free(response);

    std.debug.print("User: {s}\n", .{user_input});
    std.debug.print("Agent: {s}\n", .{response});
}
