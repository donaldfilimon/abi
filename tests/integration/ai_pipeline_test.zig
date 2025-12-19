const std = @import("std");
const abi = @import("abi");
const testing = std.testing;

test "AI pipeline: agent initialization and processing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Create and test agent
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(allocator, .{ .name = "TestAgent" });
    defer agent.deinit();

    const response = try agent.process("test input", allocator);
    defer allocator.free(@constCast(response));

    try testing.expect(response.len > 0);
}

test "AI pipeline: multiple agents coordination" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const Agent = abi.ai.agent.Agent;

    var agent1 = try Agent.init(allocator, .{ .name = "Agent1" });
    defer agent1.deinit();

    var agent2 = try Agent.init(allocator, .{ .name = "Agent2" });
    defer agent2.deinit();

    const response1 = try agent1.process("query", allocator);
    defer allocator.free(@constCast(response1));

    const response2 = try agent2.process("query", allocator);
    defer allocator.free(@constCast(response2));

    try testing.expect(response1.len > 0);
    try testing.expect(response2.len > 0);
}
