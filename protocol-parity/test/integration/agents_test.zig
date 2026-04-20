const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "agents public echo lifecycle" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    var agent = try abi.ai.agents.Agent.init(std.testing.allocator, .{
        .name = "integration-agent",
        .backend = .echo,
    });
    defer agent.deinit();

    const response = try agent.chat("hello from integration", std.testing.allocator);
    defer std.testing.allocator.free(response);

    try std.testing.expectEqualStrings("Echo: hello from integration", response);
    try std.testing.expectEqual(@as(usize, 2), agent.historyCount());
}

test "agents public context manages named agents" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const ctx = try abi.ai.agents.Context.init(std.testing.allocator, .{ .max_agents = 2 });
    defer ctx.deinit();

    const agent_ptr = try ctx.createAgent("integration-context-agent");
    try std.testing.expectEqual(agent_ptr, ctx.getAgent("integration-context-agent").?);

    const registry1 = try ctx.getToolRegistry();
    const registry2 = try ctx.getToolRegistry();
    try std.testing.expectEqual(registry1, registry2);
}

test {
    std.testing.refAllDecls(@This());
}
