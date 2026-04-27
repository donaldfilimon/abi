const std = @import("std");
const agents = @import("mod.zig");
const shared_types = @import("types.zig");

test "agents shared type exports stay aligned" {
    try std.testing.expectEqual(shared_types.DEFAULT_MAX_TOKENS, agents.DEFAULT_MAX_TOKENS);
    try std.testing.expectEqual(shared_types.AgentBackend.echo, agents.AgentBackend.echo);
}

test "agents context initialization" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    const ctx = try agents.Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 0), ctx.agents.count());
    try std.testing.expect(ctx.tool_registry == null);
}

test "agents context create and get agent" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    const ctx = try agents.Context.init(std.testing.allocator, .{ .max_agents = 5 });
    defer ctx.deinit();

    const agent_ptr = try ctx.createAgent("test-agent");
    try std.testing.expectEqual(@as(usize, 1), ctx.agents.count());

    const retrieved = ctx.getAgent("test-agent");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(agent_ptr, retrieved.?);
    try std.testing.expect(ctx.getAgent("missing") == null);
}

test "agents context max agents limit" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    const ctx = try agents.Context.init(std.testing.allocator, .{ .max_agents = 2 });
    defer ctx.deinit();

    _ = try ctx.createAgent("agent-1");
    _ = try ctx.createAgent("agent-2");

    try std.testing.expectError(error.MaxAgentsReached, ctx.createAgent("agent-3"));
}

test "agents context tool registry lazy initialization" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    const ctx = try agents.Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    try std.testing.expect(ctx.tool_registry == null);

    const registry1 = try ctx.getToolRegistry();
    try std.testing.expect(ctx.tool_registry != null);

    const registry2 = try ctx.getToolRegistry();
    try std.testing.expectEqual(registry1, registry2);
}

test "agent history controls" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    var agent = try agents.Agent.init(std.testing.allocator, .{ .name = "test-agent" });
    defer agent.deinit();

    const response = try agent.process("hello", std.testing.allocator);
    defer std.testing.allocator.free(response);

    try std.testing.expectEqual(@as(usize, 2), agent.historyCount());

    agent.clearHistory();
    try std.testing.expectEqual(@as(usize, 0), agent.historyCount());

    try agent.setTemperature(0.8);
    try agent.setTopP(0.5);
    agent.setHistoryEnabled(false);
}

test "agent rejects invalid configuration" {
    try std.testing.expectError(
        agents.AgentError.InvalidConfiguration,
        agents.Agent.init(std.testing.allocator, .{ .name = "" }),
    );
    try std.testing.expectError(
        agents.AgentError.InvalidConfiguration,
        agents.Agent.init(std.testing.allocator, .{ .name = "test", .temperature = -0.5 }),
    );
    try std.testing.expectError(
        agents.AgentError.InvalidConfiguration,
        agents.Agent.init(std.testing.allocator, .{ .name = "test", .temperature = 3.0 }),
    );
    try std.testing.expectError(
        agents.AgentError.InvalidConfiguration,
        agents.Agent.init(std.testing.allocator, .{ .name = "test", .top_p = -0.1 }),
    );
    try std.testing.expectError(
        agents.AgentError.InvalidConfiguration,
        agents.Agent.init(std.testing.allocator, .{ .name = "test", .top_p = 1.5 }),
    );
}

test "agent backend selection" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    var agent = try agents.Agent.init(std.testing.allocator, .{
        .name = "test-agent",
        .backend = .echo,
    });
    defer agent.deinit();

    const response = try agent.process("test input", std.testing.allocator);
    defer std.testing.allocator.free(response);

    try std.testing.expectEqualStrings("Echo: test input", response);
}

test "agent with system prompt" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    var agent = try agents.Agent.init(std.testing.allocator, .{
        .name = "test-agent",
        .system_prompt = "You are a helpful assistant.",
    });
    defer agent.deinit();

    try std.testing.expectEqual(@as(usize, 1), agent.historyCount());
    try std.testing.expectEqual(agents.Message.Role.system, agent.history.items[0].role);
}

test "agent stats" {
    if (!agents.isEnabled()) return error.SkipZigTest;

    var agent = try agents.Agent.init(std.testing.allocator, .{ .name = "test-agent" });
    defer agent.deinit();

    const response = try agent.process("hello", std.testing.allocator);
    defer std.testing.allocator.free(response);

    const stats = agent.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.user_messages);
    try std.testing.expectEqual(@as(usize, 1), stats.assistant_messages);
}

test "error context creation and formatting" {
    const api_ctx = agents.ErrorContext.apiError(
        agents.AgentError.HttpRequestFailed,
        .openai,
        "https://api.openai.com/v1/chat/completions",
        500,
        "gpt-4",
    );
    try std.testing.expectEqual(agents.AgentError.HttpRequestFailed, api_ctx.@"error");
    try std.testing.expectEqual(agents.AgentBackend.openai, api_ctx.backend);
    try std.testing.expectEqual(agents.OperationContext.api_request, api_ctx.operation);
    try std.testing.expectEqual(@as(?u16, 500), api_ctx.http_status);

    const config_ctx = agents.ErrorContext.configError(
        agents.AgentError.InvalidConfiguration,
        "temperature out of range",
    );
    try std.testing.expectEqual(agents.AgentError.InvalidConfiguration, config_ctx.@"error");
    try std.testing.expectEqual(agents.OperationContext.configuration_validation, config_ctx.operation);

    const gen_ctx = agents.ErrorContext.generationError(
        agents.AgentError.GenerationFailed,
        .ollama,
        "llama3.2",
        "model not loaded",
    );
    try std.testing.expectEqual(agents.AgentError.GenerationFailed, gen_ctx.@"error");
    try std.testing.expectEqual(agents.AgentBackend.ollama, gen_ctx.backend);

    const retry_ctx = agents.ErrorContext.retryError(
        agents.AgentError.RateLimitExceeded,
        .huggingface,
        "https://api-inference.huggingface.co/models/gpt2",
        2,
        3,
    );
    try std.testing.expectEqual(@as(?u32, 2), retry_ctx.retry_attempt);
    try std.testing.expectEqual(@as(?u32, 3), retry_ctx.max_retries);

    const formatted = try api_ctx.formatToString(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "HttpRequestFailed") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "openai") != null);
}

test "operation context toString" {
    try std.testing.expectEqualStrings("initialization", agents.OperationContext.initialization.toString());
    try std.testing.expectEqualStrings("API request", agents.OperationContext.api_request.toString());
    try std.testing.expectEqualStrings("JSON parsing", agents.OperationContext.json_parsing.toString());
}

test {
    std.testing.refAllDecls(@This());
}
