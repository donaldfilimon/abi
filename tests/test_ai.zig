const std = @import("std");
const testing = std.testing;
const ai = @import("../src/ai/mod.zig");

// Test AI module functionality
test "AI Persona functionality" {
    // Test persona creation and properties
    const persona = ai.PersonaType.adaptive;

    const description = persona.getDescription();
    try testing.expect(description.len > 0);

    const system_prompt = persona.getSystemPrompt();
    try testing.expect(system_prompt.len > 0);

    // Test all persona types
    inline for (std.meta.fields(ai.PersonaType)) |field| {
        const p = @as(ai.PersonaType, @enumFromInt(field.value));
        try testing.expect(p.getDescription().len > 0);
        try testing.expect(p.getSystemPrompt().len > 0);
    }
}

test "AI Message structure" {
    const allocator = testing.allocator;

    // Test message creation
    const content = "Hello, AI!";
    const message = ai.Message{
        .role = .user,
        .content = try allocator.dupe(u8, content),
    };
    defer allocator.free(message.content);

    try testing.expectEqual(ai.Message.Role.user, message.role);
    try testing.expectEqualStrings(content, message.content);

    // Test message duplication
    const duped = try message.dupe(allocator);
    defer duped.deinit(allocator);

    try testing.expectEqual(message.role, duped.role);
    try testing.expectEqualStrings(message.content, duped.content);
    try testing.expect(duped.content.ptr != message.content.ptr); // Different memory
}

test "AI Context management" {
    const allocator = testing.allocator;

    // Test context creation
    var context = try ai.Context.init(allocator, .adaptive);
    defer context.deinit();

    try testing.expectEqual(ai.PersonaType.adaptive, context.persona);
    try testing.expectEqual(@as(usize, 0), context.messages.items.len);

    // Test message addition
    const user_message = ai.Message{
        .role = .user,
        .content = try allocator.dupe(u8, "Test message"),
    };
    defer allocator.free(user_message.content);

    try context.addMessage(user_message);
    try testing.expectEqual(@as(usize, 1), context.messages.items.len);

    // Test history clearing
    context.clearHistory();
    try testing.expectEqual(@as(usize, 0), context.messages.items.len);

    // Test system prompt
    const prompt = context.getSystemPrompt();
    try testing.expect(prompt.len > 0);
}

test "AI Agent basic operations" {
    const allocator = testing.allocator;

    // Test agent creation
    var agent = try ai.Agent.init(allocator, .adaptive);
    defer agent.deinit();

    try testing.expectEqual(ai.PersonaType.adaptive, agent.context.persona);

    // Test persona changes
    agent.setPersona(.technical);
    try testing.expectEqual(ai.PersonaType.technical, agent.context.persona);

    // Test message handling
    const result = try agent.generate("Hello", .{});
    try testing.expect(result.content.len > 0);
    try testing.expect(result.usage != null);

    // Check that messages were added to context
    try testing.expect(agent.context.messages.items.len >= 2); // User + assistant
}

test "AI Agent conversation flow" {
    const allocator = testing.allocator;

    var agent = try ai.Agent.init(allocator, .adaptive);
    defer agent.deinit();

    // Test multiple interactions
    const messages = [_][]const u8{
        "What is the capital of France?",
        "Tell me about machine learning",
        "How does AI work?",
    };

    for (messages) |msg| {
        const result = try agent.generate(msg, .{});
        try testing.expect(result.content.len > 0);
        try testing.expect(result.usage != null);
        try testing.expect(result.usage.?.prompt_tokens > 0);
        try testing.expect(result.usage.?.completion_tokens > 0);
        try testing.expect(result.usage.?.total_tokens > 0);
    }

    // Check conversation history
    try testing.expect(agent.context.messages.items.len >= 6); // 3 user + 3 assistant messages

    // Test history clearing
    agent.clearHistory();
    try testing.expectEqual(@as(usize, 0), agent.context.messages.items.len);
}

test "AI Model configuration" {
    // Test model config creation
    const config = ai.ModelConfig{
        .model_id = "test-model",
        .backend = .local,
        .capabilities = .{
            .streaming = false,
            .context_window = 4096,
        },
    };

    try testing.expectEqualStrings("test-model", config.model_id);
    try testing.expectEqual(ai.Backend.local, config.backend);
    try testing.expectEqual(@as(usize, 4096), config.capabilities.context_window);
}

test "AI Backend enumeration" {
    // Test all backend types
    inline for (std.meta.fields(ai.Backend)) |field| {
        const backend = @as(ai.Backend, @enumFromInt(field.value));
        const description = backend.getDescription();
        try testing.expect(description.len > 0);
    }

    // Test specific backend descriptions
    try testing.expectEqualStrings("Local AI model", ai.Backend.local.getDescription());
    try testing.expectEqualStrings("OpenAI API", ai.Backend.openai.getDescription());
    try testing.expectEqualStrings("Anthropic Claude API", ai.Backend.anthropic.getDescription());
}

test "AI Token estimation" {
    // Test token estimation (rough approximation)
    try testing.expectEqual(@as(usize, 0), try ai.estimateTokens(""));
    try testing.expectEqual(@as(usize, 1), try ai.estimateTokens("hello"));
    try testing.expect(try ai.estimateTokens("This is a longer message with more words") > 5);
}

test "AI Generation options" {
    // Test generation options structure
    const options = ai.GenerationOptions{
        .stream_callback = null,
    };

    try testing.expect(options.stream_callback == null);
}

test "AI Error handling" {
    const allocator = testing.allocator;

    var agent = try ai.Agent.init(allocator, .adaptive);
    defer agent.deinit();

    // Test with empty input
    const result = try agent.generate("", .{});
    try testing.expect(result.content.len > 0);

    // Test with very long input
    const long_input = "a" ** 1000; // 1000 'a's
    const long_result = try agent.generate(long_input, .{});
    try testing.expect(long_result.content.len > 0);
}

test "AI Memory management" {
    const allocator = testing.allocator;

    // Test proper memory cleanup
    {
        var agent = try ai.Agent.init(allocator, .adaptive);
        defer agent.deinit();

        // Add some messages
        _ = try agent.generate("Test message 1", .{});
        _ = try agent.generate("Test message 2", .{});

        // Agent should be properly deinitialized
    }

    // Test context memory management
    {
        var context = try ai.Context.init(allocator, .adaptive);
        defer context.deinit();

        const message = ai.Message{
            .role = .user,
            .content = try allocator.dupe(u8, "Test"),
        };
        defer allocator.free(message.content);

        try context.addMessage(message);

        // Context should be properly deinitialized
    }
}

test "AI Performance characteristics" {
    const allocator = testing.allocator;

    var agent = try ai.Agent.init(allocator, .adaptive);
    defer agent.deinit();

    // Benchmark simple generation
    var timer = try std.time.Timer.start();

    const iterations = 10;
    for (0..iterations) |_| {
        _ = try agent.generate("Simple test message", .{});
    }

    const total_time = timer.read();
    const avg_time = total_time / iterations;

    // Should be reasonably fast (less than 1ms average on modern hardware)
    try testing.expect(avg_time < 1_000_000); // 1ms in nanoseconds

    std.debug.print("AI generation benchmark: {any} iterations, {any}ns average\n", .{ iterations, avg_time });
}
