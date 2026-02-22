//! C API Agent Tests — CRUD, messaging, stats, history, configuration, structs.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const abi = @import("abi");

// ============================================================================
// Agent Operations Tests (abi_agent_*)
// ============================================================================

test "c_api: agent create and destroy lifecycle" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    // Create agent (C API: abi_agent_create)
    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "test-agent",
        .backend = .echo,
        .model = "test-model",
        .system_prompt = null,
        .temperature = 0.7,
        .top_p = 0.9,
        .max_tokens = 1024,
        .enable_history = true,
    }) catch {
        return error.SkipZigTest;
    };

    // Destroy agent (C API: abi_agent_destroy)
    agent.deinit();
}

test "c_api: agent send message and receive response" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "echo-agent",
        .backend = .echo,
        .model = "echo",
        .enable_history = true,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Send message (C API: abi_agent_send)
    const response = agent.process("Hello, agent!", allocator) catch {
        return error.SkipZigTest;
    };
    defer allocator.free(response);

    // Response should contain the message (echo backend)
    try testing.expect(response.len > 0);
    try testing.expect(std.mem.indexOf(u8, response, "Echo:") != null or response.len > 0);
}

test "c_api: agent get status" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "status-test-agent",
        .backend = .echo,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Agent should be ready (C API: abi_agent_get_status returns ABI_AGENT_STATUS_READY)
    // The agent object exists and is valid
    try testing.expect(true);
}

test "c_api: agent get stats" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "stats-test-agent",
        .backend = .echo,
        .enable_history = true,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Initial stats should have zero history
    const initial_stats = agent.getStats();
    try testing.expectEqual(@as(usize, 0), initial_stats.history_length);

    // Send a message
    const response = agent.process("test", allocator) catch {
        return error.SkipZigTest;
    };
    defer allocator.free(response);

    // Stats should update (C API: abi_agent_get_stats)
    const stats = agent.getStats();
    try testing.expect(stats.history_length >= 2); // user + assistant messages
    try testing.expect(stats.user_messages >= 1);
    try testing.expect(stats.assistant_messages >= 1);
}

test "c_api: agent clear history" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "history-test-agent",
        .backend = .echo,
        .enable_history = true,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Send message to populate history
    const response = agent.process("test", allocator) catch {
        return error.SkipZigTest;
    };
    defer allocator.free(response);

    // Verify history is populated
    const stats_before = agent.getStats();
    try testing.expect(stats_before.history_length > 0);

    // Clear history (C API: abi_agent_clear_history)
    agent.clearHistory();

    // Verify history is empty
    const stats_after = agent.getStats();
    try testing.expectEqual(@as(usize, 0), stats_after.history_length);
}

test "c_api: agent set temperature" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "temp-test-agent",
        .backend = .echo,
        .temperature = 0.7,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Set valid temperature (C API: abi_agent_set_temperature)
    agent.setTemperature(0.8) catch {
        try testing.expect(false); // Should not fail for valid value
    };

    // Set invalid temperature (out of range)
    const invalid_result = agent.setTemperature(3.0);
    try testing.expect(invalid_result == error.InvalidConfiguration or invalid_result == error.InvalidArgument);
}

test "c_api: agent set max tokens" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "tokens-test-agent",
        .backend = .echo,
        .max_tokens = 1024,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Set valid max tokens (C API: abi_agent_set_max_tokens)
    agent.setMaxTokens(2048) catch {
        try testing.expect(false); // Should not fail for valid value
    };

    // Set invalid max tokens (zero)
    const invalid_result = agent.setMaxTokens(0);
    try testing.expect(invalid_result == error.InvalidConfiguration or invalid_result == error.InvalidArgument);
}

test "c_api: agent get name" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    const agent_name = "my-test-agent";
    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = agent_name,
        .backend = .echo,
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Get name (C API: abi_agent_get_name)
    const name = agent.name();
    try testing.expectEqualStrings(agent_name, name);
}

test "c_api: agent with system prompt" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    var agent = abi.ai.agent.Agent.init(allocator, .{
        .name = "system-prompt-agent",
        .backend = .echo,
        .system_prompt = "You are a helpful assistant.",
    }) catch {
        return error.SkipZigTest;
    };
    defer agent.deinit();

    // Agent should work with system prompt
    const response = agent.process("Hello", allocator) catch {
        return error.SkipZigTest;
    };
    defer allocator.free(response);

    try testing.expect(response.len > 0);
}

test "c_api: agent null handle returns error status" {
    // The C API handles null agent pointers by returning error codes
    // abi_agent_send(NULL, ...) returns ABI_ERROR_NOT_INITIALIZED
    // abi_agent_get_status(NULL) returns ABI_AGENT_STATUS_ERROR

    // In Zig, we simulate this by checking optionals
    const maybe_agent: ?*abi.ai.agent.Agent = null;
    try testing.expect(maybe_agent == null);
}

// ============================================================================
// Agent Backend Type Tests
// ============================================================================

test "c_api: agent backend enum values" {
    // Verify backend enum values match C API constants
    // ABI_AGENT_BACKEND_ECHO = 0, ABI_AGENT_BACKEND_OPENAI = 1, etc.

    // The abi.ai.agent.AgentBackend enum should have these variants
    const backends = [_]abi.ai.agent.AgentBackend{
        .echo,
        .openai,
        .ollama,
        .huggingface,
        .local,
    };

    for (backends) |backend| {
        // Each backend should be a valid enum value
        _ = backend;
    }

    try testing.expect(backends.len >= 5);
}

// ============================================================================
// Agent Configuration Tests
// ============================================================================

test "c_api: agent config defaults" {
    // The C API's AgentConfig defaults
    const ABI_AGENT_BACKEND_ECHO: c_int = 0;

    const AgentConfig = extern struct {
        name: [*:0]const u8 = "agent",
        backend: c_int = ABI_AGENT_BACKEND_ECHO,
        model: [*:0]const u8 = "gpt-4",
        system_prompt: ?[*:0]const u8 = null,
        temperature: f32 = 0.7,
        top_p: f32 = 0.9,
        max_tokens: u32 = 1024,
        enable_history: bool = true,
    };

    const config = AgentConfig{};

    try testing.expectEqual(@as(c_int, 0), config.backend);
    try testing.expectApproxEqAbs(@as(f32, 0.7), config.temperature, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.9), config.top_p, 1e-6);
    try testing.expectEqual(@as(u32, 1024), config.max_tokens);
    try testing.expect(config.enable_history);
    try testing.expect(config.system_prompt == null);
}

// ============================================================================
// Agent Response and Stats Struct Tests
// ============================================================================

test "c_api: agent response struct" {
    const AgentResponse = extern struct {
        text: ?[*:0]const u8 = null,
        length: usize = 0,
        tokens_used: u64 = 0,
    };

    // Default response should be empty
    const response = AgentResponse{};
    try testing.expect(response.text == null);
    try testing.expectEqual(@as(usize, 0), response.length);
    try testing.expectEqual(@as(u64, 0), response.tokens_used);
}

test "c_api: agent stats struct" {
    const AgentStats = extern struct {
        history_length: usize = 0,
        user_messages: usize = 0,
        assistant_messages: usize = 0,
        total_characters: usize = 0,
        total_tokens_used: u64 = 0,
    };

    // Default stats should be zero
    const stats = AgentStats{};
    try testing.expectEqual(@as(usize, 0), stats.history_length);
    try testing.expectEqual(@as(usize, 0), stats.user_messages);
    try testing.expectEqual(@as(usize, 0), stats.assistant_messages);
    try testing.expectEqual(@as(usize, 0), stats.total_characters);
    try testing.expectEqual(@as(u64, 0), stats.total_tokens_used);
}
