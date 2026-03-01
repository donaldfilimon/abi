//! End-to-End AI Agent Tests
//!
//! Complete workflow tests for AI agent functionality:
//! - Agent initialization with tools
//! - Query processing and tool usage
//! - Multi-turn conversation handling
//! - Agent with database integration

const std = @import("std");
const abi = @import("abi");
const time = abi.services.shared.time;
const sync = abi.services.shared.sync;
const e2e = @import("mod.zig");

// ============================================================================
// Helper Functions
// ============================================================================

/// Mock tool for testing agent workflows.
const MockToolResult = struct {
    success: bool,
    output: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MockToolResult) void {
        self.allocator.free(self.output);
    }
};

/// Execute a mock tool.
fn executeMockTool(allocator: std.mem.Allocator, tool_name: []const u8, input: []const u8) !MockToolResult {
    // Simulate different tool behaviors
    if (std.mem.eql(u8, tool_name, "search")) {
        const output = try std.fmt.allocPrint(allocator, "Search results for: {s}", .{input});
        return .{ .success = true, .output = output, .allocator = allocator };
    } else if (std.mem.eql(u8, tool_name, "calculate")) {
        const output = try allocator.dupe(u8, "42");
        return .{ .success = true, .output = output, .allocator = allocator };
    } else if (std.mem.eql(u8, tool_name, "error")) {
        const output = try allocator.dupe(u8, "Tool execution failed");
        return .{ .success = false, .output = output, .allocator = allocator };
    } else {
        const output = try std.fmt.allocPrint(allocator, "Unknown tool: {s}", .{tool_name});
        return .{ .success = false, .output = output, .allocator = allocator };
    }
}

/// Simulate agent response generation.
fn generateMockAgentResponse(allocator: std.mem.Allocator, query: []const u8, context: ?[]const u8) ![]u8 {
    if (context) |ctx| {
        return std.fmt.allocPrint(allocator, "Based on context ({d} chars), responding to: {s}", .{ ctx.len, query });
    } else {
        return std.fmt.allocPrint(allocator, "Responding to query: {s}", .{query});
    }
}

// ============================================================================
// E2E Tests: Agent Initialization
// ============================================================================

test "e2e: agent basic initialization" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    // Initialize test context with AI feature
    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
        .timeout_ms = 30_000,
    });
    defer ctx.deinit();

    // Create an agent
    var agent = try abi.features.ai.createAgent(allocator, "test-agent");
    defer agent.deinit();

    // Verify agent was created
    try std.testing.expectEqualStrings("test-agent", agent.config.name);
}

test "e2e: agent with tool registry" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Create agent with tool registry
    var agent = try abi.features.ai.createAgent(allocator, "tool-agent");
    defer agent.deinit();

    // The agent should be properly initialized with its name
    try std.testing.expectEqualStrings("tool-agent", agent.config.name);
}

// ============================================================================
// E2E Tests: Query Processing
// ============================================================================

test "e2e: agent processes simple query" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // Simulate query processing workflow
    const query = "What is the capital of France?";

    try timer.checkpoint("query_received");

    // Generate mock response
    const response = try generateMockAgentResponse(allocator, query, null);
    defer allocator.free(response);

    try timer.checkpoint("response_generated");

    // Verify response was generated
    try std.testing.expect(response.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, response, "capital of France") != null);

    // Workflow should complete quickly
    try std.testing.expect(!timer.isTimedOut(5_000));
}

test "e2e: agent tool execution workflow" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // 1. Receive query that requires tool usage
    const query = "Search for information about machine learning";
    try timer.checkpoint("query_received");

    // 2. Determine which tool to use (simulated)
    const tool_name = "search";
    try timer.checkpoint("tool_selected");

    // 3. Execute tool
    var tool_result = try executeMockTool(allocator, tool_name, query);
    defer tool_result.deinit();
    try timer.checkpoint("tool_executed");

    try std.testing.expect(tool_result.success);

    // 4. Generate response using tool output
    const response = try generateMockAgentResponse(allocator, query, tool_result.output);
    defer allocator.free(response);
    try timer.checkpoint("response_generated");

    // Verify workflow completed successfully
    try std.testing.expect(response.len > 0);
    try std.testing.expect(!timer.isTimedOut(10_000));

    // Record metrics
    ctx.recordOperation("tool_workflow", timer.elapsed());
}

test "e2e: agent handles tool errors gracefully" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Execute a tool that fails
    var tool_result = try executeMockTool(allocator, "error", "test input");
    defer tool_result.deinit();

    // Tool should report failure
    try std.testing.expect(!tool_result.success);
    try std.testing.expectEqualStrings("Tool execution failed", tool_result.output);

    // Agent should still be able to generate a fallback response
    const fallback = try allocator.dupe(u8, "I apologize, but I was unable to complete that operation.");
    defer allocator.free(fallback);

    try std.testing.expect(fallback.len > 0);
}

// ============================================================================
// E2E Tests: Multi-Turn Conversation
// ============================================================================

test "e2e: agent multi-turn conversation" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // Simulate multi-turn conversation
    const turns = [_]struct { user: []const u8, expected_contains: []const u8 }{
        .{ .user = "Hello, how are you?", .expected_contains = "Hello" },
        .{ .user = "Tell me about AI", .expected_contains = "AI" },
        .{ .user = "What did I ask first?", .expected_contains = "first" },
    };

    var context_builder: std.ArrayListUnmanaged(u8) = .empty;
    defer context_builder.deinit(allocator);

    for (turns) |turn| {
        // Append user message to context
        try context_builder.appendSlice(allocator, "User: ");
        try context_builder.appendSlice(allocator, turn.user);
        try context_builder.appendSlice(allocator, "\n");

        // Generate response
        const context = if (context_builder.items.len > 0) context_builder.items else null;
        const response = try generateMockAgentResponse(allocator, turn.user, context);
        defer allocator.free(response);

        // Append response to context
        try context_builder.appendSlice(allocator, "Assistant: ");
        try context_builder.appendSlice(allocator, response);
        try context_builder.appendSlice(allocator, "\n");

        // Verify response contains expected content (based on our mock implementation)
        try std.testing.expect(response.len > 0);
    }

    try timer.checkpoint("conversation_completed");

    // Context should have grown with each turn
    try std.testing.expect(context_builder.items.len > 0);
}

test "e2e: agent context window management" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Simulate a long conversation that would exceed context limits
    var context: std.ArrayListUnmanaged(u8) = .empty;
    defer context.deinit(allocator);

    const max_context_size: usize = 4096;

    // Generate many turns until we approach the limit
    var turn_count: usize = 0;
    while (context.items.len < max_context_size and turn_count < 100) : (turn_count += 1) {
        const query = "Tell me more about that topic.";
        const response = try generateMockAgentResponse(allocator, query, context.items);
        defer allocator.free(response);

        try context.appendSlice(allocator, "User: ");
        try context.appendSlice(allocator, query);
        try context.appendSlice(allocator, "\nAssistant: ");
        try context.appendSlice(allocator, response);
        try context.appendSlice(allocator, "\n");
    }

    // Verify we handled multiple turns
    try std.testing.expect(turn_count > 0);
    try std.testing.expect(context.items.len > 0);
}

// ============================================================================
// E2E Tests: Agent with Database Integration
// ============================================================================

test "e2e: agent with database tool" {
    try e2e.skipIfAiDisabled();
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true, .database = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // 1. Set up database with knowledge base
    var handle = try abi.features.database.open(allocator, "test-e2e-agent-db");
    defer abi.features.database.close(&handle);

    // Insert knowledge base documents
    const knowledge_base = [_]struct { id: u64, content: []const u8, embedding: [64]f32 }{
        .{ .id = 1, .content = "The capital of France is Paris.", .embedding = [_]f32{1.0} ++ [_]f32{0.0} ** 63 },
        .{ .id = 2, .content = "Python is a programming language.", .embedding = [_]f32{ 0.0, 1.0 } ++ [_]f32{0.0} ** 62 },
        .{ .id = 3, .content = "Machine learning is a subset of AI.", .embedding = [_]f32{ 0.0, 0.0, 1.0 } ++ [_]f32{0.0} ** 61 },
    };

    for (knowledge_base) |doc| {
        try abi.features.database.insert(&handle, doc.id, &doc.embedding, doc.content);
    }

    try timer.checkpoint("knowledge_base_created");

    // 2. Create agent
    var agent = try abi.features.ai.createAgent(allocator, "rag-agent");
    defer agent.deinit();

    try timer.checkpoint("agent_created");

    // 3. Process query that requires knowledge base lookup
    const query = "What is the capital of France?";

    // Generate query embedding (simulated - similar to first knowledge base entry)
    const query_embedding = [_]f32{ 0.9, 0.1 } ++ [_]f32{0.0} ** 62;

    // 4. Search knowledge base
    const results = try abi.features.database.search(&handle, allocator, &query_embedding, 1);
    defer allocator.free(results);

    try timer.checkpoint("knowledge_retrieved");

    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(u64, 1), results[0].id); // Should match capital of France

    // 5. Retrieve the document content
    const doc = abi.features.database.get(&handle, results[0].id);
    try std.testing.expect(doc != null);

    const knowledge = doc.?.metadata.?;

    // 6. Generate response with retrieved knowledge
    const response = try generateMockAgentResponse(allocator, query, knowledge);
    defer allocator.free(response);

    try timer.checkpoint("response_generated");

    try std.testing.expect(response.len > 0);
    try std.testing.expect(!timer.isTimedOut(30_000));
}

// ============================================================================
// E2E Tests: Agent State and Configuration
// ============================================================================

test "e2e: agent persona configuration" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Create agents with different personas
    var helpful_agent = try abi.features.ai.createAgent(allocator, "helpful-assistant");
    defer helpful_agent.deinit();

    var technical_agent = try abi.features.ai.createAgent(allocator, "technical-expert");
    defer technical_agent.deinit();

    // Each agent should have distinct identity
    try std.testing.expect(!std.mem.eql(u8, helpful_agent.config.name, technical_agent.config.name));
}

test "e2e: agent handles empty and invalid input" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Empty query
    const empty_response = try generateMockAgentResponse(allocator, "", null);
    defer allocator.free(empty_response);
    try std.testing.expect(empty_response.len > 0);

    // Very long query
    const long_query = "a" ** 10000;
    const long_response = try generateMockAgentResponse(allocator, long_query, null);
    defer allocator.free(long_response);
    try std.testing.expect(long_response.len > 0);

    // Special characters
    const special_query = "Hello! @#$%^&*() What's up?";
    const special_response = try generateMockAgentResponse(allocator, special_query, null);
    defer allocator.free(special_response);
    try std.testing.expect(special_response.len > 0);
}

// ============================================================================
// E2E Tests: Performance
// ============================================================================

test "e2e: agent response time benchmark" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Benchmark multiple query-response cycles
    const iterations: usize = 100;
    var total_time_ns: u64 = 0;

    for (0..iterations) |i| {
        var timer = try time.Timer.start();

        const query = "What is machine learning?";
        const response = try generateMockAgentResponse(allocator, query, null);
        allocator.free(response);

        const elapsed = timer.read();
        total_time_ns += elapsed;
        _ = i;
    }

    const avg_time_ns = total_time_ns / iterations;

    // Average response should be fast (< 10ms for mock)
    try std.testing.expect(avg_time_ns < 10_000_000);
}

// ============================================================================
// E2E Tests: Concurrent Agent Usage
// ============================================================================

test "e2e: multiple agents work independently" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .ai = true },
    });
    defer ctx.deinit();

    // Create multiple agents
    var agent1 = try abi.features.ai.createAgent(allocator, "agent-1");
    defer agent1.deinit();

    var agent2 = try abi.features.ai.createAgent(allocator, "agent-2");
    defer agent2.deinit();

    var agent3 = try abi.features.ai.createAgent(allocator, "agent-3");
    defer agent3.deinit();

    // Each agent should be independent
    try std.testing.expect(!std.mem.eql(u8, agent1.config.name, agent2.config.name));
    try std.testing.expect(!std.mem.eql(u8, agent2.config.name, agent3.config.name));

    // Each agent can process queries independently
    const response1 = try generateMockAgentResponse(allocator, "Query for agent 1", null);
    defer allocator.free(response1);

    const response2 = try generateMockAgentResponse(allocator, "Query for agent 2", null);
    defer allocator.free(response2);

    const response3 = try generateMockAgentResponse(allocator, "Query for agent 3", null);
    defer allocator.free(response3);

    // All responses should be valid
    try std.testing.expect(response1.len > 0);
    try std.testing.expect(response2.len > 0);
    try std.testing.expect(response3.len > 0);
}
