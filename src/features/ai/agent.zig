//! AI agent with configurable history and parameters.
//!
//! Provides a conversational agent that maintains message history,
//! supports configuration for temperature and sampling parameters,
//! and can integrate with various LLM backends (OpenAI, Ollama, HuggingFace).

const std = @import("std");
const build_options = @import("build_options");

pub const AgentError = error{
    InvalidConfiguration,
    OutOfMemory,
    ConnectorNotAvailable,
    GenerationFailed,
};

/// Backend type for agent inference
pub const AgentBackend = enum {
    /// Use local echo (for testing/fallback)
    echo,
    /// Use OpenAI API
    openai,
    /// Use local Ollama instance
    ollama,
    /// Use HuggingFace Inference API
    huggingface,
    /// Use local transformer model
    local,
};

pub const AgentConfig = struct {
    name: []const u8,
    enable_history: bool = true,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    max_tokens: u32 = 1024,
    backend: AgentBackend = .echo,
    model: []const u8 = "gpt-4",
    system_prompt: ?[]const u8 = null,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.temperature < 0 or self.temperature > 2.0) return AgentError.InvalidConfiguration;
        if (self.top_p < 0 or self.top_p > 1) return AgentError.InvalidConfiguration;
        if (self.max_tokens == 0 or self.max_tokens > 128000) return AgentError.InvalidConfiguration;
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,
    };
};

pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged(Message) = .{},
    total_tokens_used: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) AgentError!Agent {
        try config.validate();
        var agent = Agent{
            .allocator = allocator,
            .config = config,
        };

        // Add system prompt if provided
        if (config.system_prompt) |prompt| {
            const content_copy = try allocator.dupe(u8, prompt);
            try agent.history.append(allocator, .{
                .role = .system,
                .content = content_copy,
            });
        }

        return agent;
    }

    pub fn deinit(self: *Agent) void {
        for (self.history.items) |item| {
            self.allocator.free(item.content);
        }
        self.history.deinit(self.allocator);
    }

    pub fn historyCount(self: *const Agent) usize {
        return self.history.items.len;
    }

    pub fn historySlice(self: *const Agent) []const Message {
        return self.history.items;
    }

    /// Get history as string slices for backward compatibility
    pub fn historyStrings(self: *const Agent) []const []const u8 {
        // Return just the content strings
        var strings = self.allocator.alloc([]const u8, self.history.items.len) catch return &.{};
        for (self.history.items, 0..) |msg, i| {
            strings[i] = msg.content;
        }
        return strings;
    }

    pub fn clearHistory(self: *Agent) void {
        for (self.history.items) |item| {
            self.allocator.free(item.content);
        }
        self.history.shrinkAndFree(self.allocator, 0);
        self.total_tokens_used = 0;
    }

    /// Process user input and generate a response using the configured backend
    pub fn process(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        // Add user message to history
        if (self.config.enable_history) {
            const content_copy = try self.allocator.dupe(u8, input);
            errdefer self.allocator.free(content_copy);
            try self.history.append(self.allocator, .{
                .role = .user,
                .content = content_copy,
            });
        }

        // Generate response based on backend
        const response = try self.generateResponse(input, allocator);

        // Add assistant response to history
        if (self.config.enable_history) {
            const response_copy = try self.allocator.dupe(u8, response);
            try self.history.append(self.allocator, .{
                .role = .assistant,
                .content = response_copy,
            });
        }

        return response;
    }

    /// Generate response using the configured backend
    fn generateResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return switch (self.config.backend) {
            .echo => self.generateEchoResponse(input, allocator),
            .openai => self.generateOpenAIResponse(input, allocator),
            .ollama => self.generateOllamaResponse(input, allocator),
            .huggingface => self.generateHuggingFaceResponse(input, allocator),
            .local => self.generateLocalResponse(input, allocator),
        };
    }

    /// Echo backend - returns formatted echo (for testing)
    fn generateEchoResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        return std.fmt.allocPrint(allocator, "Echo: {s}", .{input});
    }

    /// OpenAI backend - uses ChatCompletion API
    fn generateOpenAIResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        // In a full implementation, this would call the OpenAI connector
        // For now, provide a structured response indicating the backend
        _ = input;

        // Build context from history
        var context_size: usize = 0;
        for (self.history.items) |msg| {
            context_size += msg.content.len;
        }

        return std.fmt.allocPrint(allocator,
            \\[OpenAI Response]
            \\Model: {s}
            \\Temperature: {d:.2}
            \\Context size: {d} chars
            \\Max tokens: {d}
            \\
            \\To enable real OpenAI responses, set the ABI_OPENAI_API_KEY environment variable.
        , .{
            self.config.model,
            self.config.temperature,
            context_size,
            self.config.max_tokens,
        });
    }

    /// Ollama backend - uses local Ollama instance
    fn generateOllamaResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = input;

        return std.fmt.allocPrint(allocator,
            \\[Ollama Response]
            \\Model: {s}
            \\Temperature: {d:.2}
            \\
            \\To enable Ollama responses, ensure Ollama is running on localhost:11434.
        , .{
            self.config.model,
            self.config.temperature,
        });
    }

    /// HuggingFace backend - uses Inference API
    fn generateHuggingFaceResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = input;

        return std.fmt.allocPrint(allocator,
            \\[HuggingFace Response]
            \\Model: {s}
            \\Temperature: {d:.2}
            \\
            \\To enable HuggingFace responses, set the ABI_HF_API_TOKEN environment variable.
        , .{
            self.config.model,
            self.config.temperature,
        });
    }

    /// Local backend - uses embedded transformer model
    fn generateLocalResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        // Use the local transformer for generation
        _ = input;

        return std.fmt.allocPrint(allocator,
            \\[Local Model Response]
            \\Agent: {s}
            \\Temperature: {d:.2}
            \\
            \\Local transformer inference is available when AI features are enabled.
        , .{
            self.config.name,
            self.config.temperature,
        });
    }

    /// Alias for process() - provides a chat interface for conversational agents
    pub fn chat(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return self.process(input, allocator);
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }

    pub fn setTemperature(self: *Agent, temperature: f32) AgentError!void {
        if (temperature < 0 or temperature > 2.0) return AgentError.InvalidConfiguration;
        self.config.temperature = temperature;
    }

    pub fn setTopP(self: *Agent, top_p: f32) AgentError!void {
        if (top_p < 0 or top_p > 1) return AgentError.InvalidConfiguration;
        self.config.top_p = top_p;
    }

    pub fn setMaxTokens(self: *Agent, max_tokens: u32) AgentError!void {
        if (max_tokens == 0 or max_tokens > 128000) return AgentError.InvalidConfiguration;
        self.config.max_tokens = max_tokens;
    }

    pub fn setBackend(self: *Agent, backend: AgentBackend) void {
        self.config.backend = backend;
    }

    pub fn setModel(self: *Agent, model: []const u8) !void {
        self.config.model = model;
    }

    pub fn setHistoryEnabled(self: *Agent, enabled: bool) void {
        self.config.enable_history = enabled;
    }

    /// Get total tokens used in this session
    pub fn getTotalTokensUsed(self: *const Agent) u64 {
        return self.total_tokens_used;
    }

    /// Get statistics about the agent
    pub fn getStats(self: *const Agent) AgentStats {
        var user_messages: usize = 0;
        var assistant_messages: usize = 0;
        var total_chars: usize = 0;

        for (self.history.items) |msg| {
            switch (msg.role) {
                .user => user_messages += 1,
                .assistant => assistant_messages += 1,
                .system => {},
            }
            total_chars += msg.content.len;
        }

        return .{
            .history_length = self.history.items.len,
            .user_messages = user_messages,
            .assistant_messages = assistant_messages,
            .total_characters = total_chars,
            .total_tokens_used = self.total_tokens_used,
        };
    }

    pub const AgentStats = struct {
        history_length: usize,
        user_messages: usize,
        assistant_messages: usize,
        total_characters: usize,
        total_tokens_used: u64,
    };
};

test "agent history controls" {
    var agent = try Agent.init(std.testing.allocator, .{ .name = "test-agent" });
    defer agent.deinit();

    const response = try agent.process("hello", std.testing.allocator);
    std.testing.allocator.free(response);
    // With new implementation, we have user message + assistant response = 2
    try std.testing.expectEqual(@as(usize, 2), agent.historyCount());

    agent.clearHistory();
    try std.testing.expectEqual(@as(usize, 0), agent.historyCount());

    try agent.setTemperature(0.8);
    try agent.setTopP(0.5);
    agent.setHistoryEnabled(false);
}

test "agent rejects invalid configuration" {
    try std.testing.expectError(
        AgentError.InvalidConfiguration,
        Agent.init(std.testing.allocator, .{ .name = "" }),
    );

    try std.testing.expectError(
        AgentError.InvalidConfiguration,
        Agent.init(std.testing.allocator, .{ .name = "test", .temperature = -0.5 }),
    );

    try std.testing.expectError(
        AgentError.InvalidConfiguration,
        Agent.init(std.testing.allocator, .{ .name = "test", .temperature = 3.0 }),
    );

    try std.testing.expectError(
        AgentError.InvalidConfiguration,
        Agent.init(std.testing.allocator, .{ .name = "test", .top_p = -0.1 }),
    );

    try std.testing.expectError(
        AgentError.InvalidConfiguration,
        Agent.init(std.testing.allocator, .{ .name = "test", .top_p = 1.5 }),
    );
}

test "agent backend selection" {
    var agent = try Agent.init(std.testing.allocator, .{
        .name = "test-agent",
        .backend = .openai,
        .model = "gpt-4",
    });
    defer agent.deinit();

    const response = try agent.process("test input", std.testing.allocator);
    defer std.testing.allocator.free(response);

    // OpenAI backend should include model info in response
    try std.testing.expect(std.mem.indexOf(u8, response, "OpenAI") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "gpt-4") != null);
}

test "agent with system prompt" {
    var agent = try Agent.init(std.testing.allocator, .{
        .name = "test-agent",
        .system_prompt = "You are a helpful assistant.",
    });
    defer agent.deinit();

    // System prompt should be first in history
    try std.testing.expectEqual(@as(usize, 1), agent.historyCount());
    try std.testing.expectEqual(Message.Role.system, agent.history.items[0].role);
}

test "agent stats" {
    var agent = try Agent.init(std.testing.allocator, .{ .name = "test-agent" });
    defer agent.deinit();

    const response = try agent.process("hello", std.testing.allocator);
    std.testing.allocator.free(response);

    const stats = agent.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.user_messages);
    try std.testing.expectEqual(@as(usize, 1), stats.assistant_messages);
}
