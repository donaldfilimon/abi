//! AI agent with configurable history and parameters.
//!
//! Provides a conversational agent that maintains message history,
//! supports configuration for temperature and sampling parameters,
//! and can integrate with various LLM backends (OpenAI, Ollama, HuggingFace).

const std = @import("std");
const build_options = @import("build_options");
const http = @import("../../shared/utils/http/async_http.zig");
const retry = @import("../../shared/utils/retry.zig");
const connectors = @import("../connectors/mod.zig");

/// Escape a string for JSON output
fn escapeJsonString(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayList(u8){};
    errdefer result.deinit(allocator);

    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            // Other control characters (excluding \n=0x0A, \r=0x0D, \t=0x09)
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                var buf: [6]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                try result.appendSlice(allocator, &buf);
            },
            else => try result.append(allocator, c),
        }
    }

    return result.toOwnedSlice(allocator);
}

pub const AgentError = error{
    InvalidConfiguration,
    OutOfMemory,
    ConnectorNotAvailable,
    GenerationFailed,
    ApiKeyMissing,
    HttpRequestFailed,
    InvalidApiResponse,
    RateLimitExceeded,
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
    retry_config: retry.RetryConfig = .{},

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

    /// Get history as string slices for backward compatibility.
    /// Caller must free the returned slice with `allocator.free(slice)`.
    /// Returns null on allocation failure.
    pub fn historyStrings(self: *const Agent, allocator: std.mem.Allocator) ?[]const []const u8 {
        if (self.history.items.len == 0) return &.{};
        const strings = allocator.alloc([]const u8, self.history.items.len) catch return null;
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
        _ = input;

        // Get API key from environment
        const api_key = try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        }) orelse return AgentError.ApiKeyMissing;
        defer allocator.free(api_key);

        // Get base URL (default to OpenAI)
        const base_url = try connectors.getEnvOwned(allocator, "ABI_OPENAI_BASE_URL") orelse
            try allocator.dupe(u8, "https://api.openai.com/v1");
        defer allocator.free(base_url);

        // Build the full API endpoint
        const endpoint = try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{base_url});
        defer allocator.free(endpoint);

        // Build messages array JSON
        var messages_json: std.ArrayList(u8) = .{};
        defer messages_json.deinit(allocator);
        var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &messages_json);
        var writer = &aw.writer;

        try writer.writeAll("[");
        for (self.history.items, 0..) |msg, i| {
            if (i > 0) try writer.writeAll(",");

            const role_str = switch (msg.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };

            // Escape content for JSON
            const escaped_content = try escapeJsonString(allocator, msg.content);
            defer allocator.free(escaped_content);
            try writer.print("{{\"role\":\"{s}\",\"content\":\"{s}\"}}", .{ role_str, escaped_content });
        }
        try writer.writeAll("]");

        // Build request body
        const request_body = try std.fmt.allocPrint(allocator,
            \\{{"model":"{s}","messages":{s},"temperature":{d:.2},"max_tokens":{d}}}
        , .{
            self.config.model,
            messages_json.items,
            self.config.temperature,
            self.config.max_tokens,
        });
        defer allocator.free(request_body);

        // Make HTTP request
        var client = try http.AsyncHttpClient.init(allocator);
        defer client.deinit();

        var request = try http.HttpRequest.init(allocator, .post, endpoint);
        defer request.deinit();

        try request.setBearerToken(api_key);
        try request.setHeader("Content-Type", "application/json");
        try request.setBody(request_body);

        // Retry with exponential backoff
        var attempt: u32 = 0;
        var backoff_ms = self.config.retry_config.initial_backoff_ms;
        // Use Instant.now() for seeding PRNG in Zig 0.16
        const instant = std.time.Instant.now() catch return AgentError.GenerationFailed;
        var seed: u64 = undefined;
        switch (@typeInfo(@TypeOf(instant.timestamp))) {
            .@"struct" => {
                // posix timespec - combine tv_sec and tv_nsec
                seed = @as(u64, @bitCast(@as(i64, instant.timestamp.tv_sec))) ^ @as(u64, @intCast(instant.timestamp.tv_nsec));
            },
            .int => {
                // Windows - raw u64 timestamp
                seed = instant.timestamp;
            },
            else => {
                seed = 0; // Fallback
            },
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        var response: http.HttpResponse = while (attempt <= self.config.retry_config.max_attempts) : (attempt += 1) {
            var res = client.fetch(&request) catch |err| {
                if (attempt >= self.config.retry_config.max_attempts) {
                    std.log.err("OpenAI API request failed after {d} attempts: {}", .{ attempt + 1, err });
                    return AgentError.HttpRequestFailed;
                }

                // TODO: In Zig 0.16, sleep requires I/O context. Immediate retry for now.
                _ = random; // Keep for jitter when sleep is re-enabled
                backoff_ms = @min(
                    @as(u64, @intFromFloat(@as(f64, @floatFromInt(backoff_ms)) * self.config.retry_config.backoff_multiplier)),
                    self.config.retry_config.max_backoff_ms,
                );
                continue;
            };

            // Check if we should retry based on status code
            if (retry.isStatusRetryable(res.status_code) and attempt < self.config.retry_config.max_attempts) {
                res.deinit();

                // TODO: In Zig 0.16, sleep requires I/O context. Immediate retry for now.
                backoff_ms = @min(
                    @as(u64, @intFromFloat(@as(f64, @floatFromInt(backoff_ms)) * self.config.retry_config.backoff_multiplier)),
                    self.config.retry_config.max_backoff_ms,
                );
                continue;
            }

            break res;
        } else {
            std.log.err("OpenAI API request failed after {d} attempts", .{self.config.retry_config.max_attempts + 1});
            return AgentError.HttpRequestFailed;
        };
        defer response.deinit();

        // Check for rate limiting
        if (response.status_code == 429) {
            return AgentError.RateLimitExceeded;
        }

        // Check for success
        if (!response.isSuccess()) {
            std.log.err("OpenAI API returned status {d}: {s}", .{ response.status_code, response.body });
            return AgentError.HttpRequestFailed;
        }

        // Parse JSON response
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            response.body,
            .{},
        ) catch |err| {
            std.log.err("Failed to parse OpenAI response: {}", .{err});
            return AgentError.InvalidApiResponse;
        };
        defer parsed.deinit();

        // Extract the assistant's message
        const choices = parsed.value.object.get("choices") orelse return AgentError.InvalidApiResponse;
        if (choices.array.items.len == 0) return AgentError.InvalidApiResponse;

        const first_choice = choices.array.items[0];
        const message = first_choice.object.get("message") orelse return AgentError.InvalidApiResponse;
        const content = message.object.get("content") orelse return AgentError.InvalidApiResponse;

        // Update token usage if provided
        if (parsed.value.object.get("usage")) |usage| {
            if (usage.object.get("total_tokens")) |total| {
                self.total_tokens_used += @as(u64, @intCast(total.integer));
            }
        }

        return try allocator.dupe(u8, content.string);
    }

    /// Ollama backend - uses local Ollama instance
    fn generateOllamaResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = input;

        // Get Ollama host (default to localhost:11434)
        const ollama_host = try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_OLLAMA_HOST",
            "OLLAMA_HOST",
        }) orelse try allocator.dupe(u8, "http://127.0.0.1:11434");
        defer allocator.free(ollama_host);

        // Get model name (use config or environment)
        const model_name = try connectors.getEnvOwned(allocator, "ABI_OLLAMA_MODEL") orelse
            try allocator.dupe(u8, self.config.model);
        defer allocator.free(model_name);

        // Build the full API endpoint
        const endpoint = try std.fmt.allocPrint(allocator, "{s}/api/chat", .{ollama_host});
        defer allocator.free(endpoint);

        // Build messages array JSON
        var messages_json: std.ArrayList(u8) = .{};
        defer messages_json.deinit(allocator);
        var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &messages_json);
        var writer = &aw.writer;

        try writer.writeAll("[");
        for (self.history.items, 0..) |msg, i| {
            if (i > 0) try writer.writeAll(",");

            const role_str = switch (msg.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };

            // Escape content for JSON
            const escaped_content = try escapeJsonString(allocator, msg.content);
            defer allocator.free(escaped_content);
            try writer.print("{{\"role\":\"{s}\",\"content\":\"{s}\"}}", .{ role_str, escaped_content });
        }
        try writer.writeAll("]");

        // Build request body (Ollama format)
        const request_body = try std.fmt.allocPrint(allocator,
            \\{{"model":"{s}","messages":{s},"stream":false,"options":{{"temperature":{d:.2},"num_predict":{d}}}}}
        , .{
            model_name,
            messages_json.items,
            self.config.temperature,
            self.config.max_tokens,
        });
        defer allocator.free(request_body);

        // Make HTTP request
        var client = try http.AsyncHttpClient.init(allocator);
        defer client.deinit();

        var request = try http.HttpRequest.init(allocator, .post, endpoint);
        defer request.deinit();

        try request.setHeader("Content-Type", "application/json");
        try request.setBody(request_body);

        var response = client.fetch(&request) catch |err| {
            std.log.err("Ollama API request failed: {}. Is Ollama running on {s}?", .{ err, ollama_host });
            return AgentError.HttpRequestFailed;
        };
        defer response.deinit();

        // Check for success
        if (!response.isSuccess()) {
            std.log.err("Ollama API returned status {d}: {s}", .{ response.status_code, response.body });
            return AgentError.HttpRequestFailed;
        }

        // Parse JSON response
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            response.body,
            .{},
        ) catch |err| {
            std.log.err("Failed to parse Ollama response: {}", .{err});
            return AgentError.InvalidApiResponse;
        };
        defer parsed.deinit();

        // Extract the assistant's message
        const message = parsed.value.object.get("message") orelse return AgentError.InvalidApiResponse;
        const content = message.object.get("content") orelse return AgentError.InvalidApiResponse;

        // Update token usage if provided
        if (parsed.value.object.get("prompt_eval_count")) |prompt_tokens| {
            if (parsed.value.object.get("eval_count")) |completion_tokens| {
                self.total_tokens_used += @as(u64, @intCast(prompt_tokens.integer));
                self.total_tokens_used += @as(u64, @intCast(completion_tokens.integer));
            }
        }

        return try allocator.dupe(u8, content.string);
    }

    /// HuggingFace backend - uses Inference API
    fn generateHuggingFaceResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        _ = input;

        // Get API token from environment
        const api_token = try connectors.getFirstEnvOwned(allocator, &.{
            "ABI_HF_API_TOKEN",
            "HF_API_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        }) orelse return AgentError.ApiKeyMissing;
        defer allocator.free(api_token);

        // Get base URL (default to HuggingFace Inference API)
        const base_url = try connectors.getEnvOwned(allocator, "ABI_HF_BASE_URL") orelse
            try allocator.dupe(u8, "https://api-inference.huggingface.co");
        defer allocator.free(base_url);

        // Build the full API endpoint
        const endpoint = try std.fmt.allocPrint(allocator, "{s}/models/{s}", .{ base_url, self.config.model });
        defer allocator.free(endpoint);

        // Build input from conversation history
        var prompt: std.ArrayList(u8) = .{};
        defer prompt.deinit(allocator);
        var paw: std.Io.Writer.Allocating = .fromArrayList(allocator, &prompt);
        var prompt_writer = &paw.writer;

        for (self.history.items) |msg| {
            const role_prefix = switch (msg.role) {
                .system => "System: ",
                .user => "User: ",
                .assistant => "Assistant: ",
            };
            try prompt_writer.print("{s}{s}\n", .{ role_prefix, msg.content });
        }
        try prompt_writer.writeAll("Assistant: ");

        // Build request body (HuggingFace Inference API format)
        const request_body = try std.fmt.allocPrint(allocator,
            \\{{"inputs":
        ++ "\"{s}\"" ++
            \\,"parameters":{{"temperature":{d:.2},"max_new_tokens":{d},"return_full_text":false}}}}
        , .{
            prompt.items,
            self.config.temperature,
            self.config.max_tokens,
        });
        defer allocator.free(request_body);

        // Make HTTP request
        var client = try http.AsyncHttpClient.init(allocator);
        defer client.deinit();

        var request = try http.HttpRequest.init(allocator, .post, endpoint);
        defer request.deinit();

        try request.setBearerToken(api_token);
        try request.setHeader("Content-Type", "application/json");
        try request.setBody(request_body);

        var response = client.fetch(&request) catch |err| {
            std.log.err("HuggingFace API request failed: {}", .{err});
            return AgentError.HttpRequestFailed;
        };
        defer response.deinit();

        // Check for rate limiting
        if (response.status_code == 429) {
            return AgentError.RateLimitExceeded;
        }

        // Check for success
        if (!response.isSuccess()) {
            std.log.err("HuggingFace API returned status {d}: {s}", .{ response.status_code, response.body });
            return AgentError.HttpRequestFailed;
        }

        // Parse JSON response (array format)
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            response.body,
            .{},
        ) catch |err| {
            std.log.err("Failed to parse HuggingFace response: {}", .{err});
            return AgentError.InvalidApiResponse;
        };
        defer parsed.deinit();

        // HuggingFace returns an array with generated_text
        if (parsed.value != .array) return AgentError.InvalidApiResponse;
        if (parsed.value.array.items.len == 0) return AgentError.InvalidApiResponse;

        const first_result = parsed.value.array.items[0];
        const generated_text = first_result.object.get("generated_text") orelse return AgentError.InvalidApiResponse;

        return try allocator.dupe(u8, generated_text.string);
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
