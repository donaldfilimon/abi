//! AI agent lifecycle and conversation-state orchestration.

const std = @import("std");
const advanced_cognition = @import("../abbey/advanced.zig");
const backend_dispatch = @import("agent/dispatch.zig");
const types = @import("types.zig");

pub const MIN_TEMPERATURE = types.MIN_TEMPERATURE;
pub const MAX_TEMPERATURE = types.MAX_TEMPERATURE;
pub const MIN_TOP_P = types.MIN_TOP_P;
pub const MAX_TOP_P = types.MAX_TOP_P;
pub const MAX_TOKENS_LIMIT = types.MAX_TOKENS_LIMIT;
pub const DEFAULT_TEMPERATURE = types.DEFAULT_TEMPERATURE;
pub const DEFAULT_TOP_P = types.DEFAULT_TOP_P;
pub const DEFAULT_MAX_TOKENS = types.DEFAULT_MAX_TOKENS;

pub const AgentError = types.AgentError;
pub const AgentBackend = types.AgentBackend;
pub const OperationContext = types.OperationContext;
pub const ErrorContext = types.ErrorContext;
pub const BackendMetrics = types.BackendMetrics;
pub const AgentConfig = types.AgentConfig;
pub const Message = types.Message;
pub const AgentStats = types.AgentStats;

pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged(Message) = .empty,
    total_tokens_used: u64 = 0,
    cognition: ?*advanced_cognition.AdvancedCognition = null,
    backend_metrics: [8]BackendMetrics = [_]BackendMetrics{.{}} ** 8,

    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) AgentError!Agent {
        try config.validate();

        var agent = Agent{
            .allocator = allocator,
            .config = config,
        };

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

    pub fn historyStrings(self: *const Agent, allocator: std.mem.Allocator) ?[]const []const u8 {
        if (self.history.items.len == 0) return &.{};

        const strings = allocator.alloc([]const u8, self.history.items.len) catch return null;
        for (self.history.items, 0..) |message, index| {
            strings[index] = message.content;
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

    pub fn process(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        if (self.config.enable_history) {
            const content_copy = try self.allocator.dupe(u8, input);
            errdefer self.allocator.free(content_copy);

            try self.history.append(self.allocator, .{
                .role = .user,
                .content = content_copy,
            });
        }

        const response = try self.generateResponse(input, allocator);

        if (self.config.enable_history) {
            const response_copy = try self.allocator.dupe(u8, response);
            try self.history.append(self.allocator, .{
                .role = .assistant,
                .content = response_copy,
            });
        }

        return response;
    }

    fn generateResponse(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return backend_dispatch.generateResponse(self, input, allocator);
    }

    pub fn chat(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return self.process(input, allocator);
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }

    pub fn setTemperature(self: *Agent, temperature: f32) AgentError!void {
        if (temperature < MIN_TEMPERATURE or temperature > MAX_TEMPERATURE) {
            return AgentError.InvalidConfiguration;
        }
        self.config.temperature = temperature;
    }

    pub fn setTopP(self: *Agent, top_p: f32) AgentError!void {
        if (top_p < MIN_TOP_P or top_p > MAX_TOP_P) {
            return AgentError.InvalidConfiguration;
        }
        self.config.top_p = top_p;
    }

    pub fn setMaxTokens(self: *Agent, max_tokens: u32) AgentError!void {
        if (max_tokens == 0 or max_tokens > MAX_TOKENS_LIMIT) {
            return AgentError.InvalidConfiguration;
        }
        self.config.max_tokens = max_tokens;
    }

    pub fn setBackend(self: *Agent, backend: AgentBackend) void {
        self.config.backend = backend;
    }

    pub fn setModel(self: *Agent, model: []const u8) !void {
        self.config.model = model;
    }

    pub fn recordBackendResult(
        self: *Agent,
        backend_index: usize,
        success: bool,
        latency_ms: u64,
        quality: f32,
    ) void {
        if (backend_index >= self.backend_metrics.len) return;
        self.backend_metrics[backend_index].record(success, latency_ms, quality);
    }

    pub fn setCognition(self: *Agent, cognition: *advanced_cognition.AdvancedCognition) void {
        self.cognition = cognition;
    }

    pub fn getCognitionInsight(self: *Agent, query: []const u8) ?advanced_cognition.CognitiveResult {
        const cognition = self.cognition orelse return null;
        return cognition.process("agent", query) catch return null;
    }

    pub fn setHistoryEnabled(self: *Agent, enabled: bool) void {
        self.config.enable_history = enabled;
    }

    pub fn getTotalTokensUsed(self: *const Agent) u64 {
        return self.total_tokens_used;
    }

    pub fn getStats(self: *const Agent) AgentStats {
        var user_messages: usize = 0;
        var assistant_messages: usize = 0;
        var total_characters: usize = 0;

        for (self.history.items) |message| {
            switch (message.role) {
                .user => user_messages += 1,
                .assistant => assistant_messages += 1,
                .system => {},
            }
            total_characters += message.content.len;
        }

        return .{
            .history_length = self.history.items.len,
            .user_messages = user_messages,
            .assistant_messages = assistant_messages,
            .total_characters = total_characters,
            .total_tokens_used = self.total_tokens_used,
        };
    }
};


test {
    std.testing.refAllDecls(@This());
}
