//! Summarizing memory implementation.
//!
//! Compresses older messages into summaries to maintain context
//! while reducing token usage. Uses a rolling summarization strategy.

const std = @import("std");
const mod = @import("mod.zig");
const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryStats = mod.MemoryStats;
const MemoryType = @import("manager.zig").MemoryType;

/// Summarization configuration.
pub const SummaryConfig = struct {
    /// Maximum messages before summarization.
    max_messages: usize = 20,
    /// Messages to keep unsummarized.
    keep_recent: usize = 5,
    /// Maximum tokens for summary.
    max_summary_tokens: usize = 500,
    /// Summarization strategy.
    strategy: SummarizationStrategy = .rolling,
    /// Custom summarization function (optional).
    summarize_fn: ?*const fn ([]const Message, std.mem.Allocator) (std.mem.Allocator.Error || error{SummarizationFailed})![]u8 = null,
};

/// Summarization strategy.
pub const SummarizationStrategy = enum {
    /// Summarize in rolling batches.
    rolling,
    /// Summarize when threshold reached.
    threshold,
    /// Hierarchical summarization (summaries of summaries).
    hierarchical,
};

/// A summary of conversation messages.
pub const Summary = struct {
    /// Summary text content.
    content: []const u8,
    /// Number of messages summarized.
    message_count: usize,
    /// Estimated token count of original messages.
    original_tokens: usize,
    /// Summary token count.
    summary_tokens: usize,
    /// Timestamp of first summarized message.
    start_time: i64,
    /// Timestamp of last summarized message.
    end_time: i64,

    pub fn deinit(self: *Summary, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        self.* = undefined;
    }
};

/// Summarizing memory that compresses old messages.
pub const SummarizingMemory = struct {
    allocator: std.mem.Allocator,
    config: SummaryConfig,
    messages: std.ArrayListUnmanaged(Message),
    summaries: std.ArrayListUnmanaged(Summary),
    total_tokens: usize,
    owns_messages: bool,
    system_message: ?Message,

    /// Initialize summarizing memory.
    pub fn init(allocator: std.mem.Allocator, config: SummaryConfig) SummarizingMemory {
        return .{
            .allocator = allocator,
            .config = config,
            .messages = .{},
            .summaries = .{},
            .total_tokens = 0,
            .owns_messages = false,
            .system_message = null,
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *SummarizingMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
            if (self.system_message) |*sys| {
                sys.deinit(self.allocator);
            }
        }
        for (self.summaries.items) |*s| {
            s.deinit(self.allocator);
        }
        self.messages.deinit(self.allocator);
        self.summaries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set the system message.
    pub fn setSystemMessage(self: *SummarizingMemory, message: Message) !void {
        if (self.system_message) |*old| {
            self.total_tokens -|= old.estimateTokens();
            if (self.owns_messages) {
                old.deinit(self.allocator);
            }
        }

        const msg = if (self.owns_messages)
            try message.clone(self.allocator)
        else
            message;

        self.system_message = msg;
        self.total_tokens += msg.estimateTokens();
    }

    /// Add a message, triggering summarization if needed.
    pub fn add(self: *SummarizingMemory, message: Message) !void {
        const msg_to_add = if (self.owns_messages)
            try message.clone(self.allocator)
        else
            message;

        try self.messages.append(self.allocator, msg_to_add);
        self.total_tokens += message.estimateTokens();

        // Check if summarization is needed
        if (self.messages.items.len > self.config.max_messages) {
            try self.summarize();
        }
    }

    /// Add a message and take ownership.
    pub fn addOwned(self: *SummarizingMemory, message: Message) !void {
        self.owns_messages = true;
        try self.add(message);
    }

    /// Trigger summarization of older messages.
    pub fn summarize(self: *SummarizingMemory) !void {
        if (self.messages.items.len <= self.config.keep_recent) {
            return;
        }

        const to_summarize = self.messages.items.len - self.config.keep_recent;
        const messages_to_summarize = self.messages.items[0..to_summarize];

        // Generate summary
        const summary_content = if (self.config.summarize_fn) |summarize_fn|
            try summarize_fn(messages_to_summarize, self.allocator)
        else
            try self.defaultSummarize(messages_to_summarize);

        // Calculate stats
        var original_tokens: usize = 0;
        var start_time: i64 = std.math.maxInt(i64);
        var end_time: i64 = 0;

        for (messages_to_summarize) |msg| {
            original_tokens += msg.estimateTokens();
            if (msg.timestamp < start_time) start_time = msg.timestamp;
            if (msg.timestamp > end_time) end_time = msg.timestamp;
        }

        const summary = Summary{
            .content = summary_content,
            .message_count = to_summarize,
            .original_tokens = original_tokens,
            .summary_tokens = (summary_content.len + 3) / 4,
            .start_time = start_time,
            .end_time = end_time,
        };

        try self.summaries.append(self.allocator, summary);

        // Remove summarized messages
        if (self.owns_messages) {
            for (messages_to_summarize) |*msg| {
                var m = msg.*;
                m.deinit(self.allocator);
            }
        }

        // Shift remaining messages to start
        const remaining = self.messages.items.len - to_summarize;
        std.mem.copyForwards(
            Message,
            self.messages.items[0..remaining],
            self.messages.items[to_summarize..],
        );
        self.messages.shrinkRetainingCapacity(remaining);

        // Update token count
        self.total_tokens = self.calculateTotalTokens();
    }

    /// Default summarization (simple extraction of key points).
    fn defaultSummarize(self: *SummarizingMemory, messages: []const Message) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        try result.appendSlice(self.allocator, "[Summary of ");
        var buf: [20]u8 = undefined;
        const count_str = std.fmt.bufPrint(&buf, "{d}", .{messages.len}) catch "?";
        try result.appendSlice(self.allocator, count_str);
        try result.appendSlice(self.allocator, " messages]\n");

        // Extract key exchanges
        var user_count: usize = 0;
        var assistant_count: usize = 0;

        for (messages) |msg| {
            switch (msg.role) {
                .user => user_count += 1,
                .assistant => assistant_count += 1,
                else => {},
            }
        }

        try result.appendSlice(self.allocator, "User messages: ");
        const user_str = std.fmt.bufPrint(&buf, "{d}", .{user_count}) catch "?";
        try result.appendSlice(self.allocator, user_str);
        try result.appendSlice(self.allocator, ", Assistant responses: ");
        const asst_str = std.fmt.bufPrint(&buf, "{d}", .{assistant_count}) catch "?";
        try result.appendSlice(self.allocator, asst_str);
        try result.append(self.allocator, '\n');

        // Include first and last user message for context
        for (messages) |msg| {
            if (msg.role == .user) {
                const preview_len = @min(msg.content.len, 100);
                try result.appendSlice(self.allocator, "First topic: ");
                try result.appendSlice(self.allocator, msg.content[0..preview_len]);
                if (preview_len < msg.content.len) {
                    try result.appendSlice(self.allocator, "...");
                }
                try result.append(self.allocator, '\n');
                break;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Calculate total tokens across summaries and messages.
    fn calculateTotalTokens(self: *const SummarizingMemory) usize {
        var total: usize = 0;

        if (self.system_message) |sys| {
            total += sys.estimateTokens();
        }

        for (self.summaries.items) |s| {
            total += s.summary_tokens;
        }

        for (self.messages.items) |msg| {
            total += msg.estimateTokens();
        }

        return total;
    }

    /// Get all content for context (summaries + recent messages).
    pub fn getContext(
        self: *const SummarizingMemory,
        allocator: std.mem.Allocator,
    ) ![]Message {
        var result = std.ArrayListUnmanaged(Message).empty;
        errdefer result.deinit(allocator);

        // Add system message first
        if (self.system_message) |sys| {
            try result.append(allocator, sys);
        }

        // Add summaries as system context
        for (self.summaries.items) |s| {
            try result.append(allocator, Message{
                .role = .system,
                .content = s.content,
                .name = null,
                .timestamp = s.start_time,
                .token_count = s.summary_tokens,
                .metadata = null,
            });
        }

        // Add recent messages
        for (self.messages.items) |msg| {
            try result.append(allocator, msg);
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get just the recent messages.
    pub fn getRecentMessages(self: *const SummarizingMemory) []const Message {
        return self.messages.items;
    }

    /// Get summaries.
    pub fn getSummaries(self: *const SummarizingMemory) []const Summary {
        return self.summaries.items;
    }

    /// Clear all messages and summaries.
    pub fn clear(self: *SummarizingMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
        }
        for (self.summaries.items) |*s| {
            s.deinit(self.allocator);
        }
        self.messages.clearRetainingCapacity();
        self.summaries.clearRetainingCapacity();

        if (self.system_message) |sys| {
            self.total_tokens = sys.estimateTokens();
        } else {
            self.total_tokens = 0;
        }
    }

    /// Get memory statistics.
    pub fn getStats(self: *const SummarizingMemory) MemoryStats {
        return .{
            .message_count = self.messages.items.len,
            .total_tokens = self.total_tokens,
            .memory_type = .summarizing,
            .capacity = self.config.max_messages,
            .utilization = if (self.config.max_messages > 0)
                @as(f64, @floatFromInt(self.messages.items.len)) /
                    @as(f64, @floatFromInt(self.config.max_messages))
            else
                0,
        };
    }

    /// Get compression ratio (original tokens / current tokens).
    pub fn getCompressionRatio(self: *const SummarizingMemory) f64 {
        var original_tokens: usize = self.total_tokens;

        for (self.summaries.items) |s| {
            original_tokens += s.original_tokens - s.summary_tokens;
        }

        if (self.total_tokens == 0) return 1.0;
        return @as(f64, @floatFromInt(original_tokens)) /
            @as(f64, @floatFromInt(self.total_tokens));
    }
};

test "summarizing memory basic operations" {
    const allocator = std.testing.allocator;
    var memory = SummarizingMemory.init(allocator, .{
        .max_messages = 10,
        .keep_recent = 3,
    });
    defer memory.deinit();

    try memory.add(Message.user("Hello"));
    try memory.add(Message.assistant("Hi!"));

    try std.testing.expectEqual(@as(usize, 2), memory.messages.items.len);
}

test "summarizing memory auto-summarization" {
    const allocator = std.testing.allocator;
    var memory = SummarizingMemory.init(allocator, .{
        .max_messages = 5,
        .keep_recent = 2,
    });
    defer memory.deinit();

    // Add more than max_messages
    var i: usize = 0;
    while (i < 7) : (i += 1) {
        try memory.add(Message.user("Message"));
        try memory.add(Message.assistant("Response"));
    }

    // Should have triggered summarization
    try std.testing.expect(memory.summaries.items.len > 0);
    try std.testing.expect(memory.messages.items.len <= 5);
}

test "summarizing memory context retrieval" {
    const allocator = std.testing.allocator;
    var memory = SummarizingMemory.init(allocator, .{
        .max_messages = 5,
        .keep_recent = 2,
    });
    defer memory.deinit();

    try memory.setSystemMessage(Message.system("You are helpful."));
    try memory.add(Message.user("Hello"));
    try memory.add(Message.assistant("Hi!"));

    const context = try memory.getContext(allocator);
    defer allocator.free(context);

    try std.testing.expect(context.len >= 3);
    try std.testing.expectEqual(MessageRole.system, context[0].role);
}
