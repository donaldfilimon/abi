//! Chat memory system for maintaining conversation context.
//!
//! Provides multiple memory strategies for managing conversation history:
//! - Short-term memory: Recent message buffer with fixed capacity
//! - Sliding window: Token-based window for context management
//! - Summarizing memory: Compresses old messages into summaries
//! - Long-term memory: Vector-based retrieval for relevant past context

const std = @import("std");
const short_term = @import("short_term.zig");
const window = @import("window.zig");
const summary = @import("summary.zig");
const long_term = @import("long_term.zig");
const manager = @import("manager.zig");
pub const persistence = @import("persistence.zig");

pub const ShortTermMemory = short_term.ShortTermMemory;
pub const SlidingWindowMemory = window.SlidingWindowMemory;
pub const SummarizingMemory = summary.SummarizingMemory;
pub const LongTermMemory = long_term.LongTermMemory;
pub const MemoryManager = manager.MemoryManager;
pub const MemoryConfig = manager.MemoryConfig;
pub const MemoryType = manager.MemoryType;

// Persistence types
pub const SessionStore = persistence.SessionStore;
pub const SessionData = persistence.SessionData;
pub const SessionMeta = persistence.SessionMeta;
pub const SessionConfig = persistence.SessionConfig;
pub const PersistenceError = persistence.PersistenceError;

/// Message role in a conversation.
pub const MessageRole = enum {
    system,
    user,
    assistant,
    tool,

    pub fn toString(self: MessageRole) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

/// A message in a conversation.
pub const Message = struct {
    /// Message role.
    role: MessageRole,
    /// Message content.
    content: []const u8,
    /// Optional message name (for tool calls).
    name: ?[]const u8 = null,
    /// Timestamp when message was added.
    timestamp: i64 = 0,
    /// Estimated token count.
    token_count: usize = 0,
    /// Optional metadata.
    metadata: ?[]const u8 = null,

    /// Create a user message.
    pub fn user(content: []const u8) Message {
        return .{
            .role = .user,
            .content = content,
        };
    }

    /// Create an assistant message.
    pub fn assistant(content: []const u8) Message {
        return .{
            .role = .assistant,
            .content = content,
        };
    }

    /// Create a system message.
    pub fn system(content: []const u8) Message {
        return .{
            .role = .system,
            .content = content,
        };
    }

    /// Create a tool message.
    pub fn tool(name: []const u8, content: []const u8) Message {
        return .{
            .role = .tool,
            .content = content,
            .name = name,
        };
    }

    /// Estimate token count (simple approximation: ~4 chars per token).
    pub fn estimateTokens(self: Message) usize {
        if (self.token_count > 0) return self.token_count;
        return (self.content.len + 3) / 4;
    }

    /// Deep copy a message.
    pub fn clone(self: Message, allocator: std.mem.Allocator) !Message {
        return .{
            .role = self.role,
            .content = try allocator.dupe(u8, self.content),
            .name = if (self.name) |n| try allocator.dupe(u8, n) else null,
            .timestamp = self.timestamp,
            .token_count = self.token_count,
            .metadata = if (self.metadata) |m| try allocator.dupe(u8, m) else null,
        };
    }

    /// Free cloned message resources.
    pub fn deinit(self: *Message, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.name) |n| allocator.free(n);
        if (self.metadata) |m| allocator.free(m);
        self.* = undefined;
    }
};

/// Conversation context for memory retrieval.
pub const ConversationContext = struct {
    /// Messages in context.
    messages: []const Message,
    /// Total token count.
    total_tokens: usize,
    /// Whether context was truncated.
    truncated: bool,
    /// Number of messages before truncation.
    original_count: usize,
};

/// Memory statistics.
pub const MemoryStats = struct {
    /// Number of messages stored.
    message_count: usize,
    /// Total tokens across all messages.
    total_tokens: usize,
    /// Memory type.
    memory_type: MemoryType,
    /// Maximum capacity (messages or tokens).
    capacity: usize,
    /// Current utilization (0-1).
    utilization: f64,
};

/// Create a new short-term memory buffer.
pub fn createShortTermMemory(
    allocator: std.mem.Allocator,
    capacity: usize,
) ShortTermMemory {
    return ShortTermMemory.init(allocator, capacity);
}

/// Create a new sliding window memory.
pub fn createSlidingWindowMemory(
    allocator: std.mem.Allocator,
    max_tokens: usize,
) SlidingWindowMemory {
    return SlidingWindowMemory.init(allocator, max_tokens);
}

/// Create a new summarizing memory.
pub fn createSummarizingMemory(
    allocator: std.mem.Allocator,
    config: summary.SummaryConfig,
) SummarizingMemory {
    return SummarizingMemory.init(allocator, config);
}

/// Create a new long-term memory.
pub fn createLongTermMemory(
    allocator: std.mem.Allocator,
    config: long_term.LongTermConfig,
) LongTermMemory {
    return LongTermMemory.init(allocator, config);
}

/// Create a memory manager with default configuration.
pub fn createMemoryManager(allocator: std.mem.Allocator) MemoryManager {
    return MemoryManager.init(allocator, .{});
}

/// Create a memory manager with custom configuration.
pub fn createMemoryManagerWithConfig(
    allocator: std.mem.Allocator,
    config: MemoryConfig,
) MemoryManager {
    return MemoryManager.init(allocator, config);
}

test "message creation" {
    const user_msg = Message.user("Hello!");
    try std.testing.expectEqual(MessageRole.user, user_msg.role);
    try std.testing.expectEqualStrings("Hello!", user_msg.content);

    const assistant_msg = Message.assistant("Hi there!");
    try std.testing.expectEqual(MessageRole.assistant, assistant_msg.role);

    const system_msg = Message.system("You are a helpful assistant.");
    try std.testing.expectEqual(MessageRole.system, system_msg.role);

    const tool_msg = Message.tool("calculator", "42");
    try std.testing.expectEqual(MessageRole.tool, tool_msg.role);
    try std.testing.expectEqualStrings("calculator", tool_msg.name.?);
}

test "message token estimation" {
    const msg = Message.user("Hello world!"); // 12 chars
    const estimated = msg.estimateTokens();
    try std.testing.expect(estimated > 0);
    try std.testing.expect(estimated <= 5);
}

test "message clone and deinit" {
    const allocator = std.testing.allocator;

    const original = Message{
        .role = .user,
        .content = "test content",
        .name = "test_name",
        .timestamp = 12345,
        .token_count = 3,
        .metadata = "meta",
    };

    var cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expectEqual(original.role, cloned.role);
    try std.testing.expectEqualStrings(original.content, cloned.content);
    try std.testing.expectEqualStrings(original.name.?, cloned.name.?);
    try std.testing.expectEqual(original.timestamp, cloned.timestamp);
}

test {
    _ = short_term;
    _ = window;
    _ = summary;
    _ = long_term;
    _ = manager;
    _ = persistence;
}

test {
    std.testing.refAllDecls(@This());
}
