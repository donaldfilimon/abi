//! Abbey Context Management
//!
//! Provides advanced context tracking:
//! - Topic tracking and threading
//! - Conversation arcs and themes
//! - Unresolved question tracking
//! - Context-aware retrieval

const std = @import("std");
const platform_time = @import("../../shared/time.zig");

// Platform-aware time function (works on WASM)
fn getTimestamp() i64 {
    return @intCast(platform_time.timestampSec());
}

/// A topic being discussed
pub const Topic = struct {
    name: []const u8,
    first_mentioned: i64,
    last_mentioned: i64,
    mention_count: usize,
    relevance_score: f32,
    subtopics: std.ArrayListUnmanaged([]const u8),

    /// Check if topic is still active (mentioned recently)
    pub fn isActive(self: *const Topic, current_time: i64) bool {
        const recency_threshold = 300; // 5 minutes
        return (current_time - self.last_mentioned) < recency_threshold;
    }

    /// Update topic mention
    pub fn mention(self: *Topic) void {
        self.last_mentioned = getTimestamp();
        self.mention_count += 1;
        // Boost relevance on mention
        self.relevance_score = @min(1.0, self.relevance_score + 0.1);
    }

    /// Decay relevance over time
    pub fn decay(self: *Topic, amount: f32) void {
        self.relevance_score = @max(0.0, self.relevance_score - amount);
    }
};

/// Topic tracker for conversation threading
pub const TopicTracker = struct {
    allocator: std.mem.Allocator,
    topics: std.StringHashMapUnmanaged(Topic),
    topic_order: std.ArrayListUnmanaged([]const u8),
    current_main_topic: ?[]const u8 = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .topics = .{},
            .topic_order = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.topics.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.subtopics.deinit(self.allocator);
        }
        self.topics.deinit(self.allocator);
        self.topic_order.deinit(self.allocator);
    }

    /// Update topics from a new message
    pub fn updateFromMessage(self: *Self, message: []const u8) !void {
        const topics = try self.extractTopics(message);
        defer self.allocator.free(topics);

        for (topics) |topic| {
            if (self.topics.getPtr(topic)) |existing| {
                existing.mention();
            } else {
                const now = getTimestamp();
                try self.topics.put(self.allocator, topic, .{
                    .name = topic,
                    .first_mentioned = now,
                    .last_mentioned = now,
                    .mention_count = 1,
                    .relevance_score = 0.5,
                    .subtopics = .{},
                });
                try self.topic_order.append(self.allocator, topic);
            }
        }

        // Update main topic
        self.updateMainTopic();

        // Decay old topics
        self.decayTopics();
    }

    /// Extract topics from text (simple keyword extraction)
    fn extractTopics(self: *Self, text: []const u8) ![][]const u8 {
        var topics = std.ArrayListUnmanaged([]const u8){};
        errdefer topics.deinit(self.allocator);

        // Technical topic patterns
        const tech_patterns = [_][]const u8{
            "zig",    "rust",   "python",           "javascript", "typescript",  "go",        "golang",
            "llm",    "ai",     "machine learning", "neural",     "transformer", "gpu",       "cuda",
            "vulkan", "memory", "allocator",        "database",   "vector",      "embedding", "search",
            "api",    "http",   "server",           "client",     "network",     "compile",   "build",
            "test",   "debug",
        };

        var lower_buf: [2048]u8 = undefined;
        const len = @min(text.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(text[i]);
        }
        const lower = lower_buf[0..len];

        for (tech_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                try topics.append(self.allocator, pattern);
            }
        }

        return topics.toOwnedSlice(self.allocator);
    }

    /// Update the main topic based on relevance
    fn updateMainTopic(self: *Self) void {
        var max_relevance: f32 = 0;
        var main: ?[]const u8 = null;

        var it = self.topics.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.relevance_score > max_relevance) {
                max_relevance = entry.value_ptr.relevance_score;
                main = entry.key_ptr.*;
            }
        }

        self.current_main_topic = main;
    }

    /// Decay topic relevance
    fn decayTopics(self: *Self) void {
        var it = self.topics.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.decay(0.05);
        }
    }

    /// Get current active topics
    pub fn getCurrentTopics(self: *const Self) []const []const u8 {
        return self.topic_order.items;
    }

    /// Get topic count
    pub fn getTopicCount(self: *const Self) usize {
        return self.topics.count();
    }

    /// Get main topic
    pub fn getMainTopic(self: *const Self) ?[]const u8 {
        return self.current_main_topic;
    }

    /// Check if a topic has been discussed
    pub fn hasDiscussed(self: *const Self, topic: []const u8) bool {
        return self.topics.contains(topic);
    }

    /// Clear all topics
    pub fn clear(self: *Self) void {
        var it = self.topics.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.subtopics.deinit(self.allocator);
        }
        self.topics.clearRetainingCapacity();
        self.topic_order.clearRetainingCapacity();
        self.current_main_topic = null;
    }
};

/// Conversation context with history and themes
pub const ConversationContext = struct {
    allocator: std.mem.Allocator,
    conversation_id: ?[]const u8 = null,
    started: i64,
    last_activity: i64,
    turn_count: usize = 0,
    unresolved_questions: std.ArrayListUnmanaged(UnresolvedQuestion),
    themes: std.ArrayListUnmanaged([]const u8),
    context_summary: ?[]const u8 = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        const now = getTimestamp();
        return .{
            .allocator = allocator,
            .started = now,
            .last_activity = now,
            .unresolved_questions = .{},
            .themes = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.unresolved_questions.deinit(self.allocator);
        self.themes.deinit(self.allocator);
        if (self.context_summary) |s| {
            self.allocator.free(s);
        }
        if (self.conversation_id) |id| {
            self.allocator.free(id);
        }
    }

    /// Record a new turn
    pub fn recordTurn(self: *Self) void {
        self.turn_count += 1;
        self.last_activity = getTimestamp();
    }

    /// Add an unresolved question
    pub fn addUnresolvedQuestion(self: *Self, question: []const u8, reason: []const u8) !void {
        try self.unresolved_questions.append(self.allocator, .{
            .question = question,
            .reason = reason,
            .turn_asked = self.turn_count,
        });
    }

    /// Mark a question as resolved
    pub fn resolveQuestion(self: *Self, index: usize) void {
        if (index < self.unresolved_questions.items.len) {
            _ = self.unresolved_questions.orderedRemove(index);
        }
    }

    /// Check if there are pending questions
    pub fn hasUnresolved(self: *const Self) bool {
        return self.unresolved_questions.items.len > 0;
    }

    /// Get conversation duration
    pub fn getDuration(self: *const Self) i64 {
        return self.last_activity - self.started;
    }

    /// Check if conversation is stale
    pub fn isStale(self: *const Self) bool {
        const now = getTimestamp();
        const stale_threshold = 1800; // 30 minutes
        return (now - self.last_activity) > stale_threshold;
    }

    /// Get context state
    pub fn getState(self: *const Self) ContextState {
        return .{
            .turn_count = self.turn_count,
            .duration_seconds = self.getDuration(),
            .unresolved_count = self.unresolved_questions.items.len,
            .theme_count = self.themes.items.len,
            .is_stale = self.isStale(),
        };
    }

    /// Clear context (but keep conversation ID)
    pub fn clear(self: *Self) void {
        self.unresolved_questions.clearRetainingCapacity();
        self.themes.clearRetainingCapacity();
        if (self.context_summary) |s| {
            self.allocator.free(s);
            self.context_summary = null;
        }
        self.turn_count = 0;
        const now = getTimestamp();
        self.started = now;
        self.last_activity = now;
    }
};

/// An unresolved question from the conversation
pub const UnresolvedQuestion = struct {
    question: []const u8,
    reason: []const u8,
    turn_asked: usize,
};

/// Context state summary
pub const ContextState = struct {
    turn_count: usize,
    duration_seconds: i64,
    unresolved_count: usize,
    theme_count: usize,
    is_stale: bool,
};

/// Intelligent context window manager
pub const ContextWindow = struct {
    allocator: std.mem.Allocator,
    max_tokens: usize,
    messages: std.ArrayListUnmanaged(ContextMessage),
    current_tokens: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_tokens: usize) Self {
        return .{
            .allocator = allocator,
            .max_tokens = max_tokens,
            .messages = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.messages.deinit(self.allocator);
    }

    /// Add a message, evicting old ones if needed
    pub fn addMessage(self: *Self, role: []const u8, content: []const u8, importance: f32) !void {
        const tokens = estimateTokens(content);

        // Evict until we have room
        while (self.current_tokens + tokens > self.max_tokens and self.messages.items.len > 0) {
            const evicted = self.messages.orderedRemove(0);
            self.current_tokens -= evicted.token_count;
        }

        try self.messages.append(self.allocator, .{
            .role = role,
            .content = content,
            .token_count = tokens,
            .importance = importance,
        });
        self.current_tokens += tokens;
    }

    /// Get messages for context
    pub fn getMessages(self: *const Self) []const ContextMessage {
        return self.messages.items;
    }

    /// Estimate token count (rough approximation)
    fn estimateTokens(text: []const u8) usize {
        // Rough estimate: 4 characters per token
        return (text.len + 3) / 4;
    }

    /// Get available token budget
    pub fn getAvailableTokens(self: *const Self) usize {
        return if (self.max_tokens > self.current_tokens)
            self.max_tokens - self.current_tokens
        else
            0;
    }
};

/// Message in context window
pub const ContextMessage = struct {
    role: []const u8,
    content: []const u8,
    token_count: usize,
    importance: f32,
};

// ============================================================================
// Tests
// ============================================================================

test "topic tracker" {
    const allocator = std.testing.allocator;

    var tracker = TopicTracker.init(allocator);
    defer tracker.deinit();

    try tracker.updateFromMessage("Let's discuss Zig programming and memory allocation");
    try std.testing.expect(tracker.hasDiscussed("zig"));
    try std.testing.expect(tracker.hasDiscussed("memory"));
    try std.testing.expect(tracker.hasDiscussed("allocator"));
}

test "conversation context" {
    const allocator = std.testing.allocator;

    var ctx = ConversationContext.init(allocator);
    defer ctx.deinit();

    ctx.recordTurn();
    ctx.recordTurn();
    try std.testing.expectEqual(@as(usize, 2), ctx.turn_count);

    try ctx.addUnresolvedQuestion("What about X?", "User asked but we moved on");
    try std.testing.expect(ctx.hasUnresolved());

    ctx.resolveQuestion(0);
    try std.testing.expect(!ctx.hasUnresolved());
}

test "context window" {
    const allocator = std.testing.allocator;

    var window = ContextWindow.init(allocator, 100);
    defer window.deinit();

    try window.addMessage("user", "Hello", 0.5);
    try window.addMessage("assistant", "Hi there!", 0.5);

    try std.testing.expectEqual(@as(usize, 2), window.messages.items.len);
    try std.testing.expect(window.current_tokens < window.max_tokens);
}
