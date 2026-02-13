//! Unified memory manager interface.
//!
//! Combines multiple memory types (short-term, sliding window, summarizing,
//! long-term) into a single interface for conversation management.

const std = @import("std");
const mod = @import("mod.zig");
const short_term = @import("short_term.zig");
const window = @import("window.zig");
const summary = @import("summary.zig");
const long_term = @import("long_term.zig");

const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryStats = mod.MemoryStats;
const ShortTermMemory = short_term.ShortTermMemory;
const SlidingWindowMemory = window.SlidingWindowMemory;
const SummarizingMemory = summary.SummarizingMemory;
const LongTermMemory = long_term.LongTermMemory;
const RetrievalResult = long_term.RetrievalResult;

/// Memory type enum.
pub const MemoryType = enum {
    short_term,
    sliding_window,
    summarizing,
    long_term,
    hybrid,
};

/// Memory manager configuration.
pub const MemoryConfig = struct {
    /// Primary memory type.
    primary_type: MemoryType = .sliding_window,
    /// Short-term capacity (messages).
    short_term_capacity: usize = 50,
    /// Sliding window max tokens.
    sliding_window_tokens: usize = 4000,
    /// System message token reserve.
    system_reserve: usize = 500,
    /// Summarization config.
    summary_config: summary.SummaryConfig = .{},
    /// Long-term config.
    long_term_config: long_term.LongTermConfig = .{},
    /// Enable long-term memory alongside primary.
    enable_long_term: bool = false,
    /// Auto-store messages to long-term memory.
    auto_store_long_term: bool = true,
    /// Importance threshold for long-term storage.
    long_term_importance_threshold: f32 = 0.6,
};

/// Unified memory manager.
pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    config: MemoryConfig,
    short_term: ?ShortTermMemory,
    sliding_window: ?SlidingWindowMemory,
    summarizing: ?SummarizingMemory,
    long_term_mem: ?LongTermMemory,
    message_count: u64,

    /// Initialize memory manager.
    pub fn init(allocator: std.mem.Allocator, config: MemoryConfig) MemoryManager {
        var manager = MemoryManager{
            .allocator = allocator,
            .config = config,
            .short_term = null,
            .sliding_window = null,
            .summarizing = null,
            .long_term_mem = null,
            .message_count = 0,
        };

        // Initialize primary memory
        switch (config.primary_type) {
            .short_term => {
                manager.short_term = ShortTermMemory.init(allocator, config.short_term_capacity);
            },
            .sliding_window => {
                manager.sliding_window = SlidingWindowMemory.initWithReserve(
                    allocator,
                    config.sliding_window_tokens,
                    config.system_reserve,
                );
            },
            .summarizing => {
                manager.summarizing = SummarizingMemory.init(allocator, config.summary_config);
            },
            .long_term => {
                manager.long_term_mem = LongTermMemory.init(allocator, config.long_term_config);
            },
            .hybrid => {
                manager.sliding_window = SlidingWindowMemory.initWithReserve(
                    allocator,
                    config.sliding_window_tokens,
                    config.system_reserve,
                );
                manager.long_term_mem = LongTermMemory.init(allocator, config.long_term_config);
            },
        }

        // Initialize long-term if enabled separately
        if (config.enable_long_term and manager.long_term_mem == null) {
            manager.long_term_mem = LongTermMemory.init(allocator, config.long_term_config);
        }

        return manager;
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *MemoryManager) void {
        if (self.short_term) |*st| st.deinit();
        if (self.sliding_window) |*sw| sw.deinit();
        if (self.summarizing) |*sm| sm.deinit();
        if (self.long_term_mem) |*lt| lt.deinit();
        self.* = undefined;
    }

    /// Set the system message.
    pub fn setSystemMessage(self: *MemoryManager, message: Message) !void {
        if (self.sliding_window) |*sw| {
            try sw.setSystemMessage(message);
        }
        if (self.summarizing) |*sm| {
            try sm.setSystemMessage(message);
        }
    }

    /// Add a message to memory.
    pub fn add(self: *MemoryManager, message: Message) !void {
        self.message_count += 1;

        // Add to primary memory
        if (self.short_term) |*st| {
            try st.add(message);
        }
        if (self.sliding_window) |*sw| {
            try sw.add(message);
        }
        if (self.summarizing) |*sm| {
            try sm.add(message);
        }

        // Auto-store to long-term if enabled
        if (self.config.auto_store_long_term) {
            if (self.long_term_mem) |*lt| {
                const importance = self.calculateImportance(message);
                if (importance >= self.config.long_term_importance_threshold) {
                    try lt.store(message, null, importance);
                }
            }
        }
    }

    /// Add user message.
    pub fn addUserMessage(self: *MemoryManager, content: []const u8) !void {
        try self.add(Message.user(content));
    }

    /// Add assistant message.
    pub fn addAssistantMessage(self: *MemoryManager, content: []const u8) !void {
        try self.add(Message.assistant(content));
    }

    /// Get conversation context for LLM input.
    pub fn getContext(self: *MemoryManager, max_tokens: ?usize) ![]Message {
        if (self.sliding_window) |*sw| {
            if (max_tokens) |budget| {
                return sw.getWithinBudget(budget, self.allocator);
            }
            return sw.getMessages(self.allocator);
        }
        if (self.summarizing) |*sm| {
            return sm.getContext(self.allocator);
        }
        if (self.short_term) |st| {
            // Clone messages for return
            var result = std.ArrayListUnmanaged(Message){};
            for (st.getMessages()) |msg| {
                try result.append(self.allocator, msg);
            }
            return result.toOwnedSlice(self.allocator);
        }

        return &[_]Message{};
    }

    /// Retrieve relevant memories from long-term storage.
    pub fn retrieveRelevant(
        self: *MemoryManager,
        query: []const u8,
        top_k: ?usize,
    ) ![]RetrievalResult {
        if (self.long_term_mem) |*lt| {
            return lt.retrieve(query, top_k);
        }
        return &[_]RetrievalResult{};
    }

    /// Get hybrid context (recent + relevant memories).
    pub fn getHybridContext(
        self: *MemoryManager,
        query: []const u8,
        max_tokens: usize,
        relevance_slots: usize,
    ) !HybridContext {
        var context = HybridContext{
            .system_message = null,
            .relevant_memories = &[_]RetrievalResult{},
            .recent_messages = &[_]Message{},
            .total_tokens = 0,
        };

        // Get relevant memories first
        if (self.long_term_mem) |*lt| {
            context.relevant_memories = try lt.retrieve(query, relevance_slots);
        }

        // Estimate tokens used by relevant memories
        var relevant_tokens: usize = 0;
        for (context.relevant_memories) |result| {
            relevant_tokens += result.message.estimateTokens();
        }

        // Get recent context with remaining budget
        const recent_budget = max_tokens -| relevant_tokens;
        if (self.sliding_window) |*sw| {
            context.recent_messages = try sw.getWithinBudget(recent_budget, self.allocator);
            if (sw.system_message) |sys| {
                context.system_message = sys;
            }
        }

        context.total_tokens = relevant_tokens;
        for (context.recent_messages) |msg| {
            context.total_tokens += msg.estimateTokens();
        }

        return context;
    }

    /// Clear all memories.
    pub fn clear(self: *MemoryManager) void {
        if (self.short_term) |*st| st.clear();
        if (self.sliding_window) |*sw| sw.clear();
        if (self.summarizing) |*sm| sm.clear();
        if (self.long_term_mem) |*lt| lt.clear();
        self.message_count = 0;
    }

    /// Get combined memory statistics.
    pub fn getStats(self: *const MemoryManager) CombinedStats {
        return .{
            .short_term = if (self.short_term) |st| st.getStats() else null,
            .sliding_window = if (self.sliding_window) |sw| sw.getStats() else null,
            .summarizing = if (self.summarizing) |sm| sm.getStats() else null,
            .long_term = if (self.long_term_mem) |lt| lt.getStats() else null,
            .total_messages = self.message_count,
            .primary_type = self.config.primary_type,
        };
    }

    /// Calculate importance score for a message.
    fn calculateImportance(self: *const MemoryManager, message: Message) f32 {
        _ = self;
        var importance: f32 = 0.5;

        // Longer messages may be more important
        const tokens = message.estimateTokens();
        if (tokens > 50) importance += 0.1;
        if (tokens > 100) importance += 0.1;

        // User messages slightly more important for retrieval
        if (message.role == .user) importance += 0.1;

        // System messages very important
        if (message.role == .system) importance = 0.9;

        // Cap at 1.0
        return @min(importance, 1.0);
    }
};

/// Hybrid context result.
pub const HybridContext = struct {
    system_message: ?Message,
    relevant_memories: []RetrievalResult,
    recent_messages: []Message,
    total_tokens: usize,

    pub fn deinit(self: *HybridContext, allocator: std.mem.Allocator) void {
        allocator.free(self.relevant_memories);
        allocator.free(self.recent_messages);
        self.* = undefined;
    }
};

/// Combined statistics from all memory types.
pub const CombinedStats = struct {
    short_term: ?MemoryStats,
    sliding_window: ?MemoryStats,
    summarizing: ?MemoryStats,
    long_term: ?MemoryStats,
    total_messages: u64,
    primary_type: MemoryType,

    pub fn format(
        self: CombinedStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("MemoryManager (primary={t}, total={d})", .{
            self.primary_type,
            self.total_messages,
        });
    }
};

test "memory manager basic operations" {
    const allocator = std.testing.allocator;
    var manager = MemoryManager.init(allocator, .{
        .primary_type = .sliding_window,
        .sliding_window_tokens = 1000,
    });
    defer manager.deinit();

    try manager.addUserMessage("Hello!");
    try manager.addAssistantMessage("Hi there!");

    const context = try manager.getContext(null);
    defer allocator.free(context);

    try std.testing.expectEqual(@as(usize, 2), context.len);
}

test "memory manager with system message" {
    const allocator = std.testing.allocator;
    var manager = MemoryManager.init(allocator, .{
        .primary_type = .sliding_window,
    });
    defer manager.deinit();

    try manager.setSystemMessage(Message.system("You are helpful."));
    try manager.addUserMessage("Hello");

    const context = try manager.getContext(null);
    defer allocator.free(context);

    try std.testing.expect(context.len >= 2);
    try std.testing.expectEqual(MessageRole.system, context[0].role);
}

test "memory manager hybrid mode" {
    const allocator = std.testing.allocator;
    var manager = MemoryManager.init(allocator, .{
        .primary_type = .hybrid,
        .sliding_window_tokens = 1000,
        .system_reserve = 0,
        .long_term_config = .{
            .max_memories = 100,
            .embedding_dim = 32,
            .min_similarity = 0,
        },
    });
    defer manager.deinit();

    try manager.addUserMessage("What is machine learning?");
    try manager.addAssistantMessage("Machine learning is...");

    var hybrid = try manager.getHybridContext("machine learning", 1000, 3);
    defer hybrid.deinit(allocator);

    try std.testing.expect(hybrid.recent_messages.len > 0);
}

test "memory manager statistics" {
    const allocator = std.testing.allocator;
    var manager = MemoryManager.init(allocator, .{});
    defer manager.deinit();

    try manager.addUserMessage("Test");

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_messages);
    try std.testing.expectEqual(MemoryType.sliding_window, stats.primary_type);
}
