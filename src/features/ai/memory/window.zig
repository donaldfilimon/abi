//! Sliding window memory implementation.
//!
//! Maintains a token-limited window of messages, removing oldest
//! messages when the token limit is exceeded.

const std = @import("std");
const mod = @import("mod.zig");
const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryStats = mod.MemoryStats;
const MemoryType = @import("manager.zig").MemoryType;

/// Sliding window memory with token-based capacity.
pub const SlidingWindowMemory = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayListUnmanaged(Message),
    max_tokens: usize,
    current_tokens: usize,
    owns_messages: bool,
    /// Reserve tokens for system message.
    system_reserve: usize,
    /// Stored system message (always kept).
    system_message: ?Message,

    /// Initialize sliding window memory.
    pub fn init(allocator: std.mem.Allocator, max_tokens: usize) SlidingWindowMemory {
        return .{
            .allocator = allocator,
            .messages = .{},
            .max_tokens = max_tokens,
            .current_tokens = 0,
            .owns_messages = false,
            .system_reserve = 0,
            .system_message = null,
        };
    }

    /// Initialize with system message reserve.
    pub fn initWithReserve(
        allocator: std.mem.Allocator,
        max_tokens: usize,
        system_reserve: usize,
    ) SlidingWindowMemory {
        return .{
            .allocator = allocator,
            .messages = .{},
            .max_tokens = max_tokens,
            .current_tokens = 0,
            .owns_messages = false,
            .system_reserve = system_reserve,
            .system_message = null,
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *SlidingWindowMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
            if (self.system_message) |*sys| {
                sys.deinit(self.allocator);
            }
        }
        self.messages.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set the system message (always retained).
    pub fn setSystemMessage(self: *SlidingWindowMemory, message: Message) !void {
        if (self.system_message) |*old| {
            self.current_tokens -|= old.estimateTokens();
            if (self.owns_messages) {
                old.deinit(self.allocator);
            }
        }

        const msg = if (self.owns_messages)
            try message.clone(self.allocator)
        else
            message;

        self.system_message = msg;
        self.current_tokens += msg.estimateTokens();
    }

    /// Add a message to the window.
    pub fn add(self: *SlidingWindowMemory, message: Message) !void {
        const msg_tokens = message.estimateTokens();
        const available = self.max_tokens -| self.system_reserve;

        // Evict oldest messages until we have room
        while (self.current_tokens + msg_tokens > available and self.messages.items.len > 0) {
            const removed = self.messages.orderedRemove(0);
            self.current_tokens -|= removed.estimateTokens();
            if (self.owns_messages) {
                var msg = removed;
                msg.deinit(self.allocator);
            }
        }

        // Skip if single message exceeds capacity
        if (msg_tokens > available) {
            return;
        }

        const msg_to_add = if (self.owns_messages)
            try message.clone(self.allocator)
        else
            message;

        try self.messages.append(self.allocator, msg_to_add);
        self.current_tokens += msg_tokens;
    }

    /// Add a message and take ownership.
    pub fn addOwned(self: *SlidingWindowMemory, message: Message) !void {
        self.owns_messages = true;
        try self.add(message);
    }

    /// Get all messages including system message.
    pub fn getMessages(
        self: *const SlidingWindowMemory,
        allocator: std.mem.Allocator,
    ) ![]Message {
        var result = std.ArrayListUnmanaged(Message){};
        errdefer result.deinit(allocator);

        if (self.system_message) |sys| {
            try result.append(allocator, sys);
        }

        for (self.messages.items) |msg| {
            try result.append(allocator, msg);
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get conversation messages (without system message).
    pub fn getConversationMessages(self: *const SlidingWindowMemory) []const Message {
        return self.messages.items;
    }

    /// Get messages within a token budget.
    pub fn getWithinBudget(
        self: *const SlidingWindowMemory,
        budget: usize,
        allocator: std.mem.Allocator,
    ) ![]Message {
        var result = std.ArrayListUnmanaged(Message){};
        errdefer result.deinit(allocator);

        var remaining = budget;

        // System message first if it fits
        if (self.system_message) |sys| {
            const tokens = sys.estimateTokens();
            if (tokens <= remaining) {
                try result.append(allocator, sys);
                remaining -= tokens;
            }
        }

        // Add from most recent, then reverse
        var temp = std.ArrayListUnmanaged(Message){};
        defer temp.deinit(allocator);

        var i = self.messages.items.len;
        while (i > 0) {
            i -= 1;
            const msg = self.messages.items[i];
            const tokens = msg.estimateTokens();
            if (tokens <= remaining) {
                try temp.append(allocator, msg);
                remaining -= tokens;
            } else {
                break;
            }
        }

        // Reverse to get chronological order
        var j = temp.items.len;
        while (j > 0) {
            j -= 1;
            try result.append(allocator, temp.items[j]);
        }

        return result.toOwnedSlice(allocator);
    }

    /// Clear all messages (keeps system message).
    pub fn clear(self: *SlidingWindowMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
        }
        self.messages.clearRetainingCapacity();

        // Recalculate tokens (only system message remains)
        if (self.system_message) |sys| {
            self.current_tokens = sys.estimateTokens();
        } else {
            self.current_tokens = 0;
        }
    }

    /// Get memory statistics.
    pub fn getStats(self: *const SlidingWindowMemory) MemoryStats {
        const msg_count = self.messages.items.len + (if (self.system_message != null) @as(usize, 1) else 0);
        return .{
            .message_count = msg_count,
            .total_tokens = self.current_tokens,
            .memory_type = .sliding_window,
            .capacity = self.max_tokens,
            .utilization = if (self.max_tokens > 0)
                @as(f64, @floatFromInt(self.current_tokens)) /
                    @as(f64, @floatFromInt(self.max_tokens))
            else
                0,
        };
    }

    /// Check remaining token capacity.
    pub fn remainingTokens(self: *const SlidingWindowMemory) usize {
        return self.max_tokens -| self.current_tokens;
    }

    /// Check if window is empty.
    pub fn isEmpty(self: *const SlidingWindowMemory) bool {
        return self.messages.items.len == 0 and self.system_message == null;
    }
};

test "sliding window basic operations" {
    const allocator = std.testing.allocator;
    var memory = SlidingWindowMemory.init(allocator, 1000);
    defer memory.deinit();

    try memory.add(Message.user("Hello"));
    try memory.add(Message.assistant("Hi there!"));

    const messages = memory.getConversationMessages();
    try std.testing.expectEqual(@as(usize, 2), messages.len);
}

test "sliding window token eviction" {
    const allocator = std.testing.allocator;
    // Very small token limit
    var memory = SlidingWindowMemory.init(allocator, 20);
    defer memory.deinit();

    // Each message is ~2-3 tokens
    try memory.add(Message.user("Hello"));
    try memory.add(Message.user("World"));
    try memory.add(Message.user("Test"));
    try memory.add(Message.user("Message"));

    // Should have evicted some messages
    try std.testing.expect(memory.current_tokens <= 20);
}

test "sliding window system message" {
    const allocator = std.testing.allocator;
    var memory = SlidingWindowMemory.init(allocator, 100);
    defer memory.deinit();

    try memory.setSystemMessage(Message.system("You are helpful."));
    try memory.add(Message.user("Hello"));

    const all = try memory.getMessages(allocator);
    defer allocator.free(all);

    try std.testing.expectEqual(@as(usize, 2), all.len);
    try std.testing.expectEqual(MessageRole.system, all[0].role);
}

test "sliding window get within budget" {
    const allocator = std.testing.allocator;
    var memory = SlidingWindowMemory.init(allocator, 1000);
    defer memory.deinit();

    try memory.add(Message.user("Message 1"));
    try memory.add(Message.user("Message 2"));
    try memory.add(Message.user("Message 3"));

    // Get with very small budget - should only get most recent
    // Each message is ~3 tokens ("Message N" = 9 chars / 4 ≈ 3), so budget=5 fits only 1
    const recent = try memory.getWithinBudget(5, allocator);
    defer allocator.free(recent);

    try std.testing.expect(recent.len >= 1);
    try std.testing.expect(recent.len <= 2);
}
