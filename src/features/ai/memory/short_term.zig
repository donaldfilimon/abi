//! Short-term memory implementation.
//!
//! Provides a fixed-capacity buffer for recent messages.
//! When capacity is exceeded, oldest messages are discarded.

const std = @import("std");
const mod = @import("mod.zig");
const Message = mod.Message;
const MessageRole = mod.MessageRole;
const MemoryStats = mod.MemoryStats;
const MemoryType = @import("manager.zig").MemoryType;

/// Short-term memory with fixed message capacity.
pub const ShortTermMemory = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayListUnmanaged(Message),
    capacity: usize,
    total_tokens: usize,
    owns_messages: bool,

    /// Initialize short-term memory.
    pub fn init(allocator: std.mem.Allocator, capacity: usize) ShortTermMemory {
        return .{
            .allocator = allocator,
            .messages = .{},
            .capacity = capacity,
            .total_tokens = 0,
            .owns_messages = false,
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *ShortTermMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
        }
        self.messages.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a message to memory.
    pub fn add(self: *ShortTermMemory, message: Message) !void {
        // Evict oldest if at capacity
        while (self.messages.items.len >= self.capacity) {
            if (self.messages.items.len > 0) {
                const removed = self.messages.orderedRemove(0);
                self.total_tokens -|= removed.estimateTokens();
                if (self.owns_messages) {
                    var msg = removed;
                    msg.deinit(self.allocator);
                }
            } else {
                break;
            }
        }

        // Clone message if we own messages
        const msg_to_add = if (self.owns_messages)
            try message.clone(self.allocator)
        else
            message;

        try self.messages.append(self.allocator, msg_to_add);
        self.total_tokens += message.estimateTokens();
    }

    /// Add a message and take ownership.
    pub fn addOwned(self: *ShortTermMemory, message: Message) !void {
        self.owns_messages = true;
        try self.add(message);
    }

    /// Get all messages.
    pub fn getMessages(self: *const ShortTermMemory) []const Message {
        return self.messages.items;
    }

    /// Get last N messages.
    pub fn getLastN(self: *const ShortTermMemory, n: usize) []const Message {
        const msg_count = @min(n, self.messages.items.len);
        return self.messages.items[self.messages.items.len - msg_count ..];
    }

    /// Get messages by role.
    pub fn getByRole(
        self: *const ShortTermMemory,
        role: MessageRole,
        allocator: std.mem.Allocator,
    ) ![]Message {
        var result = std.ArrayListUnmanaged(Message).empty;
        errdefer result.deinit(allocator);

        for (self.messages.items) |msg| {
            if (msg.role == role) {
                try result.append(allocator, msg);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Clear all messages.
    pub fn clear(self: *ShortTermMemory) void {
        if (self.owns_messages) {
            for (self.messages.items) |*msg| {
                msg.deinit(self.allocator);
            }
        }
        self.messages.clearRetainingCapacity();
        self.total_tokens = 0;
    }

    /// Get memory statistics.
    pub fn getStats(self: *const ShortTermMemory) MemoryStats {
        return .{
            .message_count = self.messages.items.len,
            .total_tokens = self.total_tokens,
            .memory_type = .short_term,
            .capacity = self.capacity,
            .utilization = if (self.capacity > 0)
                @as(f64, @floatFromInt(self.messages.items.len)) /
                    @as(f64, @floatFromInt(self.capacity))
            else
                0,
        };
    }

    /// Check if memory is empty.
    pub fn isEmpty(self: *const ShortTermMemory) bool {
        return self.messages.items.len == 0;
    }

    /// Get message count.
    pub fn count(self: *const ShortTermMemory) usize {
        return self.messages.items.len;
    }

    /// Format messages for display.
    pub fn format(
        self: *const ShortTermMemory,
        allocator: std.mem.Allocator,
    ) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        for (self.messages.items) |msg| {
            try result.appendSlice(allocator, msg.role.toString());
            try result.appendSlice(allocator, ": ");
            try result.appendSlice(allocator, msg.content);
            try result.append(allocator, '\n');
        }

        return result.toOwnedSlice(allocator);
    }
};

test "short-term memory basic operations" {
    const allocator = std.testing.allocator;
    var memory = ShortTermMemory.init(allocator, 5);
    defer memory.deinit();

    try memory.add(Message.user("Hello"));
    try memory.add(Message.assistant("Hi!"));

    try std.testing.expectEqual(@as(usize, 2), memory.count());
    try std.testing.expect(!memory.isEmpty());
}

test "short-term memory capacity eviction" {
    const allocator = std.testing.allocator;
    var memory = ShortTermMemory.init(allocator, 3);
    defer memory.deinit();

    try memory.add(Message.user("Message 1"));
    try memory.add(Message.user("Message 2"));
    try memory.add(Message.user("Message 3"));
    try memory.add(Message.user("Message 4"));

    // Should have evicted oldest
    try std.testing.expectEqual(@as(usize, 3), memory.count());

    const messages = memory.getMessages();
    try std.testing.expectEqualStrings("Message 2", messages[0].content);
}

test "short-term memory getLastN" {
    const allocator = std.testing.allocator;
    var memory = ShortTermMemory.init(allocator, 10);
    defer memory.deinit();

    try memory.add(Message.user("A"));
    try memory.add(Message.user("B"));
    try memory.add(Message.user("C"));
    try memory.add(Message.user("D"));

    const last2 = memory.getLastN(2);
    try std.testing.expectEqual(@as(usize, 2), last2.len);
    try std.testing.expectEqualStrings("C", last2[0].content);
    try std.testing.expectEqualStrings("D", last2[1].content);
}

test "short-term memory clear" {
    const allocator = std.testing.allocator;
    var memory = ShortTermMemory.init(allocator, 10);
    defer memory.deinit();

    try memory.add(Message.user("Test"));
    try std.testing.expect(!memory.isEmpty());

    memory.clear();
    try std.testing.expect(memory.isEmpty());
}

test {
    std.testing.refAllDecls(@This());
}
