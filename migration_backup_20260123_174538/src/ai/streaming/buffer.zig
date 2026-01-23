//! Token buffering for streaming.
//!
//! Provides various buffering strategies for managing token flow
//! in streaming scenarios.

const std = @import("std");
const mod = @import("mod.zig");
const StreamToken = mod.StreamToken;

/// Buffer strategy.
pub const BufferStrategy = enum {
    /// FIFO queue.
    fifo,
    /// LIFO stack.
    lifo,
    /// Priority queue (by token importance).
    priority,
    /// Ring buffer (overwrite oldest).
    ring,
};

/// Buffer configuration.
pub const BufferConfig = struct {
    /// Buffering strategy.
    strategy: BufferStrategy = .fifo,
    /// Maximum buffer capacity.
    capacity: usize = 100,
    /// Flush threshold (auto-flush when reached).
    flush_threshold: usize = 50,
    /// Enable auto-coalescing of consecutive tokens.
    coalesce: bool = false,
    /// Maximum coalesce length.
    max_coalesce_len: usize = 100,
};

/// Token buffer for managing streaming output.
pub const TokenBuffer = struct {
    allocator: std.mem.Allocator,
    config: BufferConfig,
    items: std.ArrayListUnmanaged(StreamToken),
    ring_head: usize,
    ring_tail: usize,
    coalesce_buffer: std.ArrayListUnmanaged(u8),
    total_pushed: u64,
    total_popped: u64,
    total_dropped: u64,

    /// Initialize token buffer.
    pub fn init(allocator: std.mem.Allocator, config: BufferConfig) TokenBuffer {
        return .{
            .allocator = allocator,
            .config = config,
            .items = .{},
            .ring_head = 0,
            .ring_tail = 0,
            .coalesce_buffer = .{},
            .total_pushed = 0,
            .total_popped = 0,
            .total_dropped = 0,
        };
    }

    /// Deinitialize buffer.
    pub fn deinit(self: *TokenBuffer) void {
        for (self.items.items) |*item| {
            self.allocator.free(item.text);
        }
        self.items.deinit(self.allocator);
        self.coalesce_buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Push a token to the buffer.
    pub fn push(self: *TokenBuffer, token: StreamToken) !void {
        return switch (self.config.strategy) {
            .fifo => self.pushFifo(token),
            .lifo => self.pushLifo(token),
            .priority => self.pushPriority(token),
            .ring => self.pushRing(token),
        };
    }

    /// Pop a token from the buffer.
    pub fn pop(self: *TokenBuffer) ?StreamToken {
        return switch (self.config.strategy) {
            .fifo => self.popFifo(),
            .lifo => self.popLifo(),
            .priority => self.popPriority(),
            .ring => self.popRing(),
        };
    }

    /// Peek at the next token without removing.
    pub fn peek(self: *const TokenBuffer) ?StreamToken {
        if (self.items.items.len == 0) return null;

        return switch (self.config.strategy) {
            .fifo, .priority => self.items.items[0],
            .lifo => self.items.items[self.items.items.len - 1],
            .ring => self.items.items[self.ring_tail],
        };
    }

    /// Get buffer length.
    pub fn len(self: *const TokenBuffer) usize {
        if (self.config.strategy == .ring) {
            if (self.ring_head >= self.ring_tail) {
                return self.ring_head - self.ring_tail;
            }
            return self.config.capacity - self.ring_tail + self.ring_head;
        }
        return self.items.items.len;
    }

    /// Check if buffer is empty.
    pub fn isEmpty(self: *const TokenBuffer) bool {
        return self.len() == 0;
    }

    /// Check if buffer is full.
    pub fn isFull(self: *const TokenBuffer) bool {
        return self.len() >= self.config.capacity;
    }

    /// Check if flush threshold reached.
    pub fn shouldFlush(self: *const TokenBuffer) bool {
        return self.len() >= self.config.flush_threshold;
    }

    /// Clear the buffer.
    pub fn clear(self: *TokenBuffer) void {
        for (self.items.items) |*item| {
            self.allocator.free(item.text);
        }
        self.items.clearRetainingCapacity();
        self.ring_head = 0;
        self.ring_tail = 0;
        self.coalesce_buffer.clearRetainingCapacity();
    }

    /// Get buffer statistics.
    pub fn getStats(self: *const TokenBuffer) BufferStats {
        return .{
            .current_size = self.len(),
            .capacity = self.config.capacity,
            .total_pushed = self.total_pushed,
            .total_popped = self.total_popped,
            .total_dropped = self.total_dropped,
            .utilization = if (self.config.capacity > 0)
                @as(f64, @floatFromInt(self.len())) /
                    @as(f64, @floatFromInt(self.config.capacity))
            else
                0,
        };
    }

    /// Flush all tokens as concatenated text.
    pub fn flushAsText(self: *TokenBuffer) ![]u8 {
        var output = std.ArrayListUnmanaged(u8){};
        errdefer output.deinit(self.allocator);

        while (self.pop()) |token| {
            try output.appendSlice(self.allocator, token.text);
            self.allocator.free(token.text);
        }

        return output.toOwnedSlice(self.allocator);
    }

    // FIFO implementation
    fn pushFifo(self: *TokenBuffer, token: StreamToken) !void {
        if (self.items.items.len >= self.config.capacity) {
            // Drop oldest
            if (self.items.items.len > 0) {
                const dropped = self.items.orderedRemove(0);
                self.allocator.free(dropped.text);
                self.total_dropped += 1;
            }
        }

        const cloned = try token.clone(self.allocator);
        try self.items.append(self.allocator, cloned);
        self.total_pushed += 1;
    }

    fn popFifo(self: *TokenBuffer) ?StreamToken {
        if (self.items.items.len == 0) return null;

        self.total_popped += 1;
        return self.items.orderedRemove(0);
    }

    // LIFO implementation
    fn pushLifo(self: *TokenBuffer, token: StreamToken) !void {
        if (self.items.items.len >= self.config.capacity) {
            // Drop oldest (bottom of stack)
            if (self.items.items.len > 0) {
                const dropped = self.items.orderedRemove(0);
                self.allocator.free(dropped.text);
                self.total_dropped += 1;
            }
        }

        const cloned = try token.clone(self.allocator);
        try self.items.append(self.allocator, cloned);
        self.total_pushed += 1;
    }

    fn popLifo(self: *TokenBuffer) ?StreamToken {
        if (self.items.items.len == 0) return null;

        self.total_popped += 1;
        return self.items.pop();
    }

    // Priority implementation (higher sequence_index = higher priority)
    fn pushPriority(self: *TokenBuffer, token: StreamToken) !void {
        if (self.items.items.len >= self.config.capacity) {
            // Drop lowest priority (last item after sort)
            if (self.items.items.len > 0) {
                var dropped = self.items.pop();
                if (dropped) |*d| {
                    self.allocator.free(d.text);
                    self.total_dropped += 1;
                }
            }
        }

        const cloned = try token.clone(self.allocator);
        try self.items.append(self.allocator, cloned);

        // Sort by sequence_index descending
        std.mem.sort(StreamToken, self.items.items, {}, struct {
            fn lessThan(_: void, a: StreamToken, b: StreamToken) bool {
                return a.sequence_index > b.sequence_index;
            }
        }.lessThan);

        self.total_pushed += 1;
    }

    fn popPriority(self: *TokenBuffer) ?StreamToken {
        if (self.items.items.len == 0) return null;

        self.total_popped += 1;
        return self.items.orderedRemove(0);
    }

    // Ring buffer implementation
    fn pushRing(self: *TokenBuffer, token: StreamToken) !void {
        // Ensure capacity
        while (self.items.items.len < self.config.capacity) {
            const empty = StreamToken{
                .id = 0,
                .text = "",
                .is_end = false,
            };
            try self.items.append(self.allocator, empty);
        }

        // Free old value if overwriting
        if (self.items.items[self.ring_head].text.len > 0) {
            self.allocator.free(self.items.items[self.ring_head].text);
            self.total_dropped += 1;
        }

        const cloned = try token.clone(self.allocator);
        self.items.items[self.ring_head] = cloned;
        self.ring_head = (self.ring_head + 1) % self.config.capacity;

        // Move tail if head catches up
        if (self.ring_head == self.ring_tail) {
            self.ring_tail = (self.ring_tail + 1) % self.config.capacity;
        }

        self.total_pushed += 1;
    }

    fn popRing(self: *TokenBuffer) ?StreamToken {
        if (self.ring_head == self.ring_tail) return null;

        const token = self.items.items[self.ring_tail];
        self.items.items[self.ring_tail] = StreamToken{
            .id = 0,
            .text = "",
            .is_end = false,
        };
        self.ring_tail = (self.ring_tail + 1) % self.config.capacity;
        self.total_popped += 1;

        return token;
    }
};

/// Buffer statistics.
pub const BufferStats = struct {
    current_size: usize,
    capacity: usize,
    total_pushed: u64,
    total_popped: u64,
    total_dropped: u64,
    utilization: f64,
};

/// Coalescing buffer that merges consecutive tokens.
pub const CoalescingBuffer = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),
    token_count: usize,
    max_length: usize,
    flush_on_punctuation: bool,

    /// Initialize coalescing buffer.
    pub fn init(allocator: std.mem.Allocator, max_length: usize) CoalescingBuffer {
        return .{
            .allocator = allocator,
            .buffer = .{},
            .token_count = 0,
            .max_length = max_length,
            .flush_on_punctuation = true,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *CoalescingBuffer) void {
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a token, returns coalesced text if ready.
    pub fn add(self: *CoalescingBuffer, text: []const u8) !?[]u8 {
        try self.buffer.appendSlice(self.allocator, text);
        self.token_count += 1;

        // Check if we should flush
        if (self.buffer.items.len >= self.max_length) {
            return try self.flush();
        }

        // Check for punctuation flush
        if (self.flush_on_punctuation and text.len > 0) {
            const last_char = text[text.len - 1];
            if (last_char == '.' or last_char == '!' or last_char == '?' or last_char == '\n') {
                return try self.flush();
            }
        }

        return null;
    }

    /// Flush the buffer.
    pub fn flush(self: *CoalescingBuffer) ![]u8 {
        if (self.buffer.items.len == 0) {
            return try self.allocator.alloc(u8, 0);
        }

        const result = try self.buffer.toOwnedSlice(self.allocator);
        self.token_count = 0;
        return result;
    }

    /// Get buffered length.
    pub fn len(self: *const CoalescingBuffer) usize {
        return self.buffer.items.len;
    }

    /// Check if buffer is empty.
    pub fn isEmpty(self: *const CoalescingBuffer) bool {
        return self.buffer.items.len == 0;
    }
};

test "token buffer fifo" {
    const allocator = std.testing.allocator;
    var buf = TokenBuffer.init(allocator, .{
        .strategy = .fifo,
        .capacity = 3,
    });
    defer buf.deinit();

    try buf.push(.{ .id = 1, .text = "a" });
    try buf.push(.{ .id = 2, .text = "b" });
    try buf.push(.{ .id = 3, .text = "c" });

    const first = buf.pop();
    try std.testing.expect(first != null);
    try std.testing.expectEqualStrings("a", first.?.text);
    allocator.free(first.?.text);
}

test "token buffer capacity" {
    const allocator = std.testing.allocator;
    var buf = TokenBuffer.init(allocator, .{
        .strategy = .fifo,
        .capacity = 2,
    });
    defer buf.deinit();

    try buf.push(.{ .id = 1, .text = "a" });
    try buf.push(.{ .id = 2, .text = "b" });
    try buf.push(.{ .id = 3, .text = "c" }); // Should drop "a"

    const stats = buf.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_dropped);
}

test "token buffer lifo" {
    const allocator = std.testing.allocator;
    var buf = TokenBuffer.init(allocator, .{
        .strategy = .lifo,
        .capacity = 10,
    });
    defer buf.deinit();

    try buf.push(.{ .id = 1, .text = "a" });
    try buf.push(.{ .id = 2, .text = "b" });

    const last = buf.pop();
    try std.testing.expect(last != null);
    try std.testing.expectEqualStrings("b", last.?.text);
    allocator.free(last.?.text);
}

test "coalescing buffer" {
    const allocator = std.testing.allocator;
    var buf = CoalescingBuffer.init(allocator, 100);
    defer buf.deinit();

    // Add tokens without triggering flush
    const result1 = try buf.add("Hello");
    try std.testing.expect(result1 == null);

    const result2 = try buf.add(" ");
    try std.testing.expect(result2 == null);

    // Add punctuation to trigger flush
    const result3 = try buf.add("world.");
    try std.testing.expect(result3 != null);
    defer allocator.free(result3.?);
    try std.testing.expectEqualStrings("Hello world.", result3.?);
}
