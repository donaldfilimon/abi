//! Multi-Ralph coordination and communication.
//!
//! All-Zig, fast: uses the runtime's lock-free Vyukov Channel for message passing
//! between multiple Ralph loop instances. Supports task handoff, result sharing,
//! and skill sharing so Ralphs can collaborate and self-improve across runs.
//!
//! Usage:
//!   - Create a RalphBus with init(); use send()/tryRecv() from multiple Ralph loops.
//!   - Each Ralph has a numeric id; to_id 0 means broadcast to all.
//!   - Run N engines in parallel (e.g. via ThreadPool); each posts to the bus and
//!     can read messages for its id or broadcast.

const std = @import("std");
const Channel = @import("../../../services/runtime/concurrency/channel.zig").Channel;

/// Maximum content length per message (fits in channel slot without heap).
pub const max_message_content_len = 1024;

/// Kind of inter-Ralph message.
pub const RalphMessageKind = enum(u8) {
    /// Partial or final task result (e.g. one Ralph finished a subtask).
    task_result,
    /// Hand off work to another Ralph (goal snippet or instruction).
    handoff,
    /// Share a learned skill so other Ralphs can adopt it.
    skill_share,
    /// Coordination (e.g. "I'm done", "take over").
    coordination,
};

/// Message sent between Ralph instances. Fixed-size for lock-free channel.
pub const RalphMessage = struct {
    /// Sender Ralph id (0 is reserved).
    from_id: u32 = 0,
    /// Recipient id (0 = broadcast to all).
    to_id: u32 = 0,
    kind: RalphMessageKind = .task_result,
    content_len: u16 = 0,
    /// Payload (only content_len bytes valid).
    content: [max_message_content_len]u8 = [_]u8{0} ** max_message_content_len,

    pub fn setContent(self: *RalphMessage, slice: []const u8) void {
        const n = @min(slice.len, max_message_content_len);
        @memcpy(self.content[0..n], slice[0..n]);
        self.content_len = @intCast(n);
    }

    pub fn getContent(self: *const RalphMessage) []const u8 {
        return self.content[0..self.content_len];
    }
};

/// Shared message bus for multiple Ralphs. Backed by a lock-free MPMC channel.
pub const RalphBus = struct {
    channel: Channel(RalphMessage),
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create a bus with a fixed capacity (e.g. 64 or 128 messages).
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
        const channel = try Channel(RalphMessage).init(allocator, capacity);
        return Self{
            .channel = channel,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.channel.deinit();
        self.* = undefined;
    }

    /// Send a message (blocking if full; backpressure).
    pub fn send(self: *Self, msg: RalphMessage) !void {
        return self.channel.send(msg);
    }

    /// Try to send without blocking. Returns false if full.
    pub fn trySend(self: *Self, msg: RalphMessage) bool {
        return self.channel.trySend(msg);
    }

    /// Receive a message for this Ralph (blocking). Caller should filter by to_id or broadcast in a loop.
    pub fn recv(self: *Self) !RalphMessage {
        return self.channel.recv();
    }

    /// Non-blocking receive. Returns null if empty.
    pub fn tryRecv(self: *Self) ?RalphMessage {
        return self.channel.tryRecv();
    }

    /// Drain messages for `ralph_id`: receive until empty or a message for this id (or broadcast).
    /// Returns the first message addressed to this Ralph or broadcast; others are re-sent (so order is best-effort).
    pub fn recvFor(self: *Self, ralph_id: u32) ?RalphMessage {
        while (self.channel.tryRecv()) |msg| {
            if (msg.to_id == 0 or msg.to_id == ralph_id) return msg;
            _ = self.channel.trySend(msg);
        }
        return null;
    }

    pub fn close(self: *Self) void {
        self.channel.close();
    }

    pub fn isClosed(self: *const Self) bool {
        return self.channel.isClosed();
    }
};

test "ralph_multi message setContent getContent" {
    var msg: RalphMessage = .{};
    msg.from_id = 1;
    msg.to_id = 2;
    msg.kind = .skill_share;
    msg.setContent("Always run tests after refactoring.");
    try std.testing.expectEqual(@as(u16, 35), msg.content_len);
    try std.testing.expectEqualStrings("Always run tests after refactoring.", msg.getContent());
}

test "ralph_multi bus send tryRecv" {
    var bus = try RalphBus.init(std.testing.allocator, 4);
    defer bus.deinit();

    var msg: RalphMessage = .{ .from_id = 1, .to_id = 0 };
    msg.setContent("hello");
    try bus.send(msg);

    const received = bus.tryRecv();
    try std.testing.expect(received != null);
    try std.testing.expectEqual(@as(u32, 1), received.?.from_id);
    try std.testing.expectEqualStrings("hello", received.?.getContent());
}
