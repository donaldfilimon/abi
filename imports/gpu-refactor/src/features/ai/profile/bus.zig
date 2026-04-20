//! ProfileBus: inter-profile collaboration messaging.
//!
//! Provides a bounded message queue for profiles to exchange opinions,
//! requests, responses, and compliance vetoes. Adapted from the lock-free
//! RalphBus pattern in abbey/ralph_multi.zig.

const std = @import("std");
const types = @import("types.zig");
const ProfileId = types.ProfileId;
const ProfileMessage = types.ProfileMessage;
const MessageKind = types.MessageKind;
const ProfileError = types.ProfileError;

const foundation = @import("../../../foundation/mod.zig");

/// Bounded message bus for inter-profile communication.
///
/// Each profile has its own inbox (bounded queue). Messages can be sent
/// to a specific profile or broadcast to all.
pub const ProfileBus = struct {
    allocator: std.mem.Allocator,
    inboxes: std.EnumArray(ProfileId, Inbox),
    total_sent: u64 = 0,
    total_dropped: u64 = 0,

    const max_inbox_size = 64;

    const Inbox = struct {
        buffer: [max_inbox_size]ProfileMessage = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,
        mutex: foundation.sync.Mutex = .{},

        fn append(self: *Inbox, msg: ProfileMessage) error{Overflow}!void {
            if (self.count >= max_inbox_size) return error.Overflow;
            self.buffer[self.tail] = msg;
            self.tail = (self.tail + 1) % max_inbox_size;
            self.count += 1;
        }

        fn pop(self: *Inbox) ?ProfileMessage {
            if (self.count == 0) return null;
            const msg = self.buffer[self.head];
            self.head = (self.head + 1) % max_inbox_size;
            self.count -= 1;
            return msg;
        }
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .inboxes = std.EnumArray(ProfileId, Inbox).initFill(.{}),
        };
    }

    /// Send a message to a specific profile's inbox.
    pub fn send(self: *Self, msg: ProfileMessage) ProfileError!void {
        const target = msg.to orelse return self.broadcast(msg);
        self.deliverTo(target, msg);
    }

    /// Broadcast a message to all profiles (except sender).
    pub fn broadcast(self: *Self, msg: ProfileMessage) ProfileError!void {
        for (std.enums.values(ProfileId)) |id| {
            if (id != msg.from) {
                self.deliverTo(id, msg);
            }
        }
    }

    fn deliverTo(self: *Self, target: ProfileId, msg: ProfileMessage) void {
        var inbox = self.inboxes.getPtr(target);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();

        inbox.append(msg) catch {
            self.total_dropped += 1;
            return;
        };
        self.total_sent += 1;
    }

    /// Receive the next message for a profile (FIFO). Returns null if empty.
    pub fn receive(self: *Self, id: ProfileId) ?ProfileMessage {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();

        return inbox.pop();
    }

    /// Check if a profile has pending messages.
    pub fn hasPending(self: *Self, id: ProfileId) bool {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();
        return inbox.count > 0;
    }

    /// Get count of pending messages for a profile.
    pub fn pendingCount(self: *Self, id: ProfileId) usize {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();
        return inbox.count;
    }

    /// Clear all inboxes.
    pub fn clear(self: *Self) void {
        for (std.enums.values(ProfileId)) |id| {
            var inbox = self.inboxes.getPtr(id);
            inbox.mutex.lock();
            defer inbox.mutex.unlock();
            inbox.head = 0;
            inbox.tail = 0;
            inbox.count = 0;
        }
    }

    /// Create a convenience message builder.
    pub fn createMessage(from: ProfileId, to: ?ProfileId, kind: MessageKind, payload: []const u8, confidence: f32) ProfileMessage {
        return .{
            .from = from,
            .to = to,
            .kind = kind,
            .payload = payload,
            .confidence = confidence,
            .timestamp = foundation.time.unixMs(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.clear();
    }
};

test "profile bus send and receive" {
    var bus = ProfileBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = ProfileBus.createMessage(.abbey, .aviva, .request, "Need fact check", 0.8);
    try bus.send(msg);

    try std.testing.expect(bus.hasPending(.aviva));
    try std.testing.expect(!bus.hasPending(.abbey));

    const received = bus.receive(.aviva);
    try std.testing.expect(received != null);
    try std.testing.expectEqual(ProfileId.abbey, received.?.from);
    try std.testing.expectEqualStrings("Need fact check", received.?.payload);
}

test "profile bus broadcast" {
    var bus = ProfileBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = ProfileBus.createMessage(.abi, null, .veto, "PII detected", 0.95);
    try bus.broadcast(msg);

    // Abi should NOT receive its own broadcast
    try std.testing.expect(!bus.hasPending(.abi));
    // Abbey and Aviva should receive it
    try std.testing.expect(bus.hasPending(.abbey));
    try std.testing.expect(bus.hasPending(.aviva));
}

test "profile bus empty receive" {
    var bus = ProfileBus.init(std.testing.allocator);
    defer bus.deinit();

    const received = bus.receive(.abbey);
    try std.testing.expect(received == null);
}

test "profile bus FIFO ordering with wrap-around" {
    var bus = ProfileBus.init(std.testing.allocator);
    defer bus.deinit();

    const capacity = ProfileBus.max_inbox_size;

    // Fill the queue to capacity
    for (0..capacity) |i| {
        const conf: f32 = @floatFromInt(i);
        const msg = ProfileBus.createMessage(.abbey, .aviva, .request, "msg", conf / 100.0);
        try bus.send(msg);
    }
    try std.testing.expectEqual(capacity, bus.pendingCount(.aviva));

    // Drain half — verifying FIFO order
    for (0..capacity / 2) |i| {
        const received = bus.receive(.aviva);
        try std.testing.expect(received != null);
        const expected: f32 = @floatFromInt(i);
        try std.testing.expectEqual(expected / 100.0, received.?.confidence);
    }
    try std.testing.expectEqual(capacity / 2, bus.pendingCount(.aviva));

    // Refill the freed slots (this wraps the tail around)
    for (0..capacity / 2) |i| {
        const conf: f32 = @floatFromInt(capacity + i);
        const msg = ProfileBus.createMessage(.abbey, .aviva, .response, "wrap", conf / 100.0);
        try bus.send(msg);
    }
    try std.testing.expectEqual(capacity, bus.pendingCount(.aviva));

    // Drain all — verify complete FIFO order across the wrap boundary
    for (0..capacity) |i| {
        const received = bus.receive(.aviva);
        try std.testing.expect(received != null);
        const expected: f32 = @floatFromInt(capacity / 2 + i);
        try std.testing.expectEqual(expected / 100.0, received.?.confidence);
    }
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount(.aviva));
    try std.testing.expect(bus.receive(.aviva) == null);
}

test {
    std.testing.refAllDecls(@This());
}
