//! PersonaBus: inter-persona collaboration messaging.
//!
//! Provides a bounded message queue for personas to exchange opinions,
//! requests, responses, and compliance vetoes. Adapted from the lock-free
//! RalphBus pattern in abbey/ralph_multi.zig.

const std = @import("std");
const types = @import("types.zig");
const PersonaId = types.PersonaId;
const PersonaMessage = types.PersonaMessage;
const MessageKind = types.MessageKind;
const PersonaError = types.PersonaError;

const foundation = @import("../../../foundation/mod.zig");

/// Bounded message bus for inter-persona communication.
///
/// Each persona has its own inbox (bounded queue). Messages can be sent
/// to a specific persona or broadcast to all.
pub const PersonaBus = struct {
    allocator: std.mem.Allocator,
    inboxes: std.EnumArray(PersonaId, Inbox),
    total_sent: u64 = 0,
    total_dropped: u64 = 0,

    const max_inbox_size = 64;

    const Inbox = struct {
        buffer: [max_inbox_size]PersonaMessage = undefined,
        len: usize = 0,
        mutex: foundation.sync.Mutex = .{},

        fn append(self: *Inbox, msg: PersonaMessage) error{Overflow}!void {
            if (self.len >= max_inbox_size) return error.Overflow;
            self.buffer[self.len] = msg;
            self.len += 1;
        }
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .inboxes = std.EnumArray(PersonaId, Inbox).initFill(.{}),
        };
    }

    /// Send a message to a specific persona's inbox.
    pub fn send(self: *Self, msg: PersonaMessage) PersonaError!void {
        const target = msg.to orelse return self.broadcast(msg);
        self.deliverTo(target, msg);
    }

    /// Broadcast a message to all personas (except sender).
    pub fn broadcast(self: *Self, msg: PersonaMessage) PersonaError!void {
        for (std.enums.values(PersonaId)) |id| {
            if (id != msg.from) {
                self.deliverTo(id, msg);
            }
        }
    }

    fn deliverTo(self: *Self, target: PersonaId, msg: PersonaMessage) void {
        var inbox = self.inboxes.getPtr(target);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();

        inbox.append(msg) catch {
            self.total_dropped += 1;
            return;
        };
        self.total_sent += 1;
    }

    /// Receive the next message for a persona (FIFO). Returns null if empty.
    pub fn receive(self: *Self, id: PersonaId) ?PersonaMessage {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();

        if (inbox.len == 0) return null;

        // Remove from front (shift)
        const msg = inbox.buffer[0];
        // Shift remaining messages left
        for (1..inbox.len) |i| {
            inbox.buffer[i - 1] = inbox.buffer[i];
        }
        inbox.len -= 1;
        return msg;
    }

    /// Check if a persona has pending messages.
    pub fn hasPending(self: *Self, id: PersonaId) bool {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();
        return inbox.len > 0;
    }

    /// Get count of pending messages for a persona.
    pub fn pendingCount(self: *Self, id: PersonaId) usize {
        var inbox = self.inboxes.getPtr(id);
        inbox.mutex.lock();
        defer inbox.mutex.unlock();
        return inbox.len;
    }

    /// Clear all inboxes.
    pub fn clear(self: *Self) void {
        for (std.enums.values(PersonaId)) |id| {
            var inbox = self.inboxes.getPtr(id);
            inbox.mutex.lock();
            defer inbox.mutex.unlock();
            inbox.len = 0;
        }
    }

    /// Create a convenience message builder.
    pub fn createMessage(from: PersonaId, to: ?PersonaId, kind: MessageKind, payload: []const u8, confidence: f32) PersonaMessage {
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

test "persona bus send and receive" {
    var bus = PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = PersonaBus.createMessage(.abbey, .aviva, .request, "Need fact check", 0.8);
    try bus.send(msg);

    try std.testing.expect(bus.hasPending(.aviva));
    try std.testing.expect(!bus.hasPending(.abbey));

    const received = bus.receive(.aviva);
    try std.testing.expect(received != null);
    try std.testing.expectEqual(PersonaId.abbey, received.?.from);
    try std.testing.expectEqualStrings("Need fact check", received.?.payload);
}

test "persona bus broadcast" {
    var bus = PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = PersonaBus.createMessage(.abi, null, .veto, "PII detected", 0.95);
    try bus.broadcast(msg);

    // Abi should NOT receive its own broadcast
    try std.testing.expect(!bus.hasPending(.abi));
    // Abbey and Aviva should receive it
    try std.testing.expect(bus.hasPending(.abbey));
    try std.testing.expect(bus.hasPending(.aviva));
}

test "persona bus empty receive" {
    var bus = PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    const received = bus.receive(.abbey);
    try std.testing.expect(received == null);
}

test {
    std.testing.refAllDecls(@This());
}
