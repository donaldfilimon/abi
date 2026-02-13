//! ═══════════════════════════════════════════════════════════════════════════
//! ABI Framework — Channel: Bounded MPMC Queue
//! Adapted from abi-system-v2.0/channel.zig
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! Lock-free bounded channel for passing data between pipeline stages.
//! Supports multiple producers and consumers with backpressure semantics.
//!
//! Implementation: Dmitry Vyukov's bounded MPMC queue with per-slot
//! sequence counters for wait-free progress.
//! ═══════════════════════════════════════════════════════════════════════════

const std = @import("std");
const Math = @import("../../shared/utils/v2_primitives.zig").Math;

// ─── Helpers ──────────────────────────────────────────────────────────────

fn nextPowerOfTwo(x: usize) usize {
    return Math.nextPowerOfTwo(usize, x);
}

// ─── Channel ──────────────────────────────────────────────────────────────

pub fn Channel(comptime T: type) type {
    return struct {
        const Self = @This();

        const Slot = struct {
            data: T = undefined,
            sequence: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        };

        slots: []Slot,
        capacity: usize,
        mask: usize,
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        allocator: std.mem.Allocator,

        /// Closed flag — no more sends allowed
        closed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        // ── Statistics ──────────────────────────────────────────────

        send_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        recv_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        send_fail_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

        // ── Lifecycle ───────────────────────────────────────────────

        pub fn init(alloc: std.mem.Allocator, requested_capacity: usize) !Self {
            const capacity = nextPowerOfTwo(@max(requested_capacity, 2));
            const slots = try alloc.alloc(Slot, capacity);

            // Initialize sequence counters: slot[i].sequence = i
            for (slots, 0..) |*slot, i| {
                slot.* = Slot{};
                slot.sequence.store(i, .release);
            }

            return Self{
                .slots = slots,
                .capacity = capacity,
                .mask = capacity - 1,
                .allocator = alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slots);
            self.* = undefined;
        }

        fn spinYieldSleepBackoff(spins: *u32) void {
            spins.* += 1;
            if (spins.* < 64) {
                std.atomic.spinLoopHint();
            } else if (spins.* < 256) {
                std.Thread.yield() catch {};
            } else {
                std.Thread.yield() catch {};
                // After many spins, back off more aggressively
                std.atomic.spinLoopHint();
                std.atomic.spinLoopHint();
            }
        }

        fn diffSeq(seq: usize, pos: usize) isize {
            return @as(isize, @intCast(seq)) - @as(isize, @intCast(pos));
        }

        // ── Send ────────────────────────────────────────────────────

        /// Non-blocking send. Returns false if the channel is full or closed.
        pub fn trySend(self: *Self, value: T) bool {
            if (self.closed.load(.acquire)) return false;

            var pos = self.tail.load(.acquire);
            while (true) {
                const slot = &self.slots[pos & self.mask];
                const seq = slot.sequence.load(.acquire);

                const diff = diffSeq(seq, pos);

                if (diff == 0) {
                    if (self.tail.cmpxchgWeak(pos, pos + 1, .acq_rel, .acquire)) |new_pos| {
                        pos = new_pos;
                        continue;
                    }
                    slot.data = value;
                    slot.sequence.store(pos + 1, .release);
                    _ = self.send_count.fetchAdd(1, .monotonic);
                    return true;
                } else if (diff < 0) {
                    _ = self.send_fail_count.fetchAdd(1, .monotonic);
                    return false;
                } else {
                    pos = self.tail.load(.acquire);
                }
            }
        }

        /// Blocking send with spin + yield backoff
        pub fn send(self: *Self, value: T) !void {
            var spins: u32 = 0;
            while (!self.trySend(value)) {
                if (self.closed.load(.acquire)) return error.ChannelClosed;
                spinYieldSleepBackoff(&spins);
            }
        }

        // ── Receive ─────────────────────────────────────────────────

        /// Non-blocking receive. Returns null if the channel is empty.
        pub fn tryRecv(self: *Self) ?T {
            var pos = self.head.load(.acquire);
            while (true) {
                const slot = &self.slots[pos & self.mask];
                const seq = slot.sequence.load(.acquire);

                const diff = diffSeq(seq, pos + 1);

                if (diff == 0) {
                    if (self.head.cmpxchgWeak(pos, pos + 1, .acq_rel, .acquire)) |new_pos| {
                        pos = new_pos;
                        continue;
                    }
                    const value = slot.data;
                    slot.sequence.store(pos + self.capacity, .release);
                    _ = self.recv_count.fetchAdd(1, .monotonic);
                    return value;
                } else if (diff < 0) {
                    return null;
                } else {
                    pos = self.head.load(.acquire);
                }
            }
        }

        /// Blocking receive with spin + yield backoff
        pub fn recv(self: *Self) !T {
            var spins: u32 = 0;
            while (true) {
                if (self.tryRecv()) |value| return value;
                if (self.closed.load(.acquire) and self.isEmpty()) return error.ChannelClosed;
                spinYieldSleepBackoff(&spins);
            }
        }

        // ── Control ─────────────────────────────────────────────────

        pub fn close(self: *Self) void {
            self.closed.store(true, .release);
        }

        pub fn isClosed(self: *const Self) bool {
            return self.closed.load(.acquire);
        }

        pub fn isEmpty(self: *const Self) bool {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.acquire);
            return h >= t;
        }

        pub fn len(self: *const Self) usize {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.acquire);
            return if (t > h) t - h else 0;
        }

        // ── Statistics ──────────────────────────────────────────────

        pub const ChannelStats = struct {
            capacity: usize,
            current_len: usize,
            total_sent: u64,
            total_received: u64,
            send_failures: u64,
            is_closed: bool,
        };

        pub fn stats(self: *const Self) ChannelStats {
            return .{
                .capacity = self.capacity,
                .current_len = self.len(),
                .total_sent = self.send_count.load(.acquire),
                .total_received = self.recv_count.load(.acquire),
                .send_failures = self.send_fail_count.load(.acquire),
                .is_closed = self.isClosed(),
            };
        }
    };
}

// ─── Convenience Aliases ────────────────────────────────────────────────────

pub const ByteChannel = Channel([]const u8);
pub const U64Channel = Channel(u64);
pub const F32Channel = Channel(f32);

// ─── Typed Message Channel ──────────────────────────────────────────────────

/// A tagged message for heterogeneous pipeline data flow.
/// Fits in 128 bytes for cache-line-friendly transfer.
pub const Message = struct {
    tag: MessageTag = .empty,
    payload: [120]u8 = .{0} ** 120,

    pub const MessageTag = enum(u8) {
        empty,
        token_ids,
        embedding_vector,
        attention_scores,
        decoded_text,
        routing_decision,
        error_signal,
    };

    pub fn initTag(tag: MessageTag) Message {
        return .{ .tag = tag };
    }

    pub fn setPayloadBytes(self: *Message, data: []const u8) void {
        const copy_len = @min(data.len, 120);
        @memcpy(self.payload[0..copy_len], data[0..copy_len]);
    }

    pub fn getPayloadAs(self: *const Message, comptime TPayload: type) *const TPayload {
        return @ptrCast(@alignCast(&self.payload));
    }
};

pub const MessageChannel = Channel(Message);

test "Channel trySend fails when full" {
    var channel = try Channel(u32).init(std.testing.allocator, 2);
    defer channel.deinit();

    try std.testing.expect(channel.trySend(1));
    try std.testing.expect(channel.trySend(2));
    try std.testing.expect(!channel.trySend(3));
}

test "Channel recv returns ChannelClosed when closed and empty" {
    var channel = try Channel(u32).init(std.testing.allocator, 2);
    defer channel.deinit();

    channel.close();
    try std.testing.expectError(error.ChannelClosed, channel.recv());
}

test "Channel preserves FIFO order for single producer/consumer" {
    var channel = try Channel(u32).init(std.testing.allocator, 4);
    defer channel.deinit();

    try channel.send(1);
    try channel.send(2);
    try channel.send(3);

    try std.testing.expectEqual(@as(u32, 1), try channel.recv());
    try std.testing.expectEqual(@as(u32, 2), try channel.recv());
    try std.testing.expectEqual(@as(u32, 3), try channel.recv());
}
