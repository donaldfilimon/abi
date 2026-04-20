const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
const time_mod = @import("../../foundation/mod.zig").time;
pub const types = @import("types.zig");

const MessagingConfig = types.MessagingConfig;

// ── Internal Types ─────────────────────────────────────────────────────

pub const Subscriber = struct {
    id: u64,
    pattern: []u8, // owned copy
    callback: types.SubscriberCallback,
    user_ctx: ?*anyopaque,
    active: bool = true,
};

pub const Topic = struct {
    name: []u8, // owned
    subscribers: std.ArrayListUnmanaged(*Subscriber),
    published: u64 = 0,
    delivered: u64 = 0,
    failed: u64 = 0,
};

pub const DLQEntry = struct {
    topic: []u8,
    payload: []u8,
    reason: []u8,
    timestamp_ns: u128,
    msg_id: u64,
};

// ── Module State ───────────────────────────────────────────────────────

pub const MessagingState = struct {
    allocator: std.mem.Allocator,
    config: MessagingConfig,
    topics: std.StringHashMapUnmanaged(*Topic),
    subscribers: std.AutoHashMapUnmanaged(u64, *Subscriber), // id → subscriber
    dead_letters: std.ArrayListUnmanaged(DLQEntry),
    rw_lock: sync.RwLock,
    next_sub_id: std.atomic.Value(u64),
    next_msg_id: std.atomic.Value(u64),

    // Stats
    stat_published: std.atomic.Value(u64),
    stat_delivered: std.atomic.Value(u64),
    stat_failed: std.atomic.Value(u64),

    pub fn create(allocator: std.mem.Allocator, config: MessagingConfig) !*MessagingState {
        const s = try allocator.create(MessagingState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .topics = .empty,
            .subscribers = .empty,
            .dead_letters = .empty,
            .rw_lock = sync.RwLock.init(),
            .next_sub_id = std.atomic.Value(u64).init(1),
            .next_msg_id = std.atomic.Value(u64).init(1),
            .stat_published = std.atomic.Value(u64).init(0),
            .stat_delivered = std.atomic.Value(u64).init(0),
            .stat_failed = std.atomic.Value(u64).init(0),
        };
        return s;
    }

    pub fn destroy(self: *MessagingState) void {
        const allocator = self.allocator;

        // Free topics
        var topic_iter = self.topics.iterator();
        while (topic_iter.next()) |entry| {
            const topic = entry.value_ptr.*;
            topic.subscribers.deinit(allocator);
            allocator.free(topic.name);
            allocator.destroy(topic);
        }
        self.topics.deinit(allocator);

        // Free subscribers
        var sub_iter = self.subscribers.iterator();
        while (sub_iter.next()) |entry| {
            const sub = entry.value_ptr.*;
            allocator.free(sub.pattern);
            allocator.destroy(sub);
        }
        self.subscribers.deinit(allocator);

        // Free dead letters
        for (self.dead_letters.items) |*dl| {
            allocator.free(dl.topic);
            allocator.free(dl.payload);
            allocator.free(dl.reason);
        }
        self.dead_letters.deinit(allocator);

        allocator.destroy(self);
    }

    pub fn getOrCreateTopic(self: *MessagingState, name: []const u8) !*Topic {
        if (self.topics.get(name)) |topic| return topic;

        if (self.topics.count() >= self.config.max_channels) return error.OutOfMemory;

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const topic = try self.allocator.create(Topic);
        topic.* = .{
            .name = owned_name,
            .subscribers = .empty,
        };

        try self.topics.put(self.allocator, owned_name, topic);
        return topic;
    }

    pub fn addDeadLetter(
        self: *MessagingState,
        topic_name: []const u8,
        payload: []const u8,
        reason: []const u8,
        msg_id: u64,
    ) void {
        // Cap dead letter queue at buffer_size
        if (self.dead_letters.items.len >= self.config.buffer_size) {
            // Remove oldest
            if (self.dead_letters.items.len > 0) {
                const old = self.dead_letters.orderedRemove(0);
                self.allocator.free(old.topic);
                self.allocator.free(old.payload);
                self.allocator.free(old.reason);
            }
        }

        const topic_copy = self.allocator.dupe(u8, topic_name) catch return;
        const payload_copy = self.allocator.dupe(u8, payload) catch {
            self.allocator.free(topic_copy);
            return;
        };
        const reason_copy = self.allocator.dupe(u8, reason) catch {
            self.allocator.free(topic_copy);
            self.allocator.free(payload_copy);
            return;
        };
        self.dead_letters.append(self.allocator, .{
            .topic = topic_copy,
            .payload = payload_copy,
            .reason = reason_copy,
            .timestamp_ns = (time_mod.Instant.now() catch {
                self.allocator.free(topic_copy);
                self.allocator.free(payload_copy);
                self.allocator.free(reason_copy);
                return;
            }).nanos,
            .msg_id = msg_id,
        }) catch {
            self.allocator.free(topic_copy);
            self.allocator.free(payload_copy);
            self.allocator.free(reason_copy);
        };
    }
};
