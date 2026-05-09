const std = @import("std");
const state_mod = @import("state.zig");
pub const types = @import("types.zig");

const MessagingState = state_mod.MessagingState;
const DeadLetter = types.DeadLetter;
const MessagingError = types.MessagingError;
const MessagingStats = types.MessagingStats;
const TopicInfo = types.TopicInfo;

pub fn listTopics(s: *MessagingState, allocator: std.mem.Allocator) MessagingError![][]const u8 {
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const count = s.topics.count();
    if (count == 0) return &.{};

    const result = allocator.alloc([]const u8, count) catch return error.OutOfMemory;
    var i: usize = 0;
    var iter = s.topics.iterator();
    while (iter.next()) |entry| {
        if (i < count) {
            result[i] = entry.key_ptr.*;
            i += 1;
        }
    }

    return result[0..i];
}

pub fn topicStats(s: *MessagingState, topic_name: []const u8) MessagingError!TopicInfo {
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const topic = s.topics.get(topic_name) orelse return error.TopicNotFound;
    return .{
        .name = topic.name,
        .subscriber_count = @intCast(topic.subscribers.items.len),
        .messages_published = topic.published,
        .messages_delivered = topic.delivered,
        .messages_failed = topic.failed,
    };
}

pub fn getDeadLetters(s: *MessagingState, allocator: std.mem.Allocator) MessagingError![]DeadLetter {
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    if (s.dead_letters.items.len == 0) return &.{};

    const result = allocator.alloc(DeadLetter, s.dead_letters.items.len) catch
        return error.OutOfMemory;
    for (s.dead_letters.items, 0..) |dl, i| {
        result[i] = .{
            .message = .{
                .topic = dl.topic,
                .payload = dl.payload,
                .id = dl.msg_id,
                .timestamp = @intCast(dl.timestamp_ns / std.time.ns_per_ms),
            },
            .reason = dl.reason,
            .timestamp = @intCast(dl.timestamp_ns / std.time.ns_per_ms),
        };
    }
    return result;
}

pub fn clearDeadLetters(s: *MessagingState) void {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    for (s.dead_letters.items) |*dl| {
        s.allocator.free(dl.topic);
        s.allocator.free(dl.payload);
        s.allocator.free(dl.reason);
    }
    s.dead_letters.clearRetainingCapacity();
}

pub fn messagingStats(s: *MessagingState) MessagingStats {
    return .{
        .total_published = s.stat_published.load(.monotonic),
        .total_delivered = s.stat_delivered.load(.monotonic),
        .total_failed = s.stat_failed.load(.monotonic),
        .active_topics = @intCast(s.topics.count()),
        .active_subscribers = @intCast(s.subscribers.count()),
        .dead_letter_count = @intCast(s.dead_letters.items.len),
    };
}
