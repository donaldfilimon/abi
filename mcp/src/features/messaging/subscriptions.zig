const std = @import("std");
const state_mod = @import("state.zig");
pub const types = @import("types.zig");

const MessagingState = state_mod.MessagingState;
const Subscriber = state_mod.Subscriber;
const MessagingError = types.MessagingError;
const SubscriberCallback = types.SubscriberCallback;

pub fn subscribe(
    s: *MessagingState,
    topic_pattern: []const u8,
    callback: SubscriberCallback,
    user_ctx: ?*anyopaque,
) MessagingError!u64 {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const sub_id = s.next_sub_id.fetchAdd(1, .monotonic);

    const pattern_owned = s.allocator.dupe(u8, topic_pattern) catch return error.OutOfMemory;
    errdefer s.allocator.free(pattern_owned);

    const sub = s.allocator.create(Subscriber) catch return error.OutOfMemory;
    sub.* = .{
        .id = sub_id,
        .pattern = pattern_owned,
        .callback = callback,
        .user_ctx = user_ctx,
    };

    s.subscribers.put(s.allocator, sub_id, sub) catch {
        s.allocator.destroy(sub);
        s.allocator.free(pattern_owned);
        return error.OutOfMemory;
    };

    if (std.mem.indexOf(u8, topic_pattern, "*") == null and
        std.mem.indexOf(u8, topic_pattern, "#") == null)
    {
        if (s.getOrCreateTopic(topic_pattern)) |topic| {
            topic.subscribers.append(s.allocator, sub) catch |err| {
                std.log.warn("messaging: failed to add subscriber to topic index: {t}", .{err});
            };
        } else |_| {}
    }

    return sub_id;
}

pub fn unsubscribe(s: *MessagingState, subscriber_id: u64) MessagingError!bool {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.subscribers.fetchRemove(subscriber_id)) |kv| {
        const sub = kv.value;
        var topic_iter = s.topics.iterator();
        while (topic_iter.next()) |entry| {
            const topic = entry.value_ptr.*;
            for (topic.subscribers.items, 0..) |t_sub, i| {
                if (t_sub.id == subscriber_id) {
                    _ = topic.subscribers.swapRemove(i);
                    break;
                }
            }
        }
        s.allocator.free(sub.pattern);
        s.allocator.destroy(sub);
        return true;
    }

    return false;
}
