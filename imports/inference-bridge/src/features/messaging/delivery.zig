const std = @import("std");
const state = @import("state.zig");
const pattern_matching = @import("pattern_matching.zig");
pub const types = @import("types.zig");

const MessagingState = state.MessagingState;
const Message = types.Message;
const DeliveryResult = types.DeliveryResult;
const MessagingError = types.MessagingError;

/// Deliver a message to all matching subscribers, updating stats and
/// routing discards to the dead letter queue.
pub fn deliverToSubscribers(
    s: *MessagingState,
    topic_name: []const u8,
    payload: []const u8,
    msg_id: u64,
) void {
    const msg = Message{
        .topic = topic_name,
        .payload = payload,
        .timestamp = pattern_matching.nowMs(),
        .id = msg_id,
    };

    var sub_iter = s.subscribers.iterator();
    while (sub_iter.next()) |entry| {
        const sub = entry.value_ptr.*;
        if (!sub.active) continue;

        if (pattern_matching.patternMatches(sub.pattern, topic_name)) {
            const result = sub.callback(msg, sub.user_ctx);
            switch (result) {
                .ok => {
                    _ = s.stat_delivered.fetchAdd(1, .monotonic);
                    // Update topic stats
                    if (s.topics.get(topic_name)) |topic| {
                        topic.delivered += 1;
                    }
                },
                .retry, .discard => {
                    _ = s.stat_failed.fetchAdd(1, .monotonic);
                    if (s.topics.get(topic_name)) |topic| {
                        topic.failed += 1;
                    }
                    if (result == .discard) {
                        s.addDeadLetter(topic_name, payload, "subscriber_discard", msg_id);
                    }
                },
            }
        }
    }
}
