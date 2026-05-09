const state_mod = @import("state.zig");
const delivery = @import("delivery.zig");
pub const types = @import("types.zig");

const MessagingState = state_mod.MessagingState;
const MessagingError = types.MessagingError;

pub fn publish(
    s: *MessagingState,
    topic_name: []const u8,
    payload: []const u8,
) MessagingError!void {
    const msg_id = s.next_msg_id.fetchAdd(1, .monotonic);
    _ = s.stat_published.fetchAdd(1, .monotonic);

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    delivery.deliverToSubscribers(s, topic_name, payload, msg_id);

    if (s.topics.get(topic_name)) |topic| {
        topic.published += 1;
    } else {
        const topic = s.getOrCreateTopic(topic_name) catch return error.OutOfMemory;
        topic.published = 1;
    }
}
