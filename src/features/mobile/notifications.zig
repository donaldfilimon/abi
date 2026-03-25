//! Notification tracking for the mobile feature.
//!
//! Provides notification sending, counting, and clearing backed by
//! an in-memory log of NotificationEntry records.

const std = @import("std");
const types = @import("types.zig");

const Notification = types.Notification;
const NotificationEntry = types.NotificationEntry;
const MobileError = types.MobileError;

/// Send a notification and append it to the log.
pub fn sendNotification(
    log: *std.ArrayListUnmanaged(NotificationEntry),
    allocator: std.mem.Allocator,
    title: []const u8,
    body_text: []const u8,
    priority: Notification.Priority,
) MobileError!void {
    var entry: NotificationEntry = .{
        .priority = priority,
    };

    const t_len: u8 = @intCast(@min(title.len, entry.title_buf.len));
    @memcpy(entry.title_buf[0..t_len], title[0..t_len]);
    entry.title_len = t_len;

    const b_len: u16 = @intCast(@min(body_text.len, entry.body_buf.len));
    @memcpy(entry.body_buf[0..b_len], body_text[0..b_len]);
    entry.body_len = b_len;

    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    entry.sent_at = @intCast(ts.sec);

    log.append(allocator, entry) catch return error.OutOfMemory;
}

/// Return the number of tracked notifications.
pub fn getNotificationCount(log: *const std.ArrayListUnmanaged(NotificationEntry)) usize {
    return log.items.len;
}

/// Clear all tracked notifications.
pub fn clearNotifications(log: *std.ArrayListUnmanaged(NotificationEntry)) void {
    log.clearRetainingCapacity();
}

/// Legacy module-level sendNotification (no tracking).
pub fn sendNotificationLegacy(_: []const u8, _: []const u8) MobileError!void {}
