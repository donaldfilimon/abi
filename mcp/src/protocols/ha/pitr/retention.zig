const time = @import("../../../foundation/mod.zig").time;

pub fn Retention(comptime Manager: type, comptime PitrEvent: type) type {
    return struct {
        pub fn applyRetention(self: *Manager) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const now = time.timestampSec();
            const retention_sec = @as(u64, self.config.retention_hours) * 3600;
            const cutoff = now - retention_sec;

            var pruned_count: u32 = 0;
            var freed_bytes: u64 = 0;

            var i: usize = 0;
            while (i < self.recovery_points.items.len) {
                const point = self.recovery_points.items[i];
                if (point.timestamp < cutoff) {
                    freed_bytes += point.size_bytes;
                    pruned_count += 1;

                    emitEvent(self, .{ .checkpoint_pruned = .{ .sequence = point.sequence } });
                    _ = self.recovery_points.orderedRemove(i);
                } else {
                    i += 1;
                }
            }

            if (pruned_count > 0) {
                emitEvent(self, .{ .retention_applied = .{
                    .pruned_count = pruned_count,
                    .freed_bytes = freed_bytes,
                } });
            }
        }

        pub fn emitEvent(self: *Manager, event: PitrEvent) void {
            if (self.config.on_event) |callback| {
                callback(event);
            }
        }
    };
}
