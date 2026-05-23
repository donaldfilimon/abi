const std = @import("std");
const sync = @import("../sync.zig");
const time = @import("../time.zig");

pub const IOStats = struct {
    bytes_read: u64 = 0,
    bytes_written: u64 = 0,
    read_ops: u64 = 0,
    write_ops: u64 = 0,
    errors: u64 = 0,
    last_activity_ms: i64 = 0,
    lock: sync.SpinLock = sync.SpinLock{},

    pub fn recordRead(self: *IOStats, bytes: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_read += bytes;
        self.read_ops += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn recordWrite(self: *IOStats, bytes: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_written += bytes;
        self.write_ops += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn recordError(self: *IOStats) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.errors += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn snapshot(self: *IOStats) IOStatsSnapshot {
        self.lock.lock();
        defer self.lock.unlock();
        return IOStatsSnapshot{
            .bytes_read = self.bytes_read,
            .bytes_written = self.bytes_written,
            .read_ops = self.read_ops,
            .write_ops = self.write_ops,
            .errors = self.errors,
            .last_activity_ms = self.last_activity_ms,
        };
    }

    pub fn reset(self: *IOStats) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_read = 0;
        self.bytes_written = 0;
        self.read_ops = 0;
        self.write_ops = 0;
        self.errors = 0;
        self.last_activity_ms = 0;
    }
};

pub const IOStatsSnapshot = struct {
    bytes_read: u64,
    bytes_written: u64,
    read_ops: u64,
    write_ops: u64,
    errors: u64,
    last_activity_ms: i64,
};

test {
    std.testing.refAllDecls(@This());
}

test "IOStats record and snapshot" {
    var iostats = IOStats{};
    iostats.recordRead(100);
    iostats.recordRead(200);
    iostats.recordWrite(50);
    iostats.recordError();

    const snap = iostats.snapshot();
    try std.testing.expectEqual(@as(u64, 300), snap.bytes_read);
    try std.testing.expectEqual(@as(u64, 50), snap.bytes_written);
    try std.testing.expectEqual(@as(u64, 2), snap.read_ops);
    try std.testing.expectEqual(@as(u64, 1), snap.write_ops);
    try std.testing.expectEqual(@as(u64, 1), snap.errors);
    try std.testing.expect(snap.last_activity_ms > 0);
}

test "IOStats reset" {
    var iostats = IOStats{};
    iostats.recordRead(100);
    iostats.reset();

    const snap = iostats.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.bytes_read);
    try std.testing.expectEqual(@as(u64, 0), snap.bytes_written);
    try std.testing.expectEqual(@as(u64, 0), snap.read_ops);
}
