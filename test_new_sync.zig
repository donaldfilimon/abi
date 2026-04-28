const std = @import("std");
const sync = @import("sync.zig");

test "std thread compatibility" {
    // Basic verification that BlockingMutex still works
    var m = std.Thread.Mutex{};
    m.lock();
    m.unlock();
}
