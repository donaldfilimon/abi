const std = @import("std");

test "verify std.Thread primitives" {
    var m = std.Thread.Mutex{};
    m.lock();
    m.unlock();

    var rw = std.Thread.RwLock{};
    rw.lockShared();
    rw.unlockShared();
}
