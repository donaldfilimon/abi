const std = @import("std");
pub fn main() void {
    var m = std.Thread.Mutex{};
    m.lock();
    m.unlock();
}
