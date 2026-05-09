//! Focused messaging unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const messaging = @import("features/messaging/mod.zig");
const messaging_tests = @import("features/messaging/tests.zig");

test {
    std.testing.refAllDecls(messaging_tests);
}
