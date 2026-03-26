//! Focused tasks unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const tasks = @import("tasks/mod.zig");

test {
    std.testing.refAllDecls(tasks);
}
