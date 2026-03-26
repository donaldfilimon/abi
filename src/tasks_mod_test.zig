//! Focused tasks unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const tasks = @import("tasks/mod.zig");

test {
    _ = tasks;
    std.testing.refAllDecls(tasks);
}
