//! Focused database unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const database = @import("features/database/mod.zig");

test {
    _ = database;
    std.testing.refAllDecls(database);
}
