//! Focused desktop unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const desktop = @import("features/desktop/mod.zig");

test {
    std.testing.refAllDecls(desktop);
}
