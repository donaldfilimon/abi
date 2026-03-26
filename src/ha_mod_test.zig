//! Focused HA unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const ha = @import("protocols/ha/mod.zig");

test {
    std.testing.refAllDecls(ha);
}
