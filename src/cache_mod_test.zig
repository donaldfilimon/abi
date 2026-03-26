//! Focused cache unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const cache = @import("features/cache/mod.zig");

test {
    std.testing.refAllDecls(cache);
}
