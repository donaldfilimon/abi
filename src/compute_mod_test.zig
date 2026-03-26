//! Focused compute unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const compute = @import("features/compute/mod.zig");

test {
    std.testing.refAllDecls(compute);
}
