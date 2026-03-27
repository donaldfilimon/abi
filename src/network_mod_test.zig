//! Focused network unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const network = @import("features/network/mod.zig");

test {
    std.testing.refAllDecls(network);
}
