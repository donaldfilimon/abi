//! Focused GPU unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const gpu = @import("features/gpu/mod.zig");

test {
    std.testing.refAllDecls(gpu);
}
