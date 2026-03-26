//! Focused inference unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const inference = @import("inference/mod.zig");

test {
    std.testing.refAllDecls(inference);
}
