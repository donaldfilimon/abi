//! Focused pipeline unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const pipeline = @import("features/ai/pipeline/mod.zig");

test {
    std.testing.refAllDecls(pipeline);
}
