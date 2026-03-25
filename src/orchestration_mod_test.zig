//! Focused orchestration unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const orchestration = @import("features/ai/orchestration/mod.zig");
const orchestration_tests = @import("features/ai/orchestration/tests.zig");

test {
    _ = orchestration;
    std.testing.refAllDecls(orchestration_tests);
}
