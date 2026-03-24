//! Focused agents unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const agents = @import("features/ai/agents/mod.zig");
const agents_tests = @import("features/ai/agents/tests.zig");

test {
    _ = agents;
    std.testing.refAllDecls(agents_tests);
}
