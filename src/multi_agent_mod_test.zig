//! Focused multi-agent unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const multi_agent_tests = @import("features/ai/multi_agent/tests.zig");

test {
    std.testing.refAllDecls(multi_agent_tests);
}
