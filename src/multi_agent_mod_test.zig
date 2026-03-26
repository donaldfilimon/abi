//! Focused multi-agent unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const multi_agent = @import("features/ai/multi_agent/mod.zig");

test {
    std.testing.refAllDecls(multi_agent);
}
