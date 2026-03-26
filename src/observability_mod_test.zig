//! Focused observability unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const observability = @import("features/observability/mod.zig");

test {
    std.testing.refAllDecls(observability);
}
