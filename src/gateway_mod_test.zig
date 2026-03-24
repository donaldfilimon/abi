//! Focused gateway unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const gateway = @import("features/gateway/mod.zig");

test {
    _ = gateway;
    std.testing.refAllDecls(gateway);
}
