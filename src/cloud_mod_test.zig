//! Focused cloud unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const cloud = @import("features/cloud/mod.zig");

test {
    _ = cloud;
    std.testing.refAllDecls(cloud);
}
