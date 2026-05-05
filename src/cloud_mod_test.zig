//! Focused cloud unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const cloud = @import("features/cloud/mod.zig");

test {
    refAllDecls(cloud);
}
