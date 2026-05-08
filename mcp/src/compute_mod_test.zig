//! Focused compute unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const compute = @import("features/compute/mod.zig");

test {
    refAllDecls(compute);
}
