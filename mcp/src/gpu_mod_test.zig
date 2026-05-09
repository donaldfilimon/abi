//! Focused GPU unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const gpu = @import("features/gpu/mod.zig");

test {
    refAllDecls(gpu);
}
