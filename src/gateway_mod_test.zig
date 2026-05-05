//! Focused gateway unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const gateway = @import("features/gateway/mod.zig");

test {
    refAllDecls(gateway);
}
