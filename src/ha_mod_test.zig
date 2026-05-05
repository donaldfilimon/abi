//! Focused HA unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const ha = @import("protocols/ha/mod.zig");

test {
    refAllDecls(ha);
}
