//! Focused desktop unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const desktop = @import("features/desktop/mod.zig");

test {
    refAllDecls(desktop);
}
