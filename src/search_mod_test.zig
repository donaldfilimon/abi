//! Focused search unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const search = @import("features/search/mod.zig");

test {
    refAllDecls(search);
}
