//! Focused network unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const network = @import("features/network/mod.zig");

test {
    refAllDecls(network);
}
