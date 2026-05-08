//! Focused connectors unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const connectors = @import("connectors/mod.zig");

test {
    refAllDecls(connectors);
}
