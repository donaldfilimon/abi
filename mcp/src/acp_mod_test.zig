//! Focused ACP unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const acp = @import("protocols/acp/mod.zig");

test {
    refAllDecls(acp);
}
