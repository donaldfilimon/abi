//! Focused web unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const web = @import("features/web/mod.zig");

test {
    refAllDecls(web);
}
