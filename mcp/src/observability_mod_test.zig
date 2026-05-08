//! Focused observability unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const observability = @import("features/observability/mod.zig");

test {
    refAllDecls(observability);
}
