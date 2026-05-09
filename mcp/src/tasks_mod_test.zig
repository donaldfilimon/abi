//! Focused tasks unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const tasks = @import("tasks/mod.zig");

test {
    refAllDecls(tasks);
}
