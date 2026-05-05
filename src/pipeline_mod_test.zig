//! Focused pipeline unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const pipeline = @import("features/ai/pipeline/mod.zig");

test {
    refAllDecls(pipeline);
}
