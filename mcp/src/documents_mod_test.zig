//! Focused documents unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const documents = @import("features/documents/mod.zig");

test {
    refAllDecls(documents);
}
