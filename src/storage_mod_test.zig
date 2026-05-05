//! Focused storage unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const storage = @import("features/storage/mod.zig");

test {
    refAllDecls(storage);
}
