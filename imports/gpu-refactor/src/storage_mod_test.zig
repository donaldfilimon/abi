//! Focused storage unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const storage = @import("features/storage/mod.zig");

test {
    std.testing.refAllDecls(storage);
}
