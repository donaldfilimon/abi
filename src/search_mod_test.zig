//! Focused search unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const search = @import("features/search/mod.zig");

test {
    _ = search;
    std.testing.refAllDecls(search);
}
