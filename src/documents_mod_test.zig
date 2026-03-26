//! Focused documents unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const documents = @import("features/documents/mod.zig");

test {
    std.testing.refAllDecls(documents);
}
