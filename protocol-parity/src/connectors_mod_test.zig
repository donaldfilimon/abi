//! Focused connectors unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const connectors = @import("connectors/mod.zig");

test {
    std.testing.refAllDecls(connectors);
}
