//! Focused ACP unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const acp = @import("protocols/acp/mod.zig");

test {
    _ = acp;
    std.testing.refAllDecls(acp);
}
