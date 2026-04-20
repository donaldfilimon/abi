//! Focused web unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const web = @import("features/web/mod.zig");

test {
    std.testing.refAllDecls(web);
}
