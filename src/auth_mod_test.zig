//! Focused auth unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const auth = @import("features/auth/mod.zig");

test {
    _ = auth;
    std.testing.refAllDecls(auth);
}
