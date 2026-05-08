//! Focused PITR unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const pitr = @import("protocols/ha/pitr.zig");
const pitr_tests = @import("protocols/ha/pitr/tests.zig");

test {
    std.testing.refAllDecls(pitr_tests);
}
