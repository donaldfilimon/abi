//! Focused database integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const database_tests = @import("integration/database_test.zig");
const database_core_tests = @import("integration/database_core_test.zig");
const database_surface_tests = @import("integration/database_surface_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
