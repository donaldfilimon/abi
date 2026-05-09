//! Focused secrets unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const secrets = @import("foundation/security/secrets.zig");
const secrets_tests = @import("foundation/security/secrets/tests.zig");

test {
    std.testing.refAllDecls(secrets_tests);
}
