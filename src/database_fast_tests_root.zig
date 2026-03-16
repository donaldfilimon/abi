const std = @import("std");
const database = @import("core/database/mod.zig");

test {
    std.testing.refAllDecls(database);
}
