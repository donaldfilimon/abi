const std = @import("std");
const database = @import("database/mod.zig");

test {
    std.testing.refAllDecls(database);
}
