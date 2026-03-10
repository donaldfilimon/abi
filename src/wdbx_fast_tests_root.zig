const std = @import("std");
const wdbx = @import("wdbx");

test {
    // Pull in all WDBX declarations for fast compile-check / unit test
    std.testing.refAllDecls(wdbx);
}
