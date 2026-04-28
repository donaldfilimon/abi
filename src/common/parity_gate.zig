const std = @import("std");

pub fn canRunAuth() bool {
    const v = std.c.getenv("ABI_JWT_SECRET");
    return v != null;
}

pub fn canRunTest() bool {
    const v = std.c.getenv("ABI_JWT_SECRET");
    return v != null;
}
