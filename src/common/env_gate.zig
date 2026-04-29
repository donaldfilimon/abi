const std = @import("std");

pub fn isAuthConfigured() bool {
    const c = std.c.getenv("ABI_JWT_SECRET");
    return c != null;
}

pub fn isDevAuthConfigured() bool {
    const c = std.c.getenv("ABI_DEV_JWT_SECRET");
    return c != null;
}
