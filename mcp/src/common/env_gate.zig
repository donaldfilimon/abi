const std = @import("std");

pub fn isAuthConfigured() bool {
    return std.c.getenv("ABI_JWT_SECRET") != null;
}

pub fn isDevAuthConfigured() bool {
    return std.c.getenv("ABI_DEV_JWT_SECRET") != null;
}

/// Check if auth can run (alias for isAuthConfigured for parity checks)
pub fn canRunAuth() bool {
    return isAuthConfigured();
}

/// Check if tests can run with auth configured
pub fn canRunTest() bool {
    return isAuthConfigured();
}
