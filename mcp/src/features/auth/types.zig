//! Shared types for the auth feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

pub const AuthError = error{
    AuthDisabled,
    FeatureDisabled,
    InvalidCredentials,
    TokenExpired,
    Unauthorized,
    OutOfMemory,
};

pub const Token = struct {
    raw: []const u8 = "",
    claims: Claims = .{},

    pub const Claims = struct {
        sub: []const u8 = "",
        exp: u64 = 0,
        iat: u64 = 0,
    };
};

pub const Session = struct {
    id: []const u8 = "",
    user_id: []const u8 = "",
    created_at: u64 = 0,
    expires_at: u64 = 0,
};

pub const Permission = enum { read, write, admin };
