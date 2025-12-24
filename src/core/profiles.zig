//! Legacy compatibility wrapper for `profile.zig`.
//!
//! Prefer importing `profile.zig` directly. This module re-exports the new
//! implementation to avoid duplication while keeping older imports compiling
//! during the migration window.

const std = @import("std");
const builtin = @import("builtin");
const profile = @import("profile.zig");

comptime {
    if (!builtin.is_test) {
        @compileLog("`core/profiles.zig` is deprecated; import `core/profile.zig` instead.");
    }
}

pub const LoggingSink = profile.LoggingSink;
pub const ProfileKind = profile.ProfileKind;
pub const PartialProfileConfig = profile.PartialProfileConfig;
pub const ProfileConfig = profile.ProfileConfig;

pub const parseProfileKind = profile.parseProfileKind;
pub const kindToString = profile.kindToString;
pub const kindFromString = profile.kindFromString;
pub const resolve = profile.resolve;

test {
    std.testing.refAllDecls(@This());
}
