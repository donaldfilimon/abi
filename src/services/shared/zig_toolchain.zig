const std = @import("std");

/// Resolve a home directory from platform environment variables.
///
/// Search order:
/// 1. `HOME`
/// 2. `USERPROFILE` (Windows)
pub fn resolveHomeDir() ?[]const u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
    return std.mem.span(home_ptr);
}

/// Allocate the canonical ZVM master Zig compiler path:
/// `~/.zvm/master/zig`.
pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ home, ".zvm", "master", "zig" });
}

