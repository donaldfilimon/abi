//! Core infrastructure providing platform detection, versioning, and memory utilities.
//!
//! Note: This is a minimal utility module. For most use cases:
//! - Runtime infrastructure: use `src/services/runtime/`
//! - Shared utilities: use `src/internal/` or `src/services/shared/`
//!
//! This module includes:
//! - Platform information detection (OS, architecture, CPU threads)
//! - Version management and comparison
//! - Aligned buffer allocation for cache-line alignment
//! - Hardware topology detection
//!
//! Example:
//! ```zig
//! const info = core.PlatformInfo.detect();
//! std.debug.print("OS: {t}, Arch: {t}\n", .{ info.os, info.arch });
//! ```
const std = @import("std");
const platform = @import("../platform.zig");
const simd = @import("../simd.zig");

pub const Allocator = std.mem.Allocator;
pub const PlatformInfo = platform.PlatformInfo;
pub const Os = platform.Os;
pub const Arch = platform.Arch;
// SIMD functions available through abi.simd

pub const CoreError = error{
    InvalidState,
    OutOfMemory,
};

pub const Version = struct {
    major: u16,
    minor: u16,
    patch: u16,

    pub fn format(self: Version, buffer: []u8) ![]u8 {
        return formatVersion(buffer, self);
    }

    pub fn isZero(self: Version) bool {
        return self.major == 0 and self.minor == 0 and self.patch == 0;
    }

    pub fn toInt(self: Version) u64 {
        return (@as(u64, self.major) << 32) |
            (@as(u64, self.minor) << 16) |
            @as(u64, self.patch);
    }
};

pub const CacheLineBytes: usize = 64;

pub fn AlignedBuffer(comptime alignment: usize) type {
    comptime {
        if (!std.math.isPowerOfTwo(alignment)) {
            @compileError("alignment must be a power of two");
        }
    }

    return struct {
        allocator: std.mem.Allocator,
        bytes: []align(alignment) u8,

        pub fn init(allocator: std.mem.Allocator, size: usize) !@This() {
            const align_enum = std.mem.Alignment.fromByteUnits(alignment);
            const bytes = try allocator.alignedAlloc(u8, align_enum, size);
            return .{ .allocator = allocator, .bytes = bytes };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.bytes);
            self.* = undefined;
        }

        pub fn slice(self: *@This()) []u8 {
            return self.bytes;
        }
    };
}

pub const CacheAlignedBuffer = AlignedBuffer(CacheLineBytes);

pub const HardwareTopology = struct {
    logical_cores: u32,
    physical_cores: u32,
    cache_line_bytes: u32,
    l1_cache_bytes: ?u64 = null,
    l2_cache_bytes: ?u64 = null,
    l3_cache_bytes: ?u64 = null,

    pub fn detect() HardwareTopology {
        const info = PlatformInfo.detect();
        return .{
            .logical_cores = info.max_threads,
            .physical_cores = info.max_threads,
            .cache_line_bytes = @intCast(CacheLineBytes),
            .l1_cache_bytes = null,
            .l2_cache_bytes = null,
            .l3_cache_bytes = null,
        };
    }
};

/// Parse a version string in format "major.minor.patch".
/// @param text Version string to parse (e.g., "1.2.3")
/// @return Parsed version or null if invalid format
pub fn parseVersion(text: []const u8) ?Version {
    var it = std.mem.splitScalar(u8, text, '.');
    const major = it.next() orelse return null;
    const minor = it.next() orelse return null;
    const patch = it.next() orelse return null;
    if (it.next() != null) return null;
    return Version{
        .major = std.fmt.parseInt(u16, major, 10) catch return null,
        .minor = std.fmt.parseInt(u16, minor, 10) catch return null,
        .patch = std.fmt.parseInt(u16, patch, 10) catch return null,
    };
}

/// Parse a version string with optional prefix/suffix (e.g., "v1.2.3-rc1").
/// @param text Version string to parse (may include 'v' prefix, whitespace, or suffixes)
/// @return Parsed version or null if invalid format
pub fn parseVersionLoose(text: []const u8) ?Version {
    var trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (trimmed[0] == 'v' or trimmed[0] == 'V') {
        if (trimmed.len == 1) return null;
        trimmed = trimmed[1..];
    }
    const end = std.mem.indexOfAny(u8, trimmed, "-+") orelse trimmed.len;
    if (end == 0) return null;
    return parseVersion(trimmed[0..end]);
}

/// Compare two versions for ordering.
/// @param a First version to compare
/// @param b Second version to compare
/// @return std.math.Order indicating relationship (.lt, .eq, .gt)
pub fn compareVersion(a: Version, b: Version) std.math.Order {
    if (a.major != b.major) return std.math.order(a.major, b.major);
    if (a.minor != b.minor) return std.math.order(a.minor, b.minor);
    return std.math.order(a.patch, b.patch);
}

/// Format version to string in provided buffer.
/// @param buffer Buffer to write formatted version string
/// @param version Version to format
/// @return Slice of buffer containing formatted string
pub fn formatVersion(buffer: []u8, version: Version) ![]u8 {
    return std.fmt.bufPrint(
        buffer,
        "{d}.{d}.{d}",
        .{ version.major, version.minor, version.patch },
    );
}

/// Format version to newly allocated string.
/// @param allocator Memory allocator for result string
/// @param version Version to format
/// @return Allocated string containing formatted version
pub fn formatVersionAlloc(allocator: std.mem.Allocator, version: Version) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{d}.{d}.{d}",
        .{ version.major, version.minor, version.patch },
    );
}

/// Check if current version satisfies required version constraints.
/// @param required Minimum required version
/// @param current Current version to check
/// @return True if current version is compatible with required version
pub fn isCompatible(required: Version, current: Version) bool {
    if (required.major != current.major) return false;
    return compareVersion(current, required) != .lt;
}

test "parse and format version" {
    const version = parseVersion("1.2.3") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 1), version.major);
    try std.testing.expectEqual(@as(u16, 2), version.minor);
    try std.testing.expectEqual(@as(u16, 3), version.patch);

    var buffer: [32]u8 = undefined;
    const formatted = try formatVersion(&buffer, version);
    try std.testing.expectEqualStrings("1.2.3", formatted);
}

test "compare versions" {
    const a = Version{ .major = 1, .minor = 0, .patch = 0 };
    const b = Version{ .major = 1, .minor = 1, .patch = 0 };
    try std.testing.expect(compareVersion(a, b) == .lt);
    try std.testing.expect(compareVersion(b, a) == .gt);
    try std.testing.expect(compareVersion(a, a) == .eq);
}

test "loose version parsing and compatibility" {
    const version = parseVersionLoose("v2.3.4-beta1") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 2), version.major);
    try std.testing.expectEqual(@as(u16, 3), version.minor);
    try std.testing.expectEqual(@as(u16, 4), version.patch);

    try std.testing.expect(isCompatible(.{ .major = 2, .minor = 0, .patch = 0 }, version));
    try std.testing.expect(!isCompatible(.{ .major = 3, .minor = 0, .patch = 0 }, version));

    const allocator = std.testing.allocator;
    const formatted = try formatVersionAlloc(allocator, version);
    defer allocator.free(formatted);
    try std.testing.expectEqualStrings("2.3.4", formatted);
    try std.testing.expect(!version.isZero());
    try std.testing.expect(version.toInt() > 0);
}

test "cache aligned buffer uses cache line alignment" {
    var buffer = try CacheAlignedBuffer.init(std.testing.allocator, 128);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 128), buffer.bytes.len);
    try std.testing.expect(
        std.mem.isAligned(@intFromPtr(buffer.bytes.ptr), CacheLineBytes),
    );
}

test "hardware topology reports at least one core" {
    const topo = HardwareTopology.detect();
    try std.testing.expect(topo.logical_cores >= 1);
    try std.testing.expect(topo.physical_cores >= 1);
}
