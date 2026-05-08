//! Vulkan API version capability helpers.

const std = @import("std");

pub const VulkanVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,

    pub fn atLeast(self: VulkanVersion, other: VulkanVersion) bool {
        if (self.major != other.major) return self.major > other.major;
        if (self.minor != other.minor) return self.minor > other.minor;
        return self.patch >= other.patch;
    }
};

pub const min_linux_version = VulkanVersion{ .major = 1, .minor = 3, .patch = 0 };

pub fn decodeApiVersion(raw: u32) VulkanVersion {
    return .{
        .major = raw >> 22,
        .minor = (raw >> 12) & 0x3ff,
        .patch = raw & 0xfff,
    };
}

pub fn encodeApiVersion(version: VulkanVersion) u32 {
    return (version.major << 22) | (version.minor << 12) | version.patch;
}

pub fn minimumVersionForTarget(os_tag: std.Target.Os.Tag) VulkanVersion {
    return switch (os_tag) {
        .linux => min_linux_version,
        else => .{ .major = 1, .minor = 0, .patch = 0 },
    };
}

pub fn meetsTargetMinimum(os_tag: std.Target.Os.Tag, raw_version: u32) bool {
    const required = minimumVersionForTarget(os_tag);
    return decodeApiVersion(raw_version).atLeast(required);
}

pub fn queryLoaderApiVersion(lib: *std.DynLib) ?u32 {
    const EnumerateFn = *const fn (*u32) callconv(.c) i32;
    const enumerate = lib.lookup(EnumerateFn, "vkEnumerateInstanceVersion") orelse return null;
    var version: u32 = 0;
    if (enumerate(&version) != 0) return null;
    return version;
}

test "encode/decode round-trip" {
    const version = VulkanVersion{ .major = 1, .minor = 3, .patch = 261 };
    const encoded = encodeApiVersion(version);
    const decoded = decodeApiVersion(encoded);
    try std.testing.expectEqual(version.major, decoded.major);
    try std.testing.expectEqual(version.minor, decoded.minor);
    try std.testing.expectEqual(version.patch, decoded.patch);
}

test "linux requires at least vulkan 1.3" {
    try std.testing.expect(!meetsTargetMinimum(.linux, encodeApiVersion(.{ .major = 1, .minor = 2, .patch = 203 })));
    try std.testing.expect(meetsTargetMinimum(.linux, encodeApiVersion(.{ .major = 1, .minor = 3, .patch = 0 })));
}

test {
    std.testing.refAllDecls(@This());
}
