//! TPU Device Loader
//!
//! Handles TPU detection and loading.

const std = @import("std");
const builtin = @import("builtin");

pub const TpuDevice = struct {
    id: u32,
    name: []const u8,
    available: bool,
};

pub fn canLoadTPU() bool {
    if (builtin.os.tag == .windows) return false;
    if (builtin.os.tag == .freebsd) return false;
    return false;
}

pub fn detectTPUs(allocator: std.mem.Allocator) ![]TpuDevice {
    if (!canLoadTPU()) {
        return &[_]TpuDevice{};
    }
    var devices = try allocator.alloc(TpuDevice, 1);
    devices[0] = .{
        .id = 0,
        .name = "TPU0",
        .available = true,
    };
    return devices;
}

pub fn getDefaultTPU() ?TpuDevice {
    if (!canLoadTPU()) return null;
    return .{ .id = 0, .name = "TPU0", .available = true };
}
