//! DirectML Dynamic Library Loader
//!
//! Handles runtime loading of DirectML.dll and D3D12.dll on Windows.
//! On non-Windows platforms, all operations return UnsupportedPlatform.

const std = @import("std");
const builtin = @import("builtin");
const shared = @import("../shared.zig");

pub const DirectMlError = error{
    LibraryNotFound,
    DeviceCreationFailed,
    UnsupportedPlatform,
};

pub fn canLoadDirectML() bool {
    if (builtin.os.tag != .windows) return false;
    if (comptime !shared.dynlibSupported) return false;

    const paths = [_][]const u8{"DirectML.dll"};
    return shared.tryLoadAny(&paths);
}

/// DirectML device wrapper.
///
/// Encapsulates the D3D12 device, DirectML device, and command queue
/// required for executing DirectML operators.
pub const DirectMlDevice = struct {
    d3d12_device: ?*anyopaque = null,
    dml_device: ?*anyopaque = null,
    command_queue: ?*anyopaque = null,

    pub fn init() DirectMlError!DirectMlDevice {
        if (builtin.os.tag != .windows) return DirectMlError.UnsupportedPlatform;
        // Would call D3D12CreateDevice + DMLCreateDevice1
        return .{};
    }

    pub fn deinit(self: *DirectMlDevice) void {
        self.d3d12_device = null;
        self.dml_device = null;
        self.command_queue = null;
    }

    pub fn isReady(self: *const DirectMlDevice) bool {
        return self.dml_device != null and self.d3d12_device != null;
    }
};

test "canLoadDirectML platform check" {
    if (builtin.os.tag != .windows) {
        try std.testing.expect(!canLoadDirectML());
    }
}

test "DirectMlDevice init on non-windows" {
    if (builtin.os.tag != .windows) {
        try std.testing.expectError(DirectMlError.UnsupportedPlatform, DirectMlDevice.init());
    }
}

test "DirectMlDevice deinit" {
    var device = DirectMlDevice{};
    device.deinit();
    try std.testing.expect(device.d3d12_device == null);
    try std.testing.expect(device.dml_device == null);
    try std.testing.expect(device.command_queue == null);
}

test "DirectMlDevice isReady" {
    const device = DirectMlDevice{};
    try std.testing.expect(!device.isReady());
}

test {
    std.testing.refAllDecls(@This());
}
