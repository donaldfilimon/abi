const backend = @import("../backend.zig");
const std = @import("std");

pub const listBackendInfo = backend.listBackendInfo;
pub const listDevices = backend.listDevices;
pub const defaultDevice = backend.defaultDevice;
pub const defaultDeviceLabel = backend.defaultDeviceLabel;
pub const summary = backend.summary;

test {
    std.testing.refAllDecls(@This());
}
