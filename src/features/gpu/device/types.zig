const device = @import("../device.zig");
const std = @import("std");

pub const Backend = device.Backend;
pub const DeviceCapability = device.DeviceCapability;
pub const DeviceType = device.DeviceType;
pub const Vendor = device.Vendor;
pub const Device = device.Device;
pub const DeviceFeature = device.DeviceFeature;
pub const DeviceSelector = device.DeviceSelector;
pub const DeviceSelectionCriteria = device.DeviceSelectionCriteria;

test {
    std.testing.refAllDecls(@This());
}
