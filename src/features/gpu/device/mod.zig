//! GPU device abstraction layer for enumeration, selection, and management.
//! Provides `Device`, `DeviceManager`, and selection criteria types for
//! querying capabilities, vendor info, and platform-specific probing.

const std = @import("std");

pub const types = @import("types.zig");
pub const vendor = @import("vendor.zig");
pub const selector = @import("selector.zig");
pub const enumeration = @import("enumeration.zig");
pub const manager = @import("manager.zig");
pub const android_probe = @import("android_probe.zig");

pub const Device = types.Device;
pub const DeviceType = types.DeviceType;
pub const DeviceFeature = types.DeviceFeature;
pub const DeviceSelector = types.DeviceSelector;
pub const DeviceSelectionCriteria = types.DeviceSelectionCriteria;
pub const DeviceManager = manager.DeviceManager;

test {
    std.testing.refAllDecls(@This());
}
