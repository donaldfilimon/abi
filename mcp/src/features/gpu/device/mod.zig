//! GPU device abstraction layer for enumeration, selection, and management.
//!
//! Provides `Device`, `DeviceManager`, and selection criteria types for
//! querying capabilities, vendor info, and platform-specific probing.

const std = @import("std");

// ── Submodules ────────────────────────────────────────────────────────────
pub const types = @import("types.zig");
pub const vendor = @import("vendor.zig");
pub const selector = @import("selector.zig");
pub const enumeration = @import("enumeration.zig");
pub const manager = @import("manager.zig");
pub const android_probe = @import("android_probe.zig");

// ── Re-exported types ─────────────────────────────────────────────────────
pub const Backend = types.Backend;
pub const DeviceCapability = types.DeviceCapability;
pub const DeviceType = types.DeviceType;
pub const Vendor = vendor.Vendor;
pub const Device = types.Device;
pub const DeviceFeature = types.DeviceFeature;
pub const DeviceSelectionCriteria = types.DeviceSelectionCriteria;
pub const DeviceSelector = selector.DeviceSelector;
pub const DeviceManager = manager.DeviceManager;

// ── Re-exported functions ─────────────────────────────────────────────────
pub const discoverDevices = enumeration.discoverDevices;
pub const getBestKernelBackend = enumeration.getBestKernelBackend;
pub const enumerateAllDevices = enumeration.enumerateAllDevices;
pub const enumerateDevicesForBackend = enumeration.enumerateDevicesForBackend;
pub const selectBestDevice = selector.selectBestDevice;

test {
    std.testing.refAllDecls(@This());
}
