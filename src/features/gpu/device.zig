//! GPU Device Abstraction
//!
//! Thin facade re-exporting from `device/` submodules. The canonical
//! implementations live in `device/types.zig`, `device/vendor.zig`,
//! `device/selector.zig`, `device/enumeration.zig`, and `device/manager.zig`.
//!
//! ## Memory Ownership
//!
//! Functions that return `[]Device` allocate memory that the **caller must free**:
//! - `enumerateAllDevices(allocator)` -> caller owns returned slice
//! - `enumerateDevicesForBackend(allocator, backend)` -> caller owns returned slice
//! - `discoverDevices(allocator)` -> caller owns returned slice
//! - `DeviceManager.getDevicesForBackend(allocator, backend)` -> caller owns returned slice
//!
//! Functions that return `?Device` (by value) do **not** require cleanup.
//!
//! Example:
//! ```zig
//! const devices = try device.enumerateAllDevices(allocator);
//! defer allocator.free(devices);  // Caller must free
//! ```

const std = @import("std");
const device_mod = @import("device/mod.zig");

// ── Re-exported types ─────────────────────────────────────────────────────
pub const Backend = device_mod.Backend;
pub const DeviceCapability = device_mod.DeviceCapability;
pub const DeviceType = device_mod.DeviceType;
pub const Vendor = device_mod.Vendor;
pub const Device = device_mod.Device;
pub const DeviceFeature = device_mod.DeviceFeature;
pub const DeviceSelector = device_mod.DeviceSelector;
pub const DeviceSelectionCriteria = device_mod.DeviceSelectionCriteria;
pub const DeviceManager = device_mod.DeviceManager;

// ── Re-exported functions ─────────────────────────────────────────────────
pub const discoverDevices = device_mod.discoverDevices;
pub const getBestKernelBackend = device_mod.getBestKernelBackend;
pub const enumerateAllDevices = device_mod.enumerateAllDevices;
pub const enumerateDevicesForBackend = device_mod.enumerateDevicesForBackend;
pub const selectBestDevice = device_mod.selectBestDevice;

test {
    std.testing.refAllDecls(@This());
}
