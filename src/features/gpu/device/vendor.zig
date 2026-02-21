const device = @import("../device.zig");

pub const Vendor = device.Vendor;

pub const recommendedBackend = Vendor.recommendedBackend;
pub const displayName = Vendor.displayName;
pub const fromDeviceName = Vendor.fromDeviceName;
