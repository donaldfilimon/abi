const device = @import("../device.zig");
const std = @import("std");

pub const Vendor = device.Vendor;

pub const recommendedBackend = Vendor.recommendedBackend;
pub const displayName = Vendor.displayName;
pub const fromDeviceName = Vendor.fromDeviceName;

test {
    std.testing.refAllDecls(@This());
}
