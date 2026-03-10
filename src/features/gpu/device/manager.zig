const device = @import("../device");
const std = @import("std");

pub const DeviceManager = device.DeviceManager;

test {
    std.testing.refAllDecls(@This());
}
