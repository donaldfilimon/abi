const backend = @import("../backend.zig");
const std = @import("std");

pub const Backend = backend.Backend;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const BackendInfo = backend.BackendInfo;
pub const DeviceCapability = backend.DeviceCapability;
pub const DeviceInfo = backend.DeviceInfo;
pub const Summary = backend.Summary;

test {
    std.testing.refAllDecls(@This());
}
