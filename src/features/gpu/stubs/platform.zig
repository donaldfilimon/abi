const std = @import("std");

pub const PlatformCapabilities = struct {
    has_gpu: bool = false,
    has_cuda: bool = false,
    has_metal: bool = false,
    has_vulkan: bool = false,
    has_webgpu: bool = false,
};

pub const BackendSupport = struct {
    supported: bool = false,
    reason: []const u8 = "GPU disabled",
};

pub const GpuVendor = enum { unknown, nvidia, amd, intel, apple, arm, qualcomm };

pub fn isCudaSupported() bool {
    return false;
}
pub fn isMetalSupported() bool {
    return false;
}
pub fn isVulkanSupported() bool {
    return false;
}
pub fn isWebGpuSupported() bool {
    return false;
}
pub fn platformDescription() []const u8 {
    return "GPU disabled at compile time";
}

test {
    std.testing.refAllDecls(@This());
}
