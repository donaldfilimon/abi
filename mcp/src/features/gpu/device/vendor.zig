//! GPU vendor identification and backend recommendation.

const std = @import("std");
const backend_mod = @import("../backend.zig");

pub const Backend = backend_mod.Backend;

/// GPU vendor identification.
pub const Vendor = enum {
    nvidia,
    amd,
    intel,
    apple,
    qualcomm,
    arm,
    mesa,
    microsoft,
    unknown,

    /// Get vendor from device name string.
    pub fn fromDeviceName(name: []const u8) Vendor {
        // Convert to lowercase for case-insensitive matching
        var lower_buf: [256]u8 = undefined;
        const len = @min(name.len, lower_buf.len);
        for (name[0..len], 0..) |c, i| {
            lower_buf[i] = std.ascii.toLower(c);
        }
        const lower = lower_buf[0..len];

        // NVIDIA detection
        if (std.mem.indexOf(u8, lower, "nvidia") != null or
            std.mem.indexOf(u8, lower, "geforce") != null or
            std.mem.indexOf(u8, lower, "quadro") != null or
            std.mem.indexOf(u8, lower, "tesla") != null or
            std.mem.indexOf(u8, lower, "rtx") != null or
            std.mem.indexOf(u8, lower, "gtx") != null)
        {
            return .nvidia;
        }

        // AMD detection
        if (std.mem.indexOf(u8, lower, "amd") != null or
            std.mem.indexOf(u8, lower, "radeon") != null or
            std.mem.indexOf(u8, lower, "vega") != null or
            std.mem.indexOf(u8, lower, "navi") != null or
            std.mem.indexOf(u8, lower, "polaris") != null or
            std.mem.indexOf(u8, lower, "rx ") != null)
        {
            return .amd;
        }

        // Intel detection
        if (std.mem.indexOf(u8, lower, "intel") != null or
            std.mem.indexOf(u8, lower, "iris") != null or
            std.mem.indexOf(u8, lower, "arc") != null or
            std.mem.indexOf(u8, lower, "uhd graphics") != null or
            std.mem.indexOf(u8, lower, "hd graphics") != null)
        {
            return .intel;
        }

        // Apple detection
        if (std.mem.indexOf(u8, lower, "apple") != null or
            std.mem.indexOf(u8, lower, "m1") != null or
            std.mem.indexOf(u8, lower, "m2") != null or
            std.mem.indexOf(u8, lower, "m3") != null or
            std.mem.indexOf(u8, lower, "m4") != null)
        {
            return .apple;
        }

        // Qualcomm detection (Adreno)
        if (std.mem.indexOf(u8, lower, "qualcomm") != null or
            std.mem.indexOf(u8, lower, "adreno") != null)
        {
            return .qualcomm;
        }

        // ARM Mali detection
        if (std.mem.indexOf(u8, lower, "mali") != null or
            std.mem.indexOf(u8, lower, "arm") != null)
        {
            return .arm;
        }

        // Mesa (open source) detection
        if (std.mem.indexOf(u8, lower, "llvmpipe") != null or
            std.mem.indexOf(u8, lower, "softpipe") != null or
            std.mem.indexOf(u8, lower, "mesa") != null or
            std.mem.indexOf(u8, lower, "swrast") != null)
        {
            return .mesa;
        }

        // Microsoft (WARP) detection
        if (std.mem.indexOf(u8, lower, "microsoft") != null or
            std.mem.indexOf(u8, lower, "warp") != null)
        {
            return .microsoft;
        }

        return .unknown;
    }

    /// Get recommended backend for this vendor.
    pub fn recommendedBackend(self: Vendor) Backend {
        return switch (self) {
            .nvidia => .cuda, // CUDA is optimal for NVIDIA
            .amd => .vulkan, // Vulkan works well on AMD
            .intel => .vulkan, // Vulkan or OpenCL for Intel
            .apple => .metal, // Metal is native for Apple
            .qualcomm => .vulkan, // Vulkan for mobile Qualcomm
            .arm => .vulkan, // Vulkan for ARM Mali
            .mesa, .microsoft => .vulkan, // Software rasterizers
            .unknown => .stdgpu, // Fall back to std.gpu
        };
    }

    /// Get vendor display name.
    pub fn displayName(self: Vendor) []const u8 {
        return switch (self) {
            .nvidia => "NVIDIA",
            .amd => "AMD",
            .intel => "Intel",
            .apple => "Apple",
            .qualcomm => "Qualcomm",
            .arm => "ARM",
            .mesa => "Mesa/Open Source",
            .microsoft => "Microsoft",
            .unknown => "Unknown",
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
