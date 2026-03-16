//! Metal capability levels and requirement checks.
//!
//! This module centralizes capability thresholds used by backend selection.

const std = @import("std");
const builtin = @import("builtin");
const gpu_family = @import("gpu_family.zig");

pub const MetalLevel = enum(u8) {
    none = 0,
    metal3 = 3,
    metal4 = 4,

    pub fn name(self: MetalLevel) []const u8 {
        return switch (self) {
            .none => "none",
            .metal3 => "metal3",
            .metal4 => "metal4",
        };
    }

    pub fn atLeast(self: MetalLevel, required: MetalLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(required);
    }
};

/// Minimum level for basic compute shader support (M1/M2/M3+).
pub const required_runtime_level: MetalLevel = .metal3;
/// Preferred level for advanced features (mesh shaders, ray tracing).
pub const preferred_runtime_level: MetalLevel = .metal4;

pub fn levelFromFamily(family: gpu_family.MetalGpuFamily) MetalLevel {
    if (family.isMetal4()) return .metal4;
    if (family.isMetal3()) return .metal3;
    return .none;
}

pub fn levelFromFeatureSet(features: gpu_family.MetalFeatureSet) MetalLevel {
    return levelFromFamily(features.gpu_family);
}

pub fn probeSystemMetalLevel() MetalLevel {
    if (builtin.target.os.tag != .macos and builtin.target.os.tag != .ios) {
        return .none;
    }

    var metal_lib = std.DynLib.open("/System/Library/Frameworks/Metal.framework/Metal") catch {
        return .none;
    };
    defer metal_lib.close();

    var objc_lib = std.DynLib.open("/usr/lib/libobjc.A.dylib") catch {
        return .none;
    };
    defer objc_lib.close();

    const CreateDeviceFn = *const fn () callconv(.c) ?*anyopaque;
    const SelRegisterNameFn = *const fn ([*:0]const u8) callconv(.c) *anyopaque;
    const MsgSendBoolSelFn = *const fn (?*anyopaque, *anyopaque, *anyopaque) callconv(.c) bool;
    const MsgSendBoolFamilyFn = *const fn (?*anyopaque, *anyopaque, u32) callconv(.c) bool;
    const MsgSendVoidFn = *const fn (?*anyopaque, *anyopaque) callconv(.c) void;

    const create_device = metal_lib.lookup(CreateDeviceFn, "MTLCreateSystemDefaultDevice") orelse return .none;
    const sel_register_name = objc_lib.lookup(SelRegisterNameFn, "sel_registerName") orelse return .none;
    const msg_send_bool_sel = objc_lib.lookup(MsgSendBoolSelFn, "objc_msgSend") orelse return .none;
    const msg_send_bool_family = objc_lib.lookup(MsgSendBoolFamilyFn, "objc_msgSend") orelse return .none;
    const msg_send_void = objc_lib.lookup(MsgSendVoidFn, "objc_msgSend");

    const device = create_device() orelse return .none;
    defer if (msg_send_void) |release_fn| {
        const sel_release = sel_register_name("release");
        release_fn(device, sel_release);
    };

    const sel_supports_family = sel_register_name("supportsFamily:");
    const sel_responds_to_selector = sel_register_name("respondsToSelector:");
    if (!msg_send_bool_sel(device, sel_responds_to_selector, sel_supports_family)) {
        return .none;
    }

    const family = gpu_family.detectGpuFamily(
        device,
        sel_supports_family,
        msg_send_bool_family,
    );
    return levelFromFamily(family);
}

test "level mapping from family" {
    try std.testing.expectEqual(MetalLevel.none, levelFromFamily(.apple6));
    try std.testing.expectEqual(MetalLevel.metal3, levelFromFamily(.apple8));
    try std.testing.expectEqual(MetalLevel.metal4, levelFromFamily(.apple9));
}

test "required runtime level is metal3" {
    try std.testing.expectEqual(MetalLevel.metal3, required_runtime_level);
    try std.testing.expectEqual(MetalLevel.metal4, preferred_runtime_level);
}

test {
    std.testing.refAllDecls(@This());
}
