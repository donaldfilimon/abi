//! DirectML Operator Wrappers
//!
//! Provides Zig-friendly wrappers for common DirectML operators including
//! matrix multiplication (GEMM), convolution, and activation functions.
//! On non-Windows platforms, execute() returns UnsupportedPlatform.

const std = @import("std");
const builtin = @import("builtin");
const loader = @import("loader.zig");

/// DirectML matrix multiplication operator (GEMM).
pub const DmlMatMul = struct {
    device: ?*loader.DirectMlDevice = null,
    m: u32 = 0,
    n: u32 = 0,
    k: u32 = 0,

    pub fn create(m: u32, n: u32, k: u32) DmlMatMul {
        return .{ .m = m, .n = n, .k = k };
    }

    pub fn execute(self: *const DmlMatMul) loader.DirectMlError!void {
        if (self.device == null and builtin.os.tag != .windows) return loader.DirectMlError.UnsupportedPlatform;
        if (self.device == null) return loader.DirectMlError.DeviceCreationFailed;
        // Would create DML_GEMM_OPERATOR_DESC and execute via IDMLCommandRecorder
    }
};

/// DirectML convolution operator.
pub const DmlConvolution = struct {
    kernel_h: u32 = 3,
    kernel_w: u32 = 3,
    in_channels: u32 = 1,
    out_channels: u32 = 1,
    stride_h: u32 = 1,
    stride_w: u32 = 1,
    pad_h: u32 = 0,
    pad_w: u32 = 0,
};

/// DirectML activation operator.
pub const DmlActivation = struct {
    mode: ActivationMode = .relu,

    pub const ActivationMode = enum {
        relu,
        sigmoid,
        tanh_mode,
    };
};

/// DirectML element-wise addition operator.
pub const DmlElementWiseAdd = struct {
    size: u32 = 0,

    pub fn create(size: u32) DmlElementWiseAdd {
        return .{ .size = size };
    }
};

test "DmlMatMul create" {
    const mm = DmlMatMul.create(32, 64, 16);
    try std.testing.expectEqual(@as(u32, 32), mm.m);
    try std.testing.expectEqual(@as(u32, 64), mm.n);
    try std.testing.expectEqual(@as(u32, 16), mm.k);
    try std.testing.expect(mm.device == null);
}

test "DmlMatMul execute without device" {
    const mm = DmlMatMul.create(32, 32, 32);
    if (builtin.os.tag != .windows) {
        try std.testing.expectError(loader.DirectMlError.UnsupportedPlatform, mm.execute());
    } else {
        try std.testing.expectError(loader.DirectMlError.DeviceCreationFailed, mm.execute());
    }
}

test "DmlConvolution defaults" {
    const conv = DmlConvolution{};
    try std.testing.expectEqual(@as(u32, 3), conv.kernel_h);
    try std.testing.expectEqual(@as(u32, 3), conv.kernel_w);
    try std.testing.expectEqual(@as(u32, 1), conv.in_channels);
    try std.testing.expectEqual(@as(u32, 1), conv.out_channels);
    try std.testing.expectEqual(@as(u32, 1), conv.stride_h);
    try std.testing.expectEqual(@as(u32, 0), conv.pad_h);
}

test "DmlActivation defaults" {
    const act = DmlActivation{};
    try std.testing.expectEqual(DmlActivation.ActivationMode.relu, act.mode);
}

test "DmlElementWiseAdd create" {
    const add = DmlElementWiseAdd.create(256);
    try std.testing.expectEqual(@as(u32, 256), add.size);
}

test {
    std.testing.refAllDecls(@This());
}
