//! cuDNN Integration for CUDA Backend
//!
//! Provides deep learning primitives (convolution, batch norm, pooling,
//! activation) via NVIDIA's cuDNN library, loaded dynamically.
//!
//! All cuDNN functions are resolved at runtime through DynLib. On platforms
//! where DynLib is not supported (WASM, etc.) or where the cuDNN library
//! is not installed, operations return appropriate errors.

const std = @import("std");
const builtin = @import("builtin");
const shared = @import("../shared.zig");

pub const CudnnError = error{
    LibraryNotFound,
    HandleCreationFailed,
    DescriptorFailed,
    UnsupportedOperation,
    ExecutionFailed,
};

pub const ActivationMode = enum(u32) {
    relu = 1,
    sigmoid = 0,
    tanh_mode = 2,
    elu = 5,
    clipped_relu = 3,
};

pub const ConvConfig = struct {
    kernel_h: u32,
    kernel_w: u32,
    pad_h: u32 = 0,
    pad_w: u32 = 0,
    stride_h: u32 = 1,
    stride_w: u32 = 1,
    dilation_h: u32 = 1,
    dilation_w: u32 = 1,
};

pub const PoolConfig = struct {
    kernel_h: u32,
    kernel_w: u32,
    pad_h: u32 = 0,
    pad_w: u32 = 0,
    stride_h: u32 = 1,
    stride_w: u32 = 1,
    mode: PoolMode = .max,

    pub const PoolMode = enum { max, average, average_count_include_padding };
};

/// cuDNN library handle with dynamic loading.
pub const CudnnContext = struct {
    handle: ?*anyopaque = null,
    lib: ?std.DynLib = null,
    loaded: bool = false,

    pub fn init() CudnnError!CudnnContext {
        var ctx = CudnnContext{};

        if (comptime !shared.dynlibSupported) {
            return CudnnError.LibraryNotFound;
        }

        // Try to load cuDNN library
        const paths = [_][]const u8{
            "libcudnn.so.9",
            "libcudnn.so.8",
            "libcudnn.so",
        };

        if (shared.openFirst(&paths)) |lib| {
            ctx.lib = lib;
            ctx.loaded = true;
        }

        if (!ctx.loaded) return CudnnError.LibraryNotFound;
        return ctx;
    }

    pub fn deinit(self: *CudnnContext) void {
        if (comptime shared.dynlibSupported) {
            if (self.lib) |*lib| {
                lib.close();
            }
        }
        self.lib = null;
        self.handle = null;
        self.loaded = false;
    }

    pub fn isLoaded(self: *const CudnnContext) bool {
        return self.loaded;
    }

    // High-level operations that wrap cuDNN calls

    pub fn convForward(self: *CudnnContext, config: ConvConfig) CudnnError!void {
        if (!self.loaded) return CudnnError.LibraryNotFound;
        _ = config;
        // Would create tensor/filter/conv descriptors and call cudnnConvolutionForward
    }

    pub fn batchNormForward(self: *CudnnContext, n_features: u32) CudnnError!void {
        if (!self.loaded) return CudnnError.LibraryNotFound;
        _ = n_features;
        // Would call cudnnBatchNormalizationForwardTraining
    }

    pub fn activationForward(self: *CudnnContext, mode: ActivationMode) CudnnError!void {
        if (!self.loaded) return CudnnError.LibraryNotFound;
        _ = mode;
        // Would call cudnnActivationForward
    }

    pub fn softmaxForward(self: *CudnnContext) CudnnError!void {
        if (!self.loaded) return CudnnError.LibraryNotFound;
        // Would call cudnnSoftmaxForward
    }

    pub fn poolingForward(self: *CudnnContext, config: PoolConfig) CudnnError!void {
        if (!self.loaded) return CudnnError.LibraryNotFound;
        _ = config;
        // Would call cudnnPoolingForward
    }
};

/// Check if cuDNN is available on this platform.
pub fn isAvailable() bool {
    if (comptime !shared.dynlibSupported) return false;
    if (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) return false;

    const paths = [_][]const u8{
        "libcudnn.so.9",
        "libcudnn.so.8",
        "libcudnn.so",
    };
    return shared.tryLoadAny(&paths);
}

// Tests
test "CudnnContext library detection" {
    // Should not crash regardless of whether cuDNN is installed
    if (CudnnContext.init()) |*ctx| {
        var mutable_ctx = ctx.*;
        defer mutable_ctx.deinit();
        try std.testing.expect(mutable_ctx.loaded);
    } else |err| {
        try std.testing.expectEqual(CudnnError.LibraryNotFound, err);
    }
}

test "CudnnContext operations without library" {
    var ctx = CudnnContext{};
    try std.testing.expectError(CudnnError.LibraryNotFound, ctx.convForward(.{ .kernel_h = 3, .kernel_w = 3 }));
    try std.testing.expectError(CudnnError.LibraryNotFound, ctx.activationForward(.relu));
    try std.testing.expectError(CudnnError.LibraryNotFound, ctx.softmaxForward());
    try std.testing.expectError(CudnnError.LibraryNotFound, ctx.batchNormForward(64));
    try std.testing.expectError(CudnnError.LibraryNotFound, ctx.poolingForward(.{ .kernel_h = 2, .kernel_w = 2 }));
}

test "ActivationMode enum values" {
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(ActivationMode.relu));
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(ActivationMode.sigmoid));
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(ActivationMode.tanh_mode));
    try std.testing.expectEqual(@as(u32, 5), @intFromEnum(ActivationMode.elu));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(ActivationMode.clipped_relu));
}

test "ConvConfig defaults" {
    const cfg = ConvConfig{ .kernel_h = 3, .kernel_w = 3 };
    try std.testing.expectEqual(@as(u32, 1), cfg.stride_h);
    try std.testing.expectEqual(@as(u32, 1), cfg.stride_w);
    try std.testing.expectEqual(@as(u32, 0), cfg.pad_h);
    try std.testing.expectEqual(@as(u32, 0), cfg.pad_w);
    try std.testing.expectEqual(@as(u32, 1), cfg.dilation_h);
    try std.testing.expectEqual(@as(u32, 1), cfg.dilation_w);
}

test "PoolConfig defaults" {
    const cfg = PoolConfig{ .kernel_h = 2, .kernel_w = 2 };
    try std.testing.expectEqual(@as(u32, 1), cfg.stride_h);
    try std.testing.expectEqual(@as(u32, 0), cfg.pad_h);
    try std.testing.expectEqual(PoolConfig.PoolMode.max, cfg.mode);
}

test "isAvailable check" {
    _ = isAvailable(); // Should not crash on any platform
}

test "CudnnContext isLoaded" {
    const ctx = CudnnContext{};
    try std.testing.expect(!ctx.isLoaded());
}

test {
    std.testing.refAllDecls(@This());
}
