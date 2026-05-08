//! MPS Neural Network Primitives
//!
//! Wraps MPS CNN primitives: Convolution, FullyConnected, BatchNormalization.

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../../metal_types.zig");
const mps_core = @import("../mps.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;
const MpsError = mps_core.MpsError;

// ============================================================================
// MPS Convolution
// ============================================================================

/// Convolution descriptor for creating MPS convolution kernels.
pub const ConvolutionDescriptor = struct {
    kernel_width: u32,
    kernel_height: u32,
    input_channels: u32,
    output_channels: u32,
    stride_x: u32 = 1,
    stride_y: u32 = 1,
    padding: Padding = .same,

    pub const Padding = enum { valid, same };
};

/// Wraps MPSCNNConvolution for GPU-accelerated 2D convolution.
pub const MpsConvolution = struct {
    kernel: ID = null,
    device: ID = null,
    descriptor: ConvolutionDescriptor = .{
        .kernel_width = 3,
        .kernel_height = 3,
        .input_channels = 1,
        .output_channels = 1,
    },

    weights: ?[*]const f32 = null,
    weights_len: usize = 0,

    pub fn create(device: ID, desc: ConvolutionDescriptor) MpsError!MpsConvolution {
        if (device == null) return MpsError.DeviceNotSet;
        return .{
            .kernel = null,
            .device = device,
            .descriptor = desc,
        };
    }

    /// Set convolution weights. Must be called before encode().
    /// `weights` must have length >= kernel_width * kernel_height * input_channels * output_channels.
    pub fn setWeights(self: *MpsConvolution, weights: [*]const f32, len: usize) MpsError!void {
        const required = @as(usize, self.descriptor.kernel_width) *
            self.descriptor.kernel_height *
            self.descriptor.input_channels *
            self.descriptor.output_channels;
        if (len < required) return MpsError.InvalidDimensions;
        self.weights = weights;
        self.weights_len = len;
    }

    /// Encode the convolution into a command buffer.
    /// On non-macOS or when Obj-C runtime is unavailable, returns UnsupportedOperation.
    pub fn encode(self: *const MpsConvolution, command_buffer: ID, source: ID, destination: ID) MpsError!void {
        if (self.weights == null) return MpsError.InvalidDimensions;
        if (self.device == null) return MpsError.DeviceNotSet;

        // Obj-C kernel creation requires runtime — gate for actual macOS execution
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        if (self.kernel) |k| {
            const encode_sel = sel_fn("encodeToCommandBuffer:sourceImage:destinationImage:");
            const encode_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) void = @ptrCast(msg_send);
            encode_fn(k, encode_sel, command_buffer, source, destination);
        } else {
            // Without a live kernel, we cannot encode
            return MpsError.InitFailed;
        }
    }

    pub fn destroy(self: *MpsConvolution) void {
        if (self.kernel != null) {
            if (mps_core.objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, mps_core.sel_release);
            }
            self.kernel = null;
        }
    }
};

// ============================================================================
// MPS Fully Connected
// ============================================================================

/// Wraps MPSCNNFullyConnected for dense layers.
pub const MpsFullyConnected = struct {
    kernel: ID = null,
    device: ID = null,
    input_features: u32,
    output_features: u32,
    weights: ?[*]const f32 = null,
    weights_len: usize = 0,
    biases: ?[*]const f32 = null,
    biases_len: usize = 0,

    pub fn create(device: ID, input_features: u32, output_features: u32) MpsError!MpsFullyConnected {
        if (device == null) return MpsError.DeviceNotSet;
        if (input_features == 0 or output_features == 0) return MpsError.InvalidDimensions;
        return .{
            .kernel = null,
            .device = device,
            .input_features = input_features,
            .output_features = output_features,
        };
    }

    /// Set weights and optional biases.
    /// `weights` must have length >= input_features * output_features.
    /// `biases` (if non-null) must have length >= output_features.
    pub fn setWeights(
        self: *MpsFullyConnected,
        weights: [*]const f32,
        weights_len: usize,
        biases: ?[*]const f32,
        biases_len: usize,
    ) MpsError!void {
        const required_weights = @as(usize, self.input_features) * self.output_features;
        if (weights_len < required_weights) return MpsError.InvalidDimensions;
        if (biases != null and biases_len < self.output_features) return MpsError.InvalidDimensions;
        self.weights = weights;
        self.weights_len = weights_len;
        self.biases = biases;
        self.biases_len = biases_len;
    }

    /// Encode the fully-connected layer into a command buffer.
    pub fn encode(self: *const MpsFullyConnected, command_buffer: ID, source: ID, destination: ID) MpsError!void {
        if (self.weights == null) return MpsError.InvalidDimensions;
        if (self.device == null) return MpsError.DeviceNotSet;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        if (self.kernel) |k| {
            const encode_sel = sel_fn("encodeToCommandBuffer:sourceImage:destinationImage:");
            const encode_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) void = @ptrCast(msg_send);
            encode_fn(k, encode_sel, command_buffer, source, destination);
        } else {
            return MpsError.InitFailed;
        }
    }

    pub fn destroy(self: *MpsFullyConnected) void {
        if (self.kernel != null) {
            if (mps_core.objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, mps_core.sel_release);
            }
            self.kernel = null;
        }
    }
};

// ============================================================================
// MPS Batch Normalization
// ============================================================================

/// Wraps MPSCNNBatchNormalization.
pub const MpsBatchNorm = struct {
    kernel: ID = null,
    device: ID = null,
    num_features: u32,
    mean: ?[*]const f32 = null,
    variance: ?[*]const f32 = null,
    gamma: ?[*]const f32 = null,
    beta: ?[*]const f32 = null,
    stats_len: usize = 0,

    pub fn create(device: ID, num_features: u32) MpsError!MpsBatchNorm {
        if (device == null) return MpsError.DeviceNotSet;
        if (num_features == 0) return MpsError.InvalidDimensions;
        return .{
            .kernel = null,
            .device = device,
            .num_features = num_features,
        };
    }

    /// Set batch normalization statistics.
    /// All arrays must have length >= num_features.
    pub fn setStatistics(
        self: *MpsBatchNorm,
        mean: [*]const f32,
        variance: [*]const f32,
        gamma: [*]const f32,
        beta_param: [*]const f32,
        len: usize,
    ) MpsError!void {
        if (len < self.num_features) return MpsError.InvalidDimensions;
        self.mean = mean;
        self.variance = variance;
        self.gamma = gamma;
        self.beta = beta_param;
        self.stats_len = len;
    }

    /// Encode the batch normalization into a command buffer.
    pub fn encode(self: *const MpsBatchNorm, command_buffer: ID, source: ID, destination: ID) MpsError!void {
        if (self.mean == null) return MpsError.InvalidDimensions;
        if (self.device == null) return MpsError.DeviceNotSet;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        if (self.kernel) |k| {
            const encode_sel = sel_fn("encodeToCommandBuffer:sourceImage:destinationImage:");
            const encode_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) void = @ptrCast(msg_send);
            encode_fn(k, encode_sel, command_buffer, source, destination);
        } else {
            return MpsError.InitFailed;
        }
    }

    pub fn destroy(self: *MpsBatchNorm) void {
        if (self.kernel != null) {
            if (mps_core.objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, mps_core.sel_release);
            }
            self.kernel = null;
        }
    }
};
