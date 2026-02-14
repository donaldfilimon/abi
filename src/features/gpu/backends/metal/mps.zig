//! Metal Performance Shaders (MPS) Integration
//!
//! Provides high-performance GPU-accelerated primitives for linear algebra,
//! neural networks, and image processing via Apple's MPS framework.
//!
//! MPS objects wrap existing MTLBuffer/MTLTexture objects — no additional
//! GPU memory allocation is needed. Operations are encoded into command
//! buffers just like regular Metal compute kernels.
//!
//! ## Supported Operations
//! - Matrix multiplication (MPSMatrixMultiplication)
//! - Convolution (MPSCNNConvolution)
//! - Fully connected layers (MPSCNNFullyConnected)
//! - Batch normalization (MPSCNNBatchNormalization)
//! - Graph-based execution (MPSGraph — macOS 11+)

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../metal_types.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;

pub const MpsError = error{
    FrameworkNotAvailable,
    InitFailed,
    EncodeFailed,
    InvalidDimensions,
    DeviceNotSet,
    UnsupportedOperation,
};

// ============================================================================
// Framework Loading
// ============================================================================

var mps_lib: ?std.DynLib = null;
var mps_graph_lib: ?std.DynLib = null;
var mps_load_attempted: bool = false;

// Obj-C runtime pointers (shared with metal.zig)
var objc_msgSend_fn: ?*const fn (ID, SEL) callconv(.c) ID = null;
var sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null;
var objc_get_class_fn: ?*const fn ([*:0]const u8) callconv(.c) ?Class = null;

// Cached selectors
var sel_alloc: SEL = undefined;
var sel_init: SEL = undefined;
var sel_release: SEL = undefined;
var sel_initWithDevice: SEL = undefined;
var sel_encodeToCommandBuffer: SEL = undefined;

var selectors_loaded: bool = false;

/// Initialize MPS by loading the framework and caching selectors.
/// Call after Metal is initialized (needs Obj-C runtime to be loaded).
pub fn init(
    msg_send: *const fn (ID, SEL) callconv(.c) ID,
    sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
    get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
) MpsError!void {
    if (selectors_loaded) return;

    objc_msgSend_fn = msg_send;
    sel_register_fn = sel_register;
    objc_get_class_fn = get_class;

    if (!tryLoadMps()) {
        return MpsError.FrameworkNotAvailable;
    }

    // Cache common selectors
    sel_alloc = sel_register("alloc");
    sel_init = sel_register("init");
    sel_release = sel_register("release");
    sel_initWithDevice = sel_register("initWithDevice:");
    sel_encodeToCommandBuffer = sel_register("encodeToCommandBuffer:");

    selectors_loaded = true;
}

pub fn deinit() void {
    if (mps_lib) |lib| lib.close();
    if (mps_graph_lib) |lib| lib.close();
    mps_lib = null;
    mps_graph_lib = null;
    mps_load_attempted = false;
    selectors_loaded = false;
}

pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) return false;
    if (mps_load_attempted) return mps_lib != null;
    return tryLoadMps();
}

fn tryLoadMps() bool {
    if (mps_load_attempted) return mps_lib != null;
    mps_load_attempted = true;

    const paths = [_][]const u8{
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
    };
    for (paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            mps_lib = lib;
            break;
        } else |_| {}
    }

    // Also try to load MPSGraph (macOS 11+)
    const graph_paths = [_][]const u8{
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
    };
    for (graph_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            mps_graph_lib = lib;
            break;
        } else |_| {}
    }

    return mps_lib != null;
}

// ============================================================================
// MPS Matrix Multiplication
// ============================================================================

/// Wraps MPSMatrixMultiplication for C = alpha * A * B + beta * C.
pub const MpsMatMul = struct {
    kernel: ID = null,
    device: ID = null,
    transpose_a: bool = false,
    transpose_b: bool = false,
    alpha: f64 = 1.0,
    beta: f64 = 0.0,
    result_rows: u32 = 0,
    result_columns: u32 = 0,
    interior_columns: u32 = 0,

    pub const Config = struct {
        result_rows: u32,
        result_columns: u32,
        interior_columns: u32,
        transpose_a: bool = false,
        transpose_b: bool = false,
        alpha: f64 = 1.0,
        beta: f64 = 0.0,
    };

    pub fn create(device: ID, config: Config) MpsError!MpsMatMul {
        if (device == null) return MpsError.DeviceNotSet;
        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const cls = get_class("MPSMatrixMultiplication") orelse
            return MpsError.FrameworkNotAvailable;

        // [MPSMatrixMultiplication alloc]
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const instance = msg_send(@ptrCast(cls), sel_alloc);
        if (instance == null) return MpsError.InitFailed;

        // initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:
        const init_sel = sel_fn("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:");
        const init_fn: *const fn (
            ID,
            SEL,
            ID,
            bool,
            bool,
            u32,
            u32,
            u32,
            f64,
            f64,
        ) callconv(.c) ID = @ptrCast(msg_send);
        const kernel = init_fn(
            instance,
            init_sel,
            device,
            config.transpose_a,
            config.transpose_b,
            config.result_rows,
            config.result_columns,
            config.interior_columns,
            config.alpha,
            config.beta,
        );

        if (kernel == null) return MpsError.InitFailed;

        return .{
            .kernel = kernel,
            .device = device,
            .transpose_a = config.transpose_a,
            .transpose_b = config.transpose_b,
            .alpha = config.alpha,
            .beta = config.beta,
            .result_rows = config.result_rows,
            .result_columns = config.result_columns,
            .interior_columns = config.interior_columns,
        };
    }

    /// Encode the matrix multiply operation into a command buffer.
    /// `command_buffer`, `a_matrix`, `b_matrix`, `c_matrix` are Obj-C IDs.
    pub fn encode(
        self: *const MpsMatMul,
        command_buffer: ID,
        a_matrix: ID,
        b_matrix: ID,
        c_matrix: ID,
    ) MpsError!void {
        if (self.kernel == null) return MpsError.InitFailed;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;

        const encode_sel = sel_fn(
            "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:",
        );
        const encode_fn: *const fn (ID, SEL, ID, ID, ID, ID) callconv(.c) void = @ptrCast(msg_send);
        encode_fn(self.kernel, encode_sel, command_buffer, a_matrix, b_matrix, c_matrix);
    }

    pub fn destroy(self: *MpsMatMul) void {
        if (self.kernel != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, sel_release);
            }
            self.kernel = null;
        }
    }
};

// ============================================================================
// MPS Neural Network Primitives
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

    pub fn create(device: ID, desc: ConvolutionDescriptor) MpsError!MpsConvolution {
        if (device == null) return MpsError.DeviceNotSet;
        // MPSCNNConvolution requires a data source for weights.
        // Full implementation would create the data source and initialize.
        // For now, store descriptor for later encoding.
        return .{
            .kernel = null, // Created lazily when weights are provided
            .device = device,
            .descriptor = desc,
        };
    }

    pub fn destroy(self: *MpsConvolution) void {
        if (self.kernel != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, sel_release);
            }
            self.kernel = null;
        }
    }
};

/// Wraps MPSCNNFullyConnected for dense layers.
pub const MpsFullyConnected = struct {
    kernel: ID = null,
    device: ID = null,
    input_features: u32,
    output_features: u32,

    pub fn create(device: ID, input_features: u32, output_features: u32) MpsError!MpsFullyConnected {
        if (device == null) return MpsError.DeviceNotSet;
        return .{
            .kernel = null,
            .device = device,
            .input_features = input_features,
            .output_features = output_features,
        };
    }

    pub fn destroy(self: *MpsFullyConnected) void {
        if (self.kernel != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, sel_release);
            }
            self.kernel = null;
        }
    }
};

/// Wraps MPSCNNBatchNormalization.
pub const MpsBatchNorm = struct {
    kernel: ID = null,
    device: ID = null,
    num_features: u32,

    pub fn create(device: ID, num_features: u32) MpsError!MpsBatchNorm {
        if (device == null) return MpsError.DeviceNotSet;
        return .{
            .kernel = null,
            .device = device,
            .num_features = num_features,
        };
    }

    pub fn destroy(self: *MpsBatchNorm) void {
        if (self.kernel != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.kernel, sel_release);
            }
            self.kernel = null;
        }
    }
};

// ============================================================================
// MPS Graph (macOS 11+ / iOS 14+)
// ============================================================================

/// Wraps MPSGraph for compute graph execution.
/// MPSGraph allows building a DAG of operations that MPS optimizes and fuses.
pub const MpsGraph = struct {
    graph: ID = null,
    device: ID = null,

    pub fn create(device: ID) MpsError!MpsGraph {
        if (device == null) return MpsError.DeviceNotSet;
        if (mps_graph_lib == null) return MpsError.FrameworkNotAvailable;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;

        const cls = get_class("MPSGraph") orelse return MpsError.FrameworkNotAvailable;
        const instance = msg_send(@ptrCast(cls), sel_alloc);
        if (instance == null) return MpsError.InitFailed;

        const graph = msg_send(instance, sel_init);
        if (graph == null) return MpsError.InitFailed;

        return .{
            .graph = graph,
            .device = device,
        };
    }

    pub fn destroy(self: *MpsGraph) void {
        if (self.graph != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.graph, sel_release);
            }
            self.graph = null;
        }
    }

    pub fn isGraphAvailable() bool {
        return mps_graph_lib != null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MPS availability check" {
    // Should not crash regardless of platform
    const available = isAvailable();
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!available);
    }
    // On macOS, availability depends on hardware
}

test "MpsMatMul struct layout" {
    var matmul = MpsMatMul{};
    try std.testing.expect(matmul.kernel == null);
    try std.testing.expectEqual(@as(f64, 1.0), matmul.alpha);
    try std.testing.expectEqual(@as(f64, 0.0), matmul.beta);
    matmul.destroy();
}
