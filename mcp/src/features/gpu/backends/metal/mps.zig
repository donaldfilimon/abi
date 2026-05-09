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
//!
//! ## Sub-modules
//! - `mps/nn_primitives.zig` — Convolution, FullyConnected, BatchNorm
//! - `mps/graph.zig` — MPSGraph DAG execution

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
// Framework Loading State (pub for sub-module access)
// ============================================================================

pub var mps_lib: ?std.DynLib = null;
pub var mps_graph_lib: ?std.DynLib = null;
var mps_load_attempted = std.atomic.Value(bool).init(false);

// Obj-C runtime pointers (shared with metal.zig)
pub var objc_msgSend_fn: ?*const fn (ID, SEL) callconv(.c) ID = null;
pub var sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null;
pub var objc_get_class_fn: ?*const fn ([*:0]const u8) callconv(.c) ?Class = null;

// Cached selectors
pub var sel_alloc: SEL = undefined;
pub var sel_init: SEL = undefined;
pub var sel_release: SEL = undefined;
pub var sel_initWithDevice: SEL = undefined;
pub var sel_encodeToCommandBuffer: SEL = undefined;

var selectors_loaded = std.atomic.Value(bool).init(false);

/// Initialize MPS by loading the framework and caching selectors.
/// Call after Metal is initialized (needs Obj-C runtime to be loaded).
pub fn init(
    msg_send: *const fn (ID, SEL) callconv(.c) ID,
    sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
    get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
) MpsError!void {
    if (selectors_loaded.load(.acquire)) return;

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

    selectors_loaded.store(true, .release);
}

pub fn deinit() void {
    if (mps_lib) |lib| lib.close();
    if (mps_graph_lib) |lib| lib.close();
    mps_lib = null;
    mps_graph_lib = null;
    mps_load_attempted.store(false, .release);
    selectors_loaded.store(false, .release);
}

pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) return false;
    if (mps_load_attempted.load(.acquire)) return mps_lib != null;
    return tryLoadMps();
}

fn tryLoadMps() bool {
    if (mps_load_attempted.load(.acquire)) return mps_lib != null;
    mps_load_attempted.store(true, .release);

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
// MPS Matrix Multiplication (kept in core — small, tightly coupled to init)
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
// Re-exports from sub-modules
// ============================================================================

const nn = @import("mps/nn_primitives.zig");
pub const ConvolutionDescriptor = nn.ConvolutionDescriptor;
pub const MpsConvolution = nn.MpsConvolution;
pub const MpsFullyConnected = nn.MpsFullyConnected;
pub const MpsBatchNorm = nn.MpsBatchNorm;

const graph_mod = @import("mps/graph.zig");
pub const GraphDataType = graph_mod.GraphDataType;
pub const GraphTensor = graph_mod.GraphTensor;
pub const GraphFeed = graph_mod.GraphFeed;
pub const GraphResult = graph_mod.GraphResult;
pub const MpsGraph = graph_mod.MpsGraph;

// ============================================================================
// Tests
// ============================================================================

test "MPS availability check" {
    // Should not crash regardless of platform
    const available = isAvailable();
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!available);
    }
}

test "MpsMatMul struct layout" {
    var matmul = MpsMatMul{};
    try std.testing.expect(matmul.kernel == null);
    try std.testing.expectEqual(@as(f64, 1.0), matmul.alpha);
    try std.testing.expectEqual(@as(f64, 0.0), matmul.beta);
    matmul.destroy();
}

test "MpsConvolution setWeights validates dimensions" {
    var conv = MpsConvolution{
        .device = @ptrFromInt(1), // Non-null sentinel
        .descriptor = .{
            .kernel_width = 3,
            .kernel_height = 3,
            .input_channels = 3,
            .output_channels = 16,
        },
    };
    // 3*3*3*16 = 432 weights required
    var weights: [432]f32 = undefined;
    try conv.setWeights(&weights, 432);
    try std.testing.expect(conv.weights != null);
    try std.testing.expectEqual(@as(usize, 432), conv.weights_len);

    // Too few weights should fail
    var conv2 = conv;
    conv2.weights = null;
    try std.testing.expectError(MpsError.InvalidDimensions, conv2.setWeights(&weights, 100));
}

test "MpsFullyConnected setWeights validates dimensions" {
    var fc = MpsFullyConnected{
        .device = @ptrFromInt(1),
        .input_features = 128,
        .output_features = 64,
    };
    // 128*64 = 8192 weights required
    var weights: [8192]f32 = undefined;
    var biases: [64]f32 = undefined;
    try fc.setWeights(&weights, 8192, &biases, 64);
    try std.testing.expect(fc.weights != null);
    try std.testing.expect(fc.biases != null);

    // Too few biases should fail
    var fc2 = fc;
    fc2.weights = null;
    fc2.biases = null;
    try std.testing.expectError(MpsError.InvalidDimensions, fc2.setWeights(&weights, 8192, &biases, 32));
}

test "MpsBatchNorm setStatistics validates dimensions" {
    var bn = MpsBatchNorm{
        .device = @ptrFromInt(1),
        .num_features = 16,
    };
    var mean: [16]f32 = undefined;
    var variance: [16]f32 = undefined;
    var gamma: [16]f32 = undefined;
    var beta_arr: [16]f32 = undefined;
    try bn.setStatistics(&mean, &variance, &gamma, &beta_arr, 16);
    try std.testing.expect(bn.mean != null);
    try std.testing.expect(bn.gamma != null);

    // Too few elements should fail
    var bn2 = bn;
    bn2.mean = null;
    try std.testing.expectError(MpsError.InvalidDimensions, bn2.setStatistics(&mean, &variance, &gamma, &beta_arr, 8));
}

test "GraphTensor struct layout" {
    const tensor = GraphTensor{};
    try std.testing.expect(tensor.tensor == null);
    try std.testing.expectEqual(@as(u8, 0), tensor.ndim);
    try std.testing.expectEqual(GraphDataType.float32, tensor.dtype);
    for (tensor.shape) |s| {
        try std.testing.expectEqual(@as(u32, 0), s);
    }

    var t2 = GraphTensor{
        .ndim = 3,
        .dtype = .int32,
    };
    t2.shape[0] = 2;
    t2.shape[1] = 3;
    t2.shape[2] = 4;
    try std.testing.expectEqual(@as(u8, 3), t2.ndim);
    try std.testing.expectEqual(@as(u32, 3), t2.shape[1]);
}

test "GraphDataType enum values" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(GraphDataType.float32));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(GraphDataType.float16));
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(GraphDataType.int32));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(GraphDataType.bool_type));
}

test "GraphFeed initialization" {
    var feed = GraphFeed{};
    try std.testing.expectEqual(@as(u8, 0), feed.name_len);
    try std.testing.expectEqual(@as(usize, 0), feed.data_len);
    for (feed.name) |c| {
        try std.testing.expectEqual(@as(u8, 0), c);
    }

    const name = "input_tensor";
    @memcpy(feed.name[0..name.len], name);
    feed.name_len = name.len;
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    feed.data = &data;
    feed.data_len = data.len;
    try std.testing.expectEqual(@as(u8, 12), feed.name_len);
    try std.testing.expectEqual(@as(usize, 3), feed.data_len);
}

test "GraphResult getData bounds" {
    var result = GraphResult{};
    try std.testing.expect(result.getData(0) == null);
    try std.testing.expect(result.getData(1) == null);
    try std.testing.expect(result.getData(255) == null);

    const allocator = std.testing.allocator;
    var buf = try allocator.alloc(f32, 4);
    buf[0] = 1.0;
    buf[1] = 2.0;
    buf[2] = 3.0;
    buf[3] = 4.0;

    result.outputs[0] = buf;
    result.output_count = 1;
    result.allocator = allocator;

    const data = result.getData(0);
    try std.testing.expect(data != null);
    try std.testing.expectEqual(@as(usize, 4), data.?.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.?[0], 0.001);
    try std.testing.expect(result.getData(1) == null);

    result.deinit();
}

test "MpsGraph.isGraphAvailable check" {
    const available = MpsGraph.isGraphAvailable();
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!available);
    }
}

test {
    _ = @import("mps/nn_primitives.zig");
    _ = @import("mps/graph.zig");
}
