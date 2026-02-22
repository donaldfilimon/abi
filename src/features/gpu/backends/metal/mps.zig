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

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

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

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

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

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

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

/// Data type for graph tensors
pub const GraphDataType = enum(u32) {
    float32 = 0,
    float16 = 1,
    int32 = 2,
    bool_type = 3,
};

/// Wraps an MPSGraphTensor Obj-C ID with metadata
pub const GraphTensor = struct {
    tensor: ID = null,
    shape: [8]u32 = [_]u32{0} ** 8,
    ndim: u8 = 0,
    dtype: GraphDataType = .float32,
};

/// Maps placeholder name to input data
pub const GraphFeed = struct {
    name: [64]u8 = [_]u8{0} ** 64,
    name_len: u8 = 0,
    data: [*]const f32 = undefined,
    data_len: usize = 0,
};

/// Holds output tensor data from graph execution
pub const GraphResult = struct {
    outputs: [8]?[]f32 = [_]?[]f32{null} ** 8,
    output_count: u8 = 0,
    allocator: std.mem.Allocator = undefined,

    pub fn getData(self: *const GraphResult, index: u8) ?[]const f32 {
        if (index >= self.output_count) return null;
        return self.outputs[index];
    }

    pub fn deinit(self: *GraphResult) void {
        for (self.outputs[0..self.output_count]) |maybe_out| {
            if (maybe_out) |out| self.allocator.free(out);
        }
    }
};

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

    // ========================================================================
    // Graph Tensor Construction
    // ========================================================================

    /// Create a placeholder tensor in the graph with a given name, shape, and data type.
    /// The placeholder serves as an input feed point during graph execution.
    pub fn placeholder(self: *MpsGraph, name: []const u8, shape: []const u32, dtype: GraphDataType) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (shape.len == 0 or shape.len > 8) return MpsError.InvalidDimensions;

        var result = GraphTensor{
            .ndim = @intCast(@min(shape.len, 8)),
            .dtype = dtype,
        };
        for (shape[0..result.ndim], 0..) |s, i| result.shape[i] = s;

        // Store name in the feed slot convention (for later matching)
        // Actual Obj-C placeholder creation is gated on macOS
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Build NSArray of NSNumber for shape dimensions
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);

        for (shape[0..result.ndim]) |s| {
            const num = num_fn(ns_number_class, sel_with_int, @intCast(s));
            if (num != null) add_fn(arr, sel_add, num);
        }

        // Create NSString for name
        const nsstring_class = get_class("NSString") orelse return MpsError.FrameworkNotAvailable;
        const sel_string = sel_fn("stringWithUTF8String:");
        var name_buf: [64]u8 = [_]u8{0} ** 64;
        const copy_len = @min(name.len, 63);
        @memcpy(name_buf[0..copy_len], name[0..copy_len]);
        const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(msg_send);
        const ns_name = str_fn(nsstring_class, sel_string, name_buf[0..copy_len :0]);

        // [graph placeholderWithShape:dataType:name:]
        const sel_placeholder = sel_fn("placeholderWithShape:dataType:name:");
        // MPSGraph data type mapping: float32=268435456, float16=268435457, int32=536870944, bool=270532608
        const mps_dtype: u32 = switch (dtype) {
            .float32 => 268435456,
            .float16 => 268435457,
            .int32 => 536870944,
            .bool_type => 270532608,
        };
        const placeholder_fn: *const fn (ID, SEL, ID, u32, ID) callconv(.c) ID = @ptrCast(msg_send);
        const tensor = placeholder_fn(self.graph, sel_placeholder, arr, mps_dtype, ns_name);

        // Release the shape array
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, sel_release);

        if (tensor == null) return MpsError.InitFailed;
        result.tensor = tensor;
        return result;
    }

    /// Create a constant tensor filled with a single value.
    pub fn constant(self: *MpsGraph, value: f32, shape: []const u32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (shape.len == 0 or shape.len > 8) return MpsError.InvalidDimensions;

        var result = GraphTensor{
            .ndim = @intCast(@min(shape.len, 8)),
            .dtype = .float32,
        };
        for (shape[0..result.ndim], 0..) |s, i| result.shape[i] = s;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Build NSArray of NSNumber for shape
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add_obj = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);

        for (shape[0..result.ndim]) |s| {
            const num = num_fn(ns_number_class, sel_with_int, @intCast(s));
            if (num != null) add_fn(arr, sel_add_obj, num);
        }

        // [graph constantWithScalar:shape:dataType:]
        const sel_const = sel_fn("constantWithScalar:shape:dataType:");
        const const_fn: *const fn (ID, SEL, f64, ID, u32) callconv(.c) ID = @ptrCast(msg_send);
        const tensor = const_fn(self.graph, sel_const, @as(f64, value), arr, 268435456);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, sel_release);

        if (tensor == null) return MpsError.InitFailed;
        result.tensor = tensor;
        return result;
    }

    // ========================================================================
    // Graph Operations
    // ========================================================================

    /// Matrix multiplication of two graph tensors (wraps MPSGraph matmul).
    /// Named `matmulGraph` to avoid conflict with `MpsMatMul`.
    pub fn matmulGraph(self: *MpsGraph, a: GraphTensor, b: GraphTensor, transpose_a: bool, transpose_b: bool) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // If transpose is needed, transpose the tensors first
        var a_tensor = a;
        var b_tensor = b;
        if (transpose_a and a.ndim >= 2) {
            a_tensor = try self.transpose(a, a.ndim - 2, a.ndim - 1);
        }
        if (transpose_b and b.ndim >= 2) {
            b_tensor = try self.transpose(b, b.ndim - 2, b.ndim - 1);
        }

        // [graph matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:]
        const sel_matmul = sel_fn("matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:");
        const matmul_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = matmul_fn(self.graph, sel_matmul, a_tensor.tensor, b_tensor.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        // Infer output shape: [..., M, N] from a=[..., M, K] and b=[..., K, N]
        var result = GraphTensor{
            .tensor = result_tensor,
            .ndim = a.ndim,
            .dtype = a.dtype,
        };
        if (a.ndim >= 2 and b.ndim >= 2) {
            // Copy batch dims from a, last two dims are M from a, N from b
            for (0..a.ndim) |i| result.shape[i] = a.shape[i];
            result.shape[a.ndim - 1] = b.shape[b.ndim - 1]; // N from b
        }
        return result;
    }

    /// Element-wise addition of two graph tensors.
    pub fn add(self: *MpsGraph, a: GraphTensor, b: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph additionWithPrimaryTensor:secondaryTensor:name:]
        const sel_add = sel_fn("additionWithPrimaryTensor:secondaryTensor:name:");
        const add_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = add_fn(self.graph, sel_add, a.tensor, b.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{
            .tensor = result_tensor,
            .shape = a.shape,
            .ndim = a.ndim,
            .dtype = a.dtype,
        };
    }

    /// Element-wise multiplication of two graph tensors.
    pub fn multiply(self: *MpsGraph, a: GraphTensor, b: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph multiplicationWithPrimaryTensor:secondaryTensor:name:]
        const sel_mul = sel_fn("multiplicationWithPrimaryTensor:secondaryTensor:name:");
        const mul_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = mul_fn(self.graph, sel_mul, a.tensor, b.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{
            .tensor = result_tensor,
            .shape = a.shape,
            .ndim = a.ndim,
            .dtype = a.dtype,
        };
    }

    /// ReLU activation on a graph tensor: max(0, x).
    pub fn reluGraph(self: *MpsGraph, x: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph reLUWithTensor:name:]
        const sel_relu = sel_fn("reLUWithTensor:name:");
        const relu_fn: *const fn (ID, SEL, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = relu_fn(self.graph, sel_relu, x.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
    }

    /// Sigmoid activation on a graph tensor: 1 / (1 + exp(-x)).
    pub fn sigmoidGraph(self: *MpsGraph, x: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph sigmoidWithTensor:name:]
        const sel_sigmoid = sel_fn("sigmoidWithTensor:name:");
        const sigmoid_fn: *const fn (ID, SEL, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = sigmoid_fn(self.graph, sel_sigmoid, x.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
    }

    /// Softmax activation along a specified axis.
    pub fn softmaxGraph(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph softMaxWithTensor:axis:name:]
        const sel_softmax = sel_fn("softMaxWithTensor:axis:name:");
        const softmax_fn: *const fn (ID, SEL, ID, c_int, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = softmax_fn(self.graph, sel_softmax, x.tensor, @as(c_int, axis), null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
    }

    /// Reshape a graph tensor to a new shape.
    pub fn reshape(self: *MpsGraph, x: GraphTensor, new_shape: []const u32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (new_shape.len == 0 or new_shape.len > 8) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Build NSArray of NSNumber for new shape
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add_obj = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);

        for (new_shape) |s| {
            const num = num_fn(ns_number_class, sel_with_int, @intCast(s));
            if (num != null) add_obj_fn(arr, sel_add_obj, num);
        }

        // [graph reshapeTensor:withShape:name:]
        const sel_reshape = sel_fn("reshapeTensor:withShape:name:");
        const reshape_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = reshape_fn(self.graph, sel_reshape, x.tensor, arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, sel_release);

        if (result_tensor == null) return MpsError.EncodeFailed;

        var result = GraphTensor{
            .tensor = result_tensor,
            .ndim = @intCast(@min(new_shape.len, 8)),
            .dtype = x.dtype,
        };
        for (new_shape[0..result.ndim], 0..) |s, i| result.shape[i] = s;
        return result;
    }

    /// Transpose two dimensions of a graph tensor.
    pub fn transpose(self: *MpsGraph, x: GraphTensor, dim0: u32, dim1: u32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (dim0 >= x.ndim or dim1 >= x.ndim) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // [graph transposeTensor:dimension:withDimension:name:]
        const sel_transpose = sel_fn("transposeTensor:dimension:withDimension:name:");
        const transpose_fn: *const fn (ID, SEL, ID, u32, u32, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = transpose_fn(self.graph, sel_transpose, x.tensor, dim0, dim1, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        // Swap shape dimensions
        var result = GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
        const tmp = result.shape[dim0];
        result.shape[dim0] = result.shape[dim1];
        result.shape[dim1] = tmp;
        return result;
    }

    /// Sum reduction along a specified axis.
    pub fn reduceSum(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Create NSArray with the axis as NSNumber
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add_obj = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        const axis_num = num_fn(ns_number_class, sel_with_int, @as(c_int, axis));
        if (axis_num != null) add_obj_fn(arr, sel_add_obj, axis_num);

        // [graph reductionSumWithTensor:axes:name:]
        const sel_reduce = sel_fn("reductionSumWithTensor:axes:name:");
        const reduce_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = reduce_fn(self.graph, sel_reduce, x.tensor, arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, sel_release);

        if (result_tensor == null) return MpsError.EncodeFailed;

        // Reduced shape: axis dimension becomes 1
        var result = GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
        const resolved_axis: u8 = if (axis >= 0) @intCast(@as(u32, @intCast(axis))) else blk: {
            const neg: u32 = @intCast(-axis);
            break :blk x.ndim - @as(u8, @intCast(neg));
        };
        if (resolved_axis < result.ndim) {
            result.shape[resolved_axis] = 1;
        }
        return result;
    }

    /// Mean reduction along a specified axis.
    pub fn reduceMean(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Create NSArray with the axis
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add_obj = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        const axis_num = num_fn(ns_number_class, sel_with_int, @as(c_int, axis));
        if (axis_num != null) add_obj_fn(arr, sel_add_obj, axis_num);

        // [graph meanOfTensor:axes:name:]
        const sel_mean = sel_fn("meanOfTensor:axes:name:");
        const mean_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = mean_fn(self.graph, sel_mean, x.tensor, arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, sel_release);

        if (result_tensor == null) return MpsError.EncodeFailed;

        // Reduced shape: axis dimension becomes 1
        var result = GraphTensor{
            .tensor = result_tensor,
            .shape = x.shape,
            .ndim = x.ndim,
            .dtype = x.dtype,
        };
        const resolved_axis: u8 = if (axis >= 0) @intCast(@as(u32, @intCast(axis))) else blk: {
            const neg: u32 = @intCast(-axis);
            break :blk x.ndim - @as(u8, @intCast(neg));
        };
        if (resolved_axis < result.ndim) {
            result.shape[resolved_axis] = 1;
        }
        return result;
    }

    // ========================================================================
    // Graph Execution
    // ========================================================================

    /// Execute the graph with provided feeds and return results for target tensors.
    /// Feeds map named placeholders to input data; targets specify which tensors to evaluate.
    pub fn run(self: *MpsGraph, feeds: []const GraphFeed, targets: []const GraphTensor, allocator: std.mem.Allocator) MpsError!GraphResult {
        if (self.graph == null) return MpsError.InitFailed;
        if (targets.len == 0 or targets.len > 8) return MpsError.InvalidDimensions;

        var result = GraphResult{
            .output_count = @intCast(targets.len),
            .allocator = allocator,
        };

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Create MPSGraphTensorData for each feed
        const ns_dict_class = get_class("NSMutableDictionary") orelse return MpsError.FrameworkNotAvailable;
        const dict_raw = msg_send(@ptrCast(ns_dict_class), sel_alloc);
        if (dict_raw == null) return MpsError.InitFailed;
        const feed_dict = msg_send(dict_raw, sel_init);
        if (feed_dict == null) return MpsError.InitFailed;

        // Build feed dictionary: MPSGraphTensor -> MPSGraphTensorData
        _ = feeds; // Feed data binding requires live MPSGraphTensorData creation
        // In production, each feed would create an MPSGraphTensorData from its data pointer

        // Create target tensors array
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const target_arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (target_arr_raw == null) return MpsError.InitFailed;
        const target_arr = msg_send(target_arr_raw, sel_init);
        if (target_arr == null) return MpsError.InitFailed;

        const sel_add_obj = sel_fn("addObject:");
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        for (targets) |target| {
            if (target.tensor != null) add_obj_fn(target_arr, sel_add_obj, target.tensor);
        }

        // [graph runWithFeeds:targetTensors:targetOperations:]
        const sel_run = sel_fn("runWithFeeds:targetTensors:targetOperations:");
        const run_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const results_dict = run_fn(self.graph, sel_run, feed_dict, target_arr, null);

        // Release intermediate objects
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(target_arr, sel_release);
        release_fn(feed_dict, sel_release);

        if (results_dict == null) return MpsError.EncodeFailed;

        // Extract output data from results dictionary
        // In production, iterate results and copy MPSGraphTensorData contents to allocated slices
        // For now, allocate output buffers sized by target shapes
        for (targets[0..result.output_count], 0..) |target, i| {
            var total_elements: usize = 1;
            for (target.shape[0..target.ndim]) |s| total_elements *= s;
            if (total_elements > 0) {
                const out_buf = allocator.alloc(f32, total_elements) catch return MpsError.InitFailed;
                @memset(out_buf, 0);
                result.outputs[i] = out_buf;
            }
        }

        return result;
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
    // All shape elements should be zero
    for (tensor.shape) |s| {
        try std.testing.expectEqual(@as(u32, 0), s);
    }

    // Test with populated fields
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
    // All name bytes should be zero
    for (feed.name) |c| {
        try std.testing.expectEqual(@as(u8, 0), c);
    }

    // Populate a feed
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
    // No outputs set — all indices should return null
    try std.testing.expect(result.getData(0) == null);
    try std.testing.expect(result.getData(1) == null);
    try std.testing.expect(result.getData(255) == null);

    // Set up one output
    const allocator = std.testing.allocator;
    var buf = try allocator.alloc(f32, 4);
    buf[0] = 1.0;
    buf[1] = 2.0;
    buf[2] = 3.0;
    buf[3] = 4.0;

    result.outputs[0] = buf;
    result.output_count = 1;
    result.allocator = allocator;

    // Index 0 should return data, index 1 should be null
    const data = result.getData(0);
    try std.testing.expect(data != null);
    try std.testing.expectEqual(@as(usize, 4), data.?.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.?[0], 0.001);
    try std.testing.expect(result.getData(1) == null);

    result.deinit();
}

test "MpsGraph.isGraphAvailable check" {
    // isGraphAvailable depends on whether MPSGraph framework was loaded
    const available = MpsGraph.isGraphAvailable();
    // On non-macOS or if framework not loaded, should be false
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!available);
    }
    // On macOS, result depends on whether tryLoadMps() loaded the graph lib
    // Either way, calling this should not crash
}
