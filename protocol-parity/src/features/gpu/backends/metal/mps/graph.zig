//! MPS Graph (macOS 11+ / iOS 14+)
//!
//! Wraps MPSGraph for compute graph execution. MPSGraph allows building a DAG
//! of operations that MPS optimizes and fuses.

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../../metal_types.zig");
const mps_core = @import("../mps.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;
const MpsError = mps_core.MpsError;

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
        if (mps_core.mps_graph_lib == null) return MpsError.FrameworkNotAvailable;

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;

        const cls = get_class("MPSGraph") orelse return MpsError.FrameworkNotAvailable;
        const instance = msg_send(@ptrCast(cls), mps_core.sel_alloc);
        if (instance == null) return MpsError.InitFailed;

        const graph = msg_send(instance, mps_core.sel_init);
        if (graph == null) return MpsError.InitFailed;

        return .{
            .graph = graph,
            .device = device,
        };
    }

    pub fn destroy(self: *MpsGraph) void {
        if (self.graph != null) {
            if (mps_core.objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.graph, mps_core.sel_release);
            }
            self.graph = null;
        }
    }

    pub fn isGraphAvailable() bool {
        return mps_core.mps_graph_lib != null;
    }

    // ========================================================================
    // Graph Tensor Construction
    // ========================================================================

    /// Create a placeholder tensor in the graph with a given name, shape, and data type.
    pub fn placeholder(self: *MpsGraph, name: []const u8, shape: []const u32, dtype: GraphDataType) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (shape.len == 0 or shape.len > 8) return MpsError.InvalidDimensions;

        var result = GraphTensor{
            .ndim = @intCast(@min(shape.len, 8)),
            .dtype = dtype,
        };
        for (shape[0..result.ndim], 0..) |s, i| result.shape[i] = s;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Build NSArray of NSNumber for shape dimensions
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), mps_core.sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, mps_core.sel_init);
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
        release_fn(arr, mps_core.sel_release);

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

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        // Build NSArray of NSNumber for shape
        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), mps_core.sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, mps_core.sel_init);
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
        release_fn(arr, mps_core.sel_release);

        if (tensor == null) return MpsError.InitFailed;
        result.tensor = tensor;
        return result;
    }

    // ========================================================================
    // Graph Operations
    // ========================================================================

    /// Matrix multiplication of two graph tensors (wraps MPSGraph matmul).
    pub fn matmulGraph(self: *MpsGraph, a: GraphTensor, b: GraphTensor, transpose_a: bool, transpose_b: bool) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        var a_tensor = a;
        var b_tensor = b;
        if (transpose_a and a.ndim >= 2) {
            a_tensor = try self.transpose(a, a.ndim - 2, a.ndim - 1);
        }
        if (transpose_b and b.ndim >= 2) {
            b_tensor = try self.transpose(b, b.ndim - 2, b.ndim - 1);
        }

        const sel_matmul = sel_fn("matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:");
        const matmul_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = matmul_fn(self.graph, sel_matmul, a_tensor.tensor, b_tensor.tensor, null);

        if (result_tensor == null) return MpsError.EncodeFailed;

        var result = GraphTensor{
            .tensor = result_tensor,
            .ndim = a.ndim,
            .dtype = a.dtype,
        };
        if (a.ndim >= 2 and b.ndim >= 2) {
            for (0..a.ndim) |i| result.shape[i] = a.shape[i];
            result.shape[a.ndim - 1] = b.shape[b.ndim - 1];
        }
        return result;
    }

    /// Element-wise addition of two graph tensors.
    pub fn add(self: *MpsGraph, a: GraphTensor, b: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_add = sel_fn("additionWithPrimaryTensor:secondaryTensor:name:");
        const add_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = add_fn(self.graph, sel_add, a.tensor, b.tensor, null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{ .tensor = result_tensor, .shape = a.shape, .ndim = a.ndim, .dtype = a.dtype };
    }

    /// Element-wise multiplication of two graph tensors.
    pub fn multiply(self: *MpsGraph, a: GraphTensor, b: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (a.tensor == null or b.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_mul = sel_fn("multiplicationWithPrimaryTensor:secondaryTensor:name:");
        const mul_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = mul_fn(self.graph, sel_mul, a.tensor, b.tensor, null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{ .tensor = result_tensor, .shape = a.shape, .ndim = a.ndim, .dtype = a.dtype };
    }

    /// ReLU activation on a graph tensor: max(0, x).
    pub fn reluGraph(self: *MpsGraph, x: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_relu = sel_fn("reLUWithTensor:name:");
        const relu_fn: *const fn (ID, SEL, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = relu_fn(self.graph, sel_relu, x.tensor, null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{ .tensor = result_tensor, .shape = x.shape, .ndim = x.ndim, .dtype = x.dtype };
    }

    /// Sigmoid activation on a graph tensor: 1 / (1 + exp(-x)).
    pub fn sigmoidGraph(self: *MpsGraph, x: GraphTensor) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_sigmoid = sel_fn("sigmoidWithTensor:name:");
        const sigmoid_fn: *const fn (ID, SEL, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = sigmoid_fn(self.graph, sel_sigmoid, x.tensor, null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{ .tensor = result_tensor, .shape = x.shape, .ndim = x.ndim, .dtype = x.dtype };
    }

    /// Softmax activation along a specified axis.
    pub fn softmaxGraph(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_softmax = sel_fn("softMaxWithTensor:axis:name:");
        const softmax_fn: *const fn (ID, SEL, ID, c_int, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = softmax_fn(self.graph, sel_softmax, x.tensor, @as(c_int, axis), null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        return GraphTensor{ .tensor = result_tensor, .shape = x.shape, .ndim = x.ndim, .dtype = x.dtype };
    }

    /// Reshape a graph tensor to a new shape.
    pub fn reshape(self: *MpsGraph, x: GraphTensor, new_shape: []const u32) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (new_shape.len == 0 or new_shape.len > 8) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), mps_core.sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, mps_core.sel_init);
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

        const sel_reshape = sel_fn("reshapeTensor:withShape:name:");
        const reshape_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = reshape_fn(self.graph, sel_reshape, x.tensor, arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, mps_core.sel_release);

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

        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const sel_transpose = sel_fn("transposeTensor:dimension:withDimension:name:");
        const transpose_fn: *const fn (ID, SEL, ID, u32, u32, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = transpose_fn(self.graph, sel_transpose, x.tensor, dim0, dim1, null);
        if (result_tensor == null) return MpsError.EncodeFailed;

        var result = GraphTensor{ .tensor = result_tensor, .shape = x.shape, .ndim = x.ndim, .dtype = x.dtype };
        const tmp = result.shape[dim0];
        result.shape[dim0] = result.shape[dim1];
        result.shape[dim1] = tmp;
        return result;
    }

    /// Sum reduction along a specified axis.
    pub fn reduceSum(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        return self.reduceOp(x, axis, "reductionSumWithTensor:axes:name:");
    }

    /// Mean reduction along a specified axis.
    pub fn reduceMean(self: *MpsGraph, x: GraphTensor, axis: i32) MpsError!GraphTensor {
        return self.reduceOp(x, axis, "meanOfTensor:axes:name:");
    }

    fn reduceOp(self: *MpsGraph, x: GraphTensor, axis: i32, sel_name: [*:0]const u8) MpsError!GraphTensor {
        if (self.graph == null) return MpsError.InitFailed;
        if (x.tensor == null) return MpsError.InvalidDimensions;
        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), mps_core.sel_alloc);
        if (arr_raw == null) return MpsError.InitFailed;
        const arr = msg_send(arr_raw, mps_core.sel_init);
        if (arr == null) return MpsError.InitFailed;

        const ns_number_class = get_class("NSNumber") orelse return MpsError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add_obj = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        const axis_num = num_fn(ns_number_class, sel_with_int, @as(c_int, axis));
        if (axis_num != null) add_obj_fn(arr, sel_add_obj, axis_num);

        const sel_reduce = sel_fn(sel_name);
        const reduce_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const result_tensor = reduce_fn(self.graph, sel_reduce, x.tensor, arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr, mps_core.sel_release);

        if (result_tensor == null) return MpsError.EncodeFailed;

        var result = GraphTensor{ .tensor = result_tensor, .shape = x.shape, .ndim = x.ndim, .dtype = x.dtype };
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
    pub fn run(self: *MpsGraph, feeds: []const GraphFeed, targets: []const GraphTensor, allocator: std.mem.Allocator) MpsError!GraphResult {
        if (self.graph == null) return MpsError.InitFailed;
        if (targets.len == 0 or targets.len > 8) return MpsError.InvalidDimensions;

        var result = GraphResult{
            .output_count = @intCast(targets.len),
            .allocator = allocator,
        };

        if (builtin.os.tag != .macos) return MpsError.UnsupportedOperation;

        const get_class = mps_core.objc_get_class_fn orelse return MpsError.FrameworkNotAvailable;
        const msg_send = mps_core.objc_msgSend_fn orelse return MpsError.FrameworkNotAvailable;
        const sel_fn = mps_core.sel_register_fn orelse return MpsError.FrameworkNotAvailable;

        const ns_dict_class = get_class("NSMutableDictionary") orelse return MpsError.FrameworkNotAvailable;
        const dict_raw = msg_send(@ptrCast(ns_dict_class), mps_core.sel_alloc);
        if (dict_raw == null) return MpsError.InitFailed;
        const feed_dict = msg_send(dict_raw, mps_core.sel_init);
        if (feed_dict == null) return MpsError.InitFailed;

        _ = feeds; // Feed data binding requires live MPSGraphTensorData creation

        const ns_array_class = get_class("NSMutableArray") orelse return MpsError.FrameworkNotAvailable;
        const target_arr_raw = msg_send(@ptrCast(ns_array_class), mps_core.sel_alloc);
        if (target_arr_raw == null) return MpsError.InitFailed;
        const target_arr = msg_send(target_arr_raw, mps_core.sel_init);
        if (target_arr == null) return MpsError.InitFailed;

        const sel_add_obj = sel_fn("addObject:");
        const add_obj_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);
        for (targets) |target| {
            if (target.tensor != null) add_obj_fn(target_arr, sel_add_obj, target.tensor);
        }

        const sel_run = sel_fn("runWithFeeds:targetTensors:targetOperations:");
        const run_fn: *const fn (ID, SEL, ID, ID, ID) callconv(.c) ID = @ptrCast(msg_send);
        const results_dict = run_fn(self.graph, sel_run, feed_dict, target_arr, null);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(target_arr, mps_core.sel_release);
        release_fn(feed_dict, mps_core.sel_release);

        if (results_dict == null) return MpsError.EncodeFailed;

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
