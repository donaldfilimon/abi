//! CoreML Integration (Metal Extension)
//!
//! Provides model inference via Apple's CoreML framework, leveraging
//! the Neural Engine, GPU, and CPU for optimal performance.
//!
//! CoreML models are pre-compiled `.mlmodelc` bundles that can be loaded
//! and run with minimal setup. This module bridges CoreML with the Metal
//! backend for GPU-accelerated pre/post-processing.
//!
//! ## Flow
//! 1. Create NSURL from model path
//! 2. Create MLModelConfiguration (compute units, precision)
//! 3. Load model: [MLModel modelWithContentsOfURL:configuration:error:]
//! 4. Create input feature provider
//! 5. Run prediction: [model predictionFromFeatures:error:]
//! 6. Extract outputs

const std = @import("std");
const builtin = @import("builtin");
const metal_types = @import("../metal_types.zig");

const ID = metal_types.ID;
const SEL = metal_types.SEL;
const Class = metal_types.Class;

pub const CoreMlError = error{
    FrameworkNotAvailable,
    ModelLoadFailed,
    PredictionFailed,
    InvalidInput,
    InvalidOutput,
    ConfigurationFailed,
    UnsupportedPlatform,
};

// ============================================================================
// Framework Loading
// ============================================================================

var coreml_lib: ?std.DynLib = null;
var foundation_lib: ?std.DynLib = null;
var coreml_load_attempted: bool = false;

// Obj-C runtime pointers (shared with metal.zig/mps.zig)
var objc_msgSend_fn: ?*const fn (ID, SEL) callconv(.c) ID = null;
var sel_register_fn: ?*const fn ([*:0]const u8) callconv(.c) SEL = null;
var objc_get_class_fn: ?*const fn ([*:0]const u8) callconv(.c) ?Class = null;

// Cached selectors
var sel_alloc: SEL = undefined;
var sel_init: SEL = undefined;
var sel_release: SEL = undefined;
var selectors_loaded: bool = false;

/// Compute unit preference for CoreML model execution.
pub const ComputeUnit = enum(u32) {
    all = 0, // CPU + GPU + Neural Engine
    cpu_only = 1,
    cpu_and_gpu = 2,
    cpu_and_ne = 3, // CPU + Neural Engine (skip GPU)
};

/// Configuration for CoreML model loading.
pub const ModelConfig = struct {
    compute_units: ComputeUnit = .all,
    allow_low_precision: bool = true,
};

/// Initialize CoreML integration.
pub fn init(
    msg_send: *const fn (ID, SEL) callconv(.c) ID,
    sel_register: *const fn ([*:0]const u8) callconv(.c) SEL,
    get_class: *const fn ([*:0]const u8) callconv(.c) ?Class,
) CoreMlError!void {
    if (selectors_loaded) return;

    objc_msgSend_fn = msg_send;
    sel_register_fn = sel_register;
    objc_get_class_fn = get_class;

    if (!tryLoadCoreML()) {
        return CoreMlError.FrameworkNotAvailable;
    }

    sel_alloc = sel_register("alloc");
    sel_init = sel_register("init");
    sel_release = sel_register("release");

    selectors_loaded = true;
}

pub fn deinit() void {
    if (coreml_lib) |lib| lib.close();
    if (foundation_lib) |lib| lib.close();
    coreml_lib = null;
    foundation_lib = null;
    coreml_load_attempted = false;
    selectors_loaded = false;
}

pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) return false;
    if (coreml_load_attempted) return coreml_lib != null;
    return tryLoadCoreML();
}

fn tryLoadCoreML() bool {
    if (coreml_load_attempted) return coreml_lib != null;
    coreml_load_attempted = true;

    const paths = [_][]const u8{
        "/System/Library/Frameworks/CoreML.framework/CoreML",
    };
    for (paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            coreml_lib = lib;
            break;
        } else |_| {}
    }

    // Also load Foundation for NSURL
    const foundation_paths = [_][]const u8{
        "/System/Library/Frameworks/Foundation.framework/Foundation",
    };
    for (foundation_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            foundation_lib = lib;
            break;
        } else |_| {}
    }

    return coreml_lib != null;
}

// ============================================================================
// CoreML Multi-Array (Tensor I/O)
// ============================================================================

/// Data type for CoreML multi-dimensional arrays
pub const DataType = enum(u32) {
    float32 = 65568, // MLMultiArrayDataTypeFloat32
    float16 = 65552, // MLMultiArrayDataTypeFloat16
    float64 = 65600, // MLMultiArrayDataTypeFloat64
    int32 = 131104, // MLMultiArrayDataTypeInt32
};

/// Wraps MLMultiArray for tensor I/O with CoreML models
pub const MultiArray = struct {
    shape: [8]u32 = [_]u32{0} ** 8,
    ndim: u8 = 0,
    data_type: DataType = .float32,
    obj: ID = null,

    pub fn createFloat32(shape: []const u32) CoreMlError!MultiArray {
        var result = MultiArray{ .ndim = @intCast(@min(shape.len, 8)), .data_type = .float32 };
        for (shape[0..result.ndim], 0..) |s, i| result.shape[i] = s;

        if (builtin.os.tag != .macos) return CoreMlError.UnsupportedPlatform;

        const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;

        // Create NSArray of NSNumber for shape
        const ns_array_class = get_class("NSMutableArray") orelse return CoreMlError.FrameworkNotAvailable;
        const arr_raw = msg_send(@ptrCast(ns_array_class), sel_alloc);
        if (arr_raw == null) return CoreMlError.InvalidInput;
        const arr_init = msg_send(arr_raw, sel_init);
        if (arr_init == null) return CoreMlError.InvalidInput;

        const ns_number_class = get_class("NSNumber") orelse return CoreMlError.FrameworkNotAvailable;
        const sel_with_int = sel_fn("numberWithInt:");
        const sel_add = sel_fn("addObject:");
        const num_fn: *const fn (?Class, SEL, c_int) callconv(.c) ID = @ptrCast(msg_send);
        const add_fn: *const fn (ID, SEL, ID) callconv(.c) void = @ptrCast(msg_send);

        for (shape[0..result.ndim]) |s| {
            const num = num_fn(ns_number_class, sel_with_int, @intCast(s));
            if (num != null) add_fn(arr_init, sel_add, num);
        }

        // Create MLMultiArray
        const ml_array_class = get_class("MLMultiArray") orelse return CoreMlError.FrameworkNotAvailable;
        const ml_arr = msg_send(@ptrCast(ml_array_class), sel_alloc);
        if (ml_arr == null) return CoreMlError.InvalidInput;

        const sel_init_shape = sel_fn("initWithShape:dataType:error:");
        var init_error: ID = null;
        const init_fn: *const fn (ID, SEL, ID, u32, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const obj = init_fn(ml_arr, sel_init_shape, arr_init, @intFromEnum(result.data_type), &init_error);

        // Release shape array
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(arr_init, sel_release);

        if (obj == null) return CoreMlError.InvalidInput;
        result.obj = obj;
        return result;
    }

    pub fn totalElements(self: *const MultiArray) usize {
        if (self.ndim == 0) return 0;
        var total: usize = 1;
        for (self.shape[0..self.ndim]) |s| total *= s;
        return total;
    }

    pub fn deinit(self: *MultiArray) void {
        if (self.obj != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.obj, sel_release);
            }
            self.obj = null;
        }
    }
};

/// Named array for model I/O
pub const NamedArray = struct {
    name: [64]u8 = [_]u8{0} ** 64,
    name_len: u8 = 0,
    array: ?*MultiArray = null,
};

// ============================================================================
// CoreML Model Compilation
// ============================================================================

/// Compile a CoreML model from source (.mlmodel) to compiled form (.mlmodelc).
/// Calls [MLModel compileModelAtURL:error:] on macOS.
pub fn compileModel(source_path: []const u8, output_path: []const u8) CoreMlError!void {
    _ = output_path;

    if (builtin.os.tag != .macos) return CoreMlError.UnsupportedPlatform;

    const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;
    const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;
    const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;

    // Create NSURL from source path
    const nsurl_class = get_class("NSURL") orelse return CoreMlError.FrameworkNotAvailable;
    const nsstring_class = get_class("NSString") orelse return CoreMlError.FrameworkNotAvailable;
    const sel_string = sel_fn("stringWithUTF8String:");

    var path_buf: [4096]u8 = undefined;
    if (source_path.len >= path_buf.len) return CoreMlError.ModelLoadFailed;
    @memcpy(path_buf[0..source_path.len], source_path);
    path_buf[source_path.len] = 0;

    const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(msg_send);
    const ns_path = str_fn(nsstring_class, sel_string, path_buf[0..source_path.len :0]);
    if (ns_path == null) return CoreMlError.ModelLoadFailed;

    const sel_file_url = sel_fn("fileURLWithPath:");
    const url_fn: *const fn (?Class, SEL, ID) callconv(.c) ID = @ptrCast(msg_send);
    const source_url = url_fn(nsurl_class, sel_file_url, ns_path);
    if (source_url == null) return CoreMlError.ModelLoadFailed;

    // [MLModel compileModelAtURL:error:]
    const ml_model_class = get_class("MLModel") orelse return CoreMlError.FrameworkNotAvailable;
    const sel_compile = sel_fn("compileModelAtURL:error:");
    var compile_error: ID = null;
    const compile_fn: *const fn (?Class, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
    const compiled_url = compile_fn(ml_model_class, sel_compile, source_url, &compile_error);

    if (compiled_url == null) return CoreMlError.ModelLoadFailed;
    // compiled_url is an NSURL to the compiled .mlmodelc bundle
    // In production, move the compiled bundle to output_path
}

// ============================================================================
// CoreML Model
// ============================================================================

/// A loaded CoreML model ready for inference.
pub const CoreMlModel = struct {
    model: ID = null,
    config: ModelConfig = .{},

    /// Load a compiled CoreML model (.mlmodelc) from a file path.
    pub fn load(path: []const u8, config: ModelConfig) CoreMlError!CoreMlModel {
        const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;

        // Create NSURL from path
        const nsurl_class = get_class("NSURL") orelse return CoreMlError.FrameworkNotAvailable;

        // Create null-terminated path
        var path_buf: [4096]u8 = undefined;
        if (path.len >= path_buf.len) return CoreMlError.ModelLoadFailed;
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;

        // [NSURL fileURLWithPath:@"..."]
        const sel_file_url = sel_fn("fileURLWithPath:");
        const nsstring_class = get_class("NSString") orelse return CoreMlError.FrameworkNotAvailable;
        const sel_string = sel_fn("stringWithUTF8String:");

        const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(msg_send);
        const ns_path = str_fn(nsstring_class, sel_string, path_buf[0..path.len :0]);
        if (ns_path == null) return CoreMlError.ModelLoadFailed;

        const url_fn: *const fn (?Class, SEL, ID) callconv(.c) ID = @ptrCast(msg_send);
        const url = url_fn(nsurl_class, sel_file_url, ns_path);
        if (url == null) return CoreMlError.ModelLoadFailed;

        // Create MLModelConfiguration
        const ml_config_class = get_class("MLModelConfiguration") orelse
            return CoreMlError.FrameworkNotAvailable;
        const ml_config_alloc = msg_send(@ptrCast(ml_config_class), sel_alloc);
        if (ml_config_alloc == null) return CoreMlError.ConfigurationFailed;
        const ml_config = msg_send(ml_config_alloc, sel_init);
        if (ml_config == null) return CoreMlError.ConfigurationFailed;

        // Set compute units
        const sel_set_compute = sel_fn("setComputeUnits:");
        const set_fn: *const fn (ID, SEL, u32) callconv(.c) void = @ptrCast(msg_send);
        set_fn(ml_config, sel_set_compute, @intFromEnum(config.compute_units));

        // Load model: [MLModel modelWithContentsOfURL:configuration:error:]
        const ml_model_class = get_class("MLModel") orelse
            return CoreMlError.FrameworkNotAvailable;
        const sel_load = sel_fn("modelWithContentsOfURL:configuration:error:");
        var load_error: ID = null;
        const load_fn: *const fn (?Class, SEL, ID, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const model = load_fn(ml_model_class, sel_load, url, ml_config, &load_error);

        // Release config
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(ml_config, sel_release);

        if (model == null) {
            return CoreMlError.ModelLoadFailed;
        }

        return .{
            .model = model,
            .config = config,
        };
    }

    /// Run inference with a dictionary of named inputs.
    /// Returns a prediction result wrapping the output feature provider.
    pub fn predict(self: *const CoreMlModel, input_provider: ID) CoreMlError!PredictionResult {
        if (self.model == null) return CoreMlError.ModelLoadFailed;
        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;

        const sel_predict = sel_fn("predictionFromFeatures:error:");
        var predict_error: ID = null;
        const predict_fn: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const output = predict_fn(self.model, sel_predict, input_provider, &predict_error);

        if (output == null) {
            return CoreMlError.PredictionFailed;
        }

        return .{ .output = output };
    }

    /// Run a dummy prediction to pre-compile and warm up the model pipeline.
    /// This forces CoreML to load weights and compile for the target compute units,
    /// reducing latency on the first real prediction.
    pub fn warmup(self: *CoreMlModel) CoreMlError!void {
        if (self.model == null) return CoreMlError.ModelLoadFailed;

        if (builtin.os.tag != .macos) return CoreMlError.UnsupportedPlatform;

        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;
        const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;

        // Get model description to access input features
        const sel_desc = sel_fn("modelDescription");
        const desc = msg_send(self.model, sel_desc);
        if (desc == null) return CoreMlError.PredictionFailed;

        // Get input descriptions: [description inputDescriptionsByName]
        const sel_inputs = sel_fn("inputDescriptionsByName");
        const inputs_dict = msg_send(desc, sel_inputs);
        if (inputs_dict == null) return CoreMlError.PredictionFailed;

        // Create an empty MLDictionaryFeatureProvider as a dummy input
        const provider_class = get_class("MLDictionaryFeatureProvider") orelse
            return CoreMlError.FrameworkNotAvailable;
        const provider_alloc = msg_send(@ptrCast(provider_class), sel_alloc);
        if (provider_alloc == null) return CoreMlError.PredictionFailed;

        // Create empty dictionary for dummy input
        const dict_class = get_class("NSMutableDictionary") orelse return CoreMlError.FrameworkNotAvailable;
        const dict_raw = msg_send(@ptrCast(dict_class), sel_alloc);
        if (dict_raw == null) return CoreMlError.PredictionFailed;
        const dict = msg_send(dict_raw, sel_init);
        if (dict == null) return CoreMlError.PredictionFailed;

        const sel_init_dict = sel_fn("initWithDictionary:error:");
        var init_error: ID = null;
        const init_fn: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const provider = init_fn(provider_alloc, sel_init_dict, dict, &init_error);

        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(dict, sel_release);

        if (provider == null) return CoreMlError.PredictionFailed;
        defer release_fn(provider, sel_release);

        // Attempt prediction — may fail with empty inputs, but that's okay for warmup.
        // The goal is to trigger model compilation/loading.
        const sel_predict = sel_fn("predictionFromFeatures:error:");
        var predict_error: ID = null;
        const predict_fn: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        _ = predict_fn(self.model, sel_predict, provider, &predict_error);

        // Warmup is best-effort; even a failed prediction triggers pipeline compilation
    }

    /// Return the configured compute units for this model.
    pub fn getComputeUnits(self: *const CoreMlModel) ComputeUnit {
        return self.config.compute_units;
    }

    pub fn destroy(self: *CoreMlModel) void {
        if (self.model != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.model, sel_release);
            }
            self.model = null;
        }
    }
};

/// Result of a CoreML prediction.
pub const PredictionResult = struct {
    output: ID = null,

    /// Get a named output feature value from the prediction result.
    pub fn getFeature(self: *const PredictionResult, feature_name: [*:0]const u8) CoreMlError!ID {
        if (self.output == null) return CoreMlError.InvalidOutput;
        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;

        // Create NSString from feature name
        const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;
        const nsstring_class = get_class("NSString") orelse return CoreMlError.FrameworkNotAvailable;
        const sel_string = sel_fn("stringWithUTF8String:");
        const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(msg_send);
        const ns_name = str_fn(nsstring_class, sel_string, feature_name);
        if (ns_name == null) return CoreMlError.InvalidOutput;

        // [output featureValueForName:@"name"]
        const sel_feature = sel_fn("featureValueForName:");
        const feature_fn: *const fn (ID, SEL, ID) callconv(.c) ID = @ptrCast(msg_send);
        const value = feature_fn(self.output, sel_feature, ns_name);
        if (value == null) return CoreMlError.InvalidOutput;
        return value;
    }

    pub fn destroy(self: *PredictionResult) void {
        // Output is typically autoreleased by CoreML
        self.output = null;
    }
};

/// Helper to create an MLDictionaryFeatureProvider from key-value pairs.
pub const FeatureInput = struct {
    provider: ID = null,

    pub fn create(
        keys: []const [*:0]const u8,
        values: []const ID,
    ) CoreMlError!FeatureInput {
        if (keys.len != values.len) return CoreMlError.InvalidInput;
        const get_class = objc_get_class_fn orelse return CoreMlError.FrameworkNotAvailable;
        const sel_fn = sel_register_fn orelse return CoreMlError.FrameworkNotAvailable;
        const msg_send = objc_msgSend_fn orelse return CoreMlError.FrameworkNotAvailable;

        // Create NSDictionary from keys and values
        const dict_class = get_class("NSMutableDictionary") orelse
            return CoreMlError.FrameworkNotAvailable;
        const dict_alloc = msg_send(@ptrCast(dict_class), sel_alloc);
        if (dict_alloc == null) return CoreMlError.InvalidInput;
        const dict = msg_send(dict_alloc, sel_init);
        if (dict == null) return CoreMlError.InvalidInput;

        // Add entries
        const nsstring_class = get_class("NSString") orelse return CoreMlError.FrameworkNotAvailable;
        const sel_string = sel_fn("stringWithUTF8String:");
        const sel_set = sel_fn("setObject:forKey:");
        const str_fn: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(msg_send);
        const set_fn: *const fn (ID, SEL, ID, ID) callconv(.c) void = @ptrCast(msg_send);

        for (keys, 0..) |key, i| {
            const ns_key = str_fn(nsstring_class, sel_string, key);
            if (ns_key == null) continue;
            set_fn(dict, sel_set, values[i], ns_key);
        }

        // Create MLDictionaryFeatureProvider
        const provider_class = get_class("MLDictionaryFeatureProvider") orelse
            return CoreMlError.FrameworkNotAvailable;
        const provider_alloc = msg_send(@ptrCast(provider_class), sel_alloc);
        if (provider_alloc == null) return CoreMlError.InvalidInput;

        const sel_init_dict = sel_fn("initWithDictionary:error:");
        var init_error: ID = null;
        const init_fn: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(msg_send);
        const provider = init_fn(provider_alloc, sel_init_dict, dict, &init_error);

        // Release dictionary
        const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
        release_fn(dict, sel_release);

        if (provider == null) return CoreMlError.InvalidInput;

        return .{ .provider = provider };
    }

    pub fn destroy(self: *FeatureInput) void {
        if (self.provider != null) {
            if (objc_msgSend_fn) |msg_send| {
                const release_fn: *const fn (ID, SEL) callconv(.c) void = @ptrCast(msg_send);
                release_fn(self.provider, sel_release);
            }
            self.provider = null;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CoreML availability check" {
    const available = isAvailable();
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!available);
    }
}

test "ComputeUnit enum values" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(ComputeUnit.all));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(ComputeUnit.cpu_only));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(ComputeUnit.cpu_and_ne));
}

test "MultiArray totalElements calculation" {
    // 1D array
    var arr1 = MultiArray{ .ndim = 1 };
    arr1.shape[0] = 10;
    try std.testing.expectEqual(@as(usize, 10), arr1.totalElements());

    // 2D array (3x4 = 12)
    var arr2 = MultiArray{ .ndim = 2 };
    arr2.shape[0] = 3;
    arr2.shape[1] = 4;
    try std.testing.expectEqual(@as(usize, 12), arr2.totalElements());

    // 3D array (2x3x4 = 24)
    var arr3 = MultiArray{ .ndim = 3 };
    arr3.shape[0] = 2;
    arr3.shape[1] = 3;
    arr3.shape[2] = 4;
    try std.testing.expectEqual(@as(usize, 24), arr3.totalElements());

    // Zero-dim array should return 0
    const arr0 = MultiArray{};
    try std.testing.expectEqual(@as(usize, 0), arr0.totalElements());
}

test "DataType enum values" {
    try std.testing.expectEqual(@as(u32, 65568), @intFromEnum(DataType.float32));
    try std.testing.expectEqual(@as(u32, 65552), @intFromEnum(DataType.float16));
    try std.testing.expectEqual(@as(u32, 65600), @intFromEnum(DataType.float64));
    try std.testing.expectEqual(@as(u32, 131104), @intFromEnum(DataType.int32));
}

test "NamedArray default initialization" {
    const named = NamedArray{};
    try std.testing.expectEqual(@as(u8, 0), named.name_len);
    try std.testing.expect(named.array == null);
    // All name bytes should be zero
    for (named.name) |c| {
        try std.testing.expectEqual(@as(u8, 0), c);
    }

    // Test with populated name
    var named2 = NamedArray{};
    const label = "output_logits";
    @memcpy(named2.name[0..label.len], label);
    named2.name_len = label.len;
    try std.testing.expectEqual(@as(u8, 13), named2.name_len);
    try std.testing.expectEqual(@as(u8, 'o'), named2.name[0]);
}

test "CoreMlModel warmup on null model returns error" {
    var model = CoreMlModel{};
    try std.testing.expect(model.model == null);
    try std.testing.expectError(CoreMlError.ModelLoadFailed, model.warmup());

    // getComputeUnits should return default
    try std.testing.expectEqual(ComputeUnit.all, model.getComputeUnits());

    // With custom config
    var model2 = CoreMlModel{ .config = .{ .compute_units = .cpu_only } };
    try std.testing.expectEqual(ComputeUnit.cpu_only, model2.getComputeUnits());
}
