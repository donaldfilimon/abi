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
