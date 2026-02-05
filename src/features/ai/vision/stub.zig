//! Vision Stub Module
//!
//! Stub implementation when vision is disabled at compile time.

const std = @import("std");

// ============================================================================
// Errors
// ============================================================================

/// Vision module errors.
pub const Error = error{
    VisionDisabled,
    InvalidImage,
    UnsupportedFormat,
    DecodeFailed,
    EncodeFailed,
    OutOfMemory,
};

pub const ImageError = error{
    InvalidDimensions,
    InvalidChannels,
    DataSizeMismatch,
    OutOfBounds,
    OutOfMemory,
    VisionDisabled,
};

pub const PreprocessError = error{
    InvalidDimensions,
    InvalidChannels,
    DimensionMismatch,
    OutOfMemory,
    VisionDisabled,
};

// ============================================================================
// Image Types
// ============================================================================

/// Color channel constants
pub const Channels = struct {
    pub const grayscale: u8 = 1;
    pub const rgb: u8 = 3;
    pub const rgba: u8 = 4;
};

/// Stub image type.
pub const Image = struct {
    width: u32 = 0,
    height: u32 = 0,
    channels: u8 = 0,
    data: []u8 = &.{},
    allocator: std.mem.Allocator = undefined,

    pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u8) Error!Image {
        return error.VisionDisabled;
    }

    pub fn fromData(
        _: std.mem.Allocator,
        _: u32,
        _: u32,
        _: u8,
        _: []const u8,
    ) Error!Image {
        return error.VisionDisabled;
    }

    pub fn deinit(_: *Image) void {}

    pub fn getPixel(_: *const Image, _: u32, _: u32) ?[]const u8 {
        return null;
    }

    pub fn setPixel(_: *Image, _: u32, _: u32, _: []const u8) void {}

    pub fn clone(_: *const Image) Error!Image {
        return error.VisionDisabled;
    }

    pub fn fill(_: *Image, _: []const u8) void {}

    pub fn pixelCount(_: *const Image) usize {
        return 0;
    }

    pub fn dataSize(_: *const Image) usize {
        return 0;
    }

    pub fn stride(_: *const Image) usize {
        return 0;
    }

    pub fn sameDimensions(_: *const Image, _: *const Image) bool {
        return false;
    }
};

// ============================================================================
// Image Module Stub
// ============================================================================

pub const image = struct {
    pub const Image = stub_root.Image;
    pub const Channels = stub_root.Channels;
    pub const ImageError = stub_root.ImageError;
};

const stub_root = @This();

// ============================================================================
// Preprocessing Module Stub
// ============================================================================

pub const preprocessing = struct {
    pub fn resize(_: *const Image, _: u32, _: u32) Error!Image {
        return error.VisionDisabled;
    }

    pub fn normalize(_: *Image, _: [3]f32, _: [3]f32) void {}

    pub fn toFloat(_: *const Image, _: std.mem.Allocator) Error![]f32 {
        return error.VisionDisabled;
    }

    pub fn toFloatNormalized(
        _: *const Image,
        _: std.mem.Allocator,
        _: [3]f32,
        _: [3]f32,
    ) Error![]f32 {
        return error.VisionDisabled;
    }

    pub fn centerCrop(_: *const Image, _: u32) Error!Image {
        return error.VisionDisabled;
    }

    pub fn centerCropRect(_: *const Image, _: u32, _: u32) Error!Image {
        return error.VisionDisabled;
    }

    pub fn pad(_: *const Image, _: u32, _: u8) Error!Image {
        return error.VisionDisabled;
    }

    pub fn padAsymmetric(
        _: *const Image,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: u8,
    ) Error!Image {
        return error.VisionDisabled;
    }

    pub fn toGrayscale(_: *const Image) Error!Image {
        return error.VisionDisabled;
    }

    pub fn flipHorizontal(_: *const Image) Error!Image {
        return error.VisionDisabled;
    }

    pub fn flipVertical(_: *const Image) Error!Image {
        return error.VisionDisabled;
    }
};

// ============================================================================
// Preprocessing Function Re-exports
// ============================================================================

pub const resize = preprocessing.resize;
pub const normalize = preprocessing.normalize;
pub const toFloat = preprocessing.toFloat;
pub const toFloatNormalized = preprocessing.toFloatNormalized;
pub const centerCrop = preprocessing.centerCrop;
pub const centerCropRect = preprocessing.centerCropRect;
pub const pad = preprocessing.pad;
pub const padAsymmetric = preprocessing.padAsymmetric;
pub const toGrayscale = preprocessing.toGrayscale;
pub const flipHorizontal = preprocessing.flipHorizontal;
pub const flipVertical = preprocessing.flipVertical;

// ============================================================================
// Neural Network Layer Stubs
// ============================================================================

pub const conv = struct {
    pub const Conv2D = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) Error!@This() {
            return error.VisionDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ConvGradients = struct {};
    pub fn im2col(_: anytype) Error!void {
        return error.VisionDisabled;
    }
    pub fn col2im(_: anytype) Error!void {
        return error.VisionDisabled;
    }
};

pub const pooling = struct {
    pub const MaxPool2D = struct {
        pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u32) Error!@This() {
            return error.VisionDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const AvgPool2D = struct {
        pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u32) Error!@This() {
            return error.VisionDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const AdaptiveAvgPool2D = struct {
        pub fn init(_: anytype) @This() {
            return .{};
        }
    };
    pub const PoolResult = struct {};
    pub fn globalAvgPool2D(_: anytype) Error!void {
        return error.VisionDisabled;
    }
};

pub const batchnorm = struct {
    pub const BatchNorm2D = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) Error!@This() {
            return error.VisionDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const BatchNormGradients = struct {};
};

// Neural network layer type re-exports
pub const Conv2D = conv.Conv2D;
pub const ConvGradients = conv.ConvGradients;
pub const im2col = conv.im2col;
pub const col2im = conv.col2im;
pub const MaxPool2D = pooling.MaxPool2D;
pub const AvgPool2D = pooling.AvgPool2D;
pub const AdaptiveAvgPool2D = pooling.AdaptiveAvgPool2D;
pub const PoolResult = pooling.PoolResult;
pub const globalAvgPool2D = pooling.globalAvgPool2D;
pub const BatchNorm2D = batchnorm.BatchNorm2D;
pub const BatchNormGradients = batchnorm.BatchNormGradients;

// ============================================================================
// Normalization Constants
// ============================================================================

/// Common normalization values for ImageNet-pretrained models.
pub const ImageNetNorm = struct {
    pub const mean: [3]f32 = .{ 0.485, 0.456, 0.406 };
    pub const std: [3]f32 = .{ 0.229, 0.224, 0.225 };
};

/// Common normalization values for CLIP models.
pub const ClipNorm = struct {
    pub const mean: [3]f32 = .{ 0.48145466, 0.4578275, 0.40821073 };
    pub const std: [3]f32 = .{ 0.26862954, 0.26130258, 0.27577711 };
};

// ============================================================================
// Context Stub
// ============================================================================

/// Stub vision context.
pub const Context = struct {
    allocator: std.mem.Allocator = undefined,

    pub fn init(_: std.mem.Allocator) Error!*Context {
        return error.VisionDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn createImage(_: *Context, _: u32, _: u32, _: u8) Error!Image {
        return error.VisionDisabled;
    }

    pub fn preprocessForModel(
        _: *Context,
        _: *const Image,
        _: u32,
    ) Error![]f32 {
        return error.VisionDisabled;
    }

    pub fn preprocessForModelCustom(
        _: *Context,
        _: *const Image,
        _: u32,
        _: [3]f32,
        _: [3]f32,
    ) Error![]f32 {
        return error.VisionDisabled;
    }
};

/// Check if vision is enabled at compile time.
pub fn isEnabled() bool {
    return false;
}

// ============================================================================
// Vision Transformer (ViT) Stub
// ============================================================================

/// Vision Transformer configuration (stub).
pub const ViTConfig = struct {
    image_size: u32 = 224,
    patch_size: u32 = 16,
    in_channels: u32 = 3,
    hidden_size: u32 = 768,
    num_layers: u32 = 12,
    num_heads: u32 = 12,
    mlp_dim: u32 = 3072,
    dropout: f32 = 0.0,
    attention_dropout: f32 = 0.0,
    use_class_token: bool = true,
    num_classes: u32 = 0,
    layer_norm_eps: f32 = 1e-6,
    use_gelu: bool = true,
    pre_norm: bool = true,

    const Self = @This();

    pub fn numPatches(self: Self) u32 {
        const patches_per_side = self.image_size / self.patch_size;
        return patches_per_side * patches_per_side;
    }

    pub fn seqLength(self: Self) u32 {
        return self.numPatches() + @as(u32, if (self.use_class_token) 1 else 0);
    }

    pub fn tiny(image_size: u32, patch_size: u32) Self {
        return .{ .image_size = image_size, .patch_size = patch_size, .hidden_size = 192, .num_layers = 12, .num_heads = 3, .mlp_dim = 768 };
    }

    pub fn small(image_size: u32, patch_size: u32) Self {
        return .{ .image_size = image_size, .patch_size = patch_size, .hidden_size = 384, .num_layers = 12, .num_heads = 6, .mlp_dim = 1536 };
    }

    pub fn base(image_size: u32, patch_size: u32) Self {
        return .{ .image_size = image_size, .patch_size = patch_size, .hidden_size = 768, .num_layers = 12, .num_heads = 12, .mlp_dim = 3072 };
    }

    pub fn large(image_size: u32, patch_size: u32) Self {
        return .{ .image_size = image_size, .patch_size = patch_size, .hidden_size = 1024, .num_layers = 24, .num_heads = 16, .mlp_dim = 4096 };
    }

    pub fn huge(image_size: u32, patch_size: u32) Self {
        return .{ .image_size = image_size, .patch_size = patch_size, .hidden_size = 1280, .num_layers = 32, .num_heads = 16, .mlp_dim = 5120 };
    }
};
