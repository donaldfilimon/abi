//! Vision stub types — extracted from stub.zig.

const std = @import("std");

// --- Errors ---

pub const Error = error{ FeatureDisabled, InvalidImage, UnsupportedFormat, DecodeFailed, EncodeFailed, OutOfMemory };
pub const ImageError = error{ InvalidDimensions, InvalidChannels, DataSizeMismatch, OutOfBounds, OutOfMemory, FeatureDisabled };
pub const PreprocessError = error{ InvalidDimensions, InvalidChannels, DimensionMismatch, OutOfMemory, FeatureDisabled };

// --- Image Types ---

pub const Channels = struct {
    pub const grayscale: u8 = 1;
    pub const rgb: u8 = 3;
    pub const rgba: u8 = 4;
};

pub const Image = struct {
    width: u32 = 0,
    height: u32 = 0,
    channels: u8 = 0,
    data: []u8 = &.{},
    allocator: std.mem.Allocator = undefined,
    pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u8) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn fromData(_: std.mem.Allocator, _: u32, _: u32, _: u8, _: []const u8) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Image) void {}
    pub fn getPixel(_: *const Image, _: u32, _: u32) ?[]const u8 {
        return null;
    }
    pub fn setPixel(_: *Image, _: u32, _: u32, _: []const u8) void {}
    pub fn clone(_: *const Image) Error!Image {
        return error.FeatureDisabled;
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

// --- Neural Network Layer Types ---

pub const Conv2DType = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ConvGradients = struct {};

pub const MaxPool2DType = struct {
    pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u32) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const AvgPool2DType = struct {
    pub fn init(_: std.mem.Allocator, _: u32, _: u32, _: u32) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const AdaptiveAvgPool2DType = struct {
    pub fn init(_: anytype) @This() {
        return .{};
    }
};

pub const PoolResult = struct {};

pub const BatchNorm2DType = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const BatchNormGradients = struct {};

// --- Normalization Constants ---

pub const ImageNetNorm = struct {
    pub const mean: [3]f32 = .{ 0.485, 0.456, 0.406 };
    pub const std: [3]f32 = .{ 0.229, 0.224, 0.225 };
};

pub const ClipNorm = struct {
    pub const mean: [3]f32 = .{ 0.48145466, 0.4578275, 0.40821073 };
    pub const std: [3]f32 = .{ 0.26862954, 0.26130258, 0.27577711 };
};

// --- ViT Config ---

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
        const p = self.image_size / self.patch_size;
        return p * p;
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

test {
    std.testing.refAllDecls(@This());
}
