//! Vision stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const Error = types.Error;
pub const ImageError = types.ImageError;
pub const PreprocessError = types.PreprocessError;
pub const Channels = types.Channels;
pub const Image = types.Image;
pub const ConvGradients = types.ConvGradients;
pub const PoolResult = types.PoolResult;
pub const BatchNormGradients = types.BatchNormGradients;
pub const ImageNetNorm = types.ImageNetNorm;
pub const ClipNorm = types.ClipNorm;
pub const ViTConfig = types.ViTConfig;
pub const VisionPipeline = types.VisionPipelineType;

// --- Submodule Re-exports ---
pub const image = struct {
    pub const Image = types.Image;
    pub const Channels = types.Channels;
    pub const ImageError = types.ImageError;
};

// --- Preprocessing ---
pub const preprocessing = struct {
    pub fn resize(_: *const Image, _: u32, _: u32) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn normalize(_: *Image, _: [3]f32, _: [3]f32) void {}
    pub fn toFloat(_: *const Image, _: std.mem.Allocator) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn toFloatNormalized(_: *const Image, _: std.mem.Allocator, _: [3]f32, _: [3]f32) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn centerCrop(_: *const Image, _: u32) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn centerCropRect(_: *const Image, _: u32, _: u32) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn pad(_: *const Image, _: u32, _: u8) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn padAsymmetric(_: *const Image, _: u32, _: u32, _: u32, _: u32, _: u8) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn toGrayscale(_: *const Image) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn flipHorizontal(_: *const Image) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn flipVertical(_: *const Image) Error!Image {
        return error.FeatureDisabled;
    }
};

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

// --- Neural Network Layers ---
pub const conv = struct {
    pub const Conv2D = types.Conv2DType;
    pub const ConvGradients = types.ConvGradients;
    pub fn im2col(_: anytype) Error!void {
        return error.FeatureDisabled;
    }
    pub fn col2im(_: anytype) Error!void {
        return error.FeatureDisabled;
    }
};

pub const pooling = struct {
    pub const MaxPool2D = types.MaxPool2DType;
    pub const AvgPool2D = types.AvgPool2DType;
    pub const AdaptiveAvgPool2D = types.AdaptiveAvgPool2DType;
    pub const PoolResult = types.PoolResult;
    pub fn globalAvgPool2D(_: anytype) Error!void {
        return error.FeatureDisabled;
    }
};

pub const batchnorm = struct {
    pub const BatchNorm2D = types.BatchNorm2DType;
    pub const BatchNormGradients = types.BatchNormGradients;
};

pub const Conv2D = conv.Conv2D;
pub const im2col = conv.im2col;
pub const col2im = conv.col2im;
pub const MaxPool2D = pooling.MaxPool2D;
pub const AvgPool2D = pooling.AvgPool2D;
pub const AdaptiveAvgPool2D = pooling.AdaptiveAvgPool2D;
pub const globalAvgPool2D = pooling.globalAvgPool2D;
pub const BatchNorm2D = batchnorm.BatchNorm2D;

// --- ViT Re-exports ---
pub const VisionTransformer_Internal = struct {};
pub const PatchEmbedding_Internal = struct {};
pub const MultiHeadAttention_Internal = struct {};
pub const TransformerBlock_Internal = struct {};
pub const ViTLayerNorm = struct {};
pub const ViTMLP = struct {};
pub const ViTConfig_Internal = types.ViTConfig;

pub const VisionTransformer = VisionTransformer_Internal;
pub const PatchEmbedding = PatchEmbedding_Internal;
pub const MultiHeadAttention = MultiHeadAttention_Internal;
pub const TransformerBlock = TransformerBlock_Internal;

pub fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + @as(f32, @floatCast(std.math.tanh(@as(f64, 0.7978846) * @as(f64, @floatCast(x + 0.044715 * x * x * x))))));
}

// --- Multimodal Re-exports ---
pub const CLIPModel_Internal = struct {};
pub const MultiModalConfig_Internal = struct {};
pub const ContrastiveLoss_Internal = struct {};
pub const CrossAttention_Internal = struct {};
pub const TextEncoder_Internal = struct {};
pub const TextEmbedding_Internal = struct {};
pub const FusionBlock_Internal = struct {};
pub const UnifiedEmbeddingSpace_Internal = struct {};

pub const CLIPModel = CLIPModel_Internal;
pub const MultiModalConfig = MultiModalConfig_Internal;
pub const ContrastiveLoss = ContrastiveLoss_Internal;
pub const CrossAttention = CrossAttention_Internal;
pub const TextEncoder = TextEncoder_Internal;
pub const TextEmbedding = TextEmbedding_Internal;
pub const FusionBlock = FusionBlock_Internal;
pub const UnifiedEmbeddingSpace = UnifiedEmbeddingSpace_Internal;

// --- Sub-module Namespace Re-exports ---
pub const vit = struct {
    pub const embedding = struct {
        pub const PatchEmbedding = PatchEmbedding_Internal;
    };
    pub const attention = struct {
        pub const MultiHeadAttention = MultiHeadAttention_Internal;
        pub fn softmax(_: []f32) void {}
    };
    pub const layers = struct {
        pub const TransformerBlock = TransformerBlock_Internal;
        pub const MLP = ViTMLP;
        pub const LayerNorm = ViTLayerNorm;
        pub const gelu = @import("stub.zig").gelu;
        pub fn geluSlice(_: []f32) void {}
    };

    pub const VisionTransformer = VisionTransformer_Internal;
    pub const ViTConfig = ViTConfig_Internal;
    pub const PatchEmbedding = PatchEmbedding_Internal;
    pub const MultiHeadAttention = MultiHeadAttention_Internal;
    pub const TransformerBlock = TransformerBlock_Internal;
    pub const LayerNorm = ViTLayerNorm;
    pub const MLP = ViTMLP;
    pub const gelu = @import("stub.zig").gelu;
    pub const softmax = attention.softmax;
    pub fn geluSlice(x: []f32) void {
        layers.geluSlice(x);
    }
};

pub const multimodal = struct {
    pub const preprocessing = struct {
        pub const MultiModalConfig = MultiModalConfig_Internal;
        pub const UnifiedEmbeddingSpace = UnifiedEmbeddingSpace_Internal;
    };
    pub const encoders = struct {
        pub const TextEmbedding = TextEmbedding_Internal;
        pub const TextEncoder = TextEncoder_Internal;
    };
    pub const fusion = struct {
        pub const ContrastiveLoss = ContrastiveLoss_Internal;
        pub const CrossAttention = CrossAttention_Internal;
        pub const FusionBlock = FusionBlock_Internal;
        pub const CLIPModel = CLIPModel_Internal;
    };

    pub const CLIPModel = CLIPModel_Internal;
    pub const MultiModalConfig = MultiModalConfig_Internal;
    pub const ContrastiveLoss = ContrastiveLoss_Internal;
    pub const CrossAttention = CrossAttention_Internal;
    pub const TextEncoder = TextEncoder_Internal;
    pub const TextEmbedding = TextEmbedding_Internal;
    pub const FusionBlock = FusionBlock_Internal;
    pub const UnifiedEmbeddingSpace = UnifiedEmbeddingSpace_Internal;
};

// --- Context ---
pub const Context = struct {
    allocator: std.mem.Allocator = undefined,
    pub fn init(_: std.mem.Allocator) Error!*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn createImage(_: *Context, _: u32, _: u32, _: u8) Error!Image {
        return error.FeatureDisabled;
    }
    pub fn preprocessForModel(_: *Context, _: *const Image, _: u32) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn preprocessForModelCustom(_: *Context, _: *const Image, _: u32, _: [3]f32, _: [3]f32) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn createPipeline(_: *Context) Error!VisionPipeline {
        return error.FeatureDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
