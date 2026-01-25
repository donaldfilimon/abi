//! Vision Module - Image Processing Foundation
//!
//! Provides image loading, processing, and preprocessing capabilities for the ABI framework.
//! This module serves as the foundation for computer vision and image-based AI tasks.
//!
//! ## Features
//!
//! - **Image**: Core image type with pixel manipulation
//! - **Preprocessing**: Common operations (resize, normalize, crop, pad)
//! - **Format Support**: Grayscale, RGB, and RGBA images
//! - **Neural Network Layers**: Conv2D, BatchNorm, Pooling
//!
//! ## Usage
//!
//! ```zig
//! const vision = @import("abi").ai.vision;
//!
//! // Create a new image
//! var img = try vision.Image.init(allocator, 224, 224, 3);
//! defer img.deinit();
//!
//! // Resize for model input
//! var resized = try vision.preprocessing.resize(&img, 112, 112);
//! defer resized.deinit();
//!
//! // Convert to float tensor for inference
//! const tensor = try vision.preprocessing.toFloat(&resized, allocator);
//! defer allocator.free(tensor);
//! ```

const std = @import("std");
const build_options = @import("build_options");

// ============================================================================
// Image Processing Modules
// ============================================================================

/// Core image type and operations
pub const image = @import("image.zig");

/// Image preprocessing operations (resize, normalize, crop, etc.)
pub const preprocessing = @import("preprocessing.zig");

// ============================================================================
// Neural Network Layer Modules
// ============================================================================

/// 2D Convolution layer
pub const conv = @import("conv.zig");

/// Pooling layers (MaxPool, AvgPool, AdaptiveAvgPool)
pub const pooling = @import("pooling.zig");

/// Batch normalization
pub const batchnorm = @import("batchnorm.zig");

/// Vision Transformer (ViT) for image encoding
pub const vit = @import("vit.zig");

/// Multi-modal fusion and contrastive learning
pub const multimodal = @import("multimodal.zig");

// ============================================================================
// Image Type Re-exports
// ============================================================================

/// Core image struct for pixel data storage and manipulation
pub const Image = image.Image;

/// Color channel constants (grayscale=1, RGB=3, RGBA=4)
pub const Channels = image.Channels;

/// Image operation errors
pub const ImageError = image.ImageError;

// ============================================================================
// Preprocessing Function Re-exports
// ============================================================================

/// Resize an image using bilinear interpolation
pub const resize = preprocessing.resize;

/// Normalize image pixels in-place using mean and standard deviation
pub const normalize = preprocessing.normalize;

/// Convert image to floating-point tensor in CHW format
pub const toFloat = preprocessing.toFloat;

/// Convert image to normalized floating-point tensor
pub const toFloatNormalized = preprocessing.toFloatNormalized;

/// Center crop an image to a square
pub const centerCrop = preprocessing.centerCrop;

/// Center crop an image to a rectangle
pub const centerCropRect = preprocessing.centerCropRect;

/// Pad an image with a constant value
pub const pad = preprocessing.pad;

/// Pad an image with asymmetric borders
pub const padAsymmetric = preprocessing.padAsymmetric;

/// Convert RGB image to grayscale
pub const toGrayscale = preprocessing.toGrayscale;

/// Flip image horizontally (mirror)
pub const flipHorizontal = preprocessing.flipHorizontal;

/// Flip image vertically
pub const flipVertical = preprocessing.flipVertical;

/// Preprocessing operation errors
pub const PreprocessError = preprocessing.PreprocessError;

// ============================================================================
// Neural Network Layer Re-exports
// ============================================================================

/// 2D Convolutional layer
pub const Conv2D = conv.Conv2D;

/// Gradients from Conv2D backward pass
pub const ConvGradients = conv.ConvGradients;

/// Im2col transformation for efficient convolution
pub const im2col = conv.im2col;

/// Col2im transformation (inverse of im2col)
pub const col2im = conv.col2im;

/// Max pooling layer
pub const MaxPool2D = pooling.MaxPool2D;

/// Average pooling layer
pub const AvgPool2D = pooling.AvgPool2D;

/// Adaptive average pooling layer
pub const AdaptiveAvgPool2D = pooling.AdaptiveAvgPool2D;

/// Result from pooling operations
pub const PoolResult = pooling.PoolResult;

/// Global average pooling
pub const globalAvgPool2D = pooling.globalAvgPool2D;

/// Batch normalization layer
pub const BatchNorm2D = batchnorm.BatchNorm2D;

/// Gradients from BatchNorm backward pass
pub const BatchNormGradients = batchnorm.BatchNormGradients;

/// Vision Transformer model
pub const VisionTransformer = vit.VisionTransformer;

/// Vision Transformer configuration
pub const ViTConfig = vit.ViTConfig;

/// Patch embedding layer for ViT
pub const PatchEmbedding = vit.PatchEmbedding;

/// Multi-head self-attention
pub const MultiHeadAttention = vit.MultiHeadAttention;

/// Transformer encoder block
pub const TransformerBlock = vit.TransformerBlock;

/// Layer normalization
pub const ViTLayerNorm = vit.LayerNorm;

/// MLP block (feed-forward network)
pub const ViTMLP = vit.MLP;

/// GELU activation function
pub const gelu = vit.gelu;

/// CLIP-style contrastive learning model
pub const CLIPModel = multimodal.CLIPModel;

/// Multi-modal configuration
pub const MultiModalConfig = multimodal.MultiModalConfig;

/// Contrastive loss for aligning embeddings
pub const ContrastiveLoss = multimodal.ContrastiveLoss;

/// Cross-modal attention layer
pub const CrossAttention = multimodal.CrossAttention;

/// Text encoder for multi-modal learning
pub const TextEncoder = multimodal.TextEncoder;

/// Text embedding layer
pub const TextEmbedding = multimodal.TextEmbedding;

/// Fusion block for bidirectional cross-attention
pub const FusionBlock = multimodal.FusionBlock;

/// Unified embedding space for multi-modal retrieval
pub const UnifiedEmbeddingSpace = multimodal.UnifiedEmbeddingSpace;

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
// Vision Context
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

/// Vision context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Context {
        if (!isEnabled()) return error.VisionDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }

    /// Create a new blank image.
    pub fn createImage(self: *Context, width: u32, height: u32, channels: u8) !Image {
        return Image.init(self.allocator, width, height, channels);
    }

    /// Preprocess an image for model input with standard ImageNet normalization.
    /// Resizes to target_size x target_size and converts to normalized float tensor.
    pub fn preprocessForModel(
        self: *Context,
        img: *const Image,
        target_size: u32,
    ) ![]f32 {
        // Resize
        var resized = try resize(img, target_size, target_size);
        defer resized.deinit();

        // Convert to normalized float tensor
        return toFloatNormalized(&resized, self.allocator, ImageNetNorm.mean, ImageNetNorm.std);
    }

    /// Preprocess with custom normalization parameters.
    pub fn preprocessForModelCustom(
        self: *Context,
        img: *const Image,
        target_size: u32,
        mean: [3]f32,
        std_dev: [3]f32,
    ) ![]f32 {
        var resized = try resize(img, target_size, target_size);
        defer resized.deinit();

        return toFloatNormalized(&resized, self.allocator, mean, std_dev);
    }
};

/// Check if vision is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_vision;
}

// ============================================================================
// Tests
// ============================================================================

test "module exports are accessible" {
    // Verify all exports compile
    _ = Image;
    _ = Channels;
    _ = ImageError;
    _ = resize;
    _ = normalize;
    _ = toFloat;
    _ = centerCrop;
    _ = pad;
    _ = PreprocessError;
    _ = Conv2D;
    _ = MaxPool2D;
    _ = BatchNorm2D;
}

test "ImageNetNorm values are reasonable" {
    // ImageNet mean should be in [0, 1]
    for (ImageNetNorm.mean) |m| {
        try std.testing.expect(m >= 0.0 and m <= 1.0);
    }
    // ImageNet std should be positive and reasonable
    for (ImageNetNorm.std) |s| {
        try std.testing.expect(s > 0.0 and s <= 1.0);
    }
}

test "Context creates and destroys" {
    if (!isEnabled()) return;

    const allocator = std.testing.allocator;
    var ctx = try Context.init(allocator);
    defer ctx.deinit();

    var img = try ctx.createImage(10, 10, 3);
    defer img.deinit();

    try std.testing.expectEqual(@as(u32, 10), img.width);
}

test {
    // Run all imported tests
    _ = image;
    _ = preprocessing;
    _ = conv;
    _ = pooling;
    _ = batchnorm;
    _ = vit;
    _ = multimodal;
}
