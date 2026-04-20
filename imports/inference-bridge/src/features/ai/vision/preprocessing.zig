//! Image Preprocessing Operations
//!
//! Provides common preprocessing operations for images including:
//! - Resizing (bilinear interpolation)
//! - Normalization (mean/std)
//! - Float tensor conversion
//! - Center cropping
//! - Padding

const std = @import("std");
const image_mod = @import("image.zig");
const Image = image_mod.Image;
const Channels = image_mod.Channels;

/// Preprocessing errors
pub const PreprocessError = error{
    InvalidDimensions,
    InvalidChannels,
    DimensionMismatch,
    OutOfMemory,
};

/// Resize an image using bilinear interpolation.
/// Creates a new image with the specified dimensions.
pub fn resize(img: *const Image, new_width: u32, new_height: u32) !Image {
    if (new_width == 0 or new_height == 0) return error.InvalidDimensions;

    var result = try Image.init(img.allocator, new_width, new_height, img.channels);
    errdefer result.deinit();

    const x_ratio: f32 = @as(f32, @floatFromInt(img.width)) / @as(f32, @floatFromInt(new_width));
    const y_ratio: f32 = @as(f32, @floatFromInt(img.height)) / @as(f32, @floatFromInt(new_height));

    var y: u32 = 0;
    while (y < new_height) : (y += 1) {
        var x: u32 = 0;
        while (x < new_width) : (x += 1) {
            const src_x = @as(f32, @floatFromInt(x)) * x_ratio;
            const src_y = @as(f32, @floatFromInt(y)) * y_ratio;

            const pixel = bilinearSample(img, src_x, src_y);
            result.setPixel(x, y, pixel[0..img.channels]);
        }
    }

    return result;
}

/// Perform bilinear sampling at floating-point coordinates.
fn bilinearSample(img: *const Image, x: f32, y: f32) [4]u8 {
    const x0 = @as(u32, @intFromFloat(@floor(x)));
    const y0 = @as(u32, @intFromFloat(@floor(y)));
    const x1 = @min(x0 + 1, img.width - 1);
    const y1 = @min(y0 + 1, img.height - 1);

    const x_frac = x - @floor(x);
    const y_frac = y - @floor(y);

    // Get the four neighboring pixels
    const p00 = img.getPixel(x0, y0) orelse &[_]u8{ 0, 0, 0, 0 };
    const p10 = img.getPixel(x1, y0) orelse &[_]u8{ 0, 0, 0, 0 };
    const p01 = img.getPixel(x0, y1) orelse &[_]u8{ 0, 0, 0, 0 };
    const p11 = img.getPixel(x1, y1) orelse &[_]u8{ 0, 0, 0, 0 };

    var result: [4]u8 = undefined;

    var c: u8 = 0;
    while (c < img.channels) : (c += 1) {
        const v00 = @as(f32, @floatFromInt(p00[c]));
        const v10 = @as(f32, @floatFromInt(p10[c]));
        const v01 = @as(f32, @floatFromInt(p01[c]));
        const v11 = @as(f32, @floatFromInt(p11[c]));

        // Bilinear interpolation
        const top = v00 * (1.0 - x_frac) + v10 * x_frac;
        const bottom = v01 * (1.0 - x_frac) + v11 * x_frac;
        const value = top * (1.0 - y_frac) + bottom * y_frac;

        result[c] = @as(u8, @intFromFloat(@min(255.0, @max(0.0, value))));
    }

    return result;
}

/// Normalize image pixels in-place using mean and standard deviation.
/// For RGB images, applies per-channel normalization.
/// Values are scaled from [0, 255] to approximately [-2, 2] range.
///
/// Common ImageNet normalization values:
/// - mean: [0.485, 0.456, 0.406] (RGB)
/// - std: [0.229, 0.224, 0.225] (RGB)
pub fn normalize(img: *Image, mean: [3]f32, std_dev: [3]f32) void {
    const channels = @min(img.channels, 3);

    for (img.data, 0..) |*byte, i| {
        const channel_idx = i % img.channels;
        if (channel_idx >= channels) continue; // Skip alpha channel

        const value = @as(f32, @floatFromInt(byte.*)) / 255.0;
        const normalized = (value - mean[channel_idx]) / std_dev[channel_idx];

        // Scale back to [0, 255] range for storage
        // This is a lossy operation, but preserves the normalized distribution
        const scaled = (normalized + 2.0) * 63.75; // Maps [-2, 2] to [0, 255]
        byte.* = @as(u8, @intFromFloat(@min(255.0, @max(0.0, scaled))));
    }
}

/// Convert image to floating-point tensor.
/// Returns an array of f32 values in CHW format (Channels, Height, Width)
/// normalized to [0.0, 1.0] range.
pub fn toFloat(img: *const Image, allocator: std.mem.Allocator) ![]f32 {
    const size = @as(usize, img.width) * @as(usize, img.height) * @as(usize, img.channels);
    const result = try allocator.alloc(f32, size);
    errdefer allocator.free(result);

    // Convert to CHW format (standard for neural networks)
    var c: u8 = 0;
    while (c < img.channels) : (c += 1) {
        var y: u32 = 0;
        while (y < img.height) : (y += 1) {
            var x: u32 = 0;
            while (x < img.width) : (x += 1) {
                const pixel = img.getPixel(x, y).?;
                const out_idx = @as(usize, c) * @as(usize, img.height) * @as(usize, img.width) +
                    @as(usize, y) * @as(usize, img.width) +
                    @as(usize, x);
                result[out_idx] = @as(f32, @floatFromInt(pixel[c])) / 255.0;
            }
        }
    }

    return result;
}

/// Convert image to floating-point tensor with normalization.
/// Combines toFloat and normalize operations in a single pass.
pub fn toFloatNormalized(
    img: *const Image,
    allocator: std.mem.Allocator,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const size = @as(usize, img.width) * @as(usize, img.height) * @as(usize, img.channels);
    const result = try allocator.alloc(f32, size);
    errdefer allocator.free(result);

    // Convert to CHW format with normalization
    var c: u8 = 0;
    while (c < img.channels) : (c += 1) {
        const channel_mean = if (c < 3) mean[c] else 0.0;
        const channel_std = if (c < 3) std_dev[c] else 1.0;

        var y: u32 = 0;
        while (y < img.height) : (y += 1) {
            var x: u32 = 0;
            while (x < img.width) : (x += 1) {
                const pixel = img.getPixel(x, y).?;
                const out_idx = @as(usize, c) * @as(usize, img.height) * @as(usize, img.width) +
                    @as(usize, y) * @as(usize, img.width) +
                    @as(usize, x);

                const value = @as(f32, @floatFromInt(pixel[c])) / 255.0;
                result[out_idx] = (value - channel_mean) / channel_std;
            }
        }
    }

    return result;
}

/// Center crop an image to the specified size.
/// The crop is centered on the image. If crop_size exceeds image dimensions,
/// returns an error.
pub fn centerCrop(img: *const Image, crop_size: u32) !Image {
    if (crop_size > img.width or crop_size > img.height) {
        return error.InvalidDimensions;
    }

    const offset_x = (img.width - crop_size) / 2;
    const offset_y = (img.height - crop_size) / 2;

    var result = try Image.init(img.allocator, crop_size, crop_size, img.channels);
    errdefer result.deinit();

    var y: u32 = 0;
    while (y < crop_size) : (y += 1) {
        var x: u32 = 0;
        while (x < crop_size) : (x += 1) {
            const pixel = img.getPixel(x + offset_x, y + offset_y).?;
            result.setPixel(x, y, pixel);
        }
    }

    return result;
}

/// Center crop with custom width and height.
pub fn centerCropRect(img: *const Image, crop_width: u32, crop_height: u32) !Image {
    if (crop_width > img.width or crop_height > img.height) {
        return error.InvalidDimensions;
    }

    const offset_x = (img.width - crop_width) / 2;
    const offset_y = (img.height - crop_height) / 2;

    var result = try Image.init(img.allocator, crop_width, crop_height, img.channels);
    errdefer result.deinit();

    var y: u32 = 0;
    while (y < crop_height) : (y += 1) {
        var x: u32 = 0;
        while (x < crop_width) : (x += 1) {
            const pixel = img.getPixel(x + offset_x, y + offset_y).?;
            result.setPixel(x, y, pixel);
        }
    }

    return result;
}

/// Pad an image with a constant value.
/// Creates a larger image with the original centered.
pub fn pad(img: *const Image, padding: u32, fill_value: u8) !Image {
    const new_width = img.width + padding * 2;
    const new_height = img.height + padding * 2;

    var result = try Image.init(img.allocator, new_width, new_height, img.channels);
    errdefer result.deinit();

    // Fill with constant value
    @memset(result.data, fill_value);

    // Copy original image to center
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const pixel = img.getPixel(x, y).?;
            result.setPixel(x + padding, y + padding, pixel);
        }
    }

    return result;
}

/// Pad with asymmetric values.
pub fn padAsymmetric(
    img: *const Image,
    pad_top: u32,
    pad_bottom: u32,
    pad_left: u32,
    pad_right: u32,
    fill_value: u8,
) !Image {
    const new_width = img.width + pad_left + pad_right;
    const new_height = img.height + pad_top + pad_bottom;

    var result = try Image.init(img.allocator, new_width, new_height, img.channels);
    errdefer result.deinit();

    // Fill with constant value
    @memset(result.data, fill_value);

    // Copy original image to offset position
    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const pixel = img.getPixel(x, y).?;
            result.setPixel(x + pad_left, y + pad_top, pixel);
        }
    }

    return result;
}

/// Convert RGB image to grayscale using luminance formula.
/// Y = 0.299*R + 0.587*G + 0.114*B
pub fn toGrayscale(img: *const Image) !Image {
    if (img.channels < 3) {
        // Already grayscale, just clone
        return img.clone();
    }

    var result = try Image.init(img.allocator, img.width, img.height, Channels.grayscale);
    errdefer result.deinit();

    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const pixel = img.getPixel(x, y).?;
            const r = @as(f32, @floatFromInt(pixel[0]));
            const g = @as(f32, @floatFromInt(pixel[1]));
            const b = @as(f32, @floatFromInt(pixel[2]));

            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            const gray_pixel = [_]u8{@as(u8, @intFromFloat(@min(255.0, @max(0.0, gray))))};
            result.setPixel(x, y, &gray_pixel);
        }
    }

    return result;
}

/// Flip image horizontally (mirror).
pub fn flipHorizontal(img: *const Image) !Image {
    var result = try Image.init(img.allocator, img.width, img.height, img.channels);
    errdefer result.deinit();

    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const src_x = img.width - 1 - x;
            const pixel = img.getPixel(src_x, y).?;
            result.setPixel(x, y, pixel);
        }
    }

    return result;
}

/// Flip image vertically.
pub fn flipVertical(img: *const Image) !Image {
    var result = try Image.init(img.allocator, img.width, img.height, img.channels);
    errdefer result.deinit();

    var y: u32 = 0;
    while (y < img.height) : (y += 1) {
        var x: u32 = 0;
        while (x < img.width) : (x += 1) {
            const src_y = img.height - 1 - y;
            const pixel = img.getPixel(x, src_y).?;
            result.setPixel(x, y, pixel);
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "resize creates correct dimensions" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();

    // Set some pixel values
    const pixel = [_]u8{ 255, 0, 0 };
    img.fill(&pixel);

    var resized = try resize(&img, 5, 5);
    defer resized.deinit();

    try std.testing.expectEqual(@as(u32, 5), resized.width);
    try std.testing.expectEqual(@as(u32, 5), resized.height);
    try std.testing.expectEqual(@as(u8, 3), resized.channels);
}

test "resize upscales correctly" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 2, 2, 1);
    defer img.deinit();

    // Set a simple pattern
    img.setPixel(0, 0, &[_]u8{0});
    img.setPixel(1, 0, &[_]u8{255});
    img.setPixel(0, 1, &[_]u8{255});
    img.setPixel(1, 1, &[_]u8{0});

    var resized = try resize(&img, 4, 4);
    defer resized.deinit();

    try std.testing.expectEqual(@as(u32, 4), resized.width);
    try std.testing.expectEqual(@as(u32, 4), resized.height);
}

test "toFloat produces correct range" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 2, 2, 3);
    defer img.deinit();

    // Set max values
    const pixel = [_]u8{ 255, 128, 0 };
    img.fill(&pixel);

    const float_data = try toFloat(&img, allocator);
    defer allocator.free(float_data);

    // Check that values are in [0, 1] range
    for (float_data) |value| {
        try std.testing.expect(value >= 0.0 and value <= 1.0);
    }

    // CHW format: first channel should all be 1.0 (255/255)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), float_data[0], 0.01);
}

test "centerCrop produces correct dimensions" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();

    var cropped = try centerCrop(&img, 6);
    defer cropped.deinit();

    try std.testing.expectEqual(@as(u32, 6), cropped.width);
    try std.testing.expectEqual(@as(u32, 6), cropped.height);
}

test "centerCrop rejects oversized crop" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();

    try std.testing.expectError(error.InvalidDimensions, centerCrop(&img, 15));
}

test "pad adds correct border" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 4, 4, 1);
    defer img.deinit();

    const pixel = [_]u8{128};
    img.fill(&pixel);

    var padded = try pad(&img, 2, 0);
    defer padded.deinit();

    try std.testing.expectEqual(@as(u32, 8), padded.width);
    try std.testing.expectEqual(@as(u32, 8), padded.height);

    // Check border is filled with 0
    const corner = padded.getPixel(0, 0).?;
    try std.testing.expectEqual(@as(u8, 0), corner[0]);

    // Check center has original value
    const center = padded.getPixel(2, 2).?;
    try std.testing.expectEqual(@as(u8, 128), center[0]);
}

test "toGrayscale converts correctly" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 2, 2, 3);
    defer img.deinit();

    // Set to pure red
    const red = [_]u8{ 255, 0, 0 };
    img.fill(&red);

    var gray = try toGrayscale(&img);
    defer gray.deinit();

    try std.testing.expectEqual(@as(u8, 1), gray.channels);

    // Pure red should give luminance of ~76 (0.299 * 255)
    const pixel = gray.getPixel(0, 0).?;
    try std.testing.expect(pixel[0] > 70 and pixel[0] < 80);
}

test "flipHorizontal mirrors image" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 4, 2, 1);
    defer img.deinit();

    // Set left side to 255, right side to 0
    img.setPixel(0, 0, &[_]u8{255});
    img.setPixel(1, 0, &[_]u8{255});
    img.setPixel(2, 0, &[_]u8{0});
    img.setPixel(3, 0, &[_]u8{0});

    var flipped = try flipHorizontal(&img);
    defer flipped.deinit();

    // Now left side should be 0, right side should be 255
    try std.testing.expectEqual(@as(u8, 0), flipped.getPixel(0, 0).?[0]);
    try std.testing.expectEqual(@as(u8, 0), flipped.getPixel(1, 0).?[0]);
    try std.testing.expectEqual(@as(u8, 255), flipped.getPixel(2, 0).?[0]);
    try std.testing.expectEqual(@as(u8, 255), flipped.getPixel(3, 0).?[0]);
}

test {
    std.testing.refAllDecls(@This());
}
