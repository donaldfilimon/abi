//! Core Image Type
//!
//! Provides the fundamental Image struct for image processing in the ABI framework.
//! Supports grayscale (1 channel), RGB (3 channels), and RGBA (4 channels) formats.

const std = @import("std");

/// Color channel constants
pub const Channels = struct {
    pub const grayscale: u8 = 1;
    pub const rgb: u8 = 3;
    pub const rgba: u8 = 4;
};

/// Image representation with pixel data and metadata.
/// Pixel data is stored in row-major order (left to right, top to bottom).
pub const Image = struct {
    /// Image width in pixels
    width: u32,
    /// Image height in pixels
    height: u32,
    /// Number of color channels (1=grayscale, 3=RGB, 4=RGBA)
    channels: u8,
    /// Raw pixel data in row-major order
    data: []u8,
    /// Allocator used for memory management
    allocator: std.mem.Allocator,

    /// Initialize a new image with the given dimensions and channels.
    /// Allocates memory for pixel data and initializes to zero.
    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, channels: u8) !Image {
        if (width == 0 or height == 0) return error.InvalidDimensions;
        if (channels != Channels.grayscale and channels != Channels.rgb and channels != Channels.rgba) {
            return error.InvalidChannels;
        }

        const size = @as(usize, width) * @as(usize, height) * @as(usize, channels);
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);

        return .{
            .width = width,
            .height = height,
            .channels = channels,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Create an image from existing pixel data.
    /// The data is copied, so the caller retains ownership of the original.
    pub fn fromData(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        channels: u8,
        source_data: []const u8,
    ) !Image {
        const expected_size = @as(usize, width) * @as(usize, height) * @as(usize, channels);
        if (source_data.len != expected_size) return error.DataSizeMismatch;

        var img = try init(allocator, width, height, channels);
        @memcpy(img.data, source_data);
        return img;
    }

    /// Free the image data.
    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    /// Get pixel value at (x, y) coordinates.
    /// Returns a slice of length `channels` containing the pixel values.
    /// Returns null if coordinates are out of bounds.
    pub fn getPixel(self: *const Image, x: u32, y: u32) ?[]const u8 {
        if (x >= self.width or y >= self.height) return null;

        const offset = self.pixelOffset(x, y);
        return self.data[offset .. offset + self.channels];
    }

    /// Set pixel value at (x, y) coordinates.
    /// The pixel slice must have length equal to `channels`.
    pub fn setPixel(self: *Image, x: u32, y: u32, pixel: []const u8) void {
        if (x >= self.width or y >= self.height) return;
        if (pixel.len != self.channels) return;

        const offset = self.pixelOffset(x, y);
        @memcpy(self.data[offset .. offset + self.channels], pixel);
    }

    /// Create a deep copy of the image.
    pub fn clone(self: *const Image) !Image {
        const new_data = try self.allocator.alloc(u8, self.data.len);
        @memcpy(new_data, self.data);

        return .{
            .width = self.width,
            .height = self.height,
            .channels = self.channels,
            .data = new_data,
            .allocator = self.allocator,
        };
    }

    /// Fill the entire image with a single pixel value.
    pub fn fill(self: *Image, pixel: []const u8) void {
        if (pixel.len != self.channels) return;

        var y: u32 = 0;
        while (y < self.height) : (y += 1) {
            var x: u32 = 0;
            while (x < self.width) : (x += 1) {
                self.setPixel(x, y, pixel);
            }
        }
    }

    /// Get the total number of pixels in the image.
    pub fn pixelCount(self: *const Image) usize {
        return @as(usize, self.width) * @as(usize, self.height);
    }

    /// Get the total size of pixel data in bytes.
    pub fn dataSize(self: *const Image) usize {
        return self.data.len;
    }

    /// Get the stride (bytes per row) of the image.
    pub fn stride(self: *const Image) usize {
        return @as(usize, self.width) * @as(usize, self.channels);
    }

    /// Check if this image has the same dimensions as another.
    pub fn sameDimensions(self: *const Image, other: *const Image) bool {
        return self.width == other.width and
            self.height == other.height and
            self.channels == other.channels;
    }

    /// Calculate the byte offset for a pixel at (x, y).
    fn pixelOffset(self: *const Image, x: u32, y: u32) usize {
        return (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * @as(usize, self.channels);
    }
};

/// Image errors
pub const ImageError = error{
    InvalidDimensions,
    InvalidChannels,
    DataSizeMismatch,
    OutOfBounds,
    OutOfMemory,
};

// ============================================================================
// Tests
// ============================================================================

test "Image.init creates zero-filled image" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();

    try std.testing.expectEqual(@as(u32, 10), img.width);
    try std.testing.expectEqual(@as(u32, 10), img.height);
    try std.testing.expectEqual(@as(u8, 3), img.channels);
    try std.testing.expectEqual(@as(usize, 300), img.data.len);

    // Verify zero-filled
    for (img.data) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "Image.init rejects invalid dimensions" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidDimensions, Image.init(allocator, 0, 10, 3));
    try std.testing.expectError(error.InvalidDimensions, Image.init(allocator, 10, 0, 3));
    try std.testing.expectError(error.InvalidChannels, Image.init(allocator, 10, 10, 2));
    try std.testing.expectError(error.InvalidChannels, Image.init(allocator, 10, 10, 5));
}

test "Image.getPixel and setPixel work correctly" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 4, 4, 3);
    defer img.deinit();

    // Set pixel at (2, 1)
    const pixel = [_]u8{ 255, 128, 64 };
    img.setPixel(2, 1, &pixel);

    // Get the same pixel
    const retrieved = img.getPixel(2, 1).?;
    try std.testing.expectEqual(@as(u8, 255), retrieved[0]);
    try std.testing.expectEqual(@as(u8, 128), retrieved[1]);
    try std.testing.expectEqual(@as(u8, 64), retrieved[2]);

    // Out of bounds returns null
    try std.testing.expectEqual(@as(?[]const u8, null), img.getPixel(10, 10));
}

test "Image.clone creates independent copy" {
    const allocator = std.testing.allocator;

    var original = try Image.init(allocator, 4, 4, 3);
    defer original.deinit();

    const pixel = [_]u8{ 100, 150, 200 };
    original.setPixel(1, 1, &pixel);

    var copy = try original.clone();
    defer copy.deinit();

    // Modify original
    const new_pixel = [_]u8{ 0, 0, 0 };
    original.setPixel(1, 1, &new_pixel);

    // Copy should be unchanged
    const copy_pixel = copy.getPixel(1, 1).?;
    try std.testing.expectEqual(@as(u8, 100), copy_pixel[0]);
    try std.testing.expectEqual(@as(u8, 150), copy_pixel[1]);
    try std.testing.expectEqual(@as(u8, 200), copy_pixel[2]);
}

test "Image.fromData creates image from existing data" {
    const allocator = std.testing.allocator;

    const source = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128 };
    var img = try Image.fromData(allocator, 2, 2, 3, &source);
    defer img.deinit();

    const p00 = img.getPixel(0, 0).?;
    try std.testing.expectEqual(@as(u8, 255), p00[0]); // Red

    const p10 = img.getPixel(1, 0).?;
    try std.testing.expectEqual(@as(u8, 255), p10[1]); // Green

    const p01 = img.getPixel(0, 1).?;
    try std.testing.expectEqual(@as(u8, 255), p01[2]); // Blue
}

test "Image.fill sets all pixels" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 3, 3, 1);
    defer img.deinit();

    const pixel = [_]u8{128};
    img.fill(&pixel);

    for (img.data) |byte| {
        try std.testing.expectEqual(@as(u8, 128), byte);
    }
}

test "Image supports grayscale" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 5, 5, Channels.grayscale);
    defer img.deinit();

    try std.testing.expectEqual(@as(u8, 1), img.channels);
    try std.testing.expectEqual(@as(usize, 25), img.data.len);
}

test "Image supports RGBA" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 5, 5, Channels.rgba);
    defer img.deinit();

    try std.testing.expectEqual(@as(u8, 4), img.channels);
    try std.testing.expectEqual(@as(usize, 100), img.data.len);

    const pixel = [_]u8{ 255, 128, 64, 200 };
    img.setPixel(2, 2, &pixel);

    const retrieved = img.getPixel(2, 2).?;
    try std.testing.expectEqual(@as(u8, 200), retrieved[3]); // Alpha
}
