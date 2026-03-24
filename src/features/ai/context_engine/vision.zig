//! Native Vision Matrix
//!
//! Translates raw desktop screen buffers into tokenized semantic grids.
//! Tracks pixel deltas to trigger autonomous Triad analysis only when
//! the user's screen significantly changes.

const std = @import("std");

pub const VideoFrame = struct {
    width: u32,
    height: u32,
    data: []const u8,
};

pub const VideoFrameStreamer = struct {
    allocator: std.mem.Allocator,
    matrix: VisionMatrix,
    active: bool = false,

    pub fn init(allocator: std.mem.Allocator) VideoFrameStreamer {
        return .{
            .allocator = allocator,
            .matrix = VisionMatrix.init(allocator),
        };
    }

    pub fn deinit(self: *VideoFrameStreamer) void {
        self.matrix.deinit();
    }

    pub const CaptureError = error{
        PlatformNotSupported,
        LinuxCaptureNotImplemented,
        WindowsCaptureNotImplemented,
        OutOfMemory,
    };

    /// Captures a frame natively without disk I/O.
    ///
    /// Platform support:
    /// - macOS: Uses CoreGraphics (CGDisplayCreateImage) for zero-copy screen capture.
    /// - Linux: Not yet implemented. Would require X11 (XGetImage/XShmGetImage) or
    ///   Wayland (wlr-screencopy-unstable-v1 / xdg-portal RemoteDesktop) integration.
    ///   Contributions welcome — see https://github.com/ABI/abi/issues for tracking.
    /// - Windows: Not yet implemented. Would require DXGI Desktop Duplication API
    ///   (IDXGIOutputDuplication) for performant capture.
    /// - WASM/freestanding: Screen capture is not applicable on these targets.
    pub fn captureFrame(self: *VideoFrameStreamer) CaptureError!?VideoFrame {
        if (!self.active) return null;

        const builtin = @import("builtin");
        switch (builtin.os.tag) {
            .macos => return self.captureFrameMacOS(),
            .linux => {
                // X11 path: link libX11, use XOpenDisplay/XGetImage or MIT-SHM extension.
                // Wayland path: use wlr-screencopy-unstable-v1 protocol or xdg-desktop-portal.
                return error.LinuxCaptureNotImplemented;
            },
            .windows => {
                // DXGI path: IDXGIOutputDuplication::AcquireNextFrame for GPU-accelerated capture.
                return error.WindowsCaptureNotImplemented;
            },
            else => {
                return error.PlatformNotSupported;
            },
        }
    }

    /// macOS-specific screen capture via CoreGraphics.
    fn captureFrameMacOS(self: *VideoFrameStreamer) CaptureError!?VideoFrame {
        const builtin = @import("builtin");
        if (comptime builtin.os.tag != .macos) return null;

        const objc = struct {
            pub extern "c" fn CGMainDisplayID() u32;
            pub extern "c" fn CGDisplayCreateImage(displayID: u32) ?*anyopaque;
            pub extern "c" fn CGImageGetWidth(image: *anyopaque) usize;
            pub extern "c" fn CGImageGetHeight(image: *anyopaque) usize;
            pub extern "c" fn CGImageGetDataProvider(image: *anyopaque) ?*anyopaque;
            pub extern "c" fn CGDataProviderCopyData(provider: *anyopaque) ?*anyopaque;
            pub extern "c" fn CFDataGetBytePtr(data: *anyopaque) [*]const u8;
            pub extern "c" fn CFDataGetLength(data: *anyopaque) isize;
            pub extern "c" fn CFRelease(cf: *anyopaque) void;
            pub extern "c" fn CGImageRelease(image: *anyopaque) void;
        };

        const main_display = objc.CGMainDisplayID();
        const cg_image = objc.CGDisplayCreateImage(main_display) orelse return null;
        defer objc.CGImageRelease(cg_image);

        const width = objc.CGImageGetWidth(cg_image);
        const height = objc.CGImageGetHeight(cg_image);

        const provider = objc.CGImageGetDataProvider(cg_image) orelse return null;
        const cf_data = objc.CGDataProviderCopyData(provider) orelse return null;
        defer objc.CFRelease(cf_data);

        const length = objc.CFDataGetLength(cf_data);
        if (length <= 0) return null;

        const ptr = objc.CFDataGetBytePtr(cf_data);

        // Copy the raw pixel data out so the CFData can be safely released
        const pixel_buffer = try self.allocator.alloc(u8, @intCast(length));
        @memcpy(pixel_buffer, ptr[0..@intCast(length)]);

        return VideoFrame{
            .width = @intCast(width),
            .height = @intCast(height),
            .data = pixel_buffer,
        };
    }
};

pub const VisionMatrix = struct {
    allocator: std.mem.Allocator,
    last_hash: u64,

    pub fn init(allocator: std.mem.Allocator) VisionMatrix {
        return .{
            .allocator = allocator,
            .last_hash = 0,
        };
    }

    pub fn deinit(self: *VisionMatrix) void {
        _ = self;
    }

    /// Compares a raw screen buffer against the last known state.
    /// Returns true if a major visual threshold was crossed.
    pub fn detectMotion(self: *VisionMatrix, screen_data: []const u8) bool {
        // Stub: A real implementation computes a perceptual hash or delta
        const current_hash = std.hash.CityHash64.hash(screen_data);

        if (self.last_hash == 0 or current_hash != self.last_hash) {
            self.last_hash = current_hash;
            return true;
        }
        return false;
    }

    /// Translates raw pixels into a highly compressed semantic grid embedding
    /// suitable for WDBX indexing.
    pub fn encodeSemanticGrid(self: *VisionMatrix, screen_data: []const u8) ![]const f32 {
        _ = screen_data;
        // Stub: CoreML or MLX tensor projection
        const synthetic_embedding = try self.allocator.alloc(f32, 256);
        @memset(synthetic_embedding, 0.1);
        return synthetic_embedding;
    }
};

test "captureFrame returns null when inactive" {
    var streamer = VideoFrameStreamer.init(std.testing.allocator);
    defer streamer.deinit();

    // Inactive streamer should return null without platform errors
    const frame = try streamer.captureFrame();
    try std.testing.expect(frame == null);
}

test "captureFrame returns platform-specific error when active" {
    const builtin = @import("builtin");
    var streamer = VideoFrameStreamer.init(std.testing.allocator);
    defer streamer.deinit();
    streamer.active = true;

    switch (builtin.os.tag) {
        .linux => try std.testing.expectError(
            error.LinuxCaptureNotImplemented,
            streamer.captureFrame(),
        ),
        .windows => try std.testing.expectError(
            error.WindowsCaptureNotImplemented,
            streamer.captureFrame(),
        ),
        .macos => {
            // macOS may return null in headless CI (no display)
            if (streamer.captureFrame() catch null) |frame| {
                streamer.allocator.free(frame.data);
            }
        },
        else => try std.testing.expectError(
            error.PlatformNotSupported,
            streamer.captureFrame(),
        ),
    }
}

test "CaptureError includes all platform variants" {
    const E = VideoFrameStreamer.CaptureError;
    const e1: E = error.PlatformNotSupported;
    const e2: E = error.LinuxCaptureNotImplemented;
    const e3: E = error.WindowsCaptureNotImplemented;
    try std.testing.expect(e1 != e2);
    try std.testing.expect(e2 != e3);
}

test {
    std.testing.refAllDecls(@This());
}
