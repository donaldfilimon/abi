//! Shared types for the context engine module.
//!
//! Used by mod.zig, stub.zig, and sub-modules (audio.zig, etc.)
//! to avoid circular imports and prevent type drift.

const std = @import("std");

/// Represents a single frame of video or screen capture.
pub const VideoFrame = struct {
    width: u32,
    height: u32,
    /// RGB/RGBA pixel data
    data: []const u8,
    timestamp_ms: i64,

    pub fn deinit(self: *VideoFrame, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

/// Represents a chunk of raw or encoded audio.
pub const AudioChunk = struct {
    /// e.g. 16000 for 16kHz
    sample_rate: u32,
    /// 1 for mono, 2 for stereo
    channels: u8,
    /// Audio data (typically PCM f32 or s16)
    data: []const u8,
    timestamp_ms: i64,

    pub fn deinit(self: *AudioChunk, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

/// A unified message that can contain multiple data modalities.
pub const ContextMessage = struct {
    role: Role,
    text: ?[]const u8 = null,
    audio: ?[]const AudioChunk = null,
    video: ?[]const VideoFrame = null,
    timestamp_ms: i64,

    pub const Role = enum { system, user, assistant, tool };

    pub fn deinit(self: *ContextMessage, allocator: std.mem.Allocator) void {
        if (self.text) |t| allocator.free(t);
        if (self.audio) |chunks| {
            for (chunks) |*chunk| {
                chunk.deinit(allocator);
            }
            allocator.free(chunks);
        }
        if (self.video) |frames| {
            for (frames) |*frame| {
                frame.deinit(allocator);
            }
            allocator.free(frames);
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
