//! Context Engine Stub
//!
//! Provides the stub data structures and interfaces for the
//! Contextually Aware Data Eccentric System when AI is disabled.

const std = @import("std");
const shared_types = @import("types.zig");

// Re-export shared types (stub deinit is a no-op since no allocations occur).
pub const VideoFrame = shared_types.VideoFrame;
pub const AudioChunk = shared_types.AudioChunk;
pub const ContextMessage = shared_types.ContextMessage;

// Sub-modules imported directly (none back-import mod.zig).
pub const triad = @import("triad.zig");
pub const jumpstart = @import("jumpstart.zig");
pub const audio = @import("audio.zig");
pub const telemetry = @import("telemetry.zig");
pub const vision = @import("vision.zig");
pub const codebase_indexer = @import("codebase_indexer.zig");
pub const vad = @import("vad.zig");

pub const ContextProcessor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ContextProcessor {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *ContextProcessor) void {}

    pub fn processAudio(_: *ContextProcessor, _: AudioChunk) ![]const u8 {
        return error.AiDisabled;
    }

    pub fn processVideo(_: *ContextProcessor, _: VideoFrame) ![]const u8 {
        return error.AiDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
