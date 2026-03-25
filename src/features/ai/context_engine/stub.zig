//! Context Engine Stub
//!
//! Provides the stub data structures and interfaces for the
//! Contextually Aware Data Eccentric System when AI is disabled.

const std = @import("std");

pub const VideoFrame = struct {
    width: u32,
    height: u32,
    data: []const u8,
    timestamp_ms: i64,

    pub fn deinit(_: *VideoFrame, _: std.mem.Allocator) void {}
};

pub const AudioChunk = struct {
    sample_rate: u32,
    channels: u8,
    data: []const u8,
    timestamp_ms: i64,

    pub fn deinit(_: *AudioChunk, _: std.mem.Allocator) void {}
};

pub const ContextMessage = struct {
    role: Role,
    text: ?[]const u8 = null,
    audio: ?[]const AudioChunk = null,
    video: ?[]const VideoFrame = null,
    timestamp_ms: i64,

    pub const Role = enum { system, user, assistant, tool };

    pub fn deinit(_: *ContextMessage, _: std.mem.Allocator) void {}
};

// Sub-modules without back-imports to mod.zig are imported directly.
pub const triad = @import("triad.zig");
pub const jumpstart = @import("jumpstart.zig");
pub const telemetry = @import("telemetry.zig");
pub const vision = @import("vision.zig");
pub const codebase_indexer = @import("codebase_indexer.zig");
pub const vad = @import("vad.zig");

// audio.zig back-imports mod.zig, so provide an inline stub to avoid
// pulling the real context engine into the disabled compilation path.
pub const audio = struct {
    pub const TtsEngine = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator) !TtsEngine {
            _ = allocator;
            return error.AiDisabled;
        }
        pub fn deinit(_: *TtsEngine) void {}
        pub fn speak(_: *TtsEngine, _: []const u8) !void {
            return error.AiDisabled;
        }
    };
    pub const AudioStreamer = struct {
        pub fn init(_: std.mem.Allocator) AudioStreamer {
            return .{};
        }
        pub fn deinit(_: *AudioStreamer) void {}
    };
};

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
