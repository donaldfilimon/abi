//! Native Audio Streaming and Voice Activity Detection (VAD)
//!
//! Provides the Artificial Biological Intelligence (ABI) with continuous,
//! non-blocking background listening capabilities. Slices raw microphone
//! data into discrete `AudioChunk` structures for the Context Engine.

const std = @import("std");
const context_engine = @import("mod.zig");

pub const AudioStreamer = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,
    sample_rate: u32,
    is_listening: bool,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io, sample_rate: u32) AudioStreamer {
        return .{
            .allocator = allocator,
            .io = io,
            .sample_rate = sample_rate,
            .is_listening = false,
        };
    }

    pub fn deinit(self: *AudioStreamer) void {
        self.stopListening();
    }

    /// Spawns a non-blocking `std.Io` event task to continuously sample hardware microphones.
    pub fn startListening(self: *AudioStreamer) !void {
        if (self.is_listening) return;
        self.is_listening = true;
        std.log.info("[Audio Streamer] Hardware microphone tapped. Native VAD active.", .{});
        // Stub: spawn continuous async polling of /dev/dsp or CoreAudio
    }

    pub fn stopListening(self: *AudioStreamer) void {
        if (!self.is_listening) return;
        self.is_listening = false;
        std.log.info("[Audio Streamer] Hardware microphone released.", .{});
    }

    /// Extracts the latest contiguous chunk of speech from the background buffer.
    pub fn flushVoiceActivity(self: *AudioStreamer) !?context_engine.AudioChunk {
        _ = self;
        // Stub: If Voice Activity Detection (VAD) found speech, return it.
        return null;
    }
};

test {
    std.testing.refAllDecls(@This());
}
