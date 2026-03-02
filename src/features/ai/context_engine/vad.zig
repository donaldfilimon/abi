//! Voice Activity Detection (VAD) and Audio Streaming
//!
//! Provides zero-dependency native audio stream handling and basic VAD based on RMS energy.
//! Designed to feed real-time biological inputs into the Triad's Context Engine.

const std = @import("std");
const posix = std.posix;

/// Represents a simple VAD configuration.
pub const VadConfig = struct {
    sample_rate: u32 = 16000,
    channels: u8 = 1,
    /// Energy threshold to consider "active speech"
    energy_threshold: f32 = 0.01,
    /// Frame size in milliseconds
    frame_duration_ms: u32 = 30,
};

/// A simple Voice Activity Detector using RMS energy.
pub const VoiceActivityDetector = struct {
    allocator: std.mem.Allocator,
    config: VadConfig,

    pub fn init(allocator: std.mem.Allocator, config: VadConfig) VoiceActivityDetector {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *VoiceActivityDetector) void {
        _ = self;
    }

    /// Calculate RMS energy of a f32 PCM frame
    pub fn calculateRms(self: *VoiceActivityDetector, pcm_data: []const f32) f32 {
        _ = self;
        if (pcm_data.len == 0) return 0.0;
        
        var sum_squares: f32 = 0.0;
        for (pcm_data) |sample| {
            sum_squares += sample * sample;
        }

        return @sqrt(sum_squares / @as(f32, @floatFromInt(pcm_data.len)));
    }

    /// Determine if the given frame contains voice activity based on configured threshold.
    pub fn isSpeech(self: *VoiceActivityDetector, pcm_data: []const f32) bool {
        const rms = self.calculateRms(pcm_data);
        return rms > self.config.energy_threshold;
    }
};

/// An audio streamer that reads raw PCM data from a file descriptor (e.g., stdin).
pub const AudioStreamer = struct {
    fd: posix.fd_t,
    vad: VoiceActivityDetector,
    buffer: []f32,
    frame_size_bytes: usize,

    pub fn init(allocator: std.mem.Allocator, fd: posix.fd_t, config: VadConfig) !AudioStreamer {
        const vad = VoiceActivityDetector.init(allocator, config);
        
        // Frame size: sample_rate * (frame_duration_ms / 1000) * channels * size_of(f32)
        const frame_samples = (config.sample_rate * config.frame_duration_ms) / 1000;
        const frame_size_bytes = frame_samples * config.channels * @sizeOf(f32);

        const buffer = try allocator.alloc(f32, frame_samples * config.channels);

        return .{
            .fd = fd,
            .vad = vad,
            .buffer = buffer,
            .frame_size_bytes = frame_size_bytes,
        };
    }

    pub fn deinit(self: *AudioStreamer) void {
        self.vad.allocator.free(self.buffer);
        self.vad.deinit();
    }

    /// Read a single frame. Returns a slice to the f32 samples if speech is detected.
    /// Returns null if no speech or error. (Zero-copy over the internal buffer).
    pub fn readActiveFrame(self: *AudioStreamer) !?[]const f32 {
        // Read raw bytes using posix.read
        const byte_slice = std.mem.sliceAsBytes(self.buffer);
        const bytes_read = try posix.read(self.fd, byte_slice);
        if (bytes_read < self.frame_size_bytes) {
            return null; // Incomplete frame or EOF
        }

        // Reinterpret bytes as f32 (assuming platform endianness matches stream)
        const samples = self.buffer[0 .. bytes_read / @sizeOf(f32)];
        
        if (self.vad.isSpeech(samples)) {
            return samples;
        }

        return null;
    }
};

test "vad basic calculation" {
    const allocator = std.testing.allocator;
    var vad = VoiceActivityDetector.init(allocator, .{ .energy_threshold = 0.5 });
    defer vad.deinit();

    const silence = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    try std.testing.expectEqual(false, vad.isSpeech(&silence));

    const loud = [_]f32{ 0.8, -0.9, 0.7, -0.8 };
    try std.testing.expectEqual(true, vad.isSpeech(&loud));
}
