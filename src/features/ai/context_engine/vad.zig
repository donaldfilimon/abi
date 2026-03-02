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

/// A ring buffer that captures contiguous speech segments with pre/post-roll.
pub const SpeechBuffer = struct {
    allocator: std.mem.Allocator,
    frames: std.ArrayListUnmanaged([]const f32) = .empty,
    active: bool = false,
    silence_frames_count: usize = 0,
    max_silence_frames: usize = 15, // Approx 450ms of silence closes the buffer

    pub fn init(allocator: std.mem.Allocator) SpeechBuffer {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SpeechBuffer) void {
        for (self.frames.items) |frame| {
            self.allocator.free(frame);
        }
        self.frames.deinit(self.allocator);
    }

    /// Appends a frame. If `is_speech` is true, buffer is active.
    /// Returns a combined PCM slice if speech has concluded, otherwise null.
    pub fn processFrame(self: *SpeechBuffer, frame: []const f32, is_speech: bool) !?[]f32 {
        if (is_speech) {
            self.active = true;
            self.silence_frames_count = 0;
            const frame_copy = try self.allocator.dupe(f32, frame);
            try self.frames.append(self.allocator, frame_copy);
        } else if (self.active) {
            self.silence_frames_count += 1;
            const frame_copy = try self.allocator.dupe(f32, frame);
            try self.frames.append(self.allocator, frame_copy);

            if (self.silence_frames_count >= self.max_silence_frames) {
                // Speech ended. Construct final buffer and reset.
                const total_samples = self.frames.items.len * frame.len;
                const result = try self.allocator.alloc(f32, total_samples);
                var offset: usize = 0;
                for (self.frames.items) |f| {
                    std.mem.copyForwards(f32, result[offset..], f);
                    offset += f.len;
                    self.allocator.free(f);
                }
                self.frames.clearRetainingCapacity();
                self.active = false;
                self.silence_frames_count = 0;
                return result;
            }
        }
        return null;
    }
};

/// An audio streamer that reads raw PCM data from a file descriptor (e.g., stdin).
pub const AudioStreamer = struct {
    fd: posix.fd_t,
    vad: VoiceActivityDetector,
    buffer: []f32,
    frame_size_bytes: usize,
    speech_buffer: SpeechBuffer,

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
            .speech_buffer = SpeechBuffer.init(allocator),
        };
    }

    pub fn deinit(self: *AudioStreamer) void {
        self.speech_buffer.deinit();
        self.vad.allocator.free(self.buffer);
        self.vad.deinit();
    }

    /// Read a single frame and process through the speech buffer.
    /// Returns a full speech utterance slice when complete.
    pub fn readUtterance(self: *AudioStreamer) !?[]f32 {
        // Read raw bytes using posix.read
        const byte_slice = std.mem.sliceAsBytes(self.buffer);
        const bytes_read = try posix.read(self.fd, byte_slice);
        if (bytes_read < self.frame_size_bytes) {
            return null; // Incomplete frame or EOF
        }

        const samples = self.buffer[0 .. bytes_read / @sizeOf(f32)];
        const is_speech = self.vad.isSpeech(samples);
        
        return self.speech_buffer.processFrame(samples, is_speech);
    }
    
    /// Legacy direct read (kept for compatibility with context_agent)
    pub fn readActiveFrame(self: *AudioStreamer) !?[]const f32 {
        const byte_slice = std.mem.sliceAsBytes(self.buffer);
        const bytes_read = try posix.read(self.fd, byte_slice);
        if (bytes_read < self.frame_size_bytes) {
            return null; 
        }
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
