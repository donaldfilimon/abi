//! Native Audio Streaming and Voice Activity Detection (VAD)
//!
//! Provides the Artificial Biological Intelligence (ABI) with continuous,
//! non-blocking background listening capabilities. Slices raw microphone
//! data into discrete `AudioChunk` structures for the Context Engine.

const std = @import("std");
const context_engine = @import("mod.zig");
const sync = @import("../../../services/shared/sync.zig");
const abi_time = @import("../../../services/shared/time.zig");

/// Asynchronous Text-To-Speech engine.
/// Offloads blocking TTS playback to a detached thread queue.
pub const TtsEngine = struct {
    allocator: std.mem.Allocator,
    queue: std.ArrayListUnmanaged([]const u8) = .empty,
    mutex: sync.Mutex = .{},
    active: bool = true,
    worker_thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator) !TtsEngine {
        var engine = TtsEngine{ .allocator = allocator };
        engine.worker_thread = try std.Thread.spawn(.{}, workerLoop, .{&engine});
        return engine;
    }

    pub fn deinit(self: *TtsEngine) void {
        self.mutex.lock();
        self.active = false;
        self.mutex.unlock();

        if (self.worker_thread) |*t| {
            t.join();
        }

        for (self.queue.items) |msg| {
            self.allocator.free(msg);
        }
        self.queue.deinit(self.allocator);
    }

    pub fn speak(self: *TtsEngine, text: []const u8) !void {
        const text_copy = try self.allocator.dupe(u8, text);
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.queue.append(self.allocator, text_copy);
    }

    fn workerLoop(self: *TtsEngine) void {
        while (true) {
            var msg: ?[]const u8 = null;
            self.mutex.lock();
            if (!self.active) {
                self.mutex.unlock();
                break;
            }
            if (self.queue.items.len > 0) {
                msg = self.queue.orderedRemove(0);
            }
            self.mutex.unlock();

            if (msg) |text| {
                // In a full implementation, this triggers native say/espeak.
                // We mock the blocking nature here for the framework architecture.
                std.log.info("[TTS Engine] Speaking: {s}", .{text});
                
                // Estimate roughly 100ms per word as blocking audio time
                var words: usize = 1;
                for (text) |c| {
                    if (c == ' ') words += 1;
                }
                abi_time.sleepNs(words * 100 * std.time.ns_per_ms);
                
                self.allocator.free(text);
            } else {
                abi_time.sleepNs(100 * std.time.ns_per_ms);
            }
        }
    }
};

pub const AudioStreamer = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,
    sample_rate: u32,
    is_listening: bool,
    recorder_process: ?std.process.Child,
    voice_buffer: std.ArrayListUnmanaged(u8),
    energy_threshold: u32 = 1000,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io, sample_rate: u32) AudioStreamer {
        return .{
            .allocator = allocator,
            .io = io,
            .sample_rate = sample_rate,
            .is_listening = false,
            .recorder_process = null,
            .voice_buffer = .empty,
        };
    }

    pub fn deinit(self: *AudioStreamer) void {
        self.stopListening();
        self.voice_buffer.deinit(self.allocator);
    }

    /// Spawns a non-blocking background process to capture native microphone data.
    /// Uses rec/sox as a cross-platform fallback for capturing raw 16-bit PCM.
    pub fn startListening(self: *AudioStreamer) !void {
        if (self.is_listening) return;
        
        const argv = &[_][]const u8{
            "rec", "-q", "-t", "raw", "-c", "1", "-b", "16", "-r", "16000", "-"
        };
        
        self.recorder_process = try std.process.spawn(self.io.*, .{
            .argv = argv,
            .stdout = .pipe,
            .stderr = .ignore,
        });

        self.is_listening = true;
        std.log.info("[Audio Streamer] Hardware microphone tapped via native subprocess. VAD active.", .{});
    }

    pub fn stopListening(self: *AudioStreamer) void {
        if (!self.is_listening) return;
        if (self.recorder_process) |*child| {
            child.kill(self.io.*);
            _ = child.wait(self.io.*) catch {};
            self.recorder_process = null;
        }
        self.is_listening = false;
        std.log.info("[Audio Streamer] Hardware microphone released.", .{});
    }

    /// Pulls non-blocking bytes from the active stream and measures raw RMS energy.
    /// If energy exceeds the threshold, it flushes the buffer into an AudioChunk.
    pub fn flushVoiceActivity(self: *AudioStreamer) !?context_engine.AudioChunk {
        if (!self.is_listening) return null;
        
        var child = &self.recorder_process.?;
        const stdout = child.stdout orelse return null;
        
        var buf: [4096]u8 = undefined;
        // In Zig 0.16 with std.Io, reads on spawned pipes can be managed by the event loop.
        // We perform a direct read. If no data is ready, we handle the error or zero bytes.
        const bytes_read = std.posix.read(stdout.handle, &buf) catch |err| switch (err) {
            error.WouldBlock => return null,
            else => return err,
        };

        if (bytes_read == 0) return null;

        // Calculate simple amplitude/energy for VAD
        var total_energy: u64 = 0;
        var i: usize = 0;
        while (i < bytes_read) : (i += 2) {
            if (i + 1 < bytes_read) {
                const sample = std.mem.readInt(i16, buf[i..][0..2], .little);
                total_energy += @abs(sample);
            }
        }
        
        const avg_energy = @as(u32, @intCast(total_energy / (bytes_read / 2)));
        
        if (avg_energy > self.energy_threshold) {
            std.log.info("[VAD] Voice activity detected! Energy: {d}", .{avg_energy});
            try self.voice_buffer.appendSlice(self.allocator, buf[0..bytes_read]);
            
            // If we have enough data (e.g. 1 second = 32000 bytes at 16k 16-bit mono)
            if (self.voice_buffer.items.len > 16000) {
                const audio_data = try self.allocator.dupe(u8, self.voice_buffer.items);
                self.voice_buffer.clearRetainingCapacity();

                return context_engine.AudioChunk{                    .sample_rate = 16000,
                    .channels = 1,
                    .data = audio_data,
                    .timestamp_ms = @intCast(abi_time.timestampMs()),
                };
            }
        }
        
        return null;
    }
};

test {
    std.testing.refAllDecls(@This());
}
