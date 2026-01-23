// Test stub for LLM functionality
// This is a minimal GGUF-like structure for testing purposes
// In a real scenario, this would be replaced with actual model data

const std = @import("std");

pub const TestModel = struct {
    pub fn init(allocator: std.mem.Allocator) !TestModel {
        _ = allocator;
        return TestModel{};
    }

    pub fn deinit(self: *TestModel) void {
        _ = self;
    }

    pub fn generate(self: *TestModel, allocator: std.mem.Allocator, prompt: []const u8) ![]const u8 {
        _ = self;
        _ = prompt;

        // Return a simple test response
        return allocator.dupe(u8, "Hello! This is a test response from the ABI framework. The model file would normally be loaded here, but for testing purposes, we're using a stub implementation.");
    }

    pub fn getStats(self: *TestModel) TestStats {
        _ = self;
        return TestStats{
            .total_tokens = 42,
            .prefill_time_ms = 10.0,
            .decode_time_ms = 5.0,
        };
    }
};

pub const TestStats = struct {
    total_tokens: u32,
    prefill_time_ms: f64,
    decode_time_ms: f64,

    pub fn prefillTokensPerSecond(self: TestStats) f64 {
        return @as(f64, @floatFromInt(self.total_tokens)) / (self.prefill_time_ms / 1000.0);
    }

    pub fn decodeTokensPerSecond(self: TestStats) f64 {
        return @as(f64, @floatFromInt(self.total_tokens)) / (self.decode_time_ms / 1000.0);
    }
};
