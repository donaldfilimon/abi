//! Common AI configuration types and validation utilities
//!
//! This module consolidates configuration schemas, policies, and retry logic
//! previously scattered across multiple files.

const std = @import("std");

/// Validation utility functions
pub const validation = struct {
    pub fn validateNonEmpty(s: []const u8) !void {
        if (s.len == 0) return error.Empty;
    }
};

/// Configuration schemas for AI operations
pub const Schema = struct {
    pub const SummarizeInput = struct {
        doc_id: []const u8,
        max_tokens: u16 = 512,

        pub fn validate(self: *const SummarizeInput) !void {
            try validation.validateNonEmpty(self.doc_id);
            if (self.max_tokens == 0) return error.InvalidMaxTokens;
        }
    };

    pub const SummarizeOutput = struct {
        summary: []const u8,
        tokens_used: u32,

        pub fn validate(self: *const SummarizeOutput) !void {
            try validation.validateNonEmpty(self.summary);
        }
    };
};

/// Retry and limit configuration
pub const Policy = struct {
    pub const Retry = struct {
        max_attempts: u8 = 3,
        base_ms: u32 = 500,
        factor: f32 = 2.0,
    };

    pub const Limits = struct {
        max_tokens_total: u64 = 5_000_000,
        per_provider_rps: u16 = 5,
        per_provider_parallel: u8 = 4,
    };

    pub const ProviderPolicy = struct {
        name: []const u8,
        allowed: bool = true,
    };

    pub const Config = struct {
        retry: Retry = .{},
        limits: Limits = .{},
        providers: []const ProviderPolicy,
    };
};

/// Retry backoff calculation
pub fn backoff_ms(attempt: u8, base_ms: u32, factor: f32) u32 {
    const exp: f32 = @floatFromInt(attempt);
    const mult = std.math.pow(f32, factor, exp);
    const scaled = @as(f32, @floatFromInt(base_ms)) * mult;
    return @intFromFloat(scaled);
}

test "validation functions" {
    try std.testing.expectError(error.Empty, validation.validateNonEmpty(""));
    try validation.validateNonEmpty("valid");
}

test "Schema validation" {
    const input = Schema.SummarizeInput{
        .doc_id = "test-doc",
        .max_tokens = 256,
    };
    try input.validate();
}

test "Retry backoff calculation" {
    const backoff1 = backoff_ms(1, 500, 2.0);
    try std.testing.expectEqual(@as(u32, 1000), backoff1);

    const backoff2 = backoff_ms(2, 500, 2.0);
    try std.testing.expectEqual(@as(u32, 2000), backoff2);
}
