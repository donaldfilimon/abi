//! Shared Foundational Types
//!
//! Common type definitions used across multiple ABI framework domains.
//! Provides standardized representations for confidence, emotional context,
//! identity, and other cross-cutting primitives.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Confidence and Certainty Types
// ============================================================================

/// Granular confidence level with mathematical grounding.
pub const ConfidenceLevel = enum(u8) {
    /// Near-certain (>95% confidence)
    certain = 0,
    /// High confidence (80-95%)
    high = 1,
    /// Moderate confidence (60-80%)
    medium = 2,
    /// Low confidence (40-60%)
    low = 3,
    /// Very uncertain (20-40%)
    uncertain = 4,
    /// Unknown (<20%)
    unknown = 5,

    pub fn fromScore(score: f32) ConfidenceLevel {
        if (score >= 0.95) return .certain;
        if (score >= 0.80) return .high;
        if (score >= 0.60) return .medium;
        if (score >= 0.40) return .low;
        if (score >= 0.20) return .uncertain;
        return .unknown;
    }

    pub fn toScoreRange(self: ConfidenceLevel) struct { min: f32, max: f32 } {
        return switch (self) {
            .certain => .{ .min = 0.95, .max = 1.0 },
            .high => .{ .min = 0.80, .max = 0.95 },
            .medium => .{ .min = 0.60, .max = 0.80 },
            .low => .{ .min = 0.40, .max = 0.60 },
            .uncertain => .{ .min = 0.20, .max = 0.40 },
            .unknown => .{ .min = 0.0, .max = 0.20 },
        };
    }
};

/// Detailed confidence assessment with provenance.
pub const Confidence = struct {
    level: ConfidenceLevel = .medium,
    score: f32 = 0.5,
    reasoning: []const u8 = "No specific reasoning provided",
    calibrated_at: i64 = 0,

    pub fn decay(self: *Confidence, factor: f32) void {
        self.score = @max(0.0, self.score * (1.0 - factor));
        self.level = ConfidenceLevel.fromScore(self.score);
    }
};

// ============================================================================
// Emotional Intelligence Types
// ============================================================================

/// Detected emotional state types.
pub const EmotionType = enum(u8) {
    neutral = 0,
    frustrated = 1,
    excited = 2,
    confused = 3,
    stressed = 4,
    playful = 5,
    grateful = 6,
    curious = 7,
    impatient = 8,
    skeptical = 9,
    enthusiastic = 10,
    disappointed = 11,
    hopeful = 12,
    anxious = 13,
};

/// Full emotional state tracking.
pub const EmotionalState = struct {
    current: EmotionType = .neutral,
    intensity: f32 = 0.0,
    previous: EmotionType = .neutral,
    last_detected: i64 = 0,

    pub fn update(self: *EmotionalState, emotion: EmotionType, intensity: f32, timestamp: i64) void {
        self.previous = self.current;
        self.current = emotion;
        self.intensity = intensity;
        self.last_detected = timestamp;
    }
};

// ============================================================================
// Identity and Session Primitives
// ============================================================================

/// Unique identifier for framework instances or entities.
pub const InstanceId = struct {
    bytes: [16]u8,

    pub fn generate() InstanceId {
        var bytes: [16]u8 = undefined;
        // Use project CSPRNG (Zig 0.16 removed std.crypto.random)
        const csprng = @import("security/csprng.zig");
        var rng = csprng.init();
        rng.fill(&bytes);
        return .{ .bytes = bytes };
    }

    pub fn toHex(self: InstanceId) [32]u8 {
        const hex_chars = "0123456789abcdef";
        var result: [32]u8 = undefined;
        for (self.bytes, 0..) |byte, i| {
            result[i * 2] = hex_chars[byte >> 4];
            result[i * 2 + 1] = hex_chars[byte & 0x0f];
        }
        return result;
    }
};

/// Session identifier for conversation continuity.
pub const SessionId = struct {
    id: InstanceId,
    created_at: i64,
    user_id: ?[]const u8 = null,

    pub fn create(allocator: std.mem.Allocator, user_id: ?[]const u8, timestamp: i64) !SessionId {
        return .{
            .id = InstanceId.generate(),
            .created_at = timestamp,
            .user_id = if (user_id) |uid| try allocator.dupe(u8, uid) else null,
        };
    }

    pub fn deinit(self: *SessionId, allocator: std.mem.Allocator) void {
        if (self.user_id) |uid| allocator.free(uid);
    }
};

test "confidence level from score" {
    try std.testing.expectEqual(ConfidenceLevel.certain, ConfidenceLevel.fromScore(0.98));
    try std.testing.expectEqual(ConfidenceLevel.unknown, ConfidenceLevel.fromScore(0.1));
}
