//! Abbey Core Types
//!
//! Foundational type definitions for the Abbey AI system.
//! These types form the backbone of Abbey's unique architecture.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const builtin = @import("builtin");

// ============================================================================
// Time Utilities (Zig 0.16 compatible, platform-aware)
// ============================================================================

/// Check if we're on a platform that supports time.Instant
const has_instant = !isWasmTarget();

fn isWasmTarget() bool {
    return builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
}

/// Platform-aware Instant wrapper
const PlatformInstant = if (has_instant) time.Instant else struct {
    counter: u64,

    pub fn now() error{Unsupported}!@This() {
        return .{ .counter = 0 };
    }

    pub fn since(self: @This(), earlier: @This()) u64 {
        _ = self;
        _ = earlier;
        return 0;
    }
};

/// Internal: Get current instant for time calculations
fn getCurrentInstant() ?PlatformInstant {
    return PlatformInstant.now() catch null;
}

/// Application start time for relative timing (initialized lazily)
var app_start_instant: ?PlatformInstant = null;
var app_start_initialized: bool = false;

fn ensureStartInstant() ?PlatformInstant {
    if (!app_start_initialized) {
        app_start_instant = getCurrentInstant();
        app_start_initialized = true;
    }
    return app_start_instant;
}

pub fn getTimestampNs() i128 {
    if (!has_instant) return 0;
    const start = ensureStartInstant() orelse return 0;
    const now = getCurrentInstant() orelse return 0;
    return @intCast(now.since(start));
}

pub fn getTimestampMs() i64 {
    if (!has_instant) return 0;
    const start = ensureStartInstant() orelse return 0;
    const now = getCurrentInstant() orelse return 0;
    const elapsed_ns = now.since(start);
    return @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));
}

pub fn getTimestampSec() i64 {
    if (!has_instant) return 0;
    const start = ensureStartInstant() orelse return 0;
    const now = getCurrentInstant() orelse return 0;
    const elapsed_ns = now.since(start);
    return @intCast(@divTrunc(elapsed_ns, std.time.ns_per_s));
}

// ============================================================================
// Core Identity Types
// ============================================================================

/// Unique identifier for Abbey instances
pub const InstanceId = struct {
    bytes: [16]u8,

    pub fn generate() InstanceId {
        var bytes: [16]u8 = undefined;
        // Use timer-based entropy for uniqueness
        const ns = getTimestampNs();
        const ns_bytes: [16]u8 = @bitCast(ns);
        @memcpy(&bytes, &ns_bytes);
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

/// Session identifier for conversation continuity
pub const SessionId = struct {
    id: InstanceId,
    created_at: i64,
    user_id: ?[]const u8 = null,

    pub fn create(allocator: std.mem.Allocator, user_id: ?[]const u8) !SessionId {
        return .{
            .id = InstanceId.generate(),
            .created_at = getTimestampSec(),
            .user_id = if (user_id) |uid| try allocator.dupe(u8, uid) else null,
        };
    }
};

// ============================================================================
// Confidence and Certainty Types
// ============================================================================

/// Granular confidence level with mathematical grounding
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

    pub fn needsResearch(self: ConfidenceLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(ConfidenceLevel.low);
    }

    pub fn needsVerification(self: ConfidenceLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(ConfidenceLevel.medium);
    }
};

/// Detailed confidence assessment with provenance
pub const Confidence = struct {
    level: ConfidenceLevel = .medium,
    score: f32 = 0.5,
    reasoning: []const u8 = "No specific reasoning provided",
    sources: []const ConfidenceSource = &.{},
    calibrated_at: i64 = 0,

    pub const ConfidenceSource = enum {
        training_knowledge,
        logical_inference,
        pattern_recognition,
        user_provided,
        research_verified,
        cross_referenced,
    };

    pub fn combine(confidences: []const Confidence) Confidence {
        if (confidences.len == 0) return .{};

        var total_score: f32 = 0;
        var min_level: ConfidenceLevel = .certain;

        for (confidences) |conf| {
            total_score += conf.score;
            if (@intFromEnum(conf.level) > @intFromEnum(min_level)) {
                min_level = conf.level;
            }
        }

        const avg_score = total_score / @as(f32, @floatFromInt(confidences.len));
        return .{
            .level = ConfidenceLevel.fromScore(avg_score),
            .score = avg_score,
            .reasoning = "Combined from multiple sources",
            .calibrated_at = getTimestampSec(),
        };
    }

    pub fn decay(self: *Confidence, factor: f32) void {
        self.score = @max(0.0, self.score * (1.0 - factor));
        self.level = ConfidenceLevel.fromScore(self.score);
    }
};

// ============================================================================
// Emotional Intelligence Types
// ============================================================================

/// Detected emotional state
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

    pub fn getResponseTone(self: EmotionType) []const u8 {
        return switch (self) {
            .neutral => "balanced and informative",
            .frustrated => "direct, solution-focused, and empathetic",
            .excited => "enthusiastic and engaging",
            .confused => "clear, patient, and step-by-step",
            .stressed => "calm, efficient, and reassuring",
            .playful => "light, witty, and fun",
            .grateful => "warm and appreciative",
            .curious => "thorough and exploratory",
            .impatient => "concise and actionable",
            .skeptical => "evidence-based and logical",
            .enthusiastic => "energetic and supportive",
            .disappointed => "constructive and encouraging",
            .hopeful => "optimistic and supportive",
            .anxious => "calming and structured",
        };
    }

    pub fn getTemperatureModifier(self: EmotionType) f32 {
        return switch (self) {
            .neutral => 0.0,
            .frustrated => -0.15,
            .excited => 0.1,
            .confused => -0.1,
            .stressed => -0.2,
            .playful => 0.15,
            .grateful => 0.05,
            .curious => 0.1,
            .impatient => -0.25,
            .skeptical => -0.1,
            .enthusiastic => 0.1,
            .disappointed => -0.05,
            .hopeful => 0.05,
            .anxious => -0.15,
        };
    }
};

/// Full emotional state with history
pub const EmotionalState = struct {
    current: EmotionType = .neutral,
    intensity: f32 = 0.0,
    previous: EmotionType = .neutral,
    history: [8]EmotionType = [_]EmotionType{.neutral} ** 8,
    history_index: usize = 0,
    consecutive_same: usize = 0,
    last_detected: i64 = 0,

    pub fn update(self: *EmotionalState, emotion: EmotionType, intensity: f32) void {
        self.previous = self.current;
        self.current = emotion;
        self.intensity = intensity;
        self.last_detected = getTimestampSec();

        // Track history
        self.history[self.history_index] = emotion;
        self.history_index = (self.history_index + 1) % 8;

        // Track consecutive
        if (emotion == self.previous and emotion != .neutral) {
            self.consecutive_same += 1;
        } else {
            self.consecutive_same = 0;
        }
    }

    pub fn isPersistent(self: *const EmotionalState) bool {
        return self.consecutive_same >= 2;
    }

    pub fn getDominant(self: *const EmotionalState) EmotionType {
        var counts = [_]usize{0} ** 14;
        for (self.history) |emotion| {
            counts[@intFromEnum(emotion)] += 1;
        }
        var max_count: usize = 0;
        var dominant: EmotionType = .neutral;
        for (counts, 0..) |count, i| {
            if (count > max_count) {
                max_count = count;
                dominant = @enumFromInt(i);
            }
        }
        return dominant;
    }
};

// ============================================================================
// Message and Conversation Types
// ============================================================================

/// Role in a conversation
pub const Role = enum {
    system,
    user,
    assistant,
    tool,
    internal, // Abbey's internal reasoning

    pub fn toString(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
            .internal => "internal",
        };
    }
};

/// A single message in conversation
pub const Message = struct {
    role: Role,
    content: []const u8,
    name: ?[]const u8 = null,
    timestamp: i64,
    token_count: usize = 0,
    metadata: ?MessageMetadata = null,
    embedding: ?[]f32 = null,
    importance: f32 = 0.5,

    pub const MessageMetadata = struct {
        emotion_detected: ?EmotionType = null,
        confidence: ?Confidence = null,
        topics: ?[]const []const u8 = null,
        tool_calls: ?[]const []const u8 = null,
    };

    pub fn user(content: []const u8) Message {
        return .{
            .role = .user,
            .content = content,
            .timestamp = getTimestampSec(),
        };
    }

    pub fn assistant(content: []const u8) Message {
        return .{
            .role = .assistant,
            .content = content,
            .timestamp = getTimestampSec(),
        };
    }

    pub fn system(content: []const u8) Message {
        return .{
            .role = .system,
            .content = content,
            .timestamp = getTimestampSec(),
            .importance = 1.0,
        };
    }

    pub fn estimateTokens(self: *const Message) usize {
        if (self.token_count > 0) return self.token_count;
        // Rough estimate: 4 characters per token
        return (self.content.len + 3) / 4;
    }
};

// ============================================================================
// Relationship and Trust Types
// ============================================================================

/// Trust level with a user
pub const TrustLevel = enum(u8) {
    new = 0,
    familiar = 1,
    trusted = 2,
    established = 3,

    pub fn fromInteractions(count: usize, positive_ratio: f32) TrustLevel {
        if (count < 5) return .new;
        if (count < 20 or positive_ratio < 0.6) return .familiar;
        if (count < 50 or positive_ratio < 0.8) return .trusted;
        return .established;
    }
};

/// Relationship state with a user
pub const Relationship = struct {
    trust_level: TrustLevel = .new,
    interaction_count: usize = 0,
    positive_interactions: usize = 0,
    negative_interactions: usize = 0,
    last_interaction: i64 = 0,
    preferences: CommunicationPreferences = .{},
    rapport_score: f32 = 0.5,

    pub const CommunicationPreferences = struct {
        prefers_brevity: bool = false,
        prefers_detail: bool = false,
        prefers_examples: bool = true,
        prefers_formal: bool = false,
        prefers_casual: bool = false,
        technical_level: TechnicalLevel = .intermediate,
    };

    pub const TechnicalLevel = enum {
        beginner,
        intermediate,
        advanced,
        expert,
    };

    pub fn recordInteraction(self: *Relationship, positive: bool) void {
        self.interaction_count += 1;
        if (positive) {
            self.positive_interactions += 1;
            self.rapport_score = @min(1.0, self.rapport_score + 0.01);
        } else {
            self.negative_interactions += 1;
            self.rapport_score = @max(0.0, self.rapport_score - 0.02);
        }
        self.last_interaction = getTimestampSec();
        self.updateTrustLevel();
    }

    fn updateTrustLevel(self: *Relationship) void {
        const ratio = if (self.interaction_count > 0)
            @as(f32, @floatFromInt(self.positive_interactions)) /
                @as(f32, @floatFromInt(self.interaction_count))
        else
            0.5;
        self.trust_level = TrustLevel.fromInteractions(self.interaction_count, ratio);
    }
};

// ============================================================================
// Topic and Context Types
// ============================================================================

/// A topic being discussed
pub const Topic = struct {
    name: []const u8,
    category: TopicCategory,
    first_mentioned: i64,
    last_mentioned: i64,
    mention_count: usize,
    relevance: f32,
    subtopics: std.ArrayListUnmanaged([]const u8),

    pub const TopicCategory = enum {
        programming,
        ai_ml,
        systems,
        web,
        database,
        devops,
        general,
        personal,
        creative,
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, category: TopicCategory) !Topic {
        _ = allocator;
        const now = getTimestampSec();
        return .{
            .name = name,
            .category = category,
            .first_mentioned = now,
            .last_mentioned = now,
            .mention_count = 1,
            .relevance = 0.5,
            .subtopics = .{},
        };
    }

    pub fn mention(self: *Topic) void {
        self.last_mentioned = getTimestampSec();
        self.mention_count += 1;
        self.relevance = @min(1.0, self.relevance + 0.1);
    }

    pub fn decay(self: *Topic, factor: f32) void {
        self.relevance = @max(0.0, self.relevance - factor);
    }
};

// ============================================================================
// Response Types
// ============================================================================

/// Abbey's response to a query
pub const Response = struct {
    content: []const u8,
    confidence: Confidence,
    emotional_context: EmotionalState,
    reasoning_summary: ?[]const u8 = null,
    topics: []const []const u8 = &.{},
    tool_calls: []const ToolCall = &.{},
    research_performed: bool = false,
    generation_time_ms: i64 = 0,

    pub const ToolCall = struct {
        name: []const u8,
        arguments: []const u8,
        result: ?[]const u8 = null,
    };
};

// ============================================================================
// Error Types
// ============================================================================

pub const AbbeyError = error{
    OutOfMemory,
    InvalidConfiguration,
    SessionNotFound,
    MemoryFull,
    ConfidenceTooLow,
    ResearchFailed,
    ToolExecutionFailed,
    LLMConnectionFailed,
    RateLimited,
    InvalidInput,
    ContextOverflow,
    EmbeddingFailed,
};

// ============================================================================
// Tests
// ============================================================================

test "confidence level from score" {
    try std.testing.expectEqual(ConfidenceLevel.certain, ConfidenceLevel.fromScore(0.98));
    try std.testing.expectEqual(ConfidenceLevel.high, ConfidenceLevel.fromScore(0.85));
    try std.testing.expectEqual(ConfidenceLevel.medium, ConfidenceLevel.fromScore(0.70));
    try std.testing.expectEqual(ConfidenceLevel.low, ConfidenceLevel.fromScore(0.45));
    try std.testing.expectEqual(ConfidenceLevel.unknown, ConfidenceLevel.fromScore(0.1));
}

test "emotional state update" {
    var state = EmotionalState{};
    state.update(.excited, 0.8);
    try std.testing.expectEqual(EmotionType.excited, state.current);
    try std.testing.expectEqual(EmotionType.neutral, state.previous);

    state.update(.excited, 0.9);
    try std.testing.expectEqual(@as(usize, 1), state.consecutive_same);
}

test "relationship tracking" {
    var rel = Relationship{};
    rel.recordInteraction(true);
    rel.recordInteraction(true);
    rel.recordInteraction(true);

    try std.testing.expectEqual(@as(usize, 3), rel.interaction_count);
    try std.testing.expect(rel.rapport_score > 0.5);
}

test "instance id generation" {
    const id1 = InstanceId.generate();
    const id2 = InstanceId.generate();
    // IDs should be different (with very high probability)
    try std.testing.expect(!std.mem.eql(u8, &id1.bytes, &id2.bytes));
}
