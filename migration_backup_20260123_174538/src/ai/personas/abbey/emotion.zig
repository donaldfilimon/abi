//! Abbey Emotion Processor
//!
//! Processes emotional content in user messages and determines appropriate
//! emotional responses. Integrates with Abbey's empathetic approach.
//!
//! Features:
//! - Multi-emotion detection with intensity scoring
//! - Tone suggestion based on detected emotions
//! - Empathy calibration for response generation
//! - Emotional trajectory tracking

const std = @import("std");
const core_types = @import("../../core/types.zig");

/// Emotion types detected in user input.
pub const EmotionType = core_types.EmotionType;

/// Full emotional state with history.
pub const EmotionalState = core_types.EmotionalState;

/// Tone style for response generation.
pub const ToneStyle = enum {
    /// Warm, supportive, understanding
    empathetic,
    /// Clear, patient explanations
    educational,
    /// Calm, reassuring presence
    calming,
    /// Engaged, enthusiastic
    enthusiastic,
    /// Direct but kind
    balanced,
    /// Quick, to-the-point
    efficient,
    /// Celebratory, encouraging
    celebratory,
    /// Gentle redirection
    constructive,

    pub fn getDescription(self: ToneStyle) []const u8 {
        return switch (self) {
            .empathetic => "warm, supportive, and understanding",
            .educational => "clear, patient, and explanatory",
            .calming => "calm, reassuring, and structured",
            .enthusiastic => "engaged, energetic, and encouraging",
            .balanced => "direct but kind and approachable",
            .efficient => "quick, helpful, and focused",
            .celebratory => "celebratory, encouraging, and positive",
            .constructive => "gently redirecting and supportive",
        };
    }

    pub fn getTemperature(self: ToneStyle) f32 {
        return switch (self) {
            .empathetic => 0.75,
            .educational => 0.6,
            .calming => 0.5,
            .enthusiastic => 0.8,
            .balanced => 0.65,
            .efficient => 0.4,
            .celebratory => 0.85,
            .constructive => 0.55,
        };
    }
};

/// Result of emotion processing.
pub const EmotionalResponse = struct {
    /// Primary detected emotion.
    primary_emotion: EmotionType,
    /// Intensity of primary emotion (0.0 - 1.0).
    intensity: f32,
    /// Secondary emotions detected.
    secondary_emotions: [3]?EmotionType = [_]?EmotionType{null} ** 3,
    /// Suggested tone for response.
    suggested_tone: ToneStyle,
    /// Empathy level required (0.0 - 1.0).
    empathy_level: f32,
    /// Whether special care is needed.
    needs_special_care: bool,
    /// Recommended response prefix.
    suggested_prefix: ?[]const u8 = null,
};

/// Configuration for emotion processing.
pub const EmotionConfig = struct {
    /// Minimum intensity to report an emotion.
    min_intensity_threshold: f32 = 0.3,
    /// Weight for recent emotions vs history.
    recency_weight: f32 = 0.7,
    /// Whether to track emotional trajectory.
    track_trajectory: bool = true,
    /// Decay factor for older emotions.
    history_decay: f32 = 0.1,
};

/// Patterns for emotion detection with associated emotions and weights.
const EmotionPattern = struct {
    pattern: []const u8,
    emotion: EmotionType,
    weight: f32,
    requires_context: bool = false,
};

/// Emotion detection patterns.
const EMOTION_PATTERNS = [_]EmotionPattern{
    // Frustration patterns
    .{ .pattern = "frustrated", .emotion = .frustrated, .weight = 0.9 },
    .{ .pattern = "annoyed", .emotion = .frustrated, .weight = 0.8 },
    .{ .pattern = "irritated", .emotion = .frustrated, .weight = 0.8 },
    .{ .pattern = "this is so", .emotion = .frustrated, .weight = 0.5, .requires_context = true },
    .{ .pattern = "ugh", .emotion = .frustrated, .weight = 0.7 },
    .{ .pattern = "argh", .emotion = .frustrated, .weight = 0.75 },
    .{ .pattern = "keeps failing", .emotion = .frustrated, .weight = 0.85 },
    .{ .pattern = "doesn't work", .emotion = .frustrated, .weight = 0.7 },
    .{ .pattern = "won't work", .emotion = .frustrated, .weight = 0.7 },
    .{ .pattern = "sick of", .emotion = .frustrated, .weight = 0.85 },
    .{ .pattern = "tired of", .emotion = .frustrated, .weight = 0.75 },

    // Confusion patterns
    .{ .pattern = "confused", .emotion = .confused, .weight = 0.9 },
    .{ .pattern = "don't understand", .emotion = .confused, .weight = 0.85 },
    .{ .pattern = "doesn't make sense", .emotion = .confused, .weight = 0.8 },
    .{ .pattern = "what does", .emotion = .confused, .weight = 0.5, .requires_context = true },
    .{ .pattern = "how does", .emotion = .curious, .weight = 0.5 },
    .{ .pattern = "lost", .emotion = .confused, .weight = 0.6, .requires_context = true },
    .{ .pattern = "unclear", .emotion = .confused, .weight = 0.75 },

    // Stress/anxiety patterns
    .{ .pattern = "stressed", .emotion = .stressed, .weight = 0.9 },
    .{ .pattern = "overwhelmed", .emotion = .stressed, .weight = 0.95 },
    .{ .pattern = "anxious", .emotion = .anxious, .weight = 0.9 },
    .{ .pattern = "worried", .emotion = .anxious, .weight = 0.8 },
    .{ .pattern = "nervous", .emotion = .anxious, .weight = 0.75 },
    .{ .pattern = "deadline", .emotion = .stressed, .weight = 0.6 },
    .{ .pattern = "urgent", .emotion = .stressed, .weight = 0.65 },
    .{ .pattern = "asap", .emotion = .impatient, .weight = 0.7 },
    .{ .pattern = "panic", .emotion = .anxious, .weight = 0.95 },

    // Curiosity patterns
    .{ .pattern = "curious", .emotion = .curious, .weight = 0.9 },
    .{ .pattern = "wondering", .emotion = .curious, .weight = 0.8 },
    .{ .pattern = "interested in", .emotion = .curious, .weight = 0.75 },
    .{ .pattern = "want to learn", .emotion = .curious, .weight = 0.85 },
    .{ .pattern = "tell me more", .emotion = .curious, .weight = 0.8 },
    .{ .pattern = "how come", .emotion = .curious, .weight = 0.6 },

    // Excitement patterns
    .{ .pattern = "excited", .emotion = .excited, .weight = 0.9 },
    .{ .pattern = "can't wait", .emotion = .excited, .weight = 0.85 },
    .{ .pattern = "awesome", .emotion = .excited, .weight = 0.7 },
    .{ .pattern = "amazing", .emotion = .excited, .weight = 0.7 },
    .{ .pattern = "love this", .emotion = .enthusiastic, .weight = 0.8 },
    .{ .pattern = "fantastic", .emotion = .excited, .weight = 0.75 },

    // Gratitude patterns
    .{ .pattern = "thank", .emotion = .grateful, .weight = 0.8 },
    .{ .pattern = "appreciate", .emotion = .grateful, .weight = 0.85 },
    .{ .pattern = "grateful", .emotion = .grateful, .weight = 0.9 },
    .{ .pattern = "helpful", .emotion = .grateful, .weight = 0.6 },

    // Skepticism patterns
    .{ .pattern = "skeptical", .emotion = .skeptical, .weight = 0.9 },
    .{ .pattern = "doubt", .emotion = .skeptical, .weight = 0.75 },
    .{ .pattern = "not sure if", .emotion = .skeptical, .weight = 0.6 },
    .{ .pattern = "really?", .emotion = .skeptical, .weight = 0.5 },
    .{ .pattern = "are you sure", .emotion = .skeptical, .weight = 0.7 },

    // Impatience patterns
    .{ .pattern = "hurry", .emotion = .impatient, .weight = 0.8 },
    .{ .pattern = "quickly", .emotion = .impatient, .weight = 0.5 },
    .{ .pattern = "just tell me", .emotion = .impatient, .weight = 0.75 },
    .{ .pattern = "get to the point", .emotion = .impatient, .weight = 0.85 },
    .{ .pattern = "tldr", .emotion = .impatient, .weight = 0.7 },

    // Disappointment patterns
    .{ .pattern = "disappointed", .emotion = .disappointed, .weight = 0.9 },
    .{ .pattern = "expected more", .emotion = .disappointed, .weight = 0.8 },
    .{ .pattern = "let down", .emotion = .disappointed, .weight = 0.85 },
    .{ .pattern = "not what i", .emotion = .disappointed, .weight = 0.6 },

    // Hope patterns
    .{ .pattern = "hope", .emotion = .hopeful, .weight = 0.7 },
    .{ .pattern = "hopefully", .emotion = .hopeful, .weight = 0.75 },
    .{ .pattern = "fingers crossed", .emotion = .hopeful, .weight = 0.8 },
    .{ .pattern = "looking forward", .emotion = .hopeful, .weight = 0.75 },

    // Playful patterns
    .{ .pattern = "haha", .emotion = .playful, .weight = 0.7 },
    .{ .pattern = "lol", .emotion = .playful, .weight = 0.65 },
    .{ .pattern = "joking", .emotion = .playful, .weight = 0.8 },
    .{ .pattern = "kidding", .emotion = .playful, .weight = 0.75 },
    .{ .pattern = "fun", .emotion = .playful, .weight = 0.5 },
};

/// Processes emotions in user content.
pub const EmotionProcessor = struct {
    allocator: std.mem.Allocator,
    config: EmotionConfig,
    /// Tracks emotional trajectory over conversation.
    trajectory: std.ArrayList(EmotionalState),

    const Self = @This();

    /// Initialize the emotion processor.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: EmotionConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .trajectory = std.ArrayList(EmotionalState).init(allocator),
        };
    }

    /// Shutdown and free resources.
    pub fn deinit(self: *Self) void {
        self.trajectory.deinit();
    }

    /// Process text and detect emotional content.
    pub fn process(self: *Self, text: []const u8, context: EmotionalState) !EmotionalResponse {
        const lower = try self.toLower(text);
        defer self.allocator.free(lower);

        // Detect emotions from patterns
        var emotion_scores = [_]f32{0.0} ** 14;
        for (EMOTION_PATTERNS) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern.pattern) != null) {
                // Apply context requirement check
                if (pattern.requires_context) {
                    // Only count if there's supporting context
                    if (!self.hasContextSupport(lower, pattern.emotion)) continue;
                }
                const idx = @intFromEnum(pattern.emotion);
                emotion_scores[idx] = @max(emotion_scores[idx], pattern.weight);
            }
        }

        // Consider conversation context
        if (self.config.recency_weight > 0 and context.intensity > 0) {
            const ctx_idx = @intFromEnum(context.current);
            emotion_scores[ctx_idx] += context.intensity * (1.0 - self.config.recency_weight) * 0.3;
        }

        // Find primary and secondary emotions
        var primary_emotion: EmotionType = .neutral;
        var primary_intensity: f32 = 0.0;
        var secondary: [3]?EmotionType = [_]?EmotionType{null} ** 3;

        // Find top emotions
        var top_indices: [4]usize = [_]usize{0} ** 4;
        var top_scores: [4]f32 = [_]f32{0.0} ** 4;

        for (emotion_scores, 0..) |score, idx| {
            if (score > top_scores[3]) {
                // Insert in sorted position
                var insert_pos: usize = 3;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) : (insert_pos -= 1) {}

                // Shift down
                var j: usize = 3;
                while (j > insert_pos) : (j -= 1) {
                    top_scores[j] = top_scores[j - 1];
                    top_indices[j] = top_indices[j - 1];
                }
                top_scores[insert_pos] = score;
                top_indices[insert_pos] = idx;
            }
        }

        // Set primary emotion
        if (top_scores[0] >= self.config.min_intensity_threshold) {
            primary_emotion = @enumFromInt(top_indices[0]);
            primary_intensity = @min(1.0, top_scores[0]);
        }

        // Set secondary emotions
        for (1..4) |i| {
            if (top_scores[i] >= self.config.min_intensity_threshold) {
                secondary[i - 1] = @enumFromInt(top_indices[i]);
            }
        }

        // Determine tone and empathy level
        const tone = self.suggestTone(primary_emotion);
        const empathy_level = self.calibrateEmpathy(primary_emotion, primary_intensity);
        const needs_care = self.needsSpecialCare(primary_emotion, primary_intensity);

        // Track trajectory if enabled
        if (self.config.track_trajectory) {
            var new_state = context;
            new_state.update(primary_emotion, primary_intensity);
            try self.trajectory.append(new_state);

            // Limit trajectory size
            if (self.trajectory.items.len > 50) {
                _ = self.trajectory.orderedRemove(0);
            }
        }

        return .{
            .primary_emotion = primary_emotion,
            .intensity = primary_intensity,
            .secondary_emotions = secondary,
            .suggested_tone = tone,
            .empathy_level = empathy_level,
            .needs_special_care = needs_care,
            .suggested_prefix = self.getSuggestedPrefix(primary_emotion, primary_intensity),
        };
    }

    /// Suggest appropriate tone based on detected emotion.
    pub fn suggestTone(self: *const Self, emotion: EmotionType) ToneStyle {
        _ = self;
        return switch (emotion) {
            .neutral => .balanced,
            .frustrated => .empathetic,
            .excited => .enthusiastic,
            .confused => .educational,
            .stressed => .calming,
            .playful => .enthusiastic,
            .grateful => .celebratory,
            .curious => .educational,
            .impatient => .efficient,
            .skeptical => .balanced,
            .enthusiastic => .enthusiastic,
            .disappointed => .constructive,
            .hopeful => .balanced,
            .anxious => .calming,
        };
    }

    /// Calibrate empathy level based on emotion and urgency.
    pub fn calibrateEmpathy(self: *const Self, emotion: EmotionType, intensity: f32) f32 {
        _ = self;
        const base_empathy: f32 = switch (emotion) {
            .neutral => 0.3,
            .frustrated => 0.9,
            .excited => 0.5,
            .confused => 0.7,
            .stressed => 0.95,
            .playful => 0.4,
            .grateful => 0.6,
            .curious => 0.5,
            .impatient => 0.4,
            .skeptical => 0.5,
            .enthusiastic => 0.5,
            .disappointed => 0.85,
            .hopeful => 0.55,
            .anxious => 0.95,
        };

        // Scale by intensity
        return @min(1.0, base_empathy * (0.7 + intensity * 0.3));
    }

    /// Check if the emotion requires special care in response.
    fn needsSpecialCare(self: *const Self, emotion: EmotionType, intensity: f32) bool {
        _ = self;
        // High-care emotions
        const high_care_emotions = [_]EmotionType{ .frustrated, .stressed, .anxious, .disappointed };
        for (high_care_emotions) |hc| {
            if (emotion == hc and intensity >= 0.6) return true;
        }
        return false;
    }

    /// Get suggested prefix for response based on emotion.
    fn getSuggestedPrefix(self: *const Self, emotion: EmotionType, intensity: f32) ?[]const u8 {
        _ = self;
        if (intensity < 0.5) return null;

        return switch (emotion) {
            .frustrated => "I understand this is frustrating. ",
            .confused => "Let me help clarify. ",
            .stressed => "I hear you - let's work through this together. ",
            .anxious => "Take a breath - we'll figure this out. ",
            .disappointed => "I'm sorry this didn't meet expectations. ",
            .excited => "That's wonderful! ",
            .grateful => "You're very welcome! ",
            else => null,
        };
    }

    /// Check if context supports the emotion detection.
    fn hasContextSupport(self: *const Self, text: []const u8, emotion: EmotionType) bool {
        _ = self;
        // Look for supporting indicators based on emotion type
        return switch (emotion) {
            .frustrated => std.mem.indexOf(u8, text, "!") != null or
                std.mem.indexOf(u8, text, "not") != null,
            .confused => std.mem.indexOf(u8, text, "?") != null,
            else => true,
        };
    }

    /// Get emotional trajectory analysis.
    pub fn getTrajectoryTrend(self: *const Self) EmotionTrajectory {
        if (self.trajectory.items.len < 2) {
            return .{ .direction = .stable, .magnitude = 0.0 };
        }

        // Compare first and last halves
        const mid = self.trajectory.items.len / 2;
        var first_intensity: f32 = 0.0;
        var second_intensity: f32 = 0.0;

        for (self.trajectory.items[0..mid]) |state| {
            first_intensity += state.intensity;
        }
        for (self.trajectory.items[mid..]) |state| {
            second_intensity += state.intensity;
        }

        first_intensity /= @floatFromInt(mid);
        second_intensity /= @floatFromInt(self.trajectory.items.len - mid);

        const diff = second_intensity - first_intensity;
        const direction: TrajectoryDirection = if (diff > 0.1) .escalating else if (diff < -0.1) .deescalating else .stable;

        return .{
            .direction = direction,
            .magnitude = @abs(diff),
        };
    }

    /// Convert text to lowercase for pattern matching.
    fn toLower(self: *const Self, text: []const u8) ![]u8 {
        const result = try self.allocator.alloc(u8, text.len);
        for (text, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }
};

/// Direction of emotional trajectory.
pub const TrajectoryDirection = enum {
    escalating,
    stable,
    deescalating,
};

/// Emotional trajectory analysis result.
pub const EmotionTrajectory = struct {
    direction: TrajectoryDirection,
    magnitude: f32,
};

// Tests

test "emotion processor initialization" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    try std.testing.expectEqual(@as(usize, 0), processor.trajectory.items.len);
}

test "detect frustration" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    const response = try processor.process("I'm so frustrated with this code!", .{});
    try std.testing.expectEqual(EmotionType.frustrated, response.primary_emotion);
    try std.testing.expect(response.empathy_level > 0.7);
}

test "detect confusion" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    const response = try processor.process("I don't understand how this works", .{});
    try std.testing.expectEqual(EmotionType.confused, response.primary_emotion);
    try std.testing.expectEqual(ToneStyle.educational, response.suggested_tone);
}

test "neutral detection" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    const response = try processor.process("Please explain the API", .{});
    try std.testing.expectEqual(EmotionType.neutral, response.primary_emotion);
}

test "tone suggestion" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    try std.testing.expectEqual(ToneStyle.empathetic, processor.suggestTone(.frustrated));
    try std.testing.expectEqual(ToneStyle.educational, processor.suggestTone(.confused));
    try std.testing.expectEqual(ToneStyle.calming, processor.suggestTone(.stressed));
}

test "empathy calibration" {
    var processor = EmotionProcessor.init(std.testing.allocator);
    defer processor.deinit();

    const high_empathy = processor.calibrateEmpathy(.stressed, 0.9);
    const low_empathy = processor.calibrateEmpathy(.neutral, 0.2);

    try std.testing.expect(high_empathy > low_empathy);
    try std.testing.expect(high_empathy >= 0.8);
}
