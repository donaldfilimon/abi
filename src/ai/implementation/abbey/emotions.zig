//! Abbey Emotional Intelligence Module
//!
//! Provides emotional context awareness:
//! - Emotion detection from text
//! - Emotional state tracking
//! - Response tone adjustment
//! - Relationship memory

const std = @import("std");

// Zig 0.16 compatible time function
fn getTimestamp() i64 {
    var timer = std.time.Timer.start() catch return 0;
    const ns: i64 = @intCast(timer.read());
    return @divTrunc(ns, std.time.ns_per_s);
}

/// Types of emotions Abbey can detect and respond to
pub const EmotionType = enum {
    /// No strong emotion detected
    neutral,
    /// User seems frustrated or annoyed
    frustrated,
    /// User is excited or enthusiastic
    excited,
    /// User seems confused or uncertain
    confused,
    /// User appears stressed or anxious
    stressed,
    /// User is being humorous or playful
    playful,
    /// User seems grateful or appreciative
    grateful,
    /// User is curious and exploring
    curious,
    /// User seems impatient or rushed
    impatient,

    pub fn toString(self: EmotionType) []const u8 {
        return switch (self) {
            .neutral => "neutral",
            .frustrated => "frustrated",
            .excited => "excited",
            .confused => "confused",
            .stressed => "stressed",
            .playful => "playful",
            .grateful => "grateful",
            .curious => "curious",
            .impatient => "impatient",
        };
    }

    /// Get suggested response tone
    pub fn getSuggestedTone(self: EmotionType) []const u8 {
        return switch (self) {
            .neutral => "balanced",
            .frustrated => "direct and helpful",
            .excited => "enthusiastic",
            .confused => "clear and patient",
            .stressed => "calm and efficient",
            .playful => "light and witty",
            .grateful => "warm",
            .curious => "engaging and thorough",
            .impatient => "concise",
        };
    }

    /// Get intensity modifier (affects temperature)
    pub fn getIntensityModifier(self: EmotionType) f32 {
        return switch (self) {
            .neutral => 0.0,
            .frustrated => -0.1, // Lower temp for more focused responses
            .excited => 0.1, // Higher temp for more creative responses
            .confused => -0.1, // Lower temp for clearer responses
            .stressed => -0.15, // Even lower for maximum clarity
            .playful => 0.15, // Higher for fun responses
            .grateful => 0.05,
            .curious => 0.05,
            .impatient => -0.2, // Much lower for brevity
        };
    }
};

/// Emotional state with intensity and history
pub const EmotionalState = struct {
    detected: EmotionType = .neutral,
    intensity: f32 = 0.0, // 0.0 to 1.0
    previous: EmotionType = .neutral,
    turn_detected: usize = 0,
    consecutive_same: usize = 0,

    const Self = @This();

    /// Initialize neutral state
    pub fn init() Self {
        return .{};
    }

    /// Detect emotion from text using pattern matching
    pub fn detectFromText(self: *Self, text: []const u8) void {
        const previous_emotion = self.detected;

        // Convert to lowercase for matching
        var lower_buf: [2048]u8 = undefined;
        const len = @min(text.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(text[i]);
        }
        const lower = lower_buf[0..len];

        // Frustration patterns
        const frustration_patterns = [_][]const u8{
            "frustrated", "annoying", "annoyed",     "why won't", "doesn't work",
            "broken",     "stupid",   "hate",        "ugh",       "argh",
            "damn",       "wtf",      "not working", "still not", "again?!",
            "!!",
        };

        // Excitement patterns
        const excitement_patterns = [_][]const u8{
            "awesome",   "amazing", "love it",   "perfect", "brilliant",
            "fantastic", "great",   "wonderful", "excited", "can't wait",
            "yay",       "woohoo",  "!!!",       ":)",      ":D",
        };

        // Confusion patterns
        const confusion_patterns = [_][]const u8{
            "confused",           "don't understand",  "what do you mean", "unclear",
            "lost",               "huh?",              "???",              "not sure",
            "help me understand", "i'm not following", "what?",            "doesn't make sense",
        };

        // Stress patterns
        const stress_patterns = [_][]const u8{
            "urgent",        "asap",                "deadline", "stressed",  "overwhelmed",
            "need this now", "running out of time", "critical", "emergency", "help!",
            "panic",         "anxious",
        };

        // Playful patterns
        const playful_patterns = [_][]const u8{
            "lol",   "haha", "hehe", "just kidding", "joking", "funny",
            "silly", ":P",   ";)",   "xD",           "lmao",   "rofl",
        };

        // Grateful patterns
        const grateful_patterns = [_][]const u8{
            "thank you",       "thanks",    "appreciate",     "grateful", "helpful",
            "you're the best", "lifesaver", "perfect, thank",
        };

        // Curious patterns
        const curious_patterns = [_][]const u8{
            "curious",      "wondering", "interested", "how does", "why does",
            "tell me more", "explain",   "elaborate",  "what if",
        };

        // Impatient patterns
        const impatient_patterns = [_][]const u8{
            "hurry",            "quick", "fast",  "just tell me", "skip",
            "get to the point", "tldr",  "tl;dr", "briefly",
        };

        // Check patterns and score
        var emotion_scores = [_]f32{0} ** 9;

        for (frustration_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.frustrated)] += 1;
        }
        for (excitement_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.excited)] += 1;
        }
        for (confusion_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.confused)] += 1;
        }
        for (stress_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.stressed)] += 1;
        }
        for (playful_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.playful)] += 1;
        }
        for (grateful_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.grateful)] += 1;
        }
        for (curious_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.curious)] += 1;
        }
        for (impatient_patterns) |p| {
            if (std.mem.indexOf(u8, lower, p) != null) emotion_scores[@intFromEnum(EmotionType.impatient)] += 1;
        }

        // Find highest scoring emotion
        var max_score: f32 = 0;
        var max_emotion: EmotionType = .neutral;

        for (emotion_scores, 0..) |score, i| {
            if (score > max_score) {
                max_score = score;
                max_emotion = @enumFromInt(i);
            }
        }

        // Only detect emotion if score is significant
        if (max_score >= 1) {
            self.detected = max_emotion;
            self.intensity = @min(1.0, max_score / 3.0);
        } else {
            self.detected = .neutral;
            self.intensity = 0.0;
        }

        // Track emotion continuity
        self.previous = previous_emotion;
        if (self.detected == previous_emotion and self.detected != .neutral) {
            self.consecutive_same += 1;
        } else {
            self.consecutive_same = 0;
        }
        self.turn_detected += 1;
    }

    /// Get suggested response tone
    pub fn getSuggestedTone(self: *const Self) []const u8 {
        return self.detected.getSuggestedTone();
    }

    /// Get temperature adjustment
    pub fn getTemperatureModifier(self: *const Self) f32 {
        return self.detected.getIntensityModifier() * self.intensity;
    }

    /// Check if emotion persists
    pub fn isEmotionPersistent(self: *const Self) bool {
        return self.consecutive_same >= 2;
    }

    /// Get human-readable state
    pub fn describe(self: *const Self) []const u8 {
        if (self.detected == .neutral) {
            return "neutral emotional context";
        }

        if (self.intensity > 0.7) {
            return switch (self.detected) {
                .frustrated => "highly frustrated - needs immediate, clear help",
                .excited => "very excited - match enthusiasm",
                .confused => "deeply confused - use simpler explanations",
                .stressed => "under significant stress - be efficient",
                .playful => "in a playful mood - can be lighthearted",
                .grateful => "very appreciative",
                .curious => "highly curious - provide depth",
                .impatient => "very impatient - be extremely brief",
                .neutral => "neutral",
            };
        }

        return self.detected.getSuggestedTone();
    }
};

/// Relationship memory for long-term emotional context
pub const RelationshipMemory = struct {
    allocator: std.mem.Allocator,
    positive_interactions: usize = 0,
    negative_interactions: usize = 0,
    total_turns: usize = 0,
    emotion_history: std.ArrayListUnmanaged(EmotionRecord) = .{},
    communication_preferences: CommunicationPreferences = .{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.emotion_history.deinit(self.allocator);
    }

    /// Record an emotional moment
    pub fn recordEmotion(self: *Self, emotion: EmotionType, context: []const u8) !void {
        try self.emotion_history.append(self.allocator, .{
            .emotion = emotion,
            .context = context,
            .timestamp = getTimestamp(),
        });

        // Track positive/negative
        switch (emotion) {
            .grateful, .excited, .playful => self.positive_interactions += 1,
            .frustrated, .stressed, .impatient => self.negative_interactions += 1,
            else => {},
        }

        self.total_turns += 1;
    }

    /// Get relationship health score (0-1)
    pub fn getHealthScore(self: *const Self) f32 {
        if (self.total_turns == 0) return 0.5;

        const positive: f32 = @floatFromInt(self.positive_interactions);
        const negative: f32 = @floatFromInt(self.negative_interactions);
        const total: f32 = @floatFromInt(self.total_turns);

        // Weighted score favoring positive interactions
        return (positive * 1.5 - negative) / total + 0.5;
    }

    /// Check if user tends toward certain emotions
    pub fn getDominantPattern(self: *const Self) ?EmotionType {
        if (self.emotion_history.items.len < 5) return null;

        var counts = [_]usize{0} ** 9;
        for (self.emotion_history.items) |record| {
            counts[@intFromEnum(record.emotion)] += 1;
        }

        var max_count: usize = 0;
        var dominant: ?EmotionType = null;
        for (counts, 0..) |count, i| {
            if (count > max_count and count > 2) {
                max_count = count;
                dominant = @enumFromInt(i);
            }
        }

        return dominant;
    }
};

/// Record of an emotional moment
pub const EmotionRecord = struct {
    emotion: EmotionType,
    context: []const u8,
    timestamp: i64,
};

/// Learned communication preferences
pub const CommunicationPreferences = struct {
    prefers_brevity: bool = false,
    prefers_detail: bool = false,
    prefers_examples: bool = true,
    prefers_formal: bool = false,
    prefers_casual: bool = false,
};

// ============================================================================
// Tests
// ============================================================================

test "emotion detection" {
    var state = EmotionalState.init();

    state.detectFromText("This is so frustrating, why doesn't it work?!");
    try std.testing.expectEqual(EmotionType.frustrated, state.detected);
    try std.testing.expect(state.intensity > 0);

    state.detectFromText("This is amazing! I love it!");
    try std.testing.expectEqual(EmotionType.excited, state.detected);

    state.detectFromText("I don't understand what you mean, I'm confused");
    try std.testing.expectEqual(EmotionType.confused, state.detected);

    state.detectFromText("Just a normal question");
    try std.testing.expectEqual(EmotionType.neutral, state.detected);
}

test "emotion tone suggestions" {
    try std.testing.expectEqualStrings("direct and helpful", EmotionType.frustrated.getSuggestedTone());
    try std.testing.expectEqualStrings("clear and patient", EmotionType.confused.getSuggestedTone());
    try std.testing.expectEqualStrings("concise", EmotionType.impatient.getSuggestedTone());
}

test "temperature modifiers" {
    try std.testing.expect(EmotionType.frustrated.getIntensityModifier() < 0);
    try std.testing.expect(EmotionType.excited.getIntensityModifier() > 0);
    try std.testing.expectEqual(@as(f32, 0.0), EmotionType.neutral.getIntensityModifier());
}

test "relationship memory" {
    const allocator = std.testing.allocator;

    var memory = RelationshipMemory.init(allocator);
    defer memory.deinit();

    try memory.recordEmotion(.grateful, "thanks for help");
    try memory.recordEmotion(.grateful, "you're the best");
    try memory.recordEmotion(.frustrated, "not working");

    try std.testing.expectEqual(@as(usize, 2), memory.positive_interactions);
    try std.testing.expectEqual(@as(usize, 1), memory.negative_interactions);
    try std.testing.expect(memory.getHealthScore() > 0.5);
}
