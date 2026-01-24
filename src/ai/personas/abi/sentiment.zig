//! Sentiment Analysis Module for Abi Router
//!
//! Analyzes user input to detect emotional state, urgency, and technical content.
//! This data is used by the router to select the optimal persona.
//!
//! Features:
//! - Multi-emotion detection with primary and secondary emotions
//! - Weighted urgency scoring based on keyword intensity
//! - Technical content classification with domain-specific patterns
//! - Negation-aware emotion detection

const std = @import("std");
const core_types = @import("../../core/types.zig");
const time = @import("../../../shared/time.zig");

/// Result of sentiment and intent analysis.
pub const SentimentResult = struct {
    /// The primary detected emotion.
    primary_emotion: core_types.EmotionType,
    /// Any secondary emotions detected.
    secondary_emotions: []const core_types.EmotionType,
    /// Urgency score from 0.0 (low) to 1.0 (critical).
    urgency_score: f32,
    /// Confidence in the emotion detection (0.0 to 1.0).
    confidence: f32 = 0.5,
    /// Whether the user appears to need emotional support or empathy.
    requires_empathy: bool,
    /// Whether the content appears to be primarily technical or code-related.
    is_technical: bool,
    /// Detected intent category.
    intent: IntentCategory = .general,
    /// Raw emotion scores for all detected emotions.
    emotion_scores: ?EmotionScores = null,

    pub fn deinit(self: *SentimentResult, allocator: std.mem.Allocator) void {
        allocator.free(self.secondary_emotions);
    }

    /// Convert result to a core EmotionalState for use in PersonaRequests.
    pub fn toEmotionalState(self: SentimentResult) core_types.EmotionalState {
        return .{
            .current = self.primary_emotion,
            .intensity = self.urgency_score,
            .last_detected = time.unixSeconds(),
        };
    }
};

/// User intent categories for routing decisions.
pub const IntentCategory = enum {
    general,
    question,
    request,
    complaint,
    feedback,
    greeting,
    farewell,
    code_request,
    explanation_request,
    debugging,
};

/// Raw scores for each emotion type.
pub const EmotionScores = struct {
    neutral: f32 = 0.0,
    happy: f32 = 0.0,
    sad: f32 = 0.0,
    frustrated: f32 = 0.0,
    confused: f32 = 0.0,
    excited: f32 = 0.0,
    anxious: f32 = 0.0,
    stressed: f32 = 0.0,
    curious: f32 = 0.0,
    disappointed: f32 = 0.0,
};

/// Weighted keyword pattern for emotion detection.
const EmotionPattern = struct {
    keywords: []const []const u8,
    emotion: core_types.EmotionType,
    weight: f32,
    negation_sensitive: bool = true,
};

/// Urgency indicator with associated weight.
const UrgencyIndicator = struct {
    keyword: []const u8,
    weight: f32,
};

/// Technical domain keyword patterns.
const TechnicalPattern = struct {
    keywords: []const []const u8,
    domain: []const u8,
    weight: f32,
};

/// Sentiment analyzer implementation with weighted pattern matching.
pub const SentimentAnalyzer = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    // Emotion patterns with weights
    const EMOTION_PATTERNS = [_]EmotionPattern{
        // Frustrated - high weight indicators
        .{
            .keywords = &[_][]const u8{ "frustrated", "frustrating", "infuriating", "annoying", "irritating" },
            .emotion = .frustrated,
            .weight = 0.9,
        },
        .{
            .keywords = &[_][]const u8{ "angry", "furious", "mad", "pissed" },
            .emotion = .frustrated,
            .weight = 0.95,
        },
        .{
            .keywords = &[_][]const u8{ "ugh", "argh", "grrr" },
            .emotion = .frustrated,
            .weight = 0.7,
        },
        // Confused
        .{
            .keywords = &[_][]const u8{ "confused", "confusing", "lost", "bewildered" },
            .emotion = .confused,
            .weight = 0.85,
        },
        .{
            .keywords = &[_][]const u8{ "don't understand", "doesn't make sense", "what does" },
            .emotion = .confused,
            .weight = 0.8,
        },
        // Stressed/Anxious
        .{
            .keywords = &[_][]const u8{ "stressed", "stress", "overwhelming", "overwhelmed" },
            .emotion = .stressed,
            .weight = 0.85,
        },
        .{
            .keywords = &[_][]const u8{ "anxious", "worried", "nervous", "concern" },
            .emotion = .anxious,
            .weight = 0.8,
        },
        .{
            .keywords = &[_][]const u8{ "deadline", "running out of time", "due soon" },
            .emotion = .stressed,
            .weight = 0.7,
        },
        // Excited/Happy
        .{
            .keywords = &[_][]const u8{ "excited", "thrilled", "can't wait", "looking forward" },
            .emotion = .excited,
            .weight = 0.85,
        },
        .{
            .keywords = &[_][]const u8{ "happy", "glad", "pleased", "great", "awesome", "amazing" },
            .emotion = .happy,
            .weight = 0.75,
        },
        .{
            .keywords = &[_][]const u8{ "thanks", "thank you", "appreciate" },
            .emotion = .happy,
            .weight = 0.5,
        },
        // Curious
        .{
            .keywords = &[_][]const u8{ "curious", "wondering", "interested", "intrigued" },
            .emotion = .curious,
            .weight = 0.7,
        },
        .{
            .keywords = &[_][]const u8{ "how does", "why does", "what if", "could you explain" },
            .emotion = .curious,
            .weight = 0.6,
        },
        // Disappointed
        .{
            .keywords = &[_][]const u8{ "disappointed", "let down", "expected better" },
            .emotion = .disappointed,
            .weight = 0.8,
        },
        // Sad
        .{
            .keywords = &[_][]const u8{ "sad", "unhappy", "depressed", "down" },
            .emotion = .sad,
            .weight = 0.75,
        },
    };

    // Urgency indicators with weights
    const URGENCY_INDICATORS = [_]UrgencyIndicator{
        .{ .keyword = "urgent", .weight = 0.9 },
        .{ .keyword = "asap", .weight = 0.85 },
        .{ .keyword = "immediately", .weight = 0.9 },
        .{ .keyword = "emergency", .weight = 0.95 },
        .{ .keyword = "critical", .weight = 0.9 },
        .{ .keyword = "blocking", .weight = 0.85 },
        .{ .keyword = "deadline", .weight = 0.8 },
        .{ .keyword = "now", .weight = 0.5 },
        .{ .keyword = "quickly", .weight = 0.6 },
        .{ .keyword = "soon", .weight = 0.4 },
        .{ .keyword = "help", .weight = 0.5 },
        .{ .keyword = "fix", .weight = 0.5 },
        .{ .keyword = "broken", .weight = 0.6 },
        .{ .keyword = "not working", .weight = 0.65 },
        .{ .keyword = "down", .weight = 0.6 },
        .{ .keyword = "outage", .weight = 0.9 },
    };

    // Technical domain patterns
    const TECHNICAL_PATTERNS = [_]TechnicalPattern{
        // Programming languages and frameworks
        .{
            .keywords = &[_][]const u8{ "zig", "rust", "python", "javascript", "java", "c++", "go", "typescript" },
            .domain = "programming",
            .weight = 0.9,
        },
        // Code-related terms
        .{
            .keywords = &[_][]const u8{ "code", "function", "class", "method", "variable", "struct", "enum" },
            .domain = "code",
            .weight = 0.85,
        },
        .{
            .keywords = &[_][]const u8{ "compile", "build", "run", "execute", "test", "debug" },
            .domain = "development",
            .weight = 0.8,
        },
        // Error patterns
        .{
            .keywords = &[_][]const u8{ "error:", "exception", "crash", "panic", "segfault", "undefined" },
            .domain = "errors",
            .weight = 0.9,
        },
        // API/System
        .{
            .keywords = &[_][]const u8{ "api", "endpoint", "request", "response", "http", "rest", "graphql" },
            .domain = "api",
            .weight = 0.85,
        },
        // Database
        .{
            .keywords = &[_][]const u8{ "database", "query", "sql", "table", "index", "vector", "wdbx" },
            .domain = "database",
            .weight = 0.85,
        },
        // Infrastructure
        .{
            .keywords = &[_][]const u8{ "server", "deploy", "docker", "kubernetes", "cloud", "aws", "gpu" },
            .domain = "infrastructure",
            .weight = 0.8,
        },
        // Implementation requests
        .{
            .keywords = &[_][]const u8{ "implement", "create", "write", "build", "make", "add" },
            .domain = "implementation",
            .weight = 0.6,
        },
    };

    // Negation words
    const NEGATION_WORDS = [_][]const u8{
        "not",   "no",     "don't", "doesn't", "didn't",  "won't",   "wouldn't",
        "can't", "cannot", "never", "none",    "nothing", "without",
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Analyze text to determine emotional and technical context.
    pub fn analyze(self: *const Self, text: []const u8) !SentimentResult {
        const lower = try std.ascii.allocLowerString(self.allocator, text);
        defer self.allocator.free(lower);

        // Detect emotions with scoring
        var emotion_scores = EmotionScores{};
        var secondary = std.ArrayList(core_types.EmotionType).init(self.allocator);
        errdefer secondary.deinit();

        // Calculate emotion scores
        for (EMOTION_PATTERNS) |pattern| {
            var score: f32 = 0.0;
            for (pattern.keywords) |keyword| {
                if (std.mem.indexOf(u8, lower, keyword) != null) {
                    // Check for negation
                    var negated = false;
                    if (pattern.negation_sensitive) {
                        negated = self.isNegated(lower, keyword);
                    }

                    if (!negated) {
                        score = @max(score, pattern.weight);
                    }
                }
            }
            if (score > 0) {
                self.updateEmotionScore(&emotion_scores, pattern.emotion, score);
            }
        }

        // Determine primary and secondary emotions
        const primary = self.getPrimaryEmotion(emotion_scores);
        try self.getSecondaryEmotions(emotion_scores, primary, &secondary);

        // Calculate urgency
        const urgency = self.calculateUrgency(lower);

        // Detect technical content
        const technical_info = self.detectTechnical(lower);

        // Detect intent
        const intent = self.detectIntent(lower);

        // Calculate confidence based on emotion score strength
        const max_score = self.getMaxEmotionScore(emotion_scores);
        const confidence = if (max_score > 0) @min(max_score + 0.2, 1.0) else 0.3;

        // Determine if empathy is required
        const requires_empathy = self.needsEmpathy(primary, urgency, emotion_scores);

        return SentimentResult{
            .primary_emotion = primary,
            .secondary_emotions = try secondary.toOwnedSlice(),
            .urgency_score = urgency,
            .confidence = confidence,
            .requires_empathy = requires_empathy,
            .is_technical = technical_info.is_technical,
            .intent = intent,
            .emotion_scores = emotion_scores,
        };
    }

    /// Check if a keyword is preceded by a negation word.
    fn isNegated(self: *const Self, text: []const u8, keyword: []const u8) bool {
        _ = self;
        const keyword_pos = std.mem.indexOf(u8, text, keyword) orelse return false;

        // Look for negation in the 20 characters before the keyword
        const search_start = if (keyword_pos > 20) keyword_pos - 20 else 0;
        const search_region = text[search_start..keyword_pos];

        for (NEGATION_WORDS) |negation| {
            if (std.mem.indexOf(u8, search_region, negation) != null) {
                return true;
            }
        }
        return false;
    }

    /// Update emotion scores based on detected patterns.
    fn updateEmotionScore(_: *const Self, scores: *EmotionScores, emotion: core_types.EmotionType, score: f32) void {
        switch (emotion) {
            .neutral => scores.neutral = @max(scores.neutral, score),
            .happy => scores.happy = @max(scores.happy, score),
            .sad => scores.sad = @max(scores.sad, score),
            .frustrated => scores.frustrated = @max(scores.frustrated, score),
            .confused => scores.confused = @max(scores.confused, score),
            .excited => scores.excited = @max(scores.excited, score),
            .anxious => scores.anxious = @max(scores.anxious, score),
            .stressed => scores.stressed = @max(scores.stressed, score),
            .curious => scores.curious = @max(scores.curious, score),
            .disappointed => scores.disappointed = @max(scores.disappointed, score),
        }
    }

    /// Get the primary emotion based on highest score.
    fn getPrimaryEmotion(_: *const Self, scores: EmotionScores) core_types.EmotionType {
        const emotion_values = [_]struct { emotion: core_types.EmotionType, score: f32 }{
            .{ .emotion = .frustrated, .score = scores.frustrated },
            .{ .emotion = .stressed, .score = scores.stressed },
            .{ .emotion = .anxious, .score = scores.anxious },
            .{ .emotion = .confused, .score = scores.confused },
            .{ .emotion = .disappointed, .score = scores.disappointed },
            .{ .emotion = .sad, .score = scores.sad },
            .{ .emotion = .excited, .score = scores.excited },
            .{ .emotion = .happy, .score = scores.happy },
            .{ .emotion = .curious, .score = scores.curious },
        };

        var max_score: f32 = 0.0;
        var primary: core_types.EmotionType = .neutral;

        for (emotion_values) |ev| {
            if (ev.score > max_score) {
                max_score = ev.score;
                primary = ev.emotion;
            }
        }

        return primary;
    }

    /// Get secondary emotions (those with significant scores).
    fn getSecondaryEmotions(_: *const Self, scores: EmotionScores, primary: core_types.EmotionType, list: *std.ArrayList(core_types.EmotionType)) !void {
        const threshold: f32 = 0.4;

        const emotion_values = [_]struct { emotion: core_types.EmotionType, score: f32 }{
            .{ .emotion = .frustrated, .score = scores.frustrated },
            .{ .emotion = .stressed, .score = scores.stressed },
            .{ .emotion = .anxious, .score = scores.anxious },
            .{ .emotion = .confused, .score = scores.confused },
            .{ .emotion = .disappointed, .score = scores.disappointed },
            .{ .emotion = .sad, .score = scores.sad },
            .{ .emotion = .excited, .score = scores.excited },
            .{ .emotion = .happy, .score = scores.happy },
            .{ .emotion = .curious, .score = scores.curious },
        };

        for (emotion_values) |ev| {
            if (ev.emotion != primary and ev.score >= threshold) {
                try list.append(ev.emotion);
            }
        }
    }

    /// Get maximum emotion score.
    fn getMaxEmotionScore(_: *const Self, scores: EmotionScores) f32 {
        return @max(@max(@max(@max(@max(@max(@max(@max(@max(
            scores.frustrated,
            scores.stressed,
        ), scores.anxious), scores.confused), scores.disappointed), scores.sad), scores.excited), scores.happy), scores.curious), scores.neutral);
    }

    /// Calculate urgency score from indicators.
    fn calculateUrgency(self: *const Self, text: []const u8) f32 {
        _ = self;
        var max_urgency: f32 = 0.1;

        for (URGENCY_INDICATORS) |indicator| {
            if (std.mem.indexOf(u8, text, indicator.keyword) != null) {
                max_urgency = @max(max_urgency, indicator.weight);
            }
        }

        // Boost urgency for multiple indicators
        var indicator_count: usize = 0;
        for (URGENCY_INDICATORS) |indicator| {
            if (std.mem.indexOf(u8, text, indicator.keyword) != null) {
                indicator_count += 1;
            }
        }

        if (indicator_count >= 3) {
            max_urgency = @min(max_urgency + 0.1, 1.0);
        }

        return max_urgency;
    }

    /// Detect technical content and domain.
    fn detectTechnical(self: *const Self, text: []const u8) struct { is_technical: bool, domain: ?[]const u8 } {
        _ = self;
        var max_weight: f32 = 0.0;
        var detected_domain: ?[]const u8 = null;

        for (TECHNICAL_PATTERNS) |pattern| {
            for (pattern.keywords) |keyword| {
                if (std.mem.indexOf(u8, text, keyword) != null) {
                    if (pattern.weight > max_weight) {
                        max_weight = pattern.weight;
                        detected_domain = pattern.domain;
                    }
                    break;
                }
            }
        }

        return .{
            .is_technical = max_weight >= 0.6,
            .domain = detected_domain,
        };
    }

    /// Detect user intent category.
    fn detectIntent(self: *const Self, text: []const u8) IntentCategory {
        _ = self;

        // Check for greetings
        const greetings = [_][]const u8{ "hello", "hi ", "hey ", "good morning", "good afternoon", "good evening" };
        for (greetings) |greeting| {
            if (std.mem.indexOf(u8, text, greeting) != null) return .greeting;
        }

        // Check for farewells
        const farewells = [_][]const u8{ "bye", "goodbye", "see you", "take care", "later" };
        for (farewells) |farewell| {
            if (std.mem.indexOf(u8, text, farewell) != null) return .farewell;
        }

        // Check for code requests
        const code_requests = [_][]const u8{ "write code", "implement", "create function", "write a function", "code for" };
        for (code_requests) |cr| {
            if (std.mem.indexOf(u8, text, cr) != null) return .code_request;
        }

        // Check for debugging
        const debug_patterns = [_][]const u8{ "debug", "fix this", "error:", "not working", "broken", "bug" };
        for (debug_patterns) |dp| {
            if (std.mem.indexOf(u8, text, dp) != null) return .debugging;
        }

        // Check for explanation requests
        const explain_patterns = [_][]const u8{ "explain", "how does", "why does", "what is", "what are" };
        for (explain_patterns) |ep| {
            if (std.mem.indexOf(u8, text, ep) != null) return .explanation_request;
        }

        // Check for questions
        if (std.mem.indexOf(u8, text, "?") != null or
            std.mem.indexOf(u8, text, "how") != null or
            std.mem.indexOf(u8, text, "what") != null or
            std.mem.indexOf(u8, text, "why") != null or
            std.mem.indexOf(u8, text, "when") != null or
            std.mem.indexOf(u8, text, "where") != null)
        {
            return .question;
        }

        // Check for requests
        const request_patterns = [_][]const u8{ "please", "can you", "could you", "would you", "i need", "i want" };
        for (request_patterns) |rp| {
            if (std.mem.indexOf(u8, text, rp) != null) return .request;
        }

        // Check for complaints
        const complaint_patterns = [_][]const u8{ "terrible", "awful", "horrible", "worst", "hate", "disappointed" };
        for (complaint_patterns) |cp| {
            if (std.mem.indexOf(u8, text, cp) != null) return .complaint;
        }

        return .general;
    }

    /// Determine if empathy is needed based on emotional state.
    fn needsEmpathy(_: *const Self, primary: core_types.EmotionType, urgency: f32, scores: EmotionScores) bool {
        // High-empathy emotions always need empathy
        switch (primary) {
            .frustrated, .stressed, .anxious, .disappointed, .sad => return true,
            else => {},
        }

        // High urgency needs empathy
        if (urgency > 0.7) return true;

        // Multiple negative emotions need empathy
        var negative_count: usize = 0;
        if (scores.frustrated > 0.3) negative_count += 1;
        if (scores.stressed > 0.3) negative_count += 1;
        if (scores.anxious > 0.3) negative_count += 1;
        if (scores.disappointed > 0.3) negative_count += 1;
        if (scores.sad > 0.3) negative_count += 1;

        return negative_count >= 2;
    }
};

// Tests

test "SentimentAnalyzer basic analysis" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var result = try analyzer.analyze("I'm frustrated with this bug");
    defer result.deinit(&result, allocator);

    try std.testing.expect(result.primary_emotion == .frustrated);
    try std.testing.expect(result.requires_empathy);
}

test "SentimentAnalyzer technical detection" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var result = try analyzer.analyze("How do I implement this function in Zig?");
    defer result.deinit(&result, allocator);

    try std.testing.expect(result.is_technical);
    try std.testing.expect(result.intent == .question or result.intent == .explanation_request);
}

test "SentimentAnalyzer urgency detection" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var result = try analyzer.analyze("This is urgent! The server is down!");
    defer result.deinit(&result, allocator);

    try std.testing.expect(result.urgency_score >= 0.8);
}

test "SentimentAnalyzer negation handling" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var result = try analyzer.analyze("I'm not frustrated at all, just curious");
    defer result.deinit(&result, allocator);

    // Should detect curious, not frustrated
    try std.testing.expect(result.primary_emotion != .frustrated);
}

test "SentimentAnalyzer multi-emotion detection" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var result = try analyzer.analyze("I'm stressed and confused about this deadline");
    defer result.deinit(&result, allocator);

    try std.testing.expect(result.primary_emotion == .stressed or result.primary_emotion == .confused);
    try std.testing.expect(result.urgency_score > 0.5); // "deadline" should boost urgency
}

test "SentimentAnalyzer intent classification" {
    const allocator = std.testing.allocator;
    const analyzer = SentimentAnalyzer.init(allocator);

    var greeting = try analyzer.analyze("Hello, how are you?");
    defer greeting.deinit(&greeting, allocator);
    try std.testing.expect(greeting.intent == .greeting);

    var code_req = try analyzer.analyze("Write a function to sort arrays");
    defer code_req.deinit(&code_req, allocator);
    try std.testing.expect(code_req.intent == .code_request);
}
