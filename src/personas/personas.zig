//! Multi-Persona System — Abbey, Aviva, and Abi
//!
//! Defines behavioral weights, persona types, sentiment/content classification,
//! and response context for the Abi framework's dual-persona architecture.

const std = @import("std");

// ============================================================================
// Behavioral Weights
// ============================================================================

pub const BehavioralWeights = struct {
    empathy: f32 = 0.5,
    directness: f32 = 0.5,
    verbosity: f32 = 0.5,
    formality: f32 = 0.5,
    technical_depth: f32 = 0.5,
    creativity: f32 = 0.5,

    /// Linear blend: result = a * alpha + b * (1 - alpha).
    pub fn blend(a: BehavioralWeights, b: BehavioralWeights, alpha: f32) BehavioralWeights {
        const clamped = std.math.clamp(alpha, 0.0, 1.0);
        const inv = 1.0 - clamped;
        return .{
            .empathy = a.empathy * clamped + b.empathy * inv,
            .directness = a.directness * clamped + b.directness * inv,
            .verbosity = a.verbosity * clamped + b.verbosity * inv,
            .formality = a.formality * clamped + b.formality * inv,
            .technical_depth = a.technical_depth * clamped + b.technical_depth * inv,
            .creativity = a.creativity * clamped + b.creativity * inv,
        };
    }

    /// Magnitude of the weight vector (useful for normalization).
    pub fn magnitude(self: BehavioralWeights) f32 {
        return @sqrt(self.empathy * self.empathy +
            self.directness * self.directness +
            self.verbosity * self.verbosity +
            self.formality * self.formality +
            self.technical_depth * self.technical_depth +
            self.creativity * self.creativity);
    }
};

// ============================================================================
// Persona Types
// ============================================================================

pub const PersonaType = enum {
    abbey, // Empathetic polymath — warm, creative, supportive
    aviva, // Direct expert — precise, technical, concise
    abi, // Adaptive moderator — routes between personas
    blended, // Weighted mix of Abbey and Aviva

    pub fn getDefaultWeights(self: PersonaType) BehavioralWeights {
        return switch (self) {
            .abbey => .{
                .empathy = 0.85,
                .directness = 0.45,
                .verbosity = 0.70,
                .formality = 0.35,
                .technical_depth = 0.60,
                .creativity = 0.80,
            },
            .aviva => .{
                .empathy = 0.25,
                .directness = 0.90,
                .verbosity = 0.30,
                .formality = 0.75,
                .technical_depth = 0.95,
                .creativity = 0.35,
            },
            .abi => .{
                .empathy = 0.55,
                .directness = 0.65,
                .verbosity = 0.50,
                .formality = 0.55,
                .technical_depth = 0.75,
                .creativity = 0.55,
            },
            .blended => .{}, // Default 0.5 across the board
        };
    }

    pub fn displayName(self: PersonaType) []const u8 {
        return switch (self) {
            .abbey => "Abbey",
            .aviva => "Aviva",
            .abi => "Abi",
            .blended => "Blended",
        };
    }
};

// ============================================================================
// Sentiment Analysis
// ============================================================================

pub const Sentiment = enum {
    positive,
    neutral,
    negative,
    frustrated,
    confused,
    excited,

    /// Simple keyword-based sentiment detection.
    pub fn fromText(text: []const u8) Sentiment {
        const lower = blk: {
            var buf: [512]u8 = undefined;
            const len = @min(text.len, buf.len);
            for (0..len) |i| {
                buf[i] = std.ascii.toLower(text[i]);
            }
            break :blk buf[0..len];
        };

        // Frustrated indicators
        const frustrated_words = [_][]const u8{ "frustrated", "annoying", "broken", "stupid", "hate", "ugh", "wtf" };
        for (frustrated_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .frustrated;
        }

        // Confused indicators
        const confused_words = [_][]const u8{ "confused", "don't understand", "what do you mean", "how does", "why does", "help me" };
        for (confused_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .confused;
        }

        // Excited indicators
        const excited_words = [_][]const u8{ "amazing", "awesome", "fantastic", "love it", "excited", "great" };
        for (excited_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .excited;
        }

        // Negative
        const negative_words = [_][]const u8{ "bad", "wrong", "error", "fail", "problem", "issue", "bug" };
        for (negative_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .negative;
        }

        // Positive
        const positive_words = [_][]const u8{ "thanks", "good", "nice", "please", "helpful", "works", "perfect" };
        for (positive_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .positive;
        }

        return .neutral;
    }
};

// ============================================================================
// Content Type Classification
// ============================================================================

pub const ContentType = enum {
    technical,
    emotional,
    creative,
    factual,
    code,
    general,

    /// Classify input text by content type.
    pub fn fromText(text: []const u8) ContentType {
        const lower = blk: {
            var buf: [512]u8 = undefined;
            const len = @min(text.len, buf.len);
            for (0..len) |i| {
                buf[i] = std.ascii.toLower(text[i]);
            }
            break :blk buf[0..len];
        };

        // Code indicators
        const code_words = [_][]const u8{ "function", "class ", "def ", "import ", "const ", "var ", "pub fn", "return ", "{", "}", "=>", "==", "!=", "compile" };
        var code_score: u32 = 0;
        for (code_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) code_score += 1;
        }
        if (code_score >= 2) return .code;

        // Technical indicators
        const tech_words = [_][]const u8{ "algorithm", "implement", "optimize", "performance", "latency", "throughput", "architecture", "database", "api", "server", "deploy", "debug" };
        var tech_score: u32 = 0;
        for (tech_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) tech_score += 1;
        }
        if (tech_score >= 2) return .technical;

        // Emotional
        const emotional_words = [_][]const u8{ "feel", "worried", "scared", "happy", "sad", "angry", "love", "miss", "hope", "afraid" };
        for (emotional_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .emotional;
        }

        // Creative
        const creative_words = [_][]const u8{ "write", "story", "poem", "imagine", "create", "design", "brainstorm", "idea" };
        for (creative_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .creative;
        }

        // Factual
        const factual_words = [_][]const u8{ "what is", "who is", "when did", "how many", "define", "explain", "describe" };
        for (factual_words) |word| {
            if (std.mem.indexOf(u8, lower, word) != null) return .factual;
        }

        return .general;
    }
};

// ============================================================================
// Response Context
// ============================================================================

pub const ResponseContext = struct {
    turn_count: u32 = 0,
    last_sentiment: Sentiment = .neutral,
    last_content_type: ContentType = .general,
    last_persona: PersonaType = .abi,
    topic_continuity: f32 = 0.0,
};

// ============================================================================
// Tests
// ============================================================================

test "behavioral weights blend" {
    const abbey = PersonaType.abbey.getDefaultWeights();
    const aviva = PersonaType.aviva.getDefaultWeights();

    // Full Abbey
    const full_abbey = BehavioralWeights.blend(abbey, aviva, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), full_abbey.empathy, 1e-5);

    // Full Aviva
    const full_aviva = BehavioralWeights.blend(abbey, aviva, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), full_aviva.empathy, 1e-5);

    // 50/50
    const half = BehavioralWeights.blend(abbey, aviva, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.55), half.empathy, 1e-5);
}

test "sentiment detection" {
    try std.testing.expectEqual(Sentiment.frustrated, Sentiment.fromText("This is so annoying and broken!"));
    try std.testing.expectEqual(Sentiment.confused, Sentiment.fromText("I don't understand how this works"));
    try std.testing.expectEqual(Sentiment.positive, Sentiment.fromText("Thanks, that was helpful!"));
    try std.testing.expectEqual(Sentiment.neutral, Sentiment.fromText("The weather is mild today."));
}

test "content type classification" {
    try std.testing.expectEqual(ContentType.code, ContentType.fromText("pub fn main() { return 0; }"));
    try std.testing.expectEqual(ContentType.technical, ContentType.fromText("How do I optimize the database API performance?"));
    try std.testing.expectEqual(ContentType.emotional, ContentType.fromText("I feel worried about the future"));
    try std.testing.expectEqual(ContentType.creative, ContentType.fromText("Write me a story about a dragon"));
}

test "persona display names" {
    try std.testing.expectEqualStrings("Abbey", PersonaType.abbey.displayName());
    try std.testing.expectEqualStrings("Aviva", PersonaType.aviva.displayName());
    try std.testing.expectEqualStrings("Abi", PersonaType.abi.displayName());
}
