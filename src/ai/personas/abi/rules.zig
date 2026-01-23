//! Routing Rules Engine for Abi Router
//!
//! Provides a declarative rule-based system for persona routing decisions.
//! Rules can boost or penalize specific personas based on request characteristics.

const std = @import("std");
const types = @import("../types.zig");
const sentiment_mod = @import("sentiment.zig");

/// Condition type for rule matching.
pub const ConditionType = enum {
    /// Match if sentiment urgency exceeds threshold.
    urgency_above,
    /// Match if sentiment urgency below threshold.
    urgency_below,
    /// Match if primary emotion matches.
    emotion_matches,
    /// Match if content is technical.
    is_technical,
    /// Match if empathy is required.
    requires_empathy,
    /// Match if content contains keyword.
    contains_keyword,
    /// Match if content matches regex pattern (simple glob).
    pattern_matches,
    /// Always match (for default rules).
    always,
};

/// A condition that determines whether a rule applies.
pub const RuleCondition = struct {
    condition_type: ConditionType,
    /// Threshold value for numeric conditions (urgency).
    threshold: f32 = 0.0,
    /// String value for keyword/pattern matching.
    string_value: ?[]const u8 = null,
    /// Emotion type for emotion matching.
    emotion: ?@import("../../core/types.zig").EmotionType = null,

    /// Evaluate the condition against sentiment and request.
    pub fn evaluate(self: RuleCondition, sentiment: sentiment_mod.SentimentResult, content: []const u8) bool {
        return switch (self.condition_type) {
            .urgency_above => sentiment.urgency_score > self.threshold,
            .urgency_below => sentiment.urgency_score < self.threshold,
            .emotion_matches => if (self.emotion) |e| sentiment.primary_emotion == e else false,
            .is_technical => sentiment.is_technical,
            .requires_empathy => sentiment.requires_empathy,
            .contains_keyword => if (self.string_value) |kw| std.mem.indexOf(u8, content, kw) != null else false,
            .pattern_matches => self.matchPattern(content),
            .always => true,
        };
    }

    fn matchPattern(self: RuleCondition, content: []const u8) bool {
        const pattern = self.string_value orelse return false;
        // Simple glob matching: * matches any sequence
        if (std.mem.indexOf(u8, pattern, "*")) |star_pos| {
            const prefix = pattern[0..star_pos];
            const suffix = pattern[star_pos + 1 ..];
            return std.mem.startsWith(u8, content, prefix) and
                (suffix.len == 0 or std.mem.endsWith(u8, content, suffix));
        }
        return std.mem.indexOf(u8, content, pattern) != null;
    }
};

/// A routing rule that affects persona selection.
pub const RoutingRule = struct {
    /// Human-readable name for the rule.
    name: []const u8,
    /// Condition that triggers this rule.
    condition: RuleCondition,
    /// Priority (higher = evaluated first, applied last).
    priority: u8 = 5,
    /// Score adjustments for each persona (positive = boost, negative = penalize).
    persona_adjustments: PersonaAdjustments = .{},
    /// Whether this rule is enabled.
    enabled: bool = true,
};

/// Score adjustments per persona.
pub const PersonaAdjustments = struct {
    abbey_boost: f32 = 0.0,
    aviva_boost: f32 = 0.0,
    abi_boost: f32 = 0.0,
    // Block routing to specific personas
    block_abbey: bool = false,
    block_aviva: bool = false,
};

/// Result of evaluating all routing rules.
pub const RoutingRulesScore = struct {
    /// Cumulative boost for Abbey persona.
    abbey_boost: f32 = 0.0,
    /// Cumulative boost for Aviva persona.
    aviva_boost: f32 = 0.0,
    /// Cumulative boost for Abi persona.
    abi_boost: f32 = 0.0,
    /// Whether content requires moderation (route to Abi).
    requires_moderation: bool = false,
    /// Names of rules that matched.
    matched_rules: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) RoutingRulesScore {
        return .{
            .matched_rules = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *RoutingRulesScore) void {
        self.matched_rules.deinit();
    }

    /// Get the persona with the highest boost.
    pub fn getBestPersona(self: RoutingRulesScore) types.PersonaType {
        if (self.requires_moderation) return .abi;

        const max_boost = @max(@max(self.abbey_boost, self.aviva_boost), self.abi_boost);

        if (max_boost == self.abi_boost and self.abi_boost > 0) return .abi;
        if (max_boost == self.aviva_boost and self.aviva_boost > 0) return .aviva;
        return .abbey; // Default to Abbey
    }
};

/// The routing rules engine.
pub const RulesEngine = struct {
    allocator: std.mem.Allocator,
    rules: std.ArrayListUnmanaged(RoutingRule),

    const Self = @This();

    /// Initialize the rules engine with default rules.
    pub fn init(allocator: std.mem.Allocator) Self {
        var engine = Self{
            .allocator = allocator,
            .rules = .empty,
        };

        // Add default routing rules
        engine.addDefaultRules() catch {};

        return engine;
    }

    /// Add default routing rules based on the architecture spec.
    fn addDefaultRules(self: *Self) !void {
        // Rule 1: High urgency + negative emotion -> Abbey
        try self.addRule(.{
            .name = "urgent_emotional_support",
            .condition = .{
                .condition_type = .urgency_above,
                .threshold = 0.7,
            },
            .priority = 8,
            .persona_adjustments = .{
                .abbey_boost = 0.3,
            },
        });

        // Rule 2: Technical query without emotional content -> Aviva
        try self.addRule(.{
            .name = "technical_direct",
            .condition = .{
                .condition_type = .is_technical,
            },
            .priority = 6,
            .persona_adjustments = .{
                .aviva_boost = 0.25,
            },
        });

        // Rule 3: Empathy needed -> Abbey
        try self.addRule(.{
            .name = "empathy_required",
            .condition = .{
                .condition_type = .requires_empathy,
            },
            .priority = 7,
            .persona_adjustments = .{
                .abbey_boost = 0.35,
                .aviva_boost = -0.1,
            },
        });

        // Rule 4: Code-related keywords -> Aviva
        try self.addRule(.{
            .name = "code_request",
            .condition = .{
                .condition_type = .contains_keyword,
                .string_value = "code",
            },
            .priority = 5,
            .persona_adjustments = .{
                .aviva_boost = 0.2,
            },
        });

        // Rule 5: Implementation request -> Aviva
        try self.addRule(.{
            .name = "implementation_request",
            .condition = .{
                .condition_type = .contains_keyword,
                .string_value = "implement",
            },
            .priority = 5,
            .persona_adjustments = .{
                .aviva_boost = 0.2,
            },
        });

        // Rule 6: Frustrated user -> Abbey with empathy
        try self.addRule(.{
            .name = "frustrated_user",
            .condition = .{
                .condition_type = .emotion_matches,
                .emotion = .frustrated,
            },
            .priority = 8,
            .persona_adjustments = .{
                .abbey_boost = 0.4,
            },
        });

        // Rule 7: Confused user -> Abbey for explanation
        try self.addRule(.{
            .name = "confused_user",
            .condition = .{
                .condition_type = .emotion_matches,
                .emotion = .confused,
            },
            .priority = 6,
            .persona_adjustments = .{
                .abbey_boost = 0.2,
            },
        });
    }

    /// Shutdown the engine and free resources.
    pub fn deinit(self: *Self) void {
        self.rules.deinit(self.allocator);
    }

    /// Add a routing rule.
    pub fn addRule(self: *Self, rule: RoutingRule) !void {
        try self.rules.append(self.allocator, rule);
    }

    /// Evaluate all rules against the given sentiment and content.
    pub fn evaluate(
        self: *const Self,
        sentiment: sentiment_mod.SentimentResult,
        content: []const u8,
    ) RoutingRulesScore {
        var score = RoutingRulesScore.init(self.allocator);

        // Sort rules by priority (higher priority = evaluated later, takes precedence)
        const sorted_rules = self.allocator.alloc(RoutingRule, self.rules.items.len) catch return score;
        defer self.allocator.free(sorted_rules);

        @memcpy(sorted_rules, self.rules.items);
        std.mem.sort(RoutingRule, sorted_rules, {}, struct {
            fn lessThan(_: void, a: RoutingRule, b: RoutingRule) bool {
                return a.priority < b.priority;
            }
        }.lessThan);

        // Evaluate each rule
        for (sorted_rules) |rule| {
            if (!rule.enabled) continue;

            if (rule.condition.evaluate(sentiment, content)) {
                score.abbey_boost += rule.persona_adjustments.abbey_boost;
                score.aviva_boost += rule.persona_adjustments.aviva_boost;
                score.abi_boost += rule.persona_adjustments.abi_boost;

                score.matched_rules.append(rule.name) catch {};
            }
        }

        return score;
    }

    /// Get the count of registered rules.
    pub fn ruleCount(self: *const Self) usize {
        return self.rules.items.len;
    }
};

test "RulesEngine default rules" {
    const allocator = std.testing.allocator;
    var engine = RulesEngine.init(allocator);
    defer engine.deinit();

    try std.testing.expect(engine.ruleCount() >= 7);
}

test "RuleCondition evaluation" {
    const sentiment = sentiment_mod.SentimentResult{
        .primary_emotion = .frustrated,
        .secondary_emotions = &[_]@import("../../core/types.zig").EmotionType{},
        .urgency_score = 0.8,
        .requires_empathy = true,
        .is_technical = false,
    };

    const urgency_condition = RuleCondition{
        .condition_type = .urgency_above,
        .threshold = 0.7,
    };
    try std.testing.expect(urgency_condition.evaluate(sentiment, "test"));

    const empathy_condition = RuleCondition{
        .condition_type = .requires_empathy,
    };
    try std.testing.expect(empathy_condition.evaluate(sentiment, "test"));

    const emotion_condition = RuleCondition{
        .condition_type = .emotion_matches,
        .emotion = .frustrated,
    };
    try std.testing.expect(emotion_condition.evaluate(sentiment, "test"));
}

test "RoutingRulesScore getBestPersona" {
    const allocator = std.testing.allocator;

    var score = RoutingRulesScore.init(allocator);
    defer score.deinit();

    score.abbey_boost = 0.5;
    score.aviva_boost = 0.2;

    try std.testing.expect(score.getBestPersona() == .abbey);

    score.aviva_boost = 0.8;
    try std.testing.expect(score.getBestPersona() == .aviva);

    score.requires_moderation = true;
    try std.testing.expect(score.getBestPersona() == .abi);
}
