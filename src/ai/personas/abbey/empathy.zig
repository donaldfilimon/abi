//! Abbey Empathy Injection Module
//!
//! Provides template-based empathy patterns and context-aware empathy
//! calibration for Abbey's responses. Ensures responses acknowledge
//! user emotions before diving into technical solutions.
//!
//! Features:
//! - Template-based empathy patterns
//! - Context-aware empathy calibration
//! - Tone adaptation based on emotional state
//! - Response prefix and suffix injection

const std = @import("std");
const emotion = @import("emotion.zig");
const core_types = @import("../../core/types.zig");

/// Empathy template for specific emotions.
pub const EmpathyTemplate = struct {
    /// Opening acknowledgment.
    acknowledgment: []const u8,
    /// Transition to solution.
    transition: []const u8,
    /// Closing encouragement.
    encouragement: []const u8,
    /// Whether to validate the emotion explicitly.
    validate_explicitly: bool,
};

/// Pre-defined empathy templates for each emotion type.
pub const EMPATHY_TEMPLATES = struct {
    pub const frustrated = EmpathyTemplate{
        .acknowledgment = "I understand this is frustrating, and your feelings are completely valid.",
        .transition = "Let's work through this together - ",
        .encouragement = "We'll get this sorted out.",
        .validate_explicitly = true,
    };

    pub const confused = EmpathyTemplate{
        .acknowledgment = "I can see this is confusing - it's a complex topic.",
        .transition = "Let me break this down step by step: ",
        .encouragement = "Feel free to ask if anything is still unclear.",
        .validate_explicitly = true,
    };

    pub const stressed = EmpathyTemplate{
        .acknowledgment = "I hear that you're under pressure, and I want to help.",
        .transition = "Here's what we can do: ",
        .encouragement = "One step at a time - you've got this.",
        .validate_explicitly = true,
    };

    pub const anxious = EmpathyTemplate{
        .acknowledgment = "I understand this feels overwhelming right now.",
        .transition = "Let's take this one piece at a time: ",
        .encouragement = "We'll work through this together at your pace.",
        .validate_explicitly = true,
    };

    pub const excited = EmpathyTemplate{
        .acknowledgment = "I love your enthusiasm!",
        .transition = "Let's dive in - ",
        .encouragement = "This is going to be great!",
        .validate_explicitly = false,
    };

    pub const curious = EmpathyTemplate{
        .acknowledgment = "Great question!",
        .transition = "Here's what I can share: ",
        .encouragement = "Curiosity is wonderful - keep exploring!",
        .validate_explicitly = false,
    };

    pub const grateful = EmpathyTemplate{
        .acknowledgment = "You're very welcome!",
        .transition = "I'm glad I could help. ",
        .encouragement = "Feel free to reach out anytime.",
        .validate_explicitly = false,
    };

    pub const disappointed = EmpathyTemplate{
        .acknowledgment = "I'm sorry this didn't meet your expectations.",
        .transition = "Let me see how we can make this better: ",
        .encouragement = "We can definitely improve on this.",
        .validate_explicitly = true,
    };

    pub const skeptical = EmpathyTemplate{
        .acknowledgment = "I understand your skepticism - let me provide more context.",
        .transition = "Here's the evidence: ",
        .encouragement = "Let me know if you need additional verification.",
        .validate_explicitly = false,
    };

    pub const impatient = EmpathyTemplate{
        .acknowledgment = "I'll be quick and direct.",
        .transition = "Here's the answer: ",
        .encouragement = "",
        .validate_explicitly = false,
    };

    pub const hopeful = EmpathyTemplate{
        .acknowledgment = "I appreciate your optimism!",
        .transition = "Let's see what we can do: ",
        .encouragement = "I'm hopeful we can make this work.",
        .validate_explicitly = false,
    };

    pub const enthusiastic = EmpathyTemplate{
        .acknowledgment = "Your energy is contagious!",
        .transition = "Let's make this happen - ",
        .encouragement = "This is exciting!",
        .validate_explicitly = false,
    };

    pub const playful = EmpathyTemplate{
        .acknowledgment = "",
        .transition = "",
        .encouragement = "",
        .validate_explicitly = false,
    };

    pub const neutral = EmpathyTemplate{
        .acknowledgment = "",
        .transition = "",
        .encouragement = "",
        .validate_explicitly = false,
    };
};

/// Configuration for empathy injection.
pub const EmpathyConfig = struct {
    /// Minimum empathy level to include acknowledgment.
    min_acknowledgment_threshold: f32 = 0.5,
    /// Whether to include transitions.
    include_transitions: bool = true,
    /// Whether to include encouragement.
    include_encouragement: bool = true,
    /// Maximum length for empathy prefix (0 = no limit).
    max_prefix_length: usize = 0,
    /// Whether to adapt based on user preferences.
    adapt_to_preferences: bool = true,
};

/// Result of empathy injection.
pub const EmpathyInjection = struct {
    /// Text to prepend to response.
    prefix: []const u8,
    /// Text to append to response.
    suffix: []const u8,
    /// Temperature adjustment for generation.
    temperature_adjustment: f32,
    /// Whether acknowledgment was included.
    includes_acknowledgment: bool,
};

/// Manages empathy injection for Abbey responses.
pub const EmpathyInjector = struct {
    allocator: std.mem.Allocator,
    config: EmpathyConfig,

    const Self = @This();

    /// Initialize the empathy injector.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: EmpathyConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Generate empathy injection based on emotional response.
    pub fn inject(
        self: *const Self,
        emotional_response: emotion.EmotionalResponse,
        user_preferences: ?UserPreferences,
    ) !EmpathyInjection {
        const template = self.getTemplate(emotional_response.primary_emotion);

        // Check if we should include empathy based on level
        const include_empathy = emotional_response.empathy_level >= self.config.min_acknowledgment_threshold;

        // Build prefix
        var prefix_builder: std.ArrayListUnmanaged(u8) = .{};
        errdefer prefix_builder.deinit(self.allocator);

        if (include_empathy) {
            // Check user preferences
            const prefers_brevity = if (user_preferences) |prefs| prefs.prefers_brevity else false;

            if (!prefers_brevity) {
                // Add acknowledgment
                if (template.acknowledgment.len > 0) {
                    try prefix_builder.appendSlice(self.allocator, template.acknowledgment);
                    try prefix_builder.appendSlice(self.allocator, " ");
                }

                // Add transition
                if (self.config.include_transitions and template.transition.len > 0) {
                    try prefix_builder.appendSlice(self.allocator, template.transition);
                }
            } else {
                // Brief acknowledgment for users who prefer brevity
                if (emotional_response.needs_special_care and template.validate_explicitly) {
                    try prefix_builder.appendSlice(self.allocator, "I understand. ");
                }
            }
        }

        // Apply max length if configured
        if (self.config.max_prefix_length > 0 and prefix_builder.items.len > self.config.max_prefix_length) {
            prefix_builder.shrinkRetainingCapacity(self.config.max_prefix_length);
        }

        // Build suffix
        var suffix_builder: std.ArrayListUnmanaged(u8) = .{};
        errdefer suffix_builder.deinit(self.allocator);

        if (include_empathy and self.config.include_encouragement) {
            const prefers_brevity = if (user_preferences) |prefs| prefs.prefers_brevity else false;
            if (!prefers_brevity and template.encouragement.len > 0) {
                try suffix_builder.appendSlice(self.allocator, " ");
                try suffix_builder.appendSlice(self.allocator, template.encouragement);
            }
        }

        // Calculate temperature adjustment
        const temp_adj = self.calculateTemperatureAdjustment(emotional_response);

        return .{
            .prefix = try prefix_builder.toOwnedSlice(self.allocator),
            .suffix = try suffix_builder.toOwnedSlice(self.allocator),
            .temperature_adjustment = temp_adj,
            .includes_acknowledgment = include_empathy and template.acknowledgment.len > 0,
        };
    }

    /// Free empathy injection resources.
    pub fn freeInjection(self: *const Self, injection: *EmpathyInjection) void {
        if (injection.prefix.len > 0) {
            self.allocator.free(injection.prefix);
        }
        if (injection.suffix.len > 0) {
            self.allocator.free(injection.suffix);
        }
    }

    /// Get the empathy template for an emotion.
    fn getTemplate(self: *const Self, emo: core_types.EmotionType) EmpathyTemplate {
        _ = self;
        return switch (emo) {
            .frustrated => EMPATHY_TEMPLATES.frustrated,
            .confused => EMPATHY_TEMPLATES.confused,
            .stressed => EMPATHY_TEMPLATES.stressed,
            .anxious => EMPATHY_TEMPLATES.anxious,
            .excited => EMPATHY_TEMPLATES.excited,
            .curious => EMPATHY_TEMPLATES.curious,
            .grateful => EMPATHY_TEMPLATES.grateful,
            .disappointed => EMPATHY_TEMPLATES.disappointed,
            .skeptical => EMPATHY_TEMPLATES.skeptical,
            .impatient => EMPATHY_TEMPLATES.impatient,
            .hopeful => EMPATHY_TEMPLATES.hopeful,
            .enthusiastic => EMPATHY_TEMPLATES.enthusiastic,
            .playful => EMPATHY_TEMPLATES.playful,
            .neutral => EMPATHY_TEMPLATES.neutral,
        };
    }

    /// Calculate temperature adjustment based on emotional response.
    fn calculateTemperatureAdjustment(self: *const Self, response: emotion.EmotionalResponse) f32 {
        _ = self;
        // Get base adjustment from tone
        const base_temp = response.suggested_tone.getTemperature();

        // Adjust based on intensity
        const intensity_factor = response.intensity * 0.1;

        // Empathetic responses need slightly higher temperature for warmth
        const empathy_factor = if (response.empathy_level > 0.7) @as(f32, 0.05) else @as(f32, 0.0);

        return base_temp + intensity_factor + empathy_factor - 0.65; // Normalize around 0
    }

    /// Generate a custom acknowledgment for specific contexts.
    pub fn customAcknowledgment(
        self: *const Self,
        template: []const u8,
        context: AcknowledgmentContext,
    ) ![]const u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < template.len) {
            if (std.mem.startsWith(u8, template[i..], "{emotion}")) {
                try result.appendSlice(self.allocator, context.emotion_name);
                i += 9;
            } else if (std.mem.startsWith(u8, template[i..], "{topic}")) {
                try result.appendSlice(self.allocator, context.topic orelse "this");
                i += 7;
            } else if (std.mem.startsWith(u8, template[i..], "{user_name}")) {
                try result.appendSlice(self.allocator, context.user_name orelse "");
                i += 11;
            } else {
                try result.append(self.allocator, template[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }
};

/// User preferences that affect empathy style.
pub const UserPreferences = struct {
    /// User prefers shorter, more direct responses.
    prefers_brevity: bool = false,
    /// User prefers detailed explanations.
    prefers_detail: bool = false,
    /// User prefers formal tone.
    prefers_formal: bool = false,
    /// User's technical level.
    technical_level: TechnicalLevel = .intermediate,

    pub const TechnicalLevel = enum {
        beginner,
        intermediate,
        advanced,
        expert,
    };
};

/// Context for custom acknowledgment generation.
pub const AcknowledgmentContext = struct {
    emotion_name: []const u8,
    topic: ?[]const u8 = null,
    user_name: ?[]const u8 = null,
};

/// Adapt response based on emotional trajectory.
pub fn adaptToTrajectory(
    trajectory: emotion.EmotionTrajectory,
    current_empathy: f32,
) f32 {
    return switch (trajectory.direction) {
        .escalating => @min(1.0, current_empathy + trajectory.magnitude * 0.2),
        .deescalating => current_empathy, // Don't reduce empathy as things calm down
        .stable => current_empathy,
    };
}

// Tests

test "empathy injector initialization" {
    const injector = EmpathyInjector.init(std.testing.allocator);
    try std.testing.expectEqual(@as(f32, 0.5), injector.config.min_acknowledgment_threshold);
}

test "inject empathy for frustration" {
    const injector = EmpathyInjector.init(std.testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .frustrated,
        .intensity = 0.8,
        .suggested_tone = .empathetic,
        .empathy_level = 0.9,
        .needs_special_care = true,
    };

    var result = try injector.inject(emotional_response, null);
    defer injector.freeInjection(&result);

    try std.testing.expect(result.includes_acknowledgment);
    try std.testing.expect(std.mem.indexOf(u8, result.prefix, "frustrating") != null);
}

test "brief empathy for brevity preference" {
    const injector = EmpathyInjector.init(std.testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .frustrated,
        .intensity = 0.8,
        .suggested_tone = .empathetic,
        .empathy_level = 0.9,
        .needs_special_care = true,
    };

    const prefs = UserPreferences{ .prefers_brevity = true };
    var result = try injector.inject(emotional_response, prefs);
    defer injector.freeInjection(&result);

    // Should be shorter than full template
    try std.testing.expect(result.prefix.len < 50);
}

test "no empathy for low intensity" {
    const injector = EmpathyInjector.init(std.testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .neutral,
        .intensity = 0.1,
        .suggested_tone = .balanced,
        .empathy_level = 0.2,
        .needs_special_care = false,
    };

    var result = try injector.inject(emotional_response, null);
    defer injector.freeInjection(&result);

    try std.testing.expect(!result.includes_acknowledgment);
    try std.testing.expectEqual(@as(usize, 0), result.prefix.len);
}

test "custom acknowledgment" {
    const injector = EmpathyInjector.init(std.testing.allocator);

    const result = try injector.customAcknowledgment(
        "I understand you're feeling {emotion} about {topic}.",
        .{
            .emotion_name = "frustrated",
            .topic = "the build errors",
        },
    );
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "frustrated") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "build errors") != null);
}

test "trajectory adaptation" {
    const escalating = emotion.EmotionTrajectory{ .direction = .escalating, .magnitude = 0.3 };
    const stable = emotion.EmotionTrajectory{ .direction = .stable, .magnitude = 0.0 };

    const escalating_empathy = adaptToTrajectory(escalating, 0.7);
    const stable_empathy = adaptToTrajectory(stable, 0.7);

    try std.testing.expect(escalating_empathy > stable_empathy);
}
