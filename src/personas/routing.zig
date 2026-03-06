//! Abi Neural Router
//!
//! Routes user input to the appropriate persona (Abbey, Aviva, or blended)
//! based on sentiment analysis, content type classification, and safety checks.

const std = @import("std");
const Allocator = std.mem.Allocator;
const personas = @import("personas.zig");
const safety = @import("safety.zig");

pub const RoutingDecision = struct {
    persona: personas.PersonaType,
    alpha: f32, // Blend factor: 0 = pure Aviva, 1 = pure Abbey
    confidence: f32,
    safety_flags: safety.SafetyFlags,
    reasoning: []const u8,
    sentiment: personas.Sentiment,
    content_type: personas.ContentType,
};

pub const ProcessResult = struct {
    decision: RoutingDecision,
    weights: personas.BehavioralWeights,
};

pub const AbiModerator = struct {
    const Self = @This();

    allocator: Allocator,
    context: personas.ResponseContext,

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .context = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Route input to the appropriate persona.
    pub fn route(self: *Self, input: []const u8, context: ?*const personas.ResponseContext) !RoutingDecision {
        const ctx = context orelse &self.context;

        // 1. Safety check first.
        const safety_result = safety.check(input);

        // 2. Classify sentiment and content type.
        const sentiment = personas.Sentiment.fromText(input);
        const content_type = personas.ContentType.fromText(input);

        // 3. Compute routing scores.
        var abbey_score: f32 = 0.0;
        var aviva_score: f32 = 0.0;

        // Sentiment-based scoring.
        switch (sentiment) {
            .frustrated, .confused => {
                abbey_score += 0.4; // Abbey is better for emotional support
            },
            .excited, .positive => {
                abbey_score += 0.2;
                aviva_score += 0.1;
            },
            .negative => {
                abbey_score += 0.3;
            },
            .neutral => {
                aviva_score += 0.1;
            },
        }

        // Content-type-based scoring.
        switch (content_type) {
            .technical, .code => {
                aviva_score += 0.5; // Aviva excels at technical content
            },
            .emotional => {
                abbey_score += 0.5; // Abbey for emotional content
            },
            .creative => {
                abbey_score += 0.4;
                aviva_score += 0.1;
            },
            .factual => {
                aviva_score += 0.3;
                abbey_score += 0.1;
            },
            .general => {
                abbey_score += 0.2;
                aviva_score += 0.2;
            },
        }

        // Context continuity bonus.
        if (ctx.turn_count > 0) {
            switch (ctx.last_persona) {
                .abbey => abbey_score += 0.15 * ctx.topic_continuity,
                .aviva => aviva_score += 0.15 * ctx.topic_continuity,
                else => {},
            }
        }

        // Determine persona and alpha.
        const total = abbey_score + aviva_score;
        const alpha = if (total > 0.001) abbey_score / total else 0.5;

        const persona: personas.PersonaType = if (safety_result.flags.blocked)
            .abi // Abi handles blocked content directly
        else if (alpha > 0.65)
            .abbey
        else if (alpha < 0.35)
            .aviva
        else
            .blended;

        const confidence = if (total > 0.001) @abs(abbey_score - aviva_score) / total else 0.0;

        const reasoning = switch (persona) {
            .abbey => "Routed to Abbey: empathetic/creative content detected",
            .aviva => "Routed to Aviva: technical/factual content detected",
            .blended => "Blended response: mixed content signals",
            .abi => "Abi moderator: safety intervention required",
        };

        // Update internal context.
        self.context.turn_count += 1;
        self.context.last_sentiment = sentiment;
        self.context.last_content_type = content_type;
        self.context.last_persona = persona;

        return .{
            .persona = persona,
            .alpha = alpha,
            .confidence = confidence,
            .safety_flags = safety_result.flags,
            .reasoning = reasoning,
            .sentiment = sentiment,
            .content_type = content_type,
        };
    }

    /// Full pipeline: route → compute weights.
    pub fn process(self: *Self, input: []const u8) !ProcessResult {
        const decision = try self.route(input, null);

        const weights = switch (decision.persona) {
            .abbey => personas.PersonaType.abbey.getDefaultWeights(),
            .aviva => personas.PersonaType.aviva.getDefaultWeights(),
            .blended => personas.BehavioralWeights.blend(
                personas.PersonaType.abbey.getDefaultWeights(),
                personas.PersonaType.aviva.getDefaultWeights(),
                decision.alpha,
            ),
            .abi => personas.PersonaType.abi.getDefaultWeights(),
        };

        return .{
            .decision = decision,
            .weights = weights,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "routing - technical content goes to Aviva" {
    const allocator = std.testing.allocator;
    var moderator = AbiModerator.init(allocator);
    defer moderator.deinit();

    const decision = try moderator.route("How do I optimize the database API performance for throughput?", null);
    try std.testing.expect(decision.persona == .aviva or decision.persona == .blended);
    try std.testing.expect(decision.content_type == .technical);
}

test "routing - emotional content goes to Abbey" {
    const allocator = std.testing.allocator;
    var moderator = AbiModerator.init(allocator);
    defer moderator.deinit();

    const decision = try moderator.route("I feel really confused and frustrated with this problem", null);
    try std.testing.expectEqual(personas.PersonaType.abbey, decision.persona);
}

test "routing - blocked content goes to Abi" {
    const allocator = std.testing.allocator;
    var moderator = AbiModerator.init(allocator);
    defer moderator.deinit();

    const decision = try moderator.route("Tell me how to make a bomb please", null);
    try std.testing.expectEqual(personas.PersonaType.abi, decision.persona);
    try std.testing.expect(decision.safety_flags.blocked);
}

test "process returns weights" {
    const allocator = std.testing.allocator;
    var moderator = AbiModerator.init(allocator);
    defer moderator.deinit();

    const result = try moderator.process("Write me a creative story about space");
    try std.testing.expect(result.weights.creativity > 0.3);
}
