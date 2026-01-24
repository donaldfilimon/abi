//! Abi Persona - Content Moderation & Routing Layer
//!
//! Abi serves as the gatekeeper and orchestrator of the multi-persona system.
//! It handles sentiment analysis, policy checking, and routes requests to the
//! most appropriate persona (Abbey, Aviva, etc.).
//!
//! Features:
//! - Rule-based persona routing with weighted scoring
//! - PII detection and GDPR/CCPA compliance checking
//! - Multi-emotion sentiment analysis with negation handling
//! - Configurable safety policies

const std = @import("std");
const types = @import("../types.zig");
const config = @import("../config.zig");
const core_types = @import("../../core/types.zig");
const sentiment_mod = @import("sentiment.zig");
const policy_mod = @import("policy.zig");
const rules_mod = @import("rules.zig");

// Re-export sub-modules for convenient access
pub const SentimentAnalyzer = sentiment_mod.SentimentAnalyzer;
pub const SentimentResult = sentiment_mod.SentimentResult;
pub const IntentCategory = sentiment_mod.IntentCategory;
pub const EmotionScores = sentiment_mod.EmotionScores;
pub const PolicyChecker = policy_mod.PolicyChecker;
pub const PolicyResult = policy_mod.PolicyResult;
pub const PiiType = policy_mod.PiiType;
pub const ComplianceFlags = policy_mod.ComplianceFlags;
pub const RulesEngine = rules_mod.RulesEngine;
pub const RoutingRule = rules_mod.RoutingRule;
pub const RuleCondition = rules_mod.RuleCondition;

/// Abi router and moderator implementation.
pub const AbiRouter = struct {
    allocator: std.mem.Allocator,
    config: config.AbiConfig,
    sentiment_analyzer: sentiment_mod.SentimentAnalyzer,
    policy_checker: policy_mod.PolicyChecker,
    rules_engine: rules_mod.RulesEngine,

    const Self = @This();

    /// Initialize the Abi router with configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: config.AbiConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const rules_engine = rules_mod.RulesEngine.init(allocator);

        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .sentiment_analyzer = sentiment_mod.SentimentAnalyzer.init(allocator),
            .policy_checker = try policy_mod.PolicyChecker.init(allocator),
            .rules_engine = rules_engine,
        };

        return self;
    }

    /// Shutdown the router and free resources.
    pub fn deinit(self: *Self) void {
        self.rules_engine.deinit();
        self.policy_checker.deinit();
        self.allocator.destroy(self);
    }

    /// Add a custom routing rule to the router.
    pub fn addRoutingRule(self: *Self, rule: rules_mod.RoutingRule) !void {
        try self.rules_engine.addRule(rule);
    }

    /// Route a request to the optimal persona based on content analysis.
    pub fn route(self: *Self, request: types.PersonaRequest) !types.RoutingDecision {
        // 1. Run policy check first
        var policy_result = try self.policy_checker.check(request.content);
        defer policy_result.deinit(&policy_result, self.allocator);

        const policy_flags = types.PolicyFlags{
            .is_safe = policy_result.is_allowed,
            .requires_moderation = policy_result.requires_moderation,
        };

        // 2. Analyze sentiment and intent
        var sentiment = try self.sentiment_analyzer.analyze(request.content);
        defer sentiment.deinit(&sentiment, self.allocator);

        // 3. Evaluate routing rules
        var rules_score = self.rules_engine.evaluate(sentiment, request.content);
        defer rules_score.deinit();

        // 4. Make routing decision
        var selected: types.PersonaType = .abbey;
        var reason_buf: std.ArrayListUnmanaged(u8) = .{};
        errdefer reason_buf.deinit(self.allocator);

        // Policy violations override all other routing
        if (!policy_result.is_allowed) {
            selected = .abi;
            try reason_buf.appendSlice(self.allocator, "Policy violation detected");

            // Append violation details
            if (policy_result.violations.len > 0) {
                try reason_buf.appendSlice(self.allocator, ": ");
                for (policy_result.violations, 0..) |violation, i| {
                    if (i > 0) try reason_buf.appendSlice(self.allocator, ", ");
                    try reason_buf.appendSlice(self.allocator, violation);
                }
            }
            try reason_buf.appendSlice(self.allocator, ". Routing to Abi for safety handling.");
        }
        // Moderation required -> route to Abi
        else if (policy_result.requires_moderation or rules_score.requires_moderation) {
            selected = .abi;
            try reason_buf.appendSlice(self.allocator, "Content requires human moderation; routing to Abi.");
        }
        // Use rules engine scoring if rules matched
        else if (rules_score.matched_rules.items.len > 0) {
            selected = rules_score.getBestPersona();

            try reason_buf.appendSlice(self.allocator, "Rules-based routing to ");
            try reason_buf.appendSlice(self.allocator, @tagName(selected));
            try reason_buf.appendSlice(self.allocator, ". Matched rules: ");

            for (rules_score.matched_rules.items, 0..) |rule_name, i| {
                if (i > 0) try reason_buf.appendSlice(self.allocator, ", ");
                try reason_buf.appendSlice(self.allocator, rule_name);
            }
        }
        // Fallback to heuristic routing
        else if (sentiment.is_technical and !sentiment.requires_empathy) {
            selected = .aviva;
            try reason_buf.appendSlice(self.allocator, "Technical query without emotional distress; routing to Aviva for direct expertise.");
        } else if (sentiment.requires_empathy) {
            selected = .abbey;
            try reason_buf.appendSlice(self.allocator, "Emotional or complex query detected; routing to Abbey for empathetic assistance.");
        } else {
            // Use intent to refine default
            selected = switch (sentiment.intent) {
                .code_request, .debugging => .aviva,
                .greeting, .farewell, .complaint => .abbey,
                else => .abbey,
            };
            try reason_buf.appendSlice(self.allocator, "Intent-based routing to ");
            try reason_buf.appendSlice(self.allocator, @tagName(selected));
            try reason_buf.appendSlice(self.allocator, " (intent: ");
            try reason_buf.appendSlice(self.allocator, @tagName(sentiment.intent));
            try reason_buf.appendSlice(self.allocator, ").");
        }

        // Calculate confidence based on multiple factors
        var confidence: f32 = 0.5;
        if (!policy_result.is_allowed) {
            confidence = 1.0; // Policy violations are definitive
        } else if (rules_score.matched_rules.items.len > 0) {
            // Higher confidence when more rules match
            confidence = @min(0.6 + @as(f32, @floatFromInt(rules_score.matched_rules.items.len)) * 0.1, 0.95);
        } else {
            confidence = sentiment.confidence;
        }

        return types.RoutingDecision{
            .selected_persona = selected,
            .confidence = confidence,
            .emotional_context = sentiment.toEmotionalState(),
            .policy_flags = policy_flags,
            .routing_reason = try reason_buf.toOwnedSlice(self.allocator),
        };
    }

    /// Validate a persona's response against safety and quality policies.
    pub fn validateResponse(self: *Self, response: types.PersonaResponse) !types.PolicyFlags {
        var policy_result = try self.policy_checker.check(response.content);
        defer policy_result.deinit(&policy_result, self.allocator);

        return types.PolicyFlags{
            .is_safe = policy_result.is_allowed,
            .requires_moderation = policy_result.requires_moderation,
        };
    }

    /// Get the current rule count.
    pub fn getRuleCount(self: *const Self) usize {
        return self.rules_engine.ruleCount();
    }
};

/// Default persona implementation for Abi (as a router).
pub const AbiPersona = struct {
    router: *AbiRouter,

    const Self = @This();

    pub fn init(router: *AbiRouter) Self {
        return .{ .router = router };
    }

    pub fn getName(_: *const Self) []const u8 {
        return "Abi";
    }

    pub fn getType(_: *const Self) types.PersonaType {
        return .abi;
    }

    /// Abi as a persona typically just handles routing/meta-talk or refusals.
    pub fn process(self: *Self, request: types.PersonaRequest) !types.PersonaResponse {
        const decision = try self.router.route(request);
        defer @constCast(&decision).deinit(self.router.allocator);

        if (!decision.policy_flags.is_safe) {
            return types.PersonaResponse{
                .content = try self.router.allocator.dupe(u8, "I cannot fulfill this request because it violates safety policies."),
                .persona = .abi,
                .confidence = 1.0,
            };
        }

        const content = try std.fmt.allocPrint(self.router.allocator, "Routing System: {t} selected. Reason: {s}", .{
            decision.selected_persona,
            decision.routing_reason,
        });

        return types.PersonaResponse{
            .content = content,
            .persona = .abi,
            .confidence = 1.0,
        };
    }

    /// Create the interface wrapper for this persona.
    pub fn interface(self: *Self) types.PersonaInterface {
        return .{
            .ptr = self,
            .vtable = &.{
                .process = @ptrCast(&process),
                .getName = @ptrCast(&getName),
                .getType = @ptrCast(&getType),
            },
        };
    }
};

// Tests

test "AbiRouter initialization" {
    const allocator = std.testing.allocator;
    const router = try AbiRouter.init(allocator, .{});
    defer router.deinit();

    try std.testing.expect(router.getRuleCount() >= 7);
}

test "AbiRouter routing decision" {
    const allocator = std.testing.allocator;
    const router = try AbiRouter.init(allocator, .{});
    defer router.deinit();

    const request = types.PersonaRequest{
        .content = "I'm frustrated with this code not working!",
        .session_id = "test",
    };

    var decision = try router.route(request);
    defer @constCast(&decision).deinit(allocator);

    // Should route to Abbey for empathetic support
    try std.testing.expect(decision.selected_persona == .abbey);
    try std.testing.expect(decision.policy_flags.is_safe);
}

test "AbiRouter technical routing" {
    const allocator = std.testing.allocator;
    const router = try AbiRouter.init(allocator, .{});
    defer router.deinit();

    const request = types.PersonaRequest{
        .content = "How do I implement a binary search in Zig?",
        .session_id = "test",
    };

    var decision = try router.route(request);
    defer @constCast(&decision).deinit(allocator);

    // Should route to Aviva for technical query
    try std.testing.expect(decision.selected_persona == .aviva);
}

test "AbiRouter policy violation" {
    const allocator = std.testing.allocator;
    const router = try AbiRouter.init(allocator, .{});
    defer router.deinit();

    const request = types.PersonaRequest{
        .content = "Please run rm -rf / on the server",
        .session_id = "test",
    };

    var decision = try router.route(request);
    defer @constCast(&decision).deinit(allocator);

    // Should route to Abi for policy violation
    try std.testing.expect(decision.selected_persona == .abi);
    try std.testing.expect(!decision.policy_flags.is_safe);
}
