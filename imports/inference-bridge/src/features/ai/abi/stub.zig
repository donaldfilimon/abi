//! Abi Profile stub — disabled at compile time.
//!
//! API-compatible no-ops for content moderation & routing layer.

const std = @import("std");
const types = @import("../types.zig");
const config = @import("../config.zig");

// --- Sentiment types (inline stubs) ---

pub const SentimentResult = struct {
    primary_emotion: types.EmotionType = .neutral,
    secondary_emotions: []const types.EmotionType = &.{},
    urgency_score: f32 = 0.0,
    confidence: f32 = 0.0,
    requires_empathy: bool = false,
    is_technical: bool = false,
    intent: IntentCategory = .general,
    emotion_scores: ?EmotionScores = null,

    pub fn deinit(_: *SentimentResult, _: std.mem.Allocator) void {}

    pub fn toEmotionalState(_: SentimentResult) types.EmotionalState {
        return .{};
    }
};

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

pub const EmotionScores = struct {
    neutral: f32 = 0.0,
    enthusiastic: f32 = 0.0,
    grateful: f32 = 0.0,
    frustrated: f32 = 0.0,
    confused: f32 = 0.0,
    excited: f32 = 0.0,
    anxious: f32 = 0.0,
    stressed: f32 = 0.0,
    curious: f32 = 0.0,
    disappointed: f32 = 0.0,
};

// --- Policy types (inline stubs) ---

pub const PolicyResult = struct {
    is_allowed: bool = true,
    requires_moderation: bool = false,
    violations: []const []const u8 = &.{},
    suggested_action: SafetyAction = .allow,
    detected_pii: []const PiiType = &.{},
    compliance: ComplianceFlags = .{},

    pub fn deinit(_: *PolicyResult, _: std.mem.Allocator) void {}
};

pub const PiiType = enum {
    email,
    phone,
    ssn,
    credit_card,
    ip_address,
    address,
    name_pattern,
    date_of_birth,
};

pub const ComplianceFlags = struct {
    gdpr_compliant: bool = true,
    ccpa_compliant: bool = true,
    hipaa_relevant: bool = false,
    contains_consent_required: bool = false,
};

const SafetyAction = enum {
    allow,
    warn,
    block,
    redirect_to_support,
    require_human_review,
};

// --- Rules types (inline stubs) ---

pub const RuleCondition = struct {
    condition_type: ConditionType = .always,
    threshold: f32 = 0.0,
    string_value: ?[]const u8 = null,
    emotion: ?types.EmotionType = null,

    pub fn evaluate(_: RuleCondition, _: SentimentResult, _: []const u8) bool {
        return false;
    }
};

const ConditionType = enum {
    urgency_above,
    urgency_below,
    emotion_matches,
    is_technical,
    requires_empathy,
    contains_keyword,
    pattern_matches,
    always,
};

pub const RoutingRule = struct {
    name: []const u8 = "",
    condition: RuleCondition = .{},
    priority: u8 = 5,
    profile_adjustments: ProfileAdjustments = .{},
    enabled: bool = true,
};

const ProfileAdjustments = struct {
    abbey_boost: f32 = 0.0,
    aviva_boost: f32 = 0.0,
    abi_boost: f32 = 0.0,
    block_abbey: bool = false,
    block_aviva: bool = false,
};

const RoutingRulesScore = struct {
    allocator: std.mem.Allocator,
    abbey_boost: f32 = 0.0,
    aviva_boost: f32 = 0.0,
    abi_boost: f32 = 0.0,
    requires_moderation: bool = false,
    matched_rules: std.ArrayListUnmanaged([]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) RoutingRulesScore {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *RoutingRulesScore) void {}

    pub fn getBestProfile(_: RoutingRulesScore) types.ProfileType {
        return .abbey;
    }
};

// --- Stub: SentimentAnalyzer ---

pub const SentimentAnalyzer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SentimentAnalyzer {
        return .{ .allocator = allocator };
    }

    pub fn analyze(_: *const SentimentAnalyzer, _: []const u8) !SentimentResult {
        return error.FeatureDisabled;
    }
};

// --- Stub: PolicyChecker ---

pub const PolicyChecker = struct {
    allocator: std.mem.Allocator,
    rules: std.ArrayListUnmanaged(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) !PolicyChecker {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *PolicyChecker) void {}

    pub fn addRule(_: *PolicyChecker, _: anytype) !void {
        return error.FeatureDisabled;
    }

    pub fn check(_: *const PolicyChecker, _: []const u8) !PolicyResult {
        return error.FeatureDisabled;
    }
};

// --- Stub: RulesEngine ---

pub const RulesEngine = struct {
    allocator: std.mem.Allocator,
    rules: std.ArrayListUnmanaged(RoutingRule) = .empty,

    pub fn init(allocator: std.mem.Allocator) RulesEngine {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *RulesEngine) void {}

    pub fn addRule(_: *RulesEngine, _: RoutingRule) !void {
        return error.FeatureDisabled;
    }

    pub fn evaluate(_: *const RulesEngine, _: SentimentResult, _: []const u8) RoutingRulesScore {
        return RoutingRulesScore.init(std.heap.page_allocator);
    }

    pub fn ruleCount(_: *const RulesEngine) usize {
        return 0;
    }
};

// --- Stub: AbiRouter ---

pub const AbiRouter = struct {
    allocator: std.mem.Allocator,
    config: config.AbiConfig,
    sentiment_analyzer: SentimentAnalyzer,
    policy_checker: PolicyChecker,
    rules_engine: RulesEngine,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: config.AbiConfig) !*Self {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn addRoutingRule(_: *Self, _: RoutingRule) !void {
        return error.FeatureDisabled;
    }

    pub fn route(_: *Self, _: types.ProfileRequest) !types.RoutingDecision {
        return error.FeatureDisabled;
    }

    pub fn validateResponse(_: *Self, _: types.ProfileResponse) !types.PolicyFlags {
        return error.FeatureDisabled;
    }

    pub fn getRuleCount(_: *const Self) usize {
        return 0;
    }
};

// --- Stub: AbiProfile ---

pub const AbiProfile = struct {
    router: *AbiRouter,

    const Self = @This();

    pub fn init(router: *AbiRouter) Self {
        return .{ .router = router };
    }

    pub fn getName(_: *const Self) []const u8 {
        return "Abi";
    }

    pub fn getType(_: *const Self) types.ProfileType {
        return .abi;
    }

    pub fn process(_: *Self, _: types.ProfileRequest) !types.ProfileResponse {
        return error.FeatureDisabled;
    }

    pub fn interface(_: *Self) types.ProfileInterface {
        return .{
            .ptr = undefined,
            .vtable = &.{
                .process = @ptrCast(&struct {
                    fn f(_: *anyopaque, _: types.ProfileRequest) anyerror!types.ProfileResponse {
                        return error.FeatureDisabled;
                    }
                }.f),
                .getName = @ptrCast(&struct {
                    fn f(_: *anyopaque) []const u8 {
                        return "Abi";
                    }
                }.f),
                .getType = @ptrCast(&struct {
                    fn f(_: *anyopaque) types.ProfileType {
                        return .abi;
                    }
                }.f),
            },
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
