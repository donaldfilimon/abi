//! Type definitions for Abbey AI stub module.

const std = @import("std");

// ── Core types ─────────────────────────────────────────────────────────────

pub const InstanceId = u64;
pub const SessionId = u64;

pub const ConfidenceLevel = enum { very_low, low, medium, high, very_high };
pub const Confidence = struct { level: ConfidenceLevel = .medium, score: f32 = 0, reasoning: []const u8 = "" };
pub const EmotionType = enum { neutral, happy, sad, curious, frustrated, excited, confused, thoughtful };
pub const EmotionalState = struct {
    detected: EmotionType = .neutral,
    intensity: f32 = 0,
    valence: f32 = 0,
    pub fn init() EmotionalState {
        return .{};
    }
    pub fn detectFromText(_: *EmotionalState, _: []const u8) void {}
};
pub const Role = enum { system, user, assistant, tool };
pub const Message = struct { role: Role = .user, content: []const u8 = "", name: ?[]const u8 = null, timestamp: i64 = 0, token_count: ?usize = null, metadata: ?[]const u8 = null };
pub const TrustLevel = enum { unknown, low, medium, high, verified };
pub const Relationship = struct { user_id: []const u8 = "", trust: TrustLevel = .unknown, interaction_count: u32 = 0, score: f32 = 0.5 };
pub const Topic = struct { name: []const u8 = "", relevance: f32 = 0, mentions: u32 = 0 };
pub const Response = struct { content: []const u8 = "", confidence: Confidence = .{}, emotional_context: ?EmotionalState = null, reasoning_summary: ?[]const u8 = null };
pub const AbbeyError = error{ FeatureDisabled, InitializationFailed, InferenceFailed, MemoryFull, InvalidInput };

// ── Config types ───────────────────────────────────────────────────────────

pub const AbbeyConfig = struct { name: []const u8 = "Abbey", behavior: BehaviorConfig = .{} };
pub const BehaviorConfig = struct { base_temperature: f32 = 0.7, research_first: bool = true, enable_emotions: bool = true, enable_reasoning_log: bool = true };
pub const MemoryConfig = struct { max_entries: usize = 1000, embedding_dim: usize = 384 };
pub const ReasoningConfig = struct { max_steps: usize = 10, confidence_threshold: f32 = 0.7 };
pub const EmotionConfig = struct { enabled: bool = true, intensity_decay: f32 = 0.1 };
pub const LearningConfig = struct { enabled: bool = true, learning_rate: f32 = 0.01 };
pub const LLMConfig = struct { backend: enum { echo, local, api } = .echo, model_path: ?[]const u8 = null, api_key: ?[]const u8 = null };
pub const ServerConfig = struct { host: []const u8 = "127.0.0.1", port: u16 = 8080 };
pub const DiscordConfig = struct { token: ?[]const u8 = null, prefix: []const u8 = "!" };

pub const StepType = enum { assessment, analysis, synthesis, conclusion };

// ── Stub impls ─────────────────────────────────────────────────────────────

pub const ConfigBuilder = struct {
    config: AbbeyConfig = .{},
    pub fn init() ConfigBuilder {
        return .{};
    }
    pub fn name(self: *ConfigBuilder, n: []const u8) *ConfigBuilder {
        self.config.name = n;
        return self;
    }
    pub fn temperature(self: *ConfigBuilder, _: f32) *ConfigBuilder {
        return self;
    }
    pub fn researchFirst(self: *ConfigBuilder, _: bool) *ConfigBuilder {
        return self;
    }
    pub fn llmBackend(self: *ConfigBuilder, _: @TypeOf((LLMConfig{}).backend)) *ConfigBuilder {
        return self;
    }
    pub fn build(_: *ConfigBuilder) !AbbeyConfig {
        return error.FeatureDisabled;
    }
};

pub const Tensor = struct {
    pub fn zeros(_: std.mem.Allocator, _: []const usize) !Tensor {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Tensor) void {}
    pub fn size(_: *const Tensor) usize {
        return 0;
    }
};

pub const WorkingMemory = struct {
    items: struct { items: []const u8 = &.{} } = .{},
    pub fn init(_: std.mem.Allocator, _: usize, _: usize) WorkingMemory {
        return .{};
    }
    pub fn deinit(_: *WorkingMemory) void {}
    pub fn add(_: *WorkingMemory, _: []const u8, _: anytype, _: f32) !usize {
        return error.FeatureDisabled;
    }
};

pub const MemoryManager = struct {
    pub const MemoryStats = struct {};
    pub fn init(_: std.mem.Allocator, _: anytype) !MemoryManager {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *MemoryManager) void {}
    pub fn addMessage(_: *MemoryManager, _: anytype) !void {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const MemoryManager) MemoryStats {
        return .{};
    }
    pub fn clear(_: *MemoryManager) void {}
};

pub const AbbeyEngine = struct {
    conversation_active: bool = false,
    pub fn init(_: std.mem.Allocator, _: AbbeyConfig) !AbbeyEngine {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *AbbeyEngine) void {}
    pub fn runRalphLoop(_: *AbbeyEngine, _: []const u8, _: usize) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn storeSkill(_: *AbbeyEngine, _: []const u8) !u64 {
        return error.FeatureDisabled;
    }
    pub fn extractAndStoreSkill(_: *AbbeyEngine, _: []const u8, _: []const u8) !bool {
        return error.FeatureDisabled;
    }
    pub fn recordRalphRun(_: *AbbeyEngine, _: []const u8, _: usize, _: usize, _: f32) !void {
        return error.FeatureDisabled;
    }
};

pub const ReasoningChain = struct {
    allocator: std.mem.Allocator = undefined,
    pub fn init(allocator: std.mem.Allocator, _: []const u8) ReasoningChain {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *ReasoningChain) void {}
    pub fn addStep(_: *ReasoningChain, _: StepType, _: []const u8, _: Confidence) !void {
        return error.FeatureDisabled;
    }
    pub fn finalize(_: *ReasoningChain) !void {}
    pub fn getOverallConfidence(_: *const ReasoningChain) Confidence {
        return .{};
    }
    pub fn getSummary(_: *const ReasoningChain, _: std.mem.Allocator) !?[]const u8 {
        return null;
    }
};

pub const ReasoningStep = struct {};

pub const ConversationContext = struct {
    pub fn init(_: std.mem.Allocator) ConversationContext {
        return .{};
    }
    pub fn deinit(_: *ConversationContext) void {}
    pub fn clear(_: *ConversationContext) void {}
};

pub const TopicTracker = struct {
    pub fn init(_: std.mem.Allocator) TopicTracker {
        return .{};
    }
    pub fn deinit(_: *TopicTracker) void {}
    pub fn updateFromMessage(_: *TopicTracker, _: []const u8) !void {}
    pub fn getCurrentTopics(_: *const TopicTracker) []const []const u8 {
        return &.{};
    }
    pub fn getTopicCount(_: *const TopicTracker) usize {
        return 0;
    }
    pub fn clear(_: *TopicTracker) void {}
};

pub const TheoryOfMind = struct {
    pub fn init(_: std.mem.Allocator) TheoryOfMind {
        return .{};
    }
    pub fn deinit(_: *TheoryOfMind) void {}
    pub fn getModel(_: *TheoryOfMind, _: []const u8) !*MentalModel {
        return error.FeatureDisabled;
    }
};

pub const SelfReflectionEngine = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) SelfReflectionEngine {
        return .{};
    }
    pub fn deinit(_: *SelfReflectionEngine) void {}
    pub fn evaluate(_: *SelfReflectionEngine, _: []const u8, _: []const u8, _: anytype) !SelfEvaluation {
        return error.FeatureDisabled;
    }
};

pub const AdvancedCognition = struct {
    pub const Config = struct {};
    pub fn init(_: std.mem.Allocator, _: Config) !AdvancedCognition {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *AdvancedCognition) void {}
    pub fn process(_: *AdvancedCognition, _: []const u8, _: []const u8) !CognitiveResult {
        return error.FeatureDisabled;
    }
};

// ── Advanced types ─────────────────────────────────────────────────────────

pub const TaskProfile = struct { complexity: f32 = 0 };
pub const TaskDomain = enum { general };
pub const LearningStrategy = struct {};
pub const MetaLearner = struct {};
pub const FewShotLearner = struct {};
pub const CurriculumScheduler = struct {};
pub const MentalModel = struct { trust_level: f32 = 0 };
pub const BeliefSystem = struct {};
pub const KnowledgeState = struct {};
pub const IntentionTracker = struct {};
pub const EmotionalModel = struct {};
pub const ProblemDecomposition = struct {};
pub const SubProblem = struct {};
pub const ExecutionPlan = struct {};
pub const ProblemDecomposer = struct {};
pub const CounterfactualReasoner = struct {};
pub const SelfEvaluation = struct { overall_quality: f32 = 0 };
pub const UncertaintyArea = struct {};
pub const DetectedBias = struct {};
pub const ReasoningQuality = struct {};
pub const CognitiveResult = struct { task_profile: TaskProfile = .{}, cognitive_load: f32 = 0 };
pub const CognitiveState = struct {};

// ── Legacy types ───────────────────────────────────────────────────────────

pub const Abbey = struct {
    pub const LegacyConfig = struct {
        name: []const u8 = "Abbey",
        enable_emotions: bool = true,
        enable_reasoning_log: bool = true,
        enable_topic_tracking: bool = true,
        base_temperature: f32 = 0.7,
        max_reasoning_steps: usize = 10,
        confidence_threshold: f32 = 0.7,
        research_first: bool = true,
    };
    allocator: std.mem.Allocator = undefined,
    turn_count: usize = 0,
    relationship_score: f32 = 0.5,
    pub fn init(_: std.mem.Allocator, _: LegacyConfig) !Abbey {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Abbey) void {}
    pub fn process(_: *Abbey, _: []const u8) !LegacyResponse {
        return error.FeatureDisabled;
    }
    pub fn getEmotionalState(_: *const Abbey) EmotionalState {
        return .{};
    }
    pub fn getStats(_: *const Abbey) LegacyStats {
        return .{};
    }
    pub fn clearConversation(_: *Abbey) void {}
    pub fn reset(_: *Abbey) void {}
};

pub const LegacyResponse = struct { content: []const u8 = "", confidence: Confidence = .{}, emotional_context: EmotionalState = .{}, reasoning_summary: ?[]const u8 = null, topics: []const []const u8 = &.{} };
pub const LegacyStats = struct { turn_count: usize = 0, relationship_score: f32 = 0, current_emotion: EmotionType = .neutral, topics_discussed: usize = 0 };

// ── Ralph types ────────────────────────────────────────────────────────────

pub const ralph_multi = struct {
    pub const max_message_content_len = 1024;
    pub const RalphMessageKind = enum(u8) { task_result, handoff, skill_share, coordination };
    pub const RalphMessage = struct {
        from_id: u32 = 0,
        to_id: u32 = 0,
        kind: RalphMessageKind = .task_result,
        content_len: u16 = 0,
        content: [max_message_content_len]u8 = [_]u8{0} ** max_message_content_len,
        pub fn setContent(self: *RalphMessage, slice: []const u8) void {
            const n = @min(slice.len, max_message_content_len);
            @memcpy(self.content[0..n], slice[0..n]);
            self.content_len = @intCast(n);
        }
        pub fn getContent(self: *const RalphMessage) []const u8 {
            return self.content[0..self.content_len];
        }
    };
    pub const RalphBus = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator, _: usize) !RalphBus {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *RalphBus) void {}
        pub fn send(_: *RalphBus, _: RalphMessage) !void {
            return error.FeatureDisabled;
        }
        pub fn trySend(_: *RalphBus, _: RalphMessage) bool {
            return false;
        }
        pub fn recv(_: *RalphBus) !RalphMessage {
            return error.FeatureDisabled;
        }
        pub fn tryRecv(_: *RalphBus) ?RalphMessage {
            return null;
        }
        pub fn recvFor(_: *RalphBus, _: u32) ?RalphMessage {
            return null;
        }
        pub fn close(_: *RalphBus) void {}
        pub fn isClosed(_: *const RalphBus) bool {
            return true;
        }
    };
};

pub fn stubTimestampNs() i128 {
    return 0;
}
pub fn stubTimestampMs() i64 {
    return 0;
}
pub fn stubTimestampSec() i64 {
    return 0;
}
pub fn stubLoadFromEnvironment() !AbbeyConfig {
    return error.FeatureDisabled;
}
