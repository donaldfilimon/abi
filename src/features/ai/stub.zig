//! Stub for AI feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.AiDisabled for all operations.

const std = @import("std");

pub const AiError = error{
    AiDisabled,
};

// Top-level type definitions
pub const Agent = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: anytype) AiError!@This() {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn process(self: *@This(), input: []const u8, allocator: std.mem.Allocator) AiError![]u8 {
        _ = self;
        _ = input;
        _ = allocator;
        return error.AiDisabled;
    }

    pub fn chat(self: *@This(), message: []const u8, allocator: std.mem.Allocator) AiError![]u8 {
        _ = self;
        _ = message;
        _ = allocator;
        return error.AiDisabled;
    }
};

pub const AgentConfig = struct {
    name: []const u8 = "agent",
    max_context_length: usize = 4096,
    temperature: f32 = 0.7,
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn register(self: *@This(), info: ModelInfo) AiError!void {
        _ = self;
        _ = info;
        return error.AiDisabled;
    }

    pub fn get(self: *@This(), name: []const u8) ?ModelInfo {
        _ = self;
        _ = name;
        return null;
    }

    pub fn list(self: *@This(), allocator: std.mem.Allocator) AiError![]ModelInfo {
        _ = self;
        _ = allocator;
        return error.AiDisabled;
    }
};

pub const ModelInfo = struct {
    name: []const u8,
    version: []const u8 = "1.0",
    size_bytes: u64 = 0,
    model_type: []const u8 = "unknown",
};

pub const TrainingConfig = struct {
    epochs: u32 = 10,
    batch_size: u32 = 32,
    learning_rate: f32 = 0.001,
};

pub const TrainingReport = struct {
    epochs_completed: u32 = 0,
    final_loss: f32 = 0.0,
    duration_ms: u64 = 0,
};

pub const TrainingResult = struct {
    report: TrainingReport,
    model_path: ?[]const u8 = null,
};

pub const TrainError = error{
    AiDisabled,
    TrainingFailed,
};

pub const CheckpointStore = struct {
    pub fn init(allocator: std.mem.Allocator, path: []const u8) @This() {
        _ = allocator;
        _ = path;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn save(self: *@This(), checkpoint: Checkpoint) TrainError!void {
        _ = self;
        _ = checkpoint;
        return error.AiDisabled;
    }
    pub fn load(self: *@This(), epoch: u32) TrainError!Checkpoint {
        _ = self;
        _ = epoch;
        return error.AiDisabled;
    }
};

pub const Checkpoint = struct {
    epoch: u32 = 0,
    loss: f32 = 0.0,
    weights: ?[]const u8 = null,
};

pub const GradientAccumulator = struct {
    pub fn init(allocator: std.mem.Allocator, steps: u32) @This() {
        _ = allocator;
        _ = steps;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn accumulate(self: *@This(), gradients: anytype) void {
        _ = self;
        _ = gradients;
    }
    pub fn step(self: *@This()) bool {
        _ = self;
        return false;
    }
};

pub const Tool = struct {
    name: []const u8,
    description: []const u8 = "",
    handler: ?*const fn ([]const u8) AiError![]u8 = null,
};

pub const ToolResult = struct {
    success: bool = false,
    output: ?[]const u8 = null,
    error_message: ?[]const u8 = null,
};

pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn register(self: *@This(), tool: Tool) AiError!void {
        _ = self;
        _ = tool;
        return error.AiDisabled;
    }

    pub fn execute(self: *@This(), name: []const u8, input: []const u8) AiError!ToolResult {
        _ = self;
        _ = name;
        _ = input;
        return error.AiDisabled;
    }
};

pub const TaskTool = struct {
    pub fn init(allocator: std.mem.Allocator) @This() {
        _ = allocator;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const Subagent = struct {
    pub fn init(allocator: std.mem.Allocator, config: anytype) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const DiscordTools = struct {
    pub fn init(allocator: std.mem.Allocator) @This() {
        _ = allocator;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const ExploreAgent = struct {
    pub fn init(allocator: std.mem.Allocator, config: ExploreConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn explore(self: *@This(), query: []const u8) AiError!ExploreResult {
        _ = self;
        _ = query;
        return error.AiDisabled;
    }
};

pub const ExploreConfig = struct {
    level: ExploreLevel = .quick,
    max_results: u32 = 100,
};

pub const ExploreLevel = enum {
    quick,
    medium,
    thorough,
};

pub const ExploreResult = struct {
    matches: []Match = &.{},
    stats: ExplorationStats = .{},
};

pub const Match = struct {
    path: []const u8 = "",
    line: u32 = 0,
    content: []const u8 = "",
    score: f32 = 0.0,
};

pub const ExplorationStats = struct {
    files_searched: u32 = 0,
    matches_found: u32 = 0,
    duration_ms: u64 = 0,
};

pub const QueryIntent = enum {
    search,
    definition,
    usage,
    unknown,
};

pub const ParsedQuery = struct {
    intent: QueryIntent = .unknown,
    terms: []const []const u8 = &.{},
};

pub const QueryUnderstanding = struct {
    pub fn parse(query: []const u8) ParsedQuery {
        _ = query;
        return .{};
    }
};

pub const LlmEngine = struct {
    pub fn init(allocator: std.mem.Allocator, config: LlmConfig) AiError!@This() {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn generate(self: *@This(), prompt: []const u8, allocator: std.mem.Allocator) AiError![]u8 {
        _ = self;
        _ = prompt;
        _ = allocator;
        return error.AiDisabled;
    }
};

pub const LlmModel = struct {
    pub fn load(allocator: std.mem.Allocator, path: []const u8) AiError!@This() {
        _ = allocator;
        _ = path;
        return error.AiDisabled;
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const LlmConfig = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
};

pub const GgufFile = struct {
    pub fn open(allocator: std.mem.Allocator, path: []const u8) AiError!@This() {
        _ = allocator;
        _ = path;
        return error.AiDisabled;
    }
    pub fn close(self: *@This()) void {
        _ = self;
    }
};

pub const BpeTokenizer = struct {
    pub fn init(allocator: std.mem.Allocator) @This() {
        _ = allocator;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn encode(self: *@This(), text: []const u8, allocator: std.mem.Allocator) AiError![]u32 {
        _ = self;
        _ = text;
        _ = allocator;
        return error.AiDisabled;
    }
    pub fn decode(self: *@This(), tokens: []const u32, allocator: std.mem.Allocator) AiError![]u8 {
        _ = self;
        _ = tokens;
        _ = allocator;
        return error.AiDisabled;
    }
};

pub const TransformerConfig = struct {
    hidden_size: u32 = 768,
    num_layers: u32 = 12,
    num_heads: u32 = 12,
    vocab_size: u32 = 32000,
};

pub const TransformerModel = struct {
    config: TransformerConfig,

    pub fn init(config: TransformerConfig) @This() {
        return .{ .config = config };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn infer(self: *@This(), allocator: std.mem.Allocator, input: []const u8) AiError![]u8 {
        _ = self;
        _ = allocator;
        _ = input;
        return error.AiDisabled;
    }

    pub fn embed(self: *@This(), allocator: std.mem.Allocator, input: []const u8) AiError![]f32 {
        _ = self;
        _ = allocator;
        _ = input;
        return error.AiDisabled;
    }

    pub fn encode(self: @This(), allocator: std.mem.Allocator, input: []const u8) AiError![]u32 {
        _ = self;
        _ = allocator;
        _ = input;
        return error.AiDisabled;
    }

    pub fn decode(self: @This(), allocator: std.mem.Allocator, tokens: []const u32) AiError![]u8 {
        _ = self;
        _ = allocator;
        _ = tokens;
        return error.AiDisabled;
    }
};

// Sub-module namespaces (reference top-level types)
pub const agent = struct {
    pub const Agent = @import("stub.zig").Agent;
    pub const AgentConfig = @import("stub.zig").AgentConfig;
};

pub const model_registry = struct {
    pub const ModelRegistry = @import("stub.zig").ModelRegistry;
    pub const ModelInfo = @import("stub.zig").ModelInfo;
};

pub const training = struct {
    pub const TrainingConfig = @import("stub.zig").TrainingConfig;
    pub const TrainingReport = @import("stub.zig").TrainingReport;
    pub const TrainingResult = @import("stub.zig").TrainingResult;
    pub const TrainError = @import("stub.zig").TrainError;
    pub const CheckpointStore = @import("stub.zig").CheckpointStore;
    pub const Checkpoint = @import("stub.zig").Checkpoint;
    pub const GradientAccumulator = @import("stub.zig").GradientAccumulator;

    pub const trainAndReport = @import("stub.zig").train;
    pub const trainWithResult = @import("stub.zig").trainWithResult;
};

pub const federated = struct {
    pub const FederatedConfig = struct {
        num_clients: u32 = 10,
        rounds: u32 = 5,
    };

    pub fn startCoordinator(allocator: std.mem.Allocator, config: FederatedConfig) AiError!void {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }
};

pub const transformer = struct {
    pub const TransformerConfig = @import("stub.zig").TransformerConfig;
    pub const TransformerModel = @import("stub.zig").TransformerModel;
};

pub const streaming = struct {
    pub const StreamCallback = *const fn (chunk: []const u8) void;

    pub fn streamResponse(allocator: std.mem.Allocator, input: []const u8, callback: StreamCallback) AiError!void {
        _ = allocator;
        _ = input;
        _ = callback;
        return error.AiDisabled;
    }
};

pub const tools = struct {
    pub const Tool = @import("stub.zig").Tool;
    pub const ToolResult = @import("stub.zig").ToolResult;
    pub const ToolRegistry = @import("stub.zig").ToolRegistry;
    pub const TaskTool = @import("stub.zig").TaskTool;
    pub const Subagent = @import("stub.zig").Subagent;
    pub const DiscordTools = @import("stub.zig").DiscordTools;

    pub const registerDiscordTools = @import("stub.zig").registerDiscordTools;
};

pub const explore = struct {
    pub const ExploreAgent = @import("stub.zig").ExploreAgent;
    pub const ExploreConfig = @import("stub.zig").ExploreConfig;
    pub const ExploreLevel = @import("stub.zig").ExploreLevel;
    pub const ExploreResult = @import("stub.zig").ExploreResult;
    pub const Match = @import("stub.zig").Match;
    pub const ExplorationStats = @import("stub.zig").ExplorationStats;
    pub const QueryIntent = @import("stub.zig").QueryIntent;
    pub const ParsedQuery = @import("stub.zig").ParsedQuery;
    pub const QueryUnderstanding = @import("stub.zig").QueryUnderstanding;
};

pub const llm = struct {
    pub const Engine = @import("stub.zig").LlmEngine;
    pub const Model = @import("stub.zig").LlmModel;
    pub const InferenceConfig = @import("stub.zig").LlmConfig;
    pub const GgufFile = @import("stub.zig").GgufFile;
    pub const BpeTokenizer = @import("stub.zig").BpeTokenizer;
};

pub const memory = struct {
    pub const MemoryStore = struct {
        pub fn init(allocator: std.mem.Allocator) @This() {
            _ = allocator;
            return .{};
        }
        pub fn deinit(self: *@This()) void {
            _ = self;
        }
    };
};

// Prompts module stub - types defined at module level to avoid ambiguity
pub const PromptPersona = struct {
    name: []const u8 = "assistant",
    description: []const u8 = "",
    system_prompt: []const u8 = "",
    temperature: f32 = 0.7,
    persona_type: PromptPersonaType = .assistant,
};

pub const PromptPersonaType = enum {
    assistant,
    coder,
    writer,
    analyst,
    reviewer,
    docs,
    companion,
    minimal,
    abbey,
    custom,
};

pub const PromptRole = enum {
    system,
    user,
    assistant,
};

pub const PromptMessage = struct {
    role: PromptRole,
    content: []const u8,
};

pub const PromptFormat = enum {
    text,
    json,
    chatml,
    llama,
    raw,
};

pub const PromptBuilder = struct {
    allocator: std.mem.Allocator,
    persona: PromptPersona,
    messages: std.ArrayListUnmanaged(PromptMessage),

    pub fn init(allocator: std.mem.Allocator, persona_type: PromptPersonaType) @This() {
        _ = persona_type;
        return .{
            .allocator = allocator,
            .persona = .{},
            .messages = .{},
        };
    }

    pub fn initCustom(allocator: std.mem.Allocator, persona: PromptPersona) @This() {
        return .{
            .allocator = allocator,
            .persona = persona,
            .messages = .{},
        };
    }

    pub fn deinit(self: *@This()) void {
        self.messages.deinit(self.allocator);
    }

    pub fn addUserMessage(self: *@This(), content: []const u8) AiError!void {
        _ = self;
        _ = content;
        return error.AiDisabled;
    }

    pub fn addAssistantMessage(self: *@This(), content: []const u8) AiError!void {
        _ = self;
        _ = content;
        return error.AiDisabled;
    }

    pub fn addSystemMessage(self: *@This(), content: []const u8) AiError!void {
        _ = self;
        _ = content;
        return error.AiDisabled;
    }

    pub fn build(self: *@This(), format: PromptFormat) AiError![]u8 {
        _ = self;
        _ = format;
        return error.AiDisabled;
    }

    pub fn exportDebug(self: *@This()) AiError![]u8 {
        _ = self;
        return error.AiDisabled;
    }

    pub fn getPersona(self: *const @This()) PromptPersona {
        return self.persona;
    }
};

pub const prompts = struct {
    pub const Persona = PromptPersona;
    pub const PersonaType = PromptPersonaType;
    pub const Role = PromptRole;
    pub const Message = PromptMessage;
    pub const Format = PromptFormat;
    pub const Builder = PromptBuilder;

    pub fn getPersona(persona_type: PromptPersonaType) PromptPersona {
        _ = persona_type;
        return .{};
    }

    pub fn listPersonas() []const PromptPersonaType {
        return &.{};
    }

    pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
        return PromptBuilder.init(allocator, .assistant);
    }

    pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: PromptPersonaType) PromptBuilder {
        return PromptBuilder.init(allocator, persona_type);
    }

    pub fn createBuilderWithCustomPersona(allocator: std.mem.Allocator, persona: PromptPersona) PromptBuilder {
        return PromptBuilder.initCustom(allocator, persona);
    }
};

// Re-export prompt types at top level with original names
pub const Persona = PromptPersona;
pub const PersonaType = PromptPersonaType;
pub const getPersona = prompts.getPersona;
pub const listPersonas = prompts.listPersonas;

// Abbey module stub - types defined at module level to avoid ambiguity
pub const AbbeyConfidenceLevel = enum {
    very_low,
    low,
    medium,
    high,
    very_high,
};

pub const AbbeyConfidence = struct {
    level: AbbeyConfidenceLevel = .medium,
    score: f32 = 0.5,
    reasoning: []const u8 = "",
};

pub const AbbeyEmotionType = enum {
    neutral,
    happy,
    sad,
    curious,
    frustrated,
    excited,
    concerned,
    thoughtful,
};

pub const AbbeyEmotionalState = struct {
    detected: AbbeyEmotionType = .neutral,
    intensity: f32 = 0.0,
    confidence: f32 = 0.0,

    pub fn init() @This() {
        return .{};
    }

    pub fn detectFromText(self: *@This(), text: []const u8) void {
        _ = self;
        _ = text;
    }
};

pub const AbbeyStepType = enum {
    analysis,
    assessment,
    synthesis,
    conclusion,
};

pub const AbbeyReasoningStep = struct {
    step_type: AbbeyStepType = .analysis,
    description: []const u8 = "",
    confidence: AbbeyConfidence = .{},
};

pub const AbbeyReasoningChain = struct {
    allocator: std.mem.Allocator,
    query: []const u8,
    steps: std.ArrayListUnmanaged(AbbeyReasoningStep),

    pub fn init(allocator: std.mem.Allocator, query: []const u8) @This() {
        _ = query;
        return .{
            .allocator = allocator,
            .query = "",
            .steps = .{},
        };
    }

    pub fn deinit(self: *@This()) void {
        self.steps.deinit(self.allocator);
    }

    pub fn addStep(self: *@This(), step_type: AbbeyStepType, description: []const u8, confidence: AbbeyConfidence) AiError!void {
        _ = self;
        _ = step_type;
        _ = description;
        _ = confidence;
        return error.AiDisabled;
    }

    pub fn finalize(self: *@This()) AiError!void {
        _ = self;
        return error.AiDisabled;
    }

    pub fn getOverallConfidence(self: *const @This()) AbbeyConfidence {
        _ = self;
        return .{};
    }

    pub fn getSummary(self: *const @This(), allocator: std.mem.Allocator) AiError!?[]const u8 {
        _ = self;
        _ = allocator;
        return error.AiDisabled;
    }
};

pub const AbbeyConversationContext = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn clear(self: *@This()) void {
        _ = self;
    }
};

pub const AbbeyTopicTracker = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn updateFromMessage(self: *@This(), message: []const u8) AiError!void {
        _ = self;
        _ = message;
        return error.AiDisabled;
    }

    pub fn getCurrentTopics(self: *const @This()) []const []const u8 {
        _ = self;
        return &.{};
    }

    pub fn getTopicCount(self: *const @This()) usize {
        _ = self;
        return 0;
    }

    pub fn clear(self: *@This()) void {
        _ = self;
    }
};

pub const AbbeyResponse = struct {
    content: []const u8 = "",
    confidence: AbbeyConfidence = .{},
    emotional_context: AbbeyEmotionalState = .{},
    reasoning_summary: ?[]const u8 = null,
    topics: []const []const u8 = &.{},
};

pub const AbbeyStats = struct {
    turn_count: usize = 0,
    relationship_score: f32 = 0.5,
    current_emotion: AbbeyEmotionType = .neutral,
    topics_discussed: usize = 0,
};

pub const AbbeyConfig = struct {
    name: []const u8 = "Abbey",
    enable_emotions: bool = true,
    enable_reasoning_log: bool = true,
    enable_topic_tracking: bool = true,
    base_temperature: f32 = 0.7,
    max_reasoning_steps: usize = 10,
    confidence_threshold: f32 = 0.7,
    research_first: bool = true,
};

pub const AbbeyInstance = struct {
    allocator: std.mem.Allocator,
    config: AbbeyConfig,

    pub fn init(allocator: std.mem.Allocator, config: AbbeyConfig) AiError!@This() {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn process(self: *@This(), message: []const u8) AiError!AbbeyResponse {
        _ = self;
        _ = message;
        return error.AiDisabled;
    }

    pub fn getEmotionalState(self: *const @This()) AbbeyEmotionalState {
        _ = self;
        return .{};
    }

    pub fn getStats(self: *const @This()) AbbeyStats {
        _ = self;
        return .{};
    }

    pub fn clearConversation(self: *@This()) void {
        _ = self;
    }

    pub fn reset(self: *@This()) void {
        _ = self;
    }
};

pub const abbey = struct {
    pub const AbbeyError = error{
        AiDisabled,
        InvalidConfig,
        MemoryFull,
        ProcessingFailed,
    };

    pub const ConfidenceLevel = AbbeyConfidenceLevel;
    pub const Confidence = AbbeyConfidence;
    pub const EmotionType = AbbeyEmotionType;
    pub const EmotionalState = AbbeyEmotionalState;
    pub const StepType = AbbeyStepType;
    pub const ReasoningStep = AbbeyReasoningStep;
    pub const ReasoningChain = AbbeyReasoningChain;
    pub const ConversationContext = AbbeyConversationContext;
    pub const TopicTracker = AbbeyTopicTracker;
    pub const Response = AbbeyResponse;
    pub const Stats = AbbeyStats;
    pub const Config = AbbeyConfig;
    pub const Abbey = AbbeyInstance;

    // Re-export reasoning module types
    pub const reasoning = struct {
        pub const ConfidenceLevel = AbbeyConfidenceLevel;
        pub const Confidence = AbbeyConfidence;
        pub const ReasoningChain = AbbeyReasoningChain;
        pub const ReasoningStep = AbbeyReasoningStep;
        pub const StepType = AbbeyStepType;
    };

    // Convenience functions
    pub fn createEngine(allocator: std.mem.Allocator) AiError!AbbeyInstance {
        _ = allocator;
        return error.AiDisabled;
    }

    pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: AbbeyConfig) AiError!AbbeyInstance {
        _ = allocator;
        _ = config;
        return error.AiDisabled;
    }
};

// Re-export abbey types at top level with original names
pub const Abbey = AbbeyInstance;
pub const ReasoningChain = AbbeyReasoningChain;
pub const ReasoningStep = AbbeyReasoningStep;
pub const Confidence = AbbeyConfidence;
pub const ConfidenceLevel = AbbeyConfidenceLevel;
pub const EmotionalState = AbbeyEmotionalState;
pub const EmotionType = AbbeyEmotionType;
pub const ConversationContext = AbbeyConversationContext;
pub const TopicTracker = AbbeyTopicTracker;

// Top-level function (referenced by tools namespace)
pub fn registerDiscordTools(registry: *ToolRegistry, allocator: std.mem.Allocator) AiError!void {
    _ = registry;
    _ = allocator;
    return error.AiDisabled;
}

// Module lifecycle
var initialized: bool = false;

pub fn init(_: std.mem.Allocator) AiError!void {
    return error.AiDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

// Convenience functions
pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    _ = allocator;
    _ = config;
    return error.AiDisabled;
}

pub fn trainWithResult(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingResult {
    _ = allocator;
    _ = config;
    return error.AiDisabled;
}

pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) AiError!Agent {
    _ = allocator;
    _ = name;
    return error.AiDisabled;
}

pub fn createAgentWithConfig(allocator: std.mem.Allocator, config: AgentConfig) AiError!Agent {
    _ = allocator;
    _ = config;
    return error.AiDisabled;
}

pub fn processMessage(allocator: std.mem.Allocator, name: []const u8, message: []const u8) AiError![]u8 {
    _ = allocator;
    _ = name;
    _ = message;
    return error.AiDisabled;
}

pub fn createTransformer(config: TransformerConfig) TransformerModel {
    return TransformerModel.init(config);
}

pub fn inferText(allocator: std.mem.Allocator, input: []const u8) AiError![]u8 {
    _ = allocator;
    _ = input;
    return error.AiDisabled;
}

pub fn embedText(allocator: std.mem.Allocator, input: []const u8) AiError![]f32 {
    _ = allocator;
    _ = input;
    return error.AiDisabled;
}

pub fn encodeTokens(allocator: std.mem.Allocator, input: []const u8) AiError![]u32 {
    _ = allocator;
    _ = input;
    return error.AiDisabled;
}

pub fn decodeTokens(allocator: std.mem.Allocator, tokens: []const u32) AiError![]u8 {
    _ = allocator;
    _ = tokens;
    return error.AiDisabled;
}
