//! AI Stub Module
//!
//! Stub implementation when AI is disabled at compile time.

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    AiDisabled,
    LlmDisabled,
    EmbeddingsDisabled,
    AgentsDisabled,
    TrainingDisabled,
    ModelNotFound,
    InferenceFailed,
    InvalidConfig,
};

// Sub-module stubs
pub const core = struct {};
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const training = @import("training/stub.zig");

// Stub types
pub const Agent = struct {};
pub const ModelRegistry = struct {
    pub fn init(_: std.mem.Allocator) ModelRegistry {
        return .{};
    }
};
pub const ModelInfo = struct {};
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = Error;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;

pub const Tool = struct {};
pub const ToolResult = struct {};
pub const ToolRegistry = struct {};
pub const TaskTool = struct {};
pub const Subagent = struct {};
pub const DiscordTools = struct {};

pub const transformer = struct {
    pub const TransformerConfig = struct {};
    pub const TransformerModel = struct {
        pub fn init(_: transformer.TransformerConfig) transformer.TransformerModel {
            return .{};
        }
    };
};
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

pub const streaming = struct {
    pub const StreamingGenerator = struct {};
    pub const StreamToken = struct {};
    pub const StreamState = enum { idle, generating, done };
    pub const GenerationConfig = struct {};
};
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;

pub const LlmEngine = struct {};
pub const LlmModel = struct {};
pub const LlmConfig = struct {};
pub const GgufFile = struct {};
pub const BpeTokenizer = struct {};

pub const prompts = struct {};
pub const PromptBuilder = struct {};
pub const Persona = struct {};
pub const PersonaType = enum { assistant, expert };
pub const PromptFormat = enum { plain, chat };

pub const abbey = struct {};
pub const Abbey = struct {};
pub const AbbeyConfig = struct {};
pub const AbbeyResponse = struct {};
pub const AbbeyStats = struct {};
pub const ReasoningChain = struct {};
pub const ReasoningStep = struct {};
pub const Confidence = struct {};
pub const ConfidenceLevel = enum { low, medium, high };
pub const EmotionalState = struct {};
pub const EmotionType = enum { neutral };
pub const ConversationContext = struct {};
pub const TopicTracker = struct {};

pub const explore = struct {};
pub const ExploreAgent = struct {};
pub const ExploreConfig = struct {};
pub const ExploreLevel = enum { shallow, deep };
pub const ExploreResult = struct {};
pub const Match = struct {};
pub const ExplorationStats = struct {};
pub const QueryIntent = enum { search };
pub const ParsedQuery = struct {};
pub const QueryUnderstanding = struct {};

pub const memory = struct {};
pub const federated = struct {};
pub const rag = struct {};
pub const templates = struct {};
pub const eval = struct {};

pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training };

    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) Error!*Context {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getLlm(_: *Context) Error!*llm.Context {
        return error.AiDisabled;
    }

    pub fn getEmbeddings(_: *Context) Error!*embeddings.Context {
        return error.AiDisabled;
    }

    pub fn getAgents(_: *Context) Error!*agents.Context {
        return error.AiDisabled;
    }

    pub fn getTraining(_: *Context) Error!*training.Context {
        return error.AiDisabled;
    }

    pub fn isSubFeatureEnabled(_: *Context, _: SubFeature) bool {
        return false;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isLlmEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.AiDisabled;
}

pub fn deinit() void {}

pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    _ = allocator;
    return .{};
}

pub fn train(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingReport {
    return error.AiDisabled;
}

pub fn trainWithResult(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingResult {
    return error.AiDisabled;
}

pub fn createAgent(_: std.mem.Allocator, _: []const u8) Error!Agent {
    return error.AiDisabled;
}

pub fn createTransformer(_: TransformerConfig) TransformerModel {
    return .{};
}

pub fn inferText(_: std.mem.Allocator, _: []const u8) Error![]u8 {
    return error.AiDisabled;
}

pub fn embedText(_: std.mem.Allocator, _: []const u8) Error![]f32 {
    return error.AiDisabled;
}

pub fn encodeTokens(_: std.mem.Allocator, _: []const u8) Error![]u32 {
    return error.AiDisabled;
}

pub fn decodeTokens(_: std.mem.Allocator, _: []const u32) Error![]u8 {
    return error.AiDisabled;
}
