//! AI Stub Module
//!
//! This module provides API-compatible no-op implementations for all public AI
//! functions when the AI feature is disabled at compile time. All functions
//! return `error.AiDisabled` or empty/default values as appropriate.
//!
//! The AI module encompasses:
//! - LLM inference and text generation
//! - Embeddings and vector operations
//! - Agent systems and multi-agent coordination
//! - Training pipelines (LLM, Vision, CLIP)
//! - Document processing and RAG
//! - Streaming response handling
//! - Model discovery and management
//!
//! To enable the real implementation, build with `-Denable-ai=true`.

const std = @import("std");
const config_module = @import("../config/mod.zig");

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
pub const core = @import("core/mod.zig");
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const training = @import("training/stub.zig");
pub const database = @import("database/stub.zig");
pub const documents = @import("documents/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const models = @import("models/stub.zig");
pub const memory = @import("memory/stub.zig");
pub const streaming = @import("streaming/stub.zig");
pub const explore = @import("explore/stub.zig");
pub const personas = @import("personas/stub.zig");
pub const rag = @import("rag/stub.zig");
pub const templates = @import("templates/stub.zig");
pub const eval = @import("eval/stub.zig");
pub const federated = struct {};

// Local stubs moved to src/ai/stubs/
pub const agent = @import("stubs/agent.zig");
pub const model_registry = @import("stubs/model_registry.zig");
pub const tools = @import("stubs/tools.zig");
pub const transformer = @import("stubs/transformer.zig");
pub const gpu_agent = @import("stubs/gpu_agent.zig");
pub const discovery = @import("stubs/discovery.zig");
pub const prompts = @import("stubs/prompts.zig");
pub const abbey = @import("stubs/abbey.zig");

// Multi-agent re-exports
pub const MultiAgentCoordinator = multi_agent.Coordinator;

// Stub types re-exports
pub const Agent = agent.Agent;
pub const ModelRegistry = model_registry.ModelRegistry;
pub const ModelInfo = model_registry.ModelInfo;
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = training.TrainError;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const TrainableModel = training.TrainableModel;
pub const trainable_model = training.trainable_model;
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;
pub const loadCheckpoint = training.loadCheckpoint;
pub const saveCheckpoint = training.saveCheckpoint;
pub const TrainableViTModel = training.TrainableViTModel;
pub const TrainableViTConfig = training.TrainableViTConfig;
pub const TrainableViTWeights = training.TrainableViTWeights;
pub const VisionTrainingError = training.VisionTrainingError;
pub const TrainableCLIPModel = training.TrainableCLIPModel;
pub const CLIPTrainingConfig = training.CLIPTrainingConfig;
pub const MultimodalTrainingError = training.MultimodalTrainingError;
pub const TokenizedDataset = training.TokenizedDataset;
pub const DataLoader = training.DataLoader;
pub const BatchIterator = training.BatchIterator;
pub const Batch = training.Batch;
pub const SequencePacker = training.SequencePacker;
pub const parseInstructionDataset = training.parseInstructionDataset;
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const exportGguf = database.exportGguf;

// Tools
pub const Tool = tools.Tool;
pub const ToolResult = tools.ToolResult;
pub const ToolRegistry = tools.ToolRegistry;
pub const TaskTool = tools.TaskTool;
pub const Subagent = tools.Subagent;
pub const DiscordTools = tools.DiscordTools;
pub const OsTools = tools.OsTools;
pub const registerDiscordTools = tools.registerDiscordTools;
pub const registerOsTools = tools.registerOsTools;

// Transformer
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

// GPU Agent
pub const GpuAgent = gpu_agent.GpuAgent;
pub const GpuAwareRequest = gpu_agent.GpuAwareRequest;
pub const GpuAwareResponse = gpu_agent.GpuAwareResponse;
pub const WorkloadType = gpu_agent.WorkloadType;
pub const GpuAgentPriority = gpu_agent.Priority;
pub const GpuAgentStats = gpu_agent.AgentStats;

// Discovery
pub const ModelDiscovery = discovery.ModelDiscovery;
pub const DiscoveredModel = discovery.DiscoveredModel;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const SystemCapabilities = discovery.SystemCapabilities;
pub const AdaptiveConfig = discovery.AdaptiveConfig;
pub const ModelRequirements = discovery.ModelRequirements;
pub const WarmupResult = discovery.WarmupResult;
pub const detectCapabilities = discovery.detectCapabilities;
pub const runWarmup = discovery.runWarmup;

// Streaming
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;
pub const ServerConfig = streaming.ServerConfig;
pub const StreamingServer = streaming.StreamingServer;
pub const StreamingServerError = streaming.StreamingServerError;
pub const BackendType = streaming.BackendType;

pub const LlmEngine = llm.Engine;
pub const LlmModel = llm.Model;
pub const LlmConfig = llm.InferenceConfig;
pub const GgufFile = llm.GgufFile;
pub const BpeTokenizer = llm.BpeTokenizer;

// Prompts
pub const PromptBuilder = prompts.PromptBuilder;
pub const Persona = prompts.Persona;
pub const PersonaType = prompts.PersonaType;
pub const PromptFormat = prompts.PromptFormat;

// Abbey
pub const AbbeyInstance = abbey.AbbeyInstance;
pub const Abbey = abbey.Abbey;
pub const AbbeyConfig = core.AbbeyConfig;
pub const AbbeyResponse = core.Response;
pub const AbbeyStats = abbey.Stats;
pub const ReasoningChain = abbey.ReasoningChain;
pub const ReasoningStep = abbey.ReasoningStep;
pub const Confidence = core.Confidence;
pub const ConfidenceLevel = core.ConfidenceLevel;
pub const EmotionalState = core.EmotionalState;
pub const EmotionType = core.EmotionType;
pub const ConversationContext = abbey.ConversationContext;
pub const TopicTracker = core.Topic;

// Explore
pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;

// Orchestration
pub const Orchestrator = orchestration.Orchestrator;
pub const OrchestrationConfig = orchestration.OrchestrationConfig;
pub const OrchestrationError = orchestration.OrchestrationError;
pub const RoutingStrategy = orchestration.RoutingStrategy;
pub const TaskType = orchestration.TaskType;
pub const RouteResult = orchestration.RouteResult;
pub const EnsembleMethod = orchestration.EnsembleMethod;
pub const EnsembleResult = orchestration.EnsembleResult;
pub const FallbackPolicy = orchestration.FallbackPolicy;
pub const HealthStatus = orchestration.HealthStatus;
pub const ModelBackend = orchestration.ModelBackend;
pub const ModelCapability = orchestration.Capability;
pub const OrchestrationModelConfig = orchestration.ModelConfig;

// Document Understanding
pub const DocumentPipeline = documents.DocumentPipeline;
pub const Document = documents.Document;
pub const DocumentFormat = documents.DocumentFormat;
pub const DocumentElement = documents.DocumentElement;
pub const ElementType = documents.ElementType;
pub const TextSegment = documents.TextSegment;
pub const TextSegmenter = documents.TextSegmenter;
pub const NamedEntity = documents.NamedEntity;
pub const EntityType = documents.EntityType;
pub const EntityExtractor = documents.EntityExtractor;
pub const LayoutAnalyzer = documents.LayoutAnalyzer;
pub const PipelineConfig = documents.PipelineConfig;
pub const SegmentationConfig = documents.SegmentationConfig;

// Context
pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training, personas };

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

    pub fn getPersonas(_: *Context) Error!*personas.Context {
        return error.AiDisabled;
    }

    pub fn isSubFeatureEnabled(_: *Context, _: SubFeature) bool {
        return false;
    }

    pub fn getDiscoveredModels(_: *Context) []discovery.DiscoveredModel {
        return &.{};
    }

    pub fn discoveredModelCount(_: *Context) usize {
        return 0;
    }

    pub fn findBestModel(_: *Context, _: discovery.ModelRequirements) ?*discovery.DiscoveredModel {
        return null;
    }

    pub fn generateAdaptiveConfig(_: *Context, _: *const discovery.DiscoveredModel) discovery.AdaptiveConfig {
        return .{};
    }

    pub fn getCapabilities(_: *const Context) discovery.SystemCapabilities {
        return .{};
    }

    pub fn addModelPath(_: *Context, _: []const u8) !void {
        return error.AiDisabled;
    }

    pub fn addModelWithSize(_: *Context, _: []const u8, _: u64) !void {
        return error.AiDisabled;
    }

    pub fn clearDiscoveredModels(_: *Context) void {}
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

// Note: loadCheckpoint is re-exported from training module at line 70
