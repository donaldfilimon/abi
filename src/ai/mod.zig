//! AI Module - Public API
//!
//! This is the primary entry point for AI functionality. Import from here for
//! Framework integration and the stable public API.
//!
//! Modular AI capabilities organized as independent sub-features:
//!
//! - **core**: Shared types, interfaces, and utilities (always available when AI enabled)
//! - **llm**: Local LLM inference (GGUF, transformer models)
//! - **embeddings**: Vector embeddings generation
//! - **agents**: AI agent runtime and tools
//! - **training**: Model training pipelines
//!
//! Each sub-feature can be independently enabled/disabled.
//!
//! ## Usage
//!
//! ```zig
//! const ai = @import("ai/mod.zig");
//!
//! // Initialize AI context
//! var ctx = try ai.Context.init(allocator, .{
//!     .llm = .{ .model_path = "./models/llama.gguf" },
//! });
//! defer ctx.deinit();
//!
//! // Use LLM
//! const response = try ctx.getLlm().generate("Hello, world!");
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// ============================================================================
// Direct module imports (no longer using implementation_mod.zig bridge)
// ============================================================================

// Direct imports for AI submodules
pub const agent = @import("agent.zig");
pub const model_registry = @import("model_registry.zig");
pub const transformer = @import("transformer/mod.zig");
pub const streaming = @import("streaming/mod.zig");
pub const tools = @import("tools/mod.zig");
pub const prompts = @import("prompts/mod.zig");
pub const abbey = @import("abbey/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const federated = @import("federated/mod.zig");
pub const rag = if (build_options.enable_ai) @import("rag/mod.zig") else @import("rag/stub.zig");
pub const templates = if (build_options.enable_ai) @import("templates/mod.zig") else @import("templates/stub.zig");
pub const eval = if (build_options.enable_ai) @import("eval/mod.zig") else @import("eval/stub.zig");
pub const explore = if (build_options.enable_explore) @import("explore/mod.zig") else @import("explore/stub.zig");

// ============================================================================
// Sub-modules (conditionally compiled)
// ============================================================================

/// Core AI types and utilities (always available when AI enabled)
pub const core = @import("core/mod.zig");

/// LLM inference module
pub const llm = if (build_options.enable_llm)
    @import("llm/mod.zig")
else
    @import("llm/stub.zig");

/// Embeddings generation module
pub const embeddings = if (build_options.enable_ai)
    @import("embeddings/mod.zig")
else
    @import("embeddings/stub.zig");

/// Agent runtime module
pub const agents = if (build_options.enable_ai)
    @import("agents/mod.zig")
else
    @import("agents/stub.zig");

/// Training pipelines module
pub const training = if (build_options.enable_ai)
    @import("training/mod.zig")
else
    @import("training/stub.zig");

/// Vision/image processing module
pub const vision = if (build_options.enable_vision)
    @import("vision/mod.zig")
else
    @import("vision/stub.zig");

// ============================================================================
// Re-exports for backward compatibility
// ============================================================================

// Agent types
pub const Agent = agent.Agent;

// Model registry types
pub const ModelRegistry = model_registry.ModelRegistry;
pub const ModelInfo = model_registry.ModelInfo;

// Training types (from training module)
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = training.TrainError;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const trainable_model = training.trainable_model;
pub const TrainableModel = training.TrainableModel;
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const loadCheckpoint = training.loadCheckpoint;
pub const saveCheckpoint = training.saveCheckpoint;

// Data loading
pub const TokenizedDataset = training.TokenizedDataset;
pub const DataLoader = training.DataLoader;
pub const BatchIterator = training.BatchIterator;
pub const Batch = training.Batch;
pub const SequencePacker = training.SequencePacker;
pub const parseInstructionDataset = training.parseInstructionDataset;

// Tools
pub const Tool = tools.Tool;
pub const ToolResult = tools.ToolResult;
pub const ToolRegistry = tools.ToolRegistry;
pub const TaskTool = tools.TaskTool;
pub const Subagent = tools.Subagent;
pub const DiscordTools = tools.DiscordTools;
pub const registerDiscordTools = tools.registerDiscordTools;
pub const OsTools = tools.OsTools;
pub const registerOsTools = tools.registerOsTools;

// Transformer
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

// Streaming
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;

// LLM Engine
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

// Abbey / Core AI
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

// ============================================================================
// Errors
// ============================================================================

pub const Error = error{
    /// AI feature is disabled at compile time
    AiDisabled,
    /// LLM sub-feature is disabled
    LlmDisabled,
    /// Embeddings sub-feature is disabled
    EmbeddingsDisabled,
    /// Agents sub-feature is disabled
    AgentsDisabled,
    /// Training sub-feature is disabled
    TrainingDisabled,
    /// Model not found
    ModelNotFound,
    /// Inference failed
    InferenceFailed,
    /// Invalid configuration
    InvalidConfig,
};

// ============================================================================
// Context - Unified interface for Framework integration
// ============================================================================

/// AI context for Framework integration.
/// Manages AI sub-features based on configuration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,

    // Sub-feature contexts (null if disabled)
    llm_ctx: ?*llm.Context = null,
    embeddings_ctx: ?*embeddings.Context = null,
    agents_ctx: ?*agents.Context = null,
    training_ctx: ?*training.Context = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AiConfig) !*Context {
        if (!isEnabled()) return error.AiDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        // Initialize enabled sub-features
        errdefer ctx.deinitSubFeatures();

        if (cfg.llm) |llm_cfg| {
            ctx.llm_ctx = try llm.Context.init(allocator, llm_cfg);
        }

        if (cfg.embeddings) |emb_cfg| {
            ctx.embeddings_ctx = try embeddings.Context.init(allocator, emb_cfg);
        }

        if (cfg.agents) |agent_cfg| {
            ctx.agents_ctx = try agents.Context.init(allocator, agent_cfg);
        }

        if (cfg.training) |train_cfg| {
            ctx.training_ctx = try training.Context.init(allocator, train_cfg);
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.deinitSubFeatures();
        self.allocator.destroy(self);
    }

    fn deinitSubFeatures(self: *Context) void {
        if (self.training_ctx) |t| {
            t.deinit();
            self.training_ctx = null;
        }
        if (self.agents_ctx) |a| {
            a.deinit();
            self.agents_ctx = null;
        }
        if (self.embeddings_ctx) |e| {
            e.deinit();
            self.embeddings_ctx = null;
        }
        if (self.llm_ctx) |l| {
            l.deinit();
            self.llm_ctx = null;
        }
    }

    /// Get LLM context (returns error if not enabled).
    pub fn getLlm(self: *Context) Error!*llm.Context {
        return self.llm_ctx orelse error.LlmDisabled;
    }

    /// Get embeddings context (returns error if not enabled).
    pub fn getEmbeddings(self: *Context) Error!*embeddings.Context {
        return self.embeddings_ctx orelse error.EmbeddingsDisabled;
    }

    /// Get agents context (returns error if not enabled).
    pub fn getAgents(self: *Context) Error!*agents.Context {
        return self.agents_ctx orelse error.AgentsDisabled;
    }

    /// Get training context (returns error if not enabled).
    pub fn getTraining(self: *Context) Error!*training.Context {
        return self.training_ctx orelse error.TrainingDisabled;
    }

    /// Check if a sub-feature is enabled.
    pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool {
        return switch (feature) {
            .llm => self.llm_ctx != null,
            .embeddings => self.embeddings_ctx != null,
            .agents => self.agents_ctx != null,
            .training => self.training_ctx != null,
        };
    }

    pub const SubFeature = enum {
        llm,
        embeddings,
        agents,
        training,
    };
};

// ============================================================================
// Module state
// ============================================================================

var initialized: bool = false;

// ============================================================================
// Module-level functions
// ============================================================================

/// Check if AI is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

/// Check if LLM is enabled at compile time.
pub fn isLlmEnabled() bool {
    return build_options.enable_llm;
}

/// Check if AI module is initialized.
pub fn isInitialized() bool {
    return initialized;
}

/// Initialize the AI module (legacy compatibility).
pub fn init(allocator: std.mem.Allocator) Error!void {
    _ = allocator;
    if (!isEnabled()) return error.AiDisabled;
    initialized = true;
}

/// Deinitialize the AI module (legacy compatibility).
pub fn deinit() void {
    initialized = false;
}

// Legacy convenience functions
pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    return training.trainAndReport(allocator, config);
}

pub fn trainWithResult(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingResult {
    return training.trainWithResult(allocator, config);
}

pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !Agent {
    if (!isEnabled()) return error.AiDisabled;
    return agent.Agent.init(allocator, .{ .name = name });
}

pub fn createTransformer(config: TransformerConfig) TransformerModel {
    return transformer.TransformerModel.init(config);
}

pub fn inferText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (!isEnabled()) return error.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.infer(allocator, input);
}

pub fn embedText(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
    if (!isEnabled()) return error.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.embed(allocator, input);
}

pub fn encodeTokens(allocator: std.mem.Allocator, input: []const u8) ![]u32 {
    if (!isEnabled()) return error.AiDisabled;
    const model = transformer.TransformerModel.init(.{});
    return model.encode(allocator, input);
}

pub fn decodeTokens(allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
    if (!isEnabled()) return error.AiDisabled;
    const model = transformer.TransformerModel.init(.{});
    return model.decode(allocator, tokens);
}

// ============================================================================
// Tests
// ============================================================================

test "isEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_ai, isEnabled());
}

test "isLlmEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_llm, isLlmEnabled());
}

test "ai module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
