//! AI Module - Public API
//!
//! This is the primary entry point for AI functionality in the ABI framework.
//! Import from here for Framework integration and the stable public API.
//!
//! ## Overview
//!
//! The AI module provides modular AI capabilities organized as independent sub-features,
//! each of which can be enabled or disabled independently:
//!
//! | Sub-feature | Description | Build Flag |
//! |-------------|-------------|------------|
//! | **core** | Shared types, interfaces, utilities | Always available |
//! | **llm** | Local LLM inference (GGUF models) | `-Denable-llm` |
//! | **embeddings** | Vector embeddings generation | `-Denable-ai` |
//! | **agents** | AI agent runtime and tools | `-Denable-ai` |
//! | **training** | Model training pipelines | `-Denable-ai` |
//! | **personas** | Multi-persona AI assistant | `-Denable-ai` |
//! | **vision** | Image processing and analysis | `-Denable-vision` |
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with AI enabled
//! var fw = try abi.Framework.init(allocator, .{
//!     .ai = .{
//!         .llm = .{ .model_path = "./models/llama-7b.gguf" },
//!         .embeddings = .{ .dimension = 768 },
//!     },
//! });
//! defer fw.deinit();
//!
//! // Access AI context
//! const ai_ctx = try fw.getAi();
//!
//! // Use LLM
//! const llm = try ai_ctx.getLlm();
//! // ... perform inference ...
//! ```
//!
//! ## Standalone Usage
//!
//! ```zig
//! const ai = abi.ai;
//!
//! // Initialize AI context directly
//! var ctx = try ai.Context.init(allocator, .{
//!     .llm = .{ .model_path = "./models/llama.gguf" },
//! });
//! defer ctx.deinit();
//!
//! // Check which sub-features are enabled
//! if (ctx.isSubFeatureEnabled(.llm)) {
//!     const llm = try ctx.getLlm();
//!     // ... use LLM ...
//! }
//! ```
//!
//! ## Sub-module Access
//!
//! Access sub-modules directly through the namespace:
//! - `abi.ai.llm` - LLM inference engine
//! - `abi.ai.embeddings` - Embedding generation
//! - `abi.ai.agents` - Agent runtime
//! - `abi.ai.training` - Training pipelines
//! - `abi.ai.personas` - Multi-persona system
//! - `abi.ai.vision` - Vision processing

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Direct module imports (no longer using implementation_mod.zig bridge)
// ============================================================================

// Direct imports for AI submodules
pub const agent = @import("agents/agent.zig");
pub const model_registry = @import("models/registry.zig");
pub const transformer = @import("transformer/mod.zig");
pub const streaming = @import("streaming/mod.zig");
pub const tools = @import("tools/mod.zig");
pub const prompts = @import("prompts/mod.zig");
pub const abbey = @import("abbey/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const federated = @import("federated/mod.zig");
pub const personas = if (build_options.enable_ai) @import("personas/mod.zig") else @import("personas/stub.zig");
pub const rag = if (build_options.enable_ai) @import("rag/mod.zig") else @import("rag/stub.zig");
pub const templates = if (build_options.enable_ai) @import("templates/mod.zig") else @import("templates/stub.zig");
pub const eval = if (build_options.enable_ai) @import("eval/mod.zig") else @import("eval/stub.zig");
pub const explore = if (build_options.enable_explore) @import("explore/mod.zig") else @import("explore/stub.zig");
pub const orchestration = if (build_options.enable_ai) @import("orchestration/mod.zig") else @import("orchestration/stub.zig");

// Tool-augmented agent with tool execution loop
pub const tool_agent = @import("tools/tool_agent.zig");

// Codebase self-awareness via indexing and RAG
pub const codebase_index = @import("explore/codebase_index.zig");

// Self-improvement and performance tracking
pub const self_improve = @import("self_improve.zig");

// GPU-aware agent (always available, uses stubs when GPU disabled)
pub const gpu_agent = @import("agents/gpu_agent.zig");

// Model auto-discovery and adaptive configuration
pub const discovery = @import("explore/discovery.zig");

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

/// AI Database module
pub const database = if (build_options.enable_ai)
    @import("database/mod.zig")
else
    @import("database/stub.zig");

// ---------------------------------------------------------------------------
// Multi‑Agent coordination (requires AI feature)
// ---------------------------------------------------------------------------
pub const multi_agent = if (build_options.enable_ai)
    @import("multi_agent/mod.zig")
else
    @import("multi_agent/stub.zig");

/// Vision/image processing module
pub const vision = if (build_options.enable_vision)
    @import("vision/mod.zig")
else
    @import("vision/stub.zig");

/// Document understanding and processing module
pub const documents = if (build_options.enable_ai)
    @import("documents/mod.zig")
else
    @import("documents/stub.zig");

// ============================================================================
// Re-exports for backward compatibility
// ============================================================================
//
// These re-exports provide convenient access to commonly-used types from
// sub-modules. They maintain backward compatibility with older code that
// imports types directly from the ai module.

/// AI agent for task execution and tool usage.
pub const Agent = agent.Agent;

/// Coordinator for orchestrating multiple AI agents.
pub const MultiAgentCoordinator = multi_agent.Coordinator;

/// Registry for managing AI model metadata and paths.
pub const ModelRegistry = model_registry.ModelRegistry;

/// Metadata information for a registered model.
pub const ModelInfo = model_registry.ModelInfo;

// Training types (from training module)

/// Configuration for model training sessions.
pub const TrainingConfig = training.TrainingConfig;

/// Summary report generated after training completes.
pub const TrainingReport = training.TrainingReport;

/// Result of a training operation including metrics and final weights.
pub const TrainingResult = training.TrainingResult;

/// Error types that can occur during training.
pub const TrainError = training.TrainError;

/// Supported optimizer algorithms (SGD, Adam, AdamW, etc.).
pub const OptimizerType = training.OptimizerType;

/// Learning rate scheduling strategies (constant, cosine, linear warmup, etc.).
pub const LearningRateSchedule = training.LearningRateSchedule;

/// Storage backend for training checkpoints.
pub const CheckpointStore = training.CheckpointStore;

/// Saved training checkpoint with model state and optimizer state.
pub const Checkpoint = training.Checkpoint;

/// LLM-specific training configuration.
pub const LlmTrainingConfig = training.LlmTrainingConfig;

/// Trainable model module with generic training infrastructure.
pub const trainable_model = training.trainable_model;

/// Generic trainable model interface.
pub const TrainableModel = training.TrainableModel;

/// Configuration for trainable model initialization.
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;

/// LLaMA architecture trainer for fine-tuning.
pub const LlamaTrainer = training.LlamaTrainer;

/// Load a training checkpoint from disk.
pub const loadCheckpoint = training.loadCheckpoint;

/// Save a training checkpoint to disk.
pub const saveCheckpoint = training.saveCheckpoint;

// Vision training types

/// Vision Transformer (ViT) model for training.
pub const TrainableViTModel = training.TrainableViTModel;

/// Configuration for ViT model training.
pub const TrainableViTConfig = training.TrainableViTConfig;

/// Trained weights for a ViT model.
pub const TrainableViTWeights = training.TrainableViTWeights;

/// Errors specific to vision model training.
pub const VisionTrainingError = training.VisionTrainingError;

// CLIP/Multimodal training types

/// CLIP model for contrastive text-image learning.
pub const TrainableCLIPModel = training.TrainableCLIPModel;

/// Configuration for CLIP training.
pub const CLIPTrainingConfig = training.CLIPTrainingConfig;

/// Errors specific to multimodal model training.
pub const MultimodalTrainingError = training.MultimodalTrainingError;

// Data loading

/// Pre-tokenized dataset for efficient training iteration.
pub const TokenizedDataset = training.TokenizedDataset;

/// Batched data loader with shuffling and prefetching.
pub const DataLoader = training.DataLoader;

/// Iterator over training batches.
pub const BatchIterator = training.BatchIterator;

/// A single training batch of tokenized sequences.
pub const Batch = training.Batch;

/// Utility for packing variable-length sequences into fixed-size batches.
pub const SequencePacker = training.SequencePacker;

/// Parse instruction-tuning datasets (Alpaca, ShareGPT formats).
pub const parseInstructionDataset = training.parseInstructionDataset;

/// WDBX-backed tokenized dataset for vector database integration.
pub const WdbxTokenDataset = database.WdbxTokenDataset;

/// Convert TokenBin format to WDBX database format.
pub const tokenBinToWdbx = database.tokenBinToWdbx;

/// Convert WDBX database format to TokenBin format.
pub const wdbxToTokenBin = database.wdbxToTokenBin;

/// Read a TokenBin file from disk.
pub const readTokenBinFile = database.readTokenBinFile;

/// Write a TokenBin file to disk.
pub const writeTokenBinFile = database.writeTokenBinFile;

/// Export model weights to GGUF format.
pub const exportGguf = database.exportGguf;

// Tools

/// Base interface for agent tools (function calling).
pub const Tool = tools.Tool;

/// Result returned by tool execution.
pub const ToolResult = tools.ToolResult;

/// Registry for managing and discovering available tools.
pub const ToolRegistry = tools.ToolRegistry;

/// Task management tool for creating and tracking tasks.
pub const TaskTool = tools.TaskTool;

/// Tool for spawning and coordinating sub-agents.
pub const Subagent = tools.Subagent;

/// Discord bot integration tools.
pub const DiscordTools = tools.DiscordTools;

/// Register Discord tools with a tool registry.
pub const registerDiscordTools = tools.registerDiscordTools;

/// Operating system integration tools (file, process, etc.).
pub const OsTools = tools.OsTools;

/// Register OS tools with a tool registry.
pub const registerOsTools = tools.registerOsTools;

/// Register all agent tools (file, search, edit, OS, process, network, system).
pub const registerAllAgentTools = tools.registerAllAgentTools;

// Tool-augmented agent

/// AI agent enhanced with tool-use capabilities (file I/O, shell, search, etc.).
pub const ToolAugmentedAgent = tool_agent.ToolAugmentedAgent;

/// Configuration for the tool-augmented agent.
pub const ToolAgentConfig = tool_agent.ToolAgentConfig;

// Codebase self-awareness

/// Index for codebase self-awareness and retrieval.
pub const CodebaseIndex = codebase_index.CodebaseIndex;

/// Result of a codebase query.
pub const CodebaseAnswer = codebase_index.CodebaseAnswer;

// Self-improvement

/// Performance tracker and improvement proposer.
pub const SelfImprover = self_improve.SelfImprover;

/// Quality metrics for a response.
pub const ResponseMetrics = self_improve.ResponseMetrics;

// Transformer

/// Configuration for transformer model architecture.
pub const TransformerConfig = transformer.TransformerConfig;

/// Transformer model for inference and embedding.
pub const TransformerModel = transformer.TransformerModel;

// Streaming

/// Generator for token-by-token streaming output.
pub const StreamingGenerator = streaming.StreamingGenerator;

/// A single token emitted during streaming generation.
pub const StreamToken = streaming.StreamToken;

/// State of a streaming generation session.
pub const StreamState = streaming.StreamState;

/// Configuration for text generation (temperature, top_p, etc.).
pub const GenerationConfig = streaming.GenerationConfig;

/// Configuration for streaming HTTP server.
pub const ServerConfig = streaming.ServerConfig;

/// HTTP server for streaming AI responses.
pub const StreamingServer = streaming.StreamingServer;

/// Errors from streaming server operations.
pub const StreamingServerError = streaming.StreamingServerError;

/// Type of backend used for streaming.
pub const BackendType = streaming.BackendType;

// LLM Engine

/// LLM inference engine for local model execution.
pub const LlmEngine = llm.Engine;

/// Loaded LLM model instance.
pub const LlmModel = llm.Model;

/// Configuration for LLM inference operations.
pub const LlmConfig = llm.InferenceConfig;

/// GGUF model file reader and parser.
pub const GgufFile = llm.GgufFile;

/// Byte-pair encoding tokenizer for LLM input processing.
pub const BpeTokenizer = llm.BpeTokenizer;

// Prompts

/// Builder for constructing formatted prompts.
pub const PromptBuilder = prompts.PromptBuilder;

/// AI persona definition with personality traits and behaviors.
pub const Persona = prompts.Persona;

/// Type of AI persona (abbey, abi, aviva, etc.).
pub const PersonaType = prompts.PersonaType;

/// Prompt format templates (ChatML, Alpaca, Llama, etc.).
pub const PromptFormat = prompts.PromptFormat;

// Abbey / Core AI

/// Abbey engine instance for advanced reasoning operations.
pub const AbbeyInstance = abbey.AbbeyEngine;

/// Abbey AI assistant with advanced reasoning capabilities.
pub const Abbey = abbey.Abbey;

/// Configuration for Abbey AI instance.
pub const AbbeyConfig = core.AbbeyConfig;

/// Response from Abbey AI including reasoning and confidence.
pub const AbbeyResponse = core.Response;

/// Runtime statistics for Abbey AI operations.
pub const AbbeyStats = abbey.Stats;

/// Chain of reasoning steps leading to a conclusion.
pub const ReasoningChain = abbey.ReasoningChain;

/// Single step in a reasoning chain.
pub const ReasoningStep = abbey.ReasoningStep;

/// Confidence score with supporting evidence.
pub const Confidence = core.Confidence;

/// Discrete confidence levels (low, medium, high, very_high).
pub const ConfidenceLevel = core.ConfidenceLevel;

/// Emotional state representation for AI responses.
pub const EmotionalState = core.EmotionalState;

/// Types of emotions that can be expressed.
pub const EmotionType = core.EmotionType;

/// Conversation context for multi-turn interactions.
pub const ConversationContext = abbey.ConversationContext;

/// Topic tracking for conversation flow.
pub const TopicTracker = core.Topic;

// Explore

/// Agent for exploring and searching codebases.
pub const ExploreAgent = explore.ExploreAgent;

/// Configuration for codebase exploration.
pub const ExploreConfig = explore.ExploreConfig;

/// Depth level for exploration (quick, medium, thorough, deep).
pub const ExploreLevel = explore.ExploreLevel;

/// Result of a codebase exploration query.
pub const ExploreResult = explore.ExploreResult;

/// A matched item from exploration search.
pub const Match = explore.Match;

/// Statistics about an exploration session.
pub const ExplorationStats = explore.ExplorationStats;

/// Detected intent from a natural language query.
pub const QueryIntent = explore.QueryIntent;

/// Parsed components of an exploration query.
pub const ParsedQuery = explore.ParsedQuery;

/// Natural language understanding for queries.
pub const QueryUnderstanding = explore.QueryUnderstanding;

// Orchestration - Multi-model coordination

/// Orchestrates requests across multiple AI models.
pub const Orchestrator = orchestration.Orchestrator;

/// Configuration for multi-model orchestration.
pub const OrchestrationConfig = orchestration.OrchestrationConfig;

/// Errors that can occur during orchestration.
pub const OrchestrationError = orchestration.OrchestrationError;

/// Strategy for routing requests to models (capability, cost, latency).
pub const RoutingStrategy = orchestration.RoutingStrategy;

/// Type of task for routing decisions.
pub const TaskType = orchestration.TaskType;

/// Result of routing a request to a model.
pub const RouteResult = orchestration.RouteResult;

/// Method for combining ensemble responses (voting, weighted, etc.).
pub const EnsembleMethod = orchestration.EnsembleMethod;

/// Result from an ensemble of model responses.
pub const EnsembleResult = orchestration.EnsembleResult;

/// Policy for handling model failures (retry, fallback, fail).
pub const FallbackPolicy = orchestration.FallbackPolicy;

/// Health status of a model backend.
pub const HealthStatus = orchestration.HealthStatus;

/// Backend type for model hosting.
pub const ModelBackend = orchestration.ModelBackend;

/// Capabilities supported by a model.
pub const ModelCapability = orchestration.Capability;

/// Configuration for a model in the orchestrator.
pub const OrchestrationModelConfig = orchestration.ModelConfig;

// GPU-Aware Agent types

/// AI agent with GPU acceleration support.
pub const GpuAgent = gpu_agent.GpuAgent;

/// Request to a GPU-aware agent with resource hints.
pub const GpuAwareRequest = gpu_agent.GpuAwareRequest;

/// Response from a GPU-aware agent with performance metrics.
pub const GpuAwareResponse = gpu_agent.GpuAwareResponse;

/// Type of workload for GPU scheduling decisions.
pub const WorkloadType = gpu_agent.WorkloadType;

/// Priority level for GPU agent requests.
pub const GpuAgentPriority = gpu_agent.Priority;

/// Statistics from GPU agent operations.
pub const GpuAgentStats = gpu_agent.AgentStats;

// Model Management

/// Model management utilities.
pub const models = if (build_options.enable_ai) @import("models/mod.zig") else @import("models/stub.zig");

// Model Auto-Discovery and Adaptive Configuration

/// Automatic model discovery from common paths.
pub const ModelDiscovery = discovery.ModelDiscovery;

/// A model discovered by the discovery system.
pub const DiscoveredModel = discovery.DiscoveredModel;

/// Configuration for model discovery behavior.
pub const DiscoveryConfig = discovery.DiscoveryConfig;

/// Detected system capabilities (CPU, RAM, GPU).
pub const SystemCapabilities = discovery.SystemCapabilities;

/// Auto-generated configuration adapted to system capabilities.
pub const AdaptiveConfig = discovery.AdaptiveConfig;

/// Requirements that a model must meet.
pub const ModelRequirements = discovery.ModelRequirements;

/// Result of model warmup/validation.
pub const WarmupResult = discovery.WarmupResult;

/// Detect system capabilities at runtime.
pub const detectCapabilities = discovery.detectCapabilities;

/// Run model warmup to verify loading and inference.
pub const runWarmup = discovery.runWarmup;

// Document Understanding

/// Pipeline for processing and understanding documents.
pub const DocumentPipeline = documents.DocumentPipeline;

/// Parsed document with extracted structure.
pub const Document = documents.Document;

/// Supported document formats (PDF, HTML, Markdown, etc.).
pub const DocumentFormat = documents.DocumentFormat;

/// Element within a document (paragraph, table, image, etc.).
pub const DocumentElement = documents.DocumentElement;

/// Type of document element.
pub const ElementType = documents.ElementType;

/// Segment of text with semantic boundaries.
pub const TextSegment = documents.TextSegment;

/// Utility for segmenting text into semantic chunks.
pub const TextSegmenter = documents.TextSegmenter;

/// Named entity extracted from text.
pub const NamedEntity = documents.NamedEntity;

/// Type of named entity (person, organization, location, etc.).
pub const EntityType = documents.EntityType;

/// Extractor for named entities in text.
pub const EntityExtractor = documents.EntityExtractor;

/// Analyzer for document layout structure.
pub const LayoutAnalyzer = documents.LayoutAnalyzer;

/// Configuration for document processing pipeline.
pub const PipelineConfig = documents.PipelineConfig;

/// Configuration for text segmentation.
pub const SegmentationConfig = documents.SegmentationConfig;

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
///
/// The Context struct manages all AI sub-features (LLM, embeddings, agents, training,
/// personas) based on the provided configuration. Each sub-feature is independently
/// initialized and can be accessed through type-safe getter methods.
///
/// ## Thread Safety
///
/// The Context itself is not thread-safe. If you need to access AI features from
/// multiple threads, use external synchronization.
///
/// ## Memory Management
///
/// The Context allocates memory for each enabled sub-feature context. All memory
/// is released when `deinit()` is called.
///
/// ## Example
///
/// ```zig
/// var ctx = try ai.Context.init(allocator, .{
///     .llm = .{ .model_path = "./models/llama.gguf" },
///     .embeddings = .{ .dimension = 768 },
/// });
/// defer ctx.deinit();
///
/// // Access sub-features
/// const llm = try ctx.getLlm();
/// const emb = try ctx.getEmbeddings();
/// ```
pub const Context = struct {
    /// Memory allocator for context resources.
    allocator: std.mem.Allocator,
    /// Configuration for AI sub-features.
    config: config_module.AiConfig,

    // Sub-feature contexts (null if disabled)
    /// LLM inference context, or null if not enabled.
    llm_ctx: ?*llm.Context = null,
    /// Embeddings generation context, or null if not enabled.
    embeddings_ctx: ?*embeddings.Context = null,
    /// Agent runtime context, or null if not enabled.
    agents_ctx: ?*agents.Context = null,
    /// Training pipeline context, or null if not enabled.
    training_ctx: ?*training.Context = null,
    /// Multi-persona system context, or null if not enabled.
    personas_ctx: ?*personas.Context = null,

    // Auto-discovery and adaptive configuration
    /// Model discovery system for automatic model detection.
    model_discovery: ?*discovery.ModelDiscovery = null,
    /// Detected system capabilities.
    capabilities: discovery.SystemCapabilities = .{},

    /// Initialize the AI context with the given configuration.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for context resources
    /// - `cfg`: AI configuration specifying which sub-features to enable
    ///
    /// ## Returns
    ///
    /// A pointer to the initialized Context.
    ///
    /// ## Errors
    ///
    /// - `error.AiDisabled`: AI is disabled at compile time
    /// - `error.OutOfMemory`: Memory allocation failed
    /// - Sub-feature specific errors during initialization
    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AiConfig) !*Context {
        if (!isEnabled()) return error.AiDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .capabilities = discovery.detectCapabilities(),
        };

        // Initialize model discovery if auto_discover is enabled
        if (cfg.auto_discover) {
            const disc = try allocator.create(discovery.ModelDiscovery);
            disc.* = discovery.ModelDiscovery.init(allocator, .{});
            disc.scanAll() catch |err| {
                std.log.debug("Model discovery scan failed (best effort): {t}", .{err});
            };
            ctx.model_discovery = disc;
        }

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

        if (cfg.personas) |personas_cfg| {
            ctx.personas_ctx = try personas.Context.init(allocator, personas_cfg);
        }

        return ctx;
    }

    /// Deinitialize the context and release all resources.
    ///
    /// This releases memory for all initialized sub-feature contexts and
    /// the model discovery system (if enabled). The Context pointer becomes
    /// invalid after this call.
    pub fn deinit(self: *Context) void {
        self.deinitSubFeatures();
        self.allocator.destroy(self);
    }

    fn deinitSubFeatures(self: *Context) void {
        if (self.model_discovery) |disc| {
            disc.deinit();
            self.allocator.destroy(disc);
            self.model_discovery = null;
        }
        if (self.personas_ctx) |p| {
            p.deinit();
            self.personas_ctx = null;
        }
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

    /// Get personas context (returns error if not enabled).
    pub fn getPersonas(self: *Context) Error!*personas.Context {
        return self.personas_ctx orelse error.AiDisabled;
    }

    /// Check if a sub-feature is enabled.
    pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool {
        return switch (feature) {
            .llm => self.llm_ctx != null,
            .embeddings => self.embeddings_ctx != null,
            .agents => self.agents_ctx != null,
            .training => self.training_ctx != null,
            .personas => self.personas_ctx != null,
        };
    }

    /// Get discovered models (returns empty slice if discovery not enabled).
    pub fn getDiscoveredModels(self: *Context) []discovery.DiscoveredModel {
        if (self.model_discovery) |disc| {
            return disc.getModels();
        }
        return &.{};
    }

    /// Get number of discovered models.
    pub fn discoveredModelCount(self: *Context) usize {
        if (self.model_discovery) |disc| {
            return disc.modelCount();
        }
        return 0;
    }

    /// Find best model matching requirements.
    pub fn findBestModel(self: *Context, requirements: discovery.ModelRequirements) ?*discovery.DiscoveredModel {
        if (self.model_discovery) |disc| {
            return disc.findBestModel(requirements);
        }
        return null;
    }

    /// Generate adaptive configuration for a model.
    pub fn generateAdaptiveConfig(self: *Context, model: *const discovery.DiscoveredModel) discovery.AdaptiveConfig {
        if (self.model_discovery) |disc| {
            return disc.generateConfig(model);
        }
        // Return defaults if no discovery
        return discovery.AdaptiveConfig{
            .num_threads = self.capabilities.recommendedThreads(),
            .batch_size = 1,
        };
    }

    /// Get system capabilities.
    pub fn getCapabilities(self: *const Context) discovery.SystemCapabilities {
        return self.capabilities;
    }

    /// Add a model path to the discovery system.
    /// Use this to register known model files.
    pub fn addModelPath(self: *Context, path: []const u8) !void {
        if (self.model_discovery) |disc| {
            try disc.addModelPath(path);
        }
    }

    /// Add a model path with known file size.
    pub fn addModelWithSize(self: *Context, path: []const u8, size_bytes: u64) !void {
        if (self.model_discovery) |disc| {
            try disc.addModelWithSize(path, size_bytes);
        }
    }

    /// Clear all discovered models.
    pub fn clearDiscoveredModels(self: *Context) void {
        if (self.model_discovery) |disc| {
            for (disc.discovered_models.items) |*model| {
                model.deinit(self.allocator);
            }
            disc.discovered_models.clearRetainingCapacity();
        }
    }

    pub const SubFeature = enum {
        llm,
        embeddings,
        agents,
        training,
        personas,
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

// ============================================================================
// Legacy convenience functions
// ============================================================================
//
// These functions provide simple entry points for common AI operations.
// For new code, prefer using the Context-based API or direct sub-module access.

/// Create a new model registry for managing AI models.
///
/// ## Example
///
/// ```zig
/// var registry = ai.createRegistry(allocator);
/// defer registry.deinit();
/// try registry.register("llama-7b", .{ .path = "./models/llama-7b.gguf" });
/// ```
pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

/// Train a model and return a training report.
///
/// This is a convenience wrapper around `training.trainAndReport()`.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for training operations
/// - `config`: Training configuration including model, data, and hyperparameters
///
/// ## Returns
///
/// A `TrainingReport` with metrics and training history, or a `TrainError`.
pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    return training.trainAndReport(allocator, config);
}

/// Train a model and return the full training result.
///
/// Similar to `train()` but returns `TrainingResult` with trained weights.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for training operations
/// - `config`: Training configuration including model, data, and hyperparameters
///
/// ## Returns
///
/// A `TrainingResult` with trained weights and metrics, or a `TrainError`.
pub fn trainWithResult(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingResult {
    return training.trainWithResult(allocator, config);
}

/// Create a new AI agent with the specified name.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for agent resources
/// - `name`: Display name for the agent
///
/// ## Returns
///
/// An initialized `Agent` instance, or an error if AI is disabled or allocation fails.
///
/// ## Errors
///
/// - `error.AiDisabled`: AI feature is disabled at compile time
/// - `error.OutOfMemory`: Memory allocation failed
pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !Agent {
    if (!isEnabled()) return error.AiDisabled;
    return agent.Agent.init(allocator, .{ .name = name });
}

/// Create a transformer model with the specified configuration.
///
/// ## Parameters
///
/// - `config`: Transformer architecture configuration
///
/// ## Returns
///
/// An initialized `TransformerModel` (may be unloaded until weights are set).
pub fn createTransformer(config: TransformerConfig) TransformerModel {
    return transformer.TransformerModel.init(config);
}

/// Perform text inference using a default transformer model.
///
/// This is a convenience function for quick inference without explicit model management.
/// For production use, prefer creating and reusing a model instance.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for output
/// - `input`: Input text to process
///
/// ## Returns
///
/// Generated text response, owned by the caller.
///
/// ## Errors
///
/// - `error.AiDisabled`: AI feature is disabled at compile time
/// - Other errors from model inference
pub fn inferText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (!isEnabled()) return error.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.infer(allocator, input);
}

/// Generate text embeddings using a default transformer model.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for output
/// - `input`: Input text to embed
///
/// ## Returns
///
/// Embedding vector as a float array, owned by the caller.
///
/// ## Errors
///
/// - `error.AiDisabled`: AI feature is disabled at compile time
/// - Other errors from model embedding
pub fn embedText(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
    if (!isEnabled()) return error.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.embed(allocator, input);
}

/// Encode text into token IDs using a default transformer model.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for output
/// - `input`: Input text to tokenize
///
/// ## Returns
///
/// Array of token IDs, owned by the caller.
///
/// ## Errors
///
/// - `error.AiDisabled`: AI feature is disabled at compile time
/// - Other errors from tokenization
pub fn encodeTokens(allocator: std.mem.Allocator, input: []const u8) ![]u32 {
    if (!isEnabled()) return error.AiDisabled;
    const model = transformer.TransformerModel.init(.{});
    return model.encode(allocator, input);
}

/// Decode token IDs back into text using a default transformer model.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for output
/// - `tokens`: Token IDs to decode
///
/// ## Returns
///
/// Decoded text string, owned by the caller.
///
/// ## Errors
///
/// - `error.AiDisabled`: AI feature is disabled at compile time
/// - Other errors from decoding
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
