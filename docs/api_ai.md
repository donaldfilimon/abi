# AI API Reference
> **Codebase Status:** Synced with repository as of 2026-01-18.

**Source:** `src/ai/mod.zig`

The AI module provides comprehensive capabilities for LLM inference, agent systems, embeddings, training pipelines, and vision processing. All features are feature-gated for minimal binary size.

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with AI enabled
    const config = abi.Config.init().withAI(true);
    var fw = try abi.Framework.init(allocator, config);
    defer fw.deinit();

    // Access AI features
    if (fw.ai()) |ai| {
        const response = try ai.generate("Hello, world!");
        std.debug.print("{s}\n", .{response});
    }
}
```

---

## Core Types

### `Agent`

Conversational AI agent with tool calling capabilities.

```zig
pub const Agent = struct {
    pub fn init(allocator: Allocator, config: AgentConfig) !Agent;
    pub fn deinit(self: *Agent) void;
    pub fn chat(self: *Agent, message: []const u8) ![]const u8;
    pub fn registerTool(self: *Agent, tool: Tool) !void;
    pub fn setSystemPrompt(self: *Agent, prompt: []const u8) void;
};
```

### `ModelRegistry`

Registry for managing AI models.

```zig
pub const ModelRegistry = struct {
    pub fn register(self: *ModelRegistry, info: ModelInfo) !void;
    pub fn get(self: *ModelRegistry, name: []const u8) ?ModelInfo;
    pub fn list(self: *ModelRegistry) []ModelInfo;
    pub fn remove(self: *ModelRegistry, name: []const u8) bool;
};
```

### `ModelInfo`

Model metadata structure.

```zig
pub const ModelInfo = struct {
    name: []const u8,
    path: []const u8,
    model_type: ModelType,
    parameters: u64,
    context_length: u32,
    quantization: ?Quantization,
};
```

---

## LLM Inference

### `LlmEngine`

Main LLM inference engine.

```zig
pub const LlmEngine = struct {
    pub fn init(allocator: Allocator, config: LlmConfig) !LlmEngine;
    pub fn deinit(self: *LlmEngine) void;
    pub fn loadModel(self: *LlmEngine, path: []const u8) !void;
    pub fn generate(self: *LlmEngine, prompt: []const u8, options: GenerateOptions) ![]const u8;
    pub fn chat(self: *LlmEngine, messages: []const Message) ![]const u8;
    pub fn tokenize(self: *LlmEngine, text: []const u8) ![]u32;
    pub fn detokenize(self: *LlmEngine, tokens: []const u32) ![]const u8;
};
```

### `LlmConfig`

LLM configuration options.

```zig
pub const LlmConfig = struct {
    model_path: ?[]const u8 = null,
    context_size: u32 = 4096,
    batch_size: u32 = 512,
    threads: u32 = 4,
    gpu_layers: u32 = 0,
    use_mmap: bool = true,
    use_mlock: bool = false,
};
```

### `LlmModel`

Loaded LLM model handle.

```zig
pub const LlmModel = struct {
    pub fn getContextSize(self: *LlmModel) u32;
    pub fn getVocabSize(self: *LlmModel) u32;
    pub fn getEmbeddingSize(self: *LlmModel) u32;
};
```

---

## Streaming Generation

### `StreamingGenerator`

Real-time token streaming for LLM output.

```zig
pub const StreamingGenerator = struct {
    pub fn init(engine: *LlmEngine, config: GenerationConfig) StreamingGenerator;
    pub fn next(self: *StreamingGenerator) !?StreamToken;
    pub fn stop(self: *StreamingGenerator) void;
    pub fn getState(self: *StreamingGenerator) StreamState;
};
```

### `StreamToken`

Individual token in streaming output.

```zig
pub const StreamToken = struct {
    text: []const u8,
    token_id: u32,
    logprob: f32,
    is_special: bool,
};
```

### `StreamState`

Current state of streaming generation.

```zig
pub const StreamState = enum {
    idle,
    generating,
    finished,
    stopped,
    error,
};
```

### `GenerationConfig`

Generation parameters.

```zig
pub const GenerationConfig = struct {
    max_tokens: u32 = 2048,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repeat_penalty: f32 = 1.1,
    stop_sequences: []const []const u8 = &.{},
};
```

---

## Transformer Models

### `TransformerConfig`

Transformer architecture configuration.

```zig
pub const TransformerConfig = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    num_heads: u32,
    intermediate_size: u32,
    max_position_embeddings: u32,
    layer_norm_eps: f32 = 1e-5,
    attention_dropout: f32 = 0.0,
    hidden_dropout: f32 = 0.0,
};
```

### `TransformerModel`

Transformer model implementation.

```zig
pub const TransformerModel = struct {
    pub fn init(allocator: Allocator, config: TransformerConfig) !TransformerModel;
    pub fn deinit(self: *TransformerModel) void;
    pub fn forward(self: *TransformerModel, input_ids: []const u32) ![]f32;
    pub fn loadWeights(self: *TransformerModel, path: []const u8) !void;
    pub fn saveWeights(self: *TransformerModel, path: []const u8) !void;
};
```

---

## Training

### `TrainingConfig`

Training hyperparameters.

```zig
pub const TrainingConfig = struct {
    learning_rate: f32 = 1e-4,
    batch_size: u32 = 8,
    epochs: u32 = 3,
    warmup_steps: u32 = 100,
    weight_decay: f32 = 0.01,
    gradient_accumulation_steps: u32 = 1,
    max_grad_norm: f32 = 1.0,
    optimizer: OptimizerType = .adamw,
    lr_schedule: LearningRateSchedule = .cosine,
    save_steps: u32 = 500,
    eval_steps: u32 = 100,
};
```

### `OptimizerType`

Available optimizers.

```zig
pub const OptimizerType = enum {
    sgd,
    adam,
    adamw,
    adagrad,
    rmsprop,
};
```

### `LearningRateSchedule`

Learning rate scheduling strategies.

```zig
pub const LearningRateSchedule = enum {
    constant,
    linear,
    cosine,
    polynomial,
    warmup_cosine,
};
```

### `TrainableModel`

Interface for trainable models.

```zig
pub const TrainableModel = struct {
    pub fn train(self: *TrainableModel, dataset: *TokenizedDataset, config: TrainingConfig) !TrainingResult;
    pub fn evaluate(self: *TrainableModel, dataset: *TokenizedDataset) !EvalResult;
    pub fn saveCheckpoint(self: *TrainableModel, path: []const u8) !void;
    pub fn loadCheckpoint(self: *TrainableModel, path: []const u8) !void;
};
```

### `TrainingResult`

Training run results.

```zig
pub const TrainingResult = struct {
    final_loss: f32,
    total_steps: u64,
    total_tokens: u64,
    training_time_seconds: f64,
    best_checkpoint_path: ?[]const u8,
};
```

### `TrainingReport`

Detailed training report.

```zig
pub const TrainingReport = struct {
    config: TrainingConfig,
    result: TrainingResult,
    loss_history: []f32,
    eval_history: []EvalMetrics,
    hardware_info: HardwareInfo,
};
```

---

## Checkpointing

### `CheckpointStore`

Manage model checkpoints.

```zig
pub const CheckpointStore = struct {
    pub fn init(allocator: Allocator, base_path: []const u8) !CheckpointStore;
    pub fn save(self: *CheckpointStore, checkpoint: Checkpoint) !void;
    pub fn load(self: *CheckpointStore, name: []const u8) !Checkpoint;
    pub fn list(self: *CheckpointStore) ![]CheckpointInfo;
    pub fn delete(self: *CheckpointStore, name: []const u8) !void;
};
```

### `Checkpoint`

Checkpoint data structure.

```zig
pub const Checkpoint = struct {
    name: []const u8,
    step: u64,
    model_state: []const u8,
    optimizer_state: ?[]const u8,
    metadata: CheckpointMetadata,
};
```

---

## Data Loading

### `TokenizedDataset`

Dataset of tokenized sequences.

```zig
pub const TokenizedDataset = struct {
    pub fn fromFile(allocator: Allocator, path: []const u8) !TokenizedDataset;
    pub fn fromTokens(allocator: Allocator, tokens: []const []u32) !TokenizedDataset;
    pub fn len(self: *TokenizedDataset) usize;
    pub fn get(self: *TokenizedDataset, index: usize) []u32;
};
```

### `DataLoader`

Batched data loading.

```zig
pub const DataLoader = struct {
    pub fn init(dataset: *TokenizedDataset, batch_size: u32, shuffle: bool) DataLoader;
    pub fn next(self: *DataLoader) ?Batch;
    pub fn reset(self: *DataLoader) void;
    pub fn epoch(self: *DataLoader) u32;
};
```

### `Batch`

Training batch.

```zig
pub const Batch = struct {
    input_ids: [][]u32,
    attention_mask: [][]u8,
    labels: ?[][]u32,
};
```

---

## Agent Tools

### `Tool`

Tool definition for agent tool calling.

```zig
pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const ToolParameter,
    handler: *const fn(args: ToolArgs) ToolResult,
};
```

### `ToolResult`

Result from tool execution.

```zig
pub const ToolResult = struct {
    success: bool,
    output: []const u8,
    error_message: ?[]const u8,
};
```

### `ToolRegistry`

Manage available tools.

```zig
pub const ToolRegistry = struct {
    pub fn init(allocator: Allocator) ToolRegistry;
    pub fn register(self: *ToolRegistry, tool: Tool) !void;
    pub fn get(self: *ToolRegistry, name: []const u8) ?Tool;
    pub fn list(self: *ToolRegistry) []Tool;
    pub fn execute(self: *ToolRegistry, name: []const u8, args: ToolArgs) !ToolResult;
};
```

### `TaskTool`

Tool for task management integration.

```zig
pub const TaskTool = struct {
    pub fn create() Tool;
};
```

### `Subagent`

Specialized sub-agents for complex tasks.

```zig
pub const Subagent = struct {
    pub fn init(allocator: Allocator, agent_type: SubagentType) !Subagent;
    pub fn execute(self: *Subagent, task: []const u8) ![]const u8;
};
```

---

## Discord Integration

### `DiscordTools`

Tools for Discord bot integration.

```zig
pub const DiscordTools = struct {
    send_message: Tool,
    read_channel: Tool,
    manage_roles: Tool,
};
```

### `registerDiscordTools`

Register Discord tools with an agent.

```zig
pub fn registerDiscordTools(registry: *ToolRegistry, config: DiscordConfig) !void;
```

---

## Feature Flags

The AI module respects these build-time feature flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | Main AI module |
| `-Denable-llm` | true | LLM inference (requires ai) |
| `-Denable-vision` | true | Vision processing (requires ai) |

---

## Related Documentation

- [AI Guide](ai.md) - Comprehensive AI module guide
- [AI Dataflow Diagram](diagrams/ai-dataflow.md) - Architecture visualization
- [Training Guide](training/abbey-fine-tuning.md) - Fine-tuning tutorial

---

*See also: [Framework API](api_abi.md) | [GPU API](api_gpu.md) | [Database API](api_database.md)*
