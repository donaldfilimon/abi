---
title: "Training"
description: "Training pipelines, federated learning, data loading, and multimodal training in the ABI Training module."
section: "AI"
order: 4
---

# Training

The Training module provides model training infrastructure: training pipelines,
fine-tuning, checkpointing, data loading and preprocessing, federated learning,
and the AI-database bridge for WDBX token datasets.

- **Build flag:** `-Denable-training=true` (default: enabled)
- **Namespace:** `abi.training`
- **Source:** `src/features/ai_training/`

## Overview

The Training module handles the full model training lifecycle -- from data
loading and preprocessing through optimizer configuration, checkpoint
management, and training execution. It also includes federated learning support
for distributed training across multiple nodes.

Key capabilities:

- **Training Pipelines** -- Configurable training loops with optimizer selection
- **Checkpointing** -- Save and restore training state
- **Data Loading** -- Tokenized datasets, batch iterators, sequence packing
- **Instruction Datasets** -- Parse instruction-tuning data formats
- **Federated Learning** -- Distributed training across nodes
- **WDBX Bridge** -- Convert between token binary format and WDBX storage
- **Multimodal Training** -- Vision (ViT), CLIP, video, and audio training
- **Self-Learning** -- Flow system-level config into self-learning pipelines

## Quick Start

```zig
const abi = @import("abi");

// Initialize training context
var ctx = try abi.training.Context.init(allocator, .{
    .training = .{
        .epochs = 10,
        .batch_size = 32,
        .learning_rate = 3e-4,
    },
});
defer ctx.deinit();

// Access the training sub-context
const train_ctx = try ctx.getTraining();
_ = train_ctx;

// Or use the module-level train function directly
const report = try abi.training.train(allocator, training_config);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Module context wrapping the training sub-context |
| `TrainingConfig` | Primary training configuration (epochs, batch size, LR, etc.) |
| `TrainingReport` | Summary report after training completes |
| `TrainingResult` | Detailed training result with metrics |
| `TrainError` | Error set for training operations |
| `OptimizerType` | Optimizer selection (SGD, Adam, AdamW, etc.) |
| `LearningRateSchedule` | LR schedule (constant, cosine, linear warmup, etc.) |
| `CheckpointStore` | Manages saved training checkpoints |
| `Checkpoint` | A single training checkpoint |
| `LlmTrainingConfig` | LLM-specific training configuration |
| `TrainableModel` | Generic trainable model interface |
| `TrainableModelConfig` | Configuration for trainable models |
| `LlamaTrainer` | Specialized trainer for LLaMA-architecture models |
| `TrainableViTModel` | Trainable Vision Transformer model |
| `TrainableViTConfig` | ViT training configuration |
| `TrainableViTWeights` | ViT model weights |
| `VisionTrainingError` | Errors specific to vision training |
| `TrainableCLIPModel` | Trainable CLIP (text + image) model |
| `CLIPTrainingConfig` | CLIP training configuration |
| `MultimodalTrainingError` | Errors specific to multimodal training |
| `TokenizedDataset` | Pre-tokenized training dataset |
| `DataLoader` | Loads and batches training data |
| `BatchIterator` | Iterates over batches of training data |
| `Batch` | A single batch of training examples |
| `SequencePacker` | Packs variable-length sequences for efficient batching |
| `WdbxTokenDataset` | WDBX-backed token dataset |

### Key Functions

| Function | Description |
|----------|-------------|
| `isEnabled() bool` | Returns `true` if training is compiled in |
| `train(allocator, config) !TrainingReport` | Run a training pipeline and return a summary report |
| `trainWithResult(allocator, config) !TrainingResult` | Run training and return detailed results |
| `loadCheckpoint(path) !Checkpoint` | Load a training checkpoint from disk |
| `saveCheckpoint(checkpoint, path) !void` | Save a training checkpoint to disk |
| `parseInstructionDataset(data) !Dataset` | Parse instruction-tuning format data |
| `tokenBinToWdbx(...)` | Convert token binary file to WDBX format |
| `wdbxToTokenBin(...)` | Convert WDBX storage to token binary file |
| `readTokenBinFile(...)` | Read a token binary file |
| `writeTokenBinFile(...)` | Write a token binary file |
| `exportGguf(...)` | Export model weights to GGUF format |

## Training Pipelines

A training pipeline is configured through `TrainingConfig` and executed via the
`train()` or `trainWithResult()` functions:

```zig
const config = abi.training.TrainingConfig{
    .epochs = 10,
    .batch_size = 32,
    .learning_rate = 3e-4,
    .optimizer = .adam,
    .lr_schedule = .cosine,
};

const report = try abi.training.train(allocator, config);
```

### Optimizer Types

The training module supports multiple optimizers:

- SGD (Stochastic Gradient Descent)
- Adam
- AdamW (Adam with weight decay)
- And other standard deep learning optimizers

### Learning Rate Schedules

- Constant
- Cosine annealing
- Linear warmup
- Step decay

## Checkpointing

Save and restore training state for resumable training:

```zig
// Save a checkpoint
try abi.training.saveCheckpoint(checkpoint, "./checkpoints/epoch_5.ckpt");

// Load a checkpoint to resume training
const restored = try abi.training.loadCheckpoint("./checkpoints/epoch_5.ckpt");
```

## Data Loading

The data loading subsystem provides efficient batching and sequence packing:

```zig
// Create a data loader
var loader = abi.training.DataLoader.init(allocator, dataset, .{
    .batch_size = 32,
    .shuffle = true,
});
defer loader.deinit();

// Iterate over batches
var iter = loader.iterator();
while (iter.next()) |batch| {
    // Process batch
    _ = batch;
}
```

### WDBX Token Datasets

The database bridge allows training from WDBX-stored token datasets:

```zig
// Convert token binary to WDBX for storage
try abi.training.tokenBinToWdbx(token_data, wdbx_path);

// Convert back for training
try abi.training.wdbxToTokenBin(wdbx_path, output_path);

// Export trained weights to GGUF
try abi.training.exportGguf(model_weights, gguf_path);
```

## Multimodal Training

The training module supports multimodal model training through system-level
configuration flags:

```zig
// System-level training config with multimodal flags
const core_config = abi.config.TrainingConfig{
    .epochs = 10,
    .batch_size = 32,
    .enable_vision = true,   // Enable image training
    .enable_video = true,    // Enable video training
    .enable_audio = true,    // Enable audio training
    .enable_all_modalities = true, // Enable all data types
};
```

### selfLearningConfigFromCore

Flow system-level `TrainingConfig` into the self-learning subsystem:

```zig
const self_learning_cfg = abi.ai.training.selfLearningConfigFromCore(core_config);
// self_learning_cfg now has enable_vision, enable_video, enable_audio,
// enable_all_modalities propagated from the core training config
```

This bridges the gap between system-level configuration (`abi.config.TrainingConfig`)
and the AI self-learning module, ensuring multimodal flags are consistent.

### Content Kinds

The `ContentKind` enum defines the data types that models can process:

| Kind | Description |
|------|-------------|
| `text` | Text data |
| `image` | Image data |
| `video` | Video data |
| `audio` | Audio data |
| `document` | Structured documents |
| `other` | Raw / unclassified payloads |

### Vision Training (ViT)

```zig
const vit_config = abi.training.TrainableViTConfig{
    // Configure Vision Transformer training
};
var vit_model = try abi.training.TrainableViTModel.init(allocator, vit_config);
defer vit_model.deinit();
```

### CLIP Training

```zig
const clip_config = abi.training.CLIPTrainingConfig{
    // Configure CLIP (text + image) training
};
var clip_model = try abi.training.TrainableCLIPModel.init(allocator, clip_config);
defer clip_model.deinit();
```

## Federated Learning

The federated learning subsystem enables distributed training across multiple
nodes:

```zig
const federated = abi.training.federated;
// Configure and run federated training sessions
```

## Configuration

Training is configured through `AiConfig.training`:

```zig
const config: abi.config.AiConfig = .{
    .training = .{
        .epochs = 20,
        .batch_size = 64,
        .learning_rate = 1e-4,
        .enable_vision = true,
        .enable_video = false,
        .enable_audio = false,
        .enable_all_modalities = false,
    },
};

var ctx = try abi.training.Context.init(allocator, config);
```

## CLI Commands

```bash
# Run a training pipeline
zig build run -- train run --config training.json

# Resume training from a checkpoint
zig build run -- train resume --checkpoint ./checkpoints/latest.ckpt

# Show training info
zig build run -- train info
```

## Disabling at Build Time

```bash
# Compile without training support
zig build -Denable-training=false
```

When disabled, `Context.init()` returns `error.TrainingDisabled`, `train()`
returns `error.TrainingDisabled`, and `isEnabled()` returns `false`. All type
signatures are preserved by the stub module.

## Related

- [AI Overview](ai-overview.html) -- Architecture of all five AI modules
- [AI Core](ai-core.html) -- Agents, tools, prompts, personas
- [Inference](ai-inference.html) -- LLM execution, embeddings, streaming
- [Reasoning](ai-reasoning.html) -- Abbey engine, RAG, orchestration
