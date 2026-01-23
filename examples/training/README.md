---
title: "README"
tags: []
---
# LLM Training Example
> **Codebase Status:** Synced with repository as of 2026-01-22.

This directory contains examples demonstrating the ABI LLM training pipeline.

## Quick Start

Run the training demo with synthetic data:

```bash
zig build run-train-demo
```

## Files

- `train_demo.zig` - Complete training example with synthetic data

## Training Pipeline Overview

The ABI training infrastructure provides a complete LLM training solution:

```
                                    Training Pipeline
+------------------+    +------------------+    +------------------+
|  Training Data   | -> |  Data Loader     | -> |  LlamaTrainer    |
|  (Tokenized)     |    |  (Batching)      |    |  (Train Loop)    |
+------------------+    +------------------+    +------------------+
                                                        |
                        +------------------+            v
                        |  TrainableModel  | <-  Gradients
                        |  (Weights)       |     & Updates
                        +------------------+
                                |
                                v
                        +------------------+
                        |  Checkpoint      |
                        |  (Save/Resume)   |
                        +------------------+
```

## Preparing Training Data

### Option 1: Binary Token Files

Create tokenized data in binary format (raw u32 little-endian):

```zig
const abi = @import("abi");

// Load from binary file
var dataset = try abi.ai.TokenizedDataset.load(allocator, "data/train.bin");
defer dataset.deinit();
```

### Option 2: In-Memory Data

Use pre-tokenized slices directly:

```zig
const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
var dataset = abi.ai.TokenizedDataset.fromSlice(allocator, &tokens);
defer dataset.deinit();
```

### Option 3: Instruction Tuning (JSONL)

Parse Alpaca/ShareGPT format instruction data:

```zig
const samples = try abi.ai.parseInstructionDataset(allocator, jsonl_content);
defer samples.deinit(allocator);

for (samples.items) |sample| {
    // sample.instruction, sample.input, sample.output
}
```

## Configuration Options

### Model Configuration (TrainableModelConfig)

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `hidden_dim` | Model dimension | 256 - 4096 |
| `num_layers` | Transformer layers | 2 - 32 |
| `num_heads` | Attention heads | 4 - 32 |
| `num_kv_heads` | KV heads (for GQA) | Same as num_heads or less |
| `intermediate_dim` | FFN dimension | 2-4x hidden_dim |
| `vocab_size` | Vocabulary size | 32000 - 128000 |
| `max_seq_len` | Maximum sequence length | 512 - 8192 |

### Training Configuration (LlmTrainingConfig)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Training epochs | 10 |
| `batch_size` | Sequences per batch | 4 |
| `max_seq_len` | Sequence length | 512 |
| `learning_rate` | Base learning rate | 1e-5 |
| `lr_schedule` | Learning rate schedule | warmup_cosine |
| `warmup_steps` | LR warmup steps | 100 |
| `decay_steps` | Total decay steps | 10000 |
| `min_learning_rate` | Minimum learning rate | 1e-7 |
| `grad_accum_steps` | Gradient accumulation | 8 |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `weight_decay` | AdamW weight decay | 0.01 |
| `optimizer` | Optimizer type | adamw |
| `checkpoint_interval` | Steps between checkpoints | 1000 |
| `checkpoint_path` | Checkpoint directory | null |
| `log_interval` | Steps between logs | 10 |
| `mixed_precision` | FP16 training | false |

### Learning Rate Schedules

- `constant` - Fixed learning rate
- `cosine` - Cosine decay
- `warmup_cosine` - Linear warmup + cosine decay (recommended)
- `step` - Step decay
- `polynomial` - Polynomial decay

### Optimizers

- `sgd` - SGD with momentum
- `adam` - Adam optimizer
- `adamw` - AdamW (decoupled weight decay, recommended)

## Running Training

### Programmatic API

```zig
const abi = @import("abi");

// 1. Create model
const model_config = abi.ai.trainable_model.TrainableModelConfig{
    .hidden_dim = 256,
    .num_layers = 4,
    .num_heads = 4,
    .num_kv_heads = 4,
    .intermediate_dim = 512,
    .vocab_size = 32000,
    .max_seq_len = 512,
};
var model = try abi.ai.TrainableModel.init(allocator, model_config);
defer model.deinit();

// 2. Load training data
var dataset = try abi.ai.TokenizedDataset.load(allocator, "train.bin");
defer dataset.deinit();

// 3. Configure training
const train_config = abi.ai.LlmTrainingConfig{
    .epochs = 3,
    .batch_size = 4,
    .learning_rate = 1e-4,
    .checkpoint_interval = 100,
    .checkpoint_path = "checkpoints",
};

// 4. Create trainer and train
var trainer = try abi.ai.LlamaTrainer.init(allocator, &model, train_config);
defer trainer.deinit();

// Training loop
for (0..train_config.epochs) |epoch| {
    var iter = try dataset.batches(allocator, train_config.batch_size, train_config.max_seq_len, true);
    defer iter.deinit();

    while (iter.next()) |batch| {
        const metrics = try trainer.trainStepWithMetrics(batch.input_ids, batch.labels);
        std.debug.print("loss={d:.4} ppl={d:.2}\n", .{metrics.loss, @exp(metrics.loss)});
    }
}
```

### CLI Training

```bash
# Run basic training
zig build run -- train run --epochs 10 --batch-size 4

# Train LLM from GGUF
zig build run -- train llm model.gguf --epochs 1 --lr 1e-5

# View default configuration
zig build run -- train info

# Resume from checkpoint
zig build run -- train resume checkpoints/step_1000.ckpt
```

## Monitoring Progress

### Training Stats

```zig
const stats = trainer.getStats();
std.debug.print("step={d} loss={d:.4} ppl={d:.2} acc={d:.2}%\n", .{
    stats.global_step,
    stats.loss,
    stats.perplexity,
    stats.accuracy * 100,
});
```

### Training Report

After training completes:

```zig
const report = trainer.getReport();
std.debug.print("Final: loss={d:.4} tokens={d} steps={d}\n", .{
    report.final_loss,
    report.total_tokens,
    report.total_steps,
});
```

## Checkpointing

### Saving Checkpoints

Enable in config:

```zig
const config = abi.ai.LlmTrainingConfig{
    .checkpoint_interval = 1000,  // Save every 1000 steps
    .checkpoint_path = "checkpoints",
    .max_checkpoints = 3,  // Keep last 3 checkpoints
};
```

### Loading Checkpoints

```zig
// Load checkpoint and resume training
try trainer.loadCheckpoint("checkpoints/llm_step_1000.ckpt");
```

## Advanced Features

### Gradient Accumulation

For larger effective batch sizes:

```zig
const config = abi.ai.LlmTrainingConfig{
    .batch_size = 2,
    .grad_accum_steps = 8,  // Effective batch size = 2 * 8 = 16
};
```

### Validation and Early Stopping

```zig
trainer.setValidationData(val_tokens);

const early_stop_config = abi.ai.EarlyStoppingConfig{
    .patience = 3,
    .min_delta = 0.001,
    .enabled = true,
};

const result = try trainer.trainEpochWithValidation(train_data, early_stop_config);
if (result.early_stopped) {
    std.debug.print("Training stopped early!\n", .{});
}
```

### LoRA Fine-tuning

For parameter-efficient fine-tuning:

```zig
const lora_config = abi.ai.LoraConfig{
    .rank = 8,
    .alpha = 16,
    .dropout = 0.1,
    .target_modules = .{ .q = true, .v = true },
};

var lora_model = try abi.ai.LoraModel.init(allocator, &model, lora_config);
defer lora_model.deinit();
```

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce sequence length
4. Use gradient accumulation with smaller batches

### Loss Not Decreasing

1. Lower learning rate
2. Increase warmup steps
3. Check data quality/tokenization
4. Ensure data is shuffled

### NaN Loss

1. Lower learning rate
2. Increase gradient clipping
3. Check for data issues (empty sequences, invalid tokens)

### Slow Training

1. Increase batch size (if memory allows)
2. Enable mixed precision training
3. Profile to identify bottlenecks

## Performance Tips

1. **Batch size**: Larger is generally better for throughput
2. **Gradient accumulation**: Use when memory-limited
3. **Learning rate**: Scale with batch size (linear scaling rule)
4. **Warmup**: Essential for stable training
5. **Weight decay**: Prevents overfitting on small datasets

## Related Documentation

- [API Reference](../../API_REFERENCE.md) - Full API documentation
- [AI Module Guide](../../docs/ai.md) - AI subsystem details
- [Troubleshooting](../../docs/troubleshooting.md) - Common issues

