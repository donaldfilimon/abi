# Multi-Model Training System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a comprehensive multi-model training system that can train LLMs, vision models, embeddings, and multimodal models with self-learning capabilities, voice/image/video generation, and full abi/abbey/aviva integration.

**Architecture:** A unified training framework connecting existing `TrainableModel`, forward/backward ops, `SelfLearningSystem`, `VisionTransformer`, `CLIPModel`, and `DocumentPipeline` through a new `UnifiedTrainer` orchestrator. Each model type has specialized training pipelines that share common infrastructure.

**Tech Stack:** Zig 0.16, existing ops (`matmul`, `attention`, `ffn`, `rmsnorm`, `rope`), backward ops, GPU acceleration (CUDA/Vulkan/Metal), GGUF import/export.

---

## Phase 1: Complete Transformer Training Loop (Tasks 1-5)

### Task 1: Implement Forward Pass in TrainableModel

**Files:**
- Modify: `src/ai/training/trainable_model.zig`

**Step 1: Add forward method signature**

Add after `prepareForTraining()` (around line 570):

```zig
/// Forward pass through the transformer model.
/// input_ids: [seq_len] - input token indices
/// logits: [seq_len * vocab_size] - output logits (pre-allocated)
/// Populates self.activations with intermediate states for backward pass.
pub fn forward(self: *TrainableModel, input_ids: []const u32, logits: []f32) !void {
    const seq_len: u32 = @intCast(input_ids.len);
    const hidden_dim = self.config.hidden_dim;
    const vocab_size = self.config.vocab_size;

    // Ensure activation cache is prepared
    if (self.activations == null) {
        try self.prepareForTraining(seq_len);
    }

    const act = &self.activations.?;
    act.input_ids = input_ids;
    act.seq_len = seq_len;

    // Step 1: Token embeddings lookup
    const hidden = act.embeddings;
    for (0..seq_len) |i| {
        const token_id = input_ids[i];
        const emb_start = @as(usize, token_id) * hidden_dim;
        const h_start = i * hidden_dim;
        @memcpy(hidden[h_start..][0..hidden_dim],
                self.weights.token_embedding[emb_start..][0..hidden_dim]);
    }

    // Step 2: Process through transformer layers
    for (self.weights.layers, 0..) |*layer, idx| {
        try self.forwardLayer(hidden, layer, &act.layer_caches[idx], seq_len);
    }

    // Step 3: Final RMS normalization
    @memcpy(act.final_hidden[0..seq_len * hidden_dim], hidden[0..seq_len * hidden_dim]);
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.rmsNormInPlace(hidden[h_start..][0..hidden_dim],
                           self.weights.final_norm, self.config.norm_eps);
    }

    // Step 4: Output projection to logits
    const out_proj = self.weights.getOutputProj();
    ops.matmul.matrixMultiply(
        hidden[0..seq_len * hidden_dim],
        out_proj,
        logits,
        seq_len, hidden_dim, vocab_size
    );
}
```

**Step 2: Run test**

```bash
zig test src/ai/training/trainable_model.zig --test-filter "trainable model forward"
```

**Step 3: Commit**

```bash
git add src/ai/training/trainable_model.zig
git commit -m "feat(training): add forward pass to TrainableModel"
```

---

### Task 2: Implement Backward Pass in TrainableModel

**Files:**
- Modify: `src/ai/training/trainable_model.zig`

**Step 1: Add backward method**

```zig
/// Backward pass through the transformer model.
/// d_logits: [seq_len * vocab_size] - gradient of loss w.r.t. logits
/// Accumulates gradients in self.weights.d_* fields.
pub fn backward(self: *TrainableModel, d_logits: []const f32) !void {
    const act = self.activations orelse return error.NotPreparedForTraining;
    const seq_len = act.seq_len;
    const hidden_dim = self.config.hidden_dim;
    const vocab_size = self.config.vocab_size;

    // Allocate gradient buffer for hidden states
    const d_hidden = try self.allocator.alloc(f32, @as(usize, seq_len) * hidden_dim);
    defer self.allocator.free(d_hidden);
    @memset(d_hidden, 0);

    // Step 1: Backward through output projection
    const out_proj = self.weights.getOutputProj();
    const d_out_proj = self.weights.getOutputProjGrad();

    ops.backward.matmul_backward.matmulBackward(
        d_logits,
        act.final_hidden[0..seq_len * hidden_dim],
        out_proj,
        d_hidden,
        d_out_proj,
        seq_len, hidden_dim, vocab_size
    );

    // Step 2: Backward through final norm
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.backward.rmsnorm_backward.rmsNormBackward(
            d_hidden[h_start..][0..hidden_dim],
            act.final_hidden[h_start..][0..hidden_dim],
            self.weights.final_norm,
            d_hidden[h_start..][0..hidden_dim],
            self.weights.d_final_norm,
            self.config.norm_eps
        );
    }

    // Step 3: Backward through layers (reverse order)
    var layer_idx: usize = self.config.num_layers;
    while (layer_idx > 0) {
        layer_idx -= 1;
        try self.backwardLayer(d_hidden,
                               &self.weights.layers[layer_idx],
                               &act.layer_caches[layer_idx],
                               seq_len);
    }

    // Step 4: Backward through embeddings
    for (0..seq_len) |i| {
        const token_id = act.input_ids[i];
        const emb_start = @as(usize, token_id) * hidden_dim;
        const h_start = i * hidden_dim;
        for (0..hidden_dim) |h| {
            self.weights.d_token_embedding[emb_start + h] += d_hidden[h_start + h];
        }
    }
}
```

**Step 2: Run test**

```bash
zig test src/ai/training/trainable_model.zig --test-filter "backward"
```

**Step 3: Commit**

```bash
git add src/ai/training/trainable_model.zig
git commit -m "feat(training): add backward pass to TrainableModel"
```

---

### Task 3: Connect Forward/Backward to LlamaTrainer

**Files:**
- Modify: `src/ai/training/llm_trainer.zig`

**Step 1: Replace placeholder in trainStepWithMetrics (lines 356-362)**

Replace:
```zig
// Placeholder: generate random logits
var rng = std.Random.DefaultPrng.init(self.stats.global_step);
for (logits) |*l| {
    l.* = rng.random().floatNorm(f32) * 0.1;
}
```

With:
```zig
// Real forward pass
try self.model.forward(input_ids, logits);
```

**Step 2: Add backward call after loss computation (around line 380)**

After `self.loss_fn.backward(d_logits);`, add:
```zig
// Backpropagate through model
try self.model.backward(d_logits);
```

**Step 3: Run integration test**

```bash
zig test src/ai/training/llm_trainer.zig --test-filter "training step"
```

**Step 4: Commit**

```bash
git add src/ai/training/llm_trainer.zig
git commit -m "feat(training): connect real forward/backward in LlamaTrainer"
```

---

### Task 4: Implement Real Optimizer Updates

**Files:**
- Modify: `src/ai/training/llm_trainer.zig`

**Step 1: Implement applySgdUpdate (around line 478)**

```zig
fn applySgdUpdate(self: *LlamaTrainer, lr: f32) void {
    const weights = &self.model.weights;

    // Update token embedding
    for (weights.d_token_embedding, 0..) |g, i| {
        weights.token_embedding[i] -= lr * g;
    }

    // Update each layer
    for (weights.layers) |*layer| {
        // Attention weights
        for (layer.d_w_q, 0..) |g, i| layer.w_q[i] -= lr * g;
        for (layer.d_w_k, 0..) |g, i| layer.w_k[i] -= lr * g;
        for (layer.d_w_v, 0..) |g, i| layer.w_v[i] -= lr * g;
        for (layer.d_w_o, 0..) |g, i| layer.w_o[i] -= lr * g;
        for (layer.d_attn_norm, 0..) |g, i| layer.attn_norm[i] -= lr * g;

        // FFN weights
        for (layer.d_w_gate, 0..) |g, i| layer.w_gate[i] -= lr * g;
        for (layer.d_w_up, 0..) |g, i| layer.w_up[i] -= lr * g;
        for (layer.d_w_down, 0..) |g, i| layer.w_down[i] -= lr * g;
        for (layer.d_ffn_norm, 0..) |g, i| layer.ffn_norm[i] -= lr * g;
    }

    // Final norm
    for (weights.d_final_norm, 0..) |g, i| {
        weights.final_norm[i] -= lr * g;
    }

    // Output projection
    if (weights.output_proj) |proj| {
        if (weights.d_output_proj) |d_proj| {
            for (d_proj, 0..) |g, i| proj[i] -= lr * g;
        }
    }
}
```

**Step 2: Implement applyAdamUpdate**

Similar structure but with momentum and velocity updates.

**Step 3: Commit**

```bash
git add src/ai/training/llm_trainer.zig
git commit -m "feat(training): implement real SGD and Adam optimizer updates"
```

---

### Task 5: End-to-End LLM Training Test

**Files:**
- Create: `src/tests/e2e/llm_training_e2e.zig`

**Step 1: Write end-to-end test**

```zig
test "llm training loss decreases" {
    const allocator = std.testing.allocator;

    const config = trainable_model.TrainableModelConfig{
        .hidden_dim = 64,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 128,
        .vocab_size = 256,
        .max_seq_len = 32,
    };

    var model = try trainable_model.TrainableModel.init(allocator, config);
    defer model.deinit();

    var trainer = try llm_trainer.LlamaTrainer.init(allocator, &model, .{
        .epochs = 5,
        .batch_size = 4,
        .max_seq_len = 16,
        .learning_rate = 0.001,
    });
    defer trainer.deinit();

    // Training data
    var tokens: [64]u32 = undefined;
    for (0..64) |i| tokens[i] = @intCast(i % 256);

    const loss1 = (try trainer.trainStepWithMetrics(&tokens, tokens[1..])).loss;

    // Train for multiple steps
    for (0..20) |_| {
        _ = try trainer.trainStepWithMetrics(&tokens, tokens[1..]);
    }

    const loss2 = (try trainer.trainStepWithMetrics(&tokens, tokens[1..])).loss;

    // Loss should decrease
    try std.testing.expect(loss2 < loss1);
}
```

**Step 2: Run test**

```bash
zig test src/tests/e2e/llm_training_e2e.zig
```

**Step 3: Test via CLI**

```bash
zig build run -- train new --hidden-dim 64 --num-layers 2 --epochs 5 --dataset-path training_data.txt
```

**Step 4: Commit**

```bash
git add src/tests/e2e/llm_training_e2e.zig
git commit -m "test(e2e): add LLM training integration test"
```

---

## Phase 2: Vision Model Training (Tasks 6-9)

### Task 6: Add Training Support to VisionTransformer

**Files:**
- Modify: `src/ai/vision/vit.zig`

**Step 1: Add trainable weights structure**

```zig
pub const TrainableViTWeights = struct {
    patch_proj: []f32,      // [patch_dim, hidden_size]
    d_patch_proj: []f32,    // gradients
    pos_embed: []f32,       // [seq_len + 1, hidden_size]
    d_pos_embed: []f32,
    class_token: []f32,     // [hidden_size]
    d_class_token: []f32,
    layers: []TrainableViTLayerWeights,
    final_norm: []f32,
    d_final_norm: []f32,
    head: ?[]f32,           // Classification head
    d_head: ?[]f32,

    pub fn init(allocator: Allocator, config: ViTConfig) !TrainableViTWeights { ... }
    pub fn deinit(self: *TrainableViTWeights) void { ... }
    pub fn zeroGradients(self: *TrainableViTWeights) void { ... }
};
```

**Step 2: Add forward/backward methods**

**Step 3: Commit**

```bash
git add src/ai/vision/vit.zig
git commit -m "feat(vision): add training support to VisionTransformer"
```

---

### Task 7: Create Vision Trainer

**Files:**
- Create: `src/ai/training/vision_trainer.zig`

**Step 1: Create VisionTrainer struct**

```zig
pub const VisionTrainer = struct {
    allocator: Allocator,
    model: *TrainableViT,
    config: VisionTrainingConfig,
    optimizer: Optimizer,
    stats: TrainingStats,

    pub const VisionTrainingConfig = struct {
        epochs: u32 = 10,
        batch_size: u32 = 32,
        learning_rate: f32 = 1e-4,
        image_size: u32 = 224,
        num_classes: u32 = 1000,
        use_augmentation: bool = true,
        mixup_alpha: f32 = 0.2,
    };

    pub fn trainStep(self: *VisionTrainer, images: []const f32, labels: []const u32) !TrainStepResult { ... }
    pub fn validate(self: *VisionTrainer, val_images: []const f32, val_labels: []const u32) !ValidationResult { ... }
};
```

**Step 2: Add image augmentation**

**Step 3: Commit**

---

### Task 8: Implement Image Classification Loss

**Files:**
- Modify: `src/ai/training/loss.zig`

Add softmax cross-entropy for multi-class classification with label smoothing.

---

### Task 9: Vision Training E2E Test

Test vision model training on synthetic data.

---

## Phase 3: Multimodal Training (Tasks 10-13)

### Task 10: Add Training to CLIPModel

**Files:**
- Modify: `src/ai/vision/multimodal.zig`

Add trainable weights and contrastive loss backward pass.

---

### Task 11: Create MultimodalTrainer

**Files:**
- Create: `src/ai/training/multimodal_trainer.zig`

Orchestrates joint vision-language training with contrastive loss.

---

### Task 12: Implement Text Encoder Training

Add trainable text encoder for CLIP-style learning.

---

### Task 13: Multimodal Training E2E Test

Test image-text alignment improves over training.

---

## Phase 4: Self-Learning Integration (Tasks 14-17)

### Task 14: Connect PolicyNetwork to TrainableModel

**Files:**
- Modify: `src/ai/training/self_learning.zig`

Make PolicyNetwork use actual model weights for actor-critic.

---

### Task 15: Implement DPO Training Loop

**Files:**
- Modify: `src/ai/training/self_learning.zig`

Connect DPOOptimizer to LlamaTrainer for preference learning.

---

### Task 16: Add Feedback-Driven Fine-tuning

Connect FeedbackIntegrator to trigger training updates.

---

### Task 17: Self-Learning Integration Test

Test that user feedback improves model responses.

---

## Phase 5: Generation Features (Tasks 18-21)

### Task 18: Implement Voice Generation Stub

**Files:**
- Create: `src/ai/generation/voice.zig`

Stub for future TTS/voice synthesis integration.

---

### Task 19: Implement Image Generation Stub

**Files:**
- Create: `src/ai/generation/image.zig`

Stub for diffusion model integration.

---

### Task 20: Implement Video Generation Stub

**Files:**
- Create: `src/ai/generation/video.zig`

Stub for video generation pipeline.

---

### Task 21: Create Unified Generation API

**Files:**
- Create: `src/ai/generation/mod.zig`

Unified API for all generation modalities.

---

## Phase 6: CLI and Demo Integration (Tasks 22-25)

### Task 22: Enhance train CLI Command

**Files:**
- Modify: `tools/cli/commands/train.zig`

Add subcommands: `train llm`, `train vision`, `train multimodal`, `train self-learn`

---

### Task 23: Add Training Dashboard to TUI

Real-time loss curves, GPU utilization, checkpoint status.

---

### Task 24: Create Training Demo

Demo script showing all training pipelines working.

---

### Task 25: Update Documentation

Update CLAUDE.md, docs/ai.md with new training capabilities.

---

## Summary

This plan implements comprehensive multi-model training in 25 tasks across 6 phases:

1. **Transformer Training** (5 tasks) - Complete forward/backward loop
2. **Vision Training** (4 tasks) - ViT with image classification
3. **Multimodal Training** (4 tasks) - CLIP-style contrastive learning
4. **Self-Learning** (4 tasks) - RLHF/DPO integration
5. **Generation** (4 tasks) - Voice/image/video stubs
6. **Integration** (4 tasks) - CLI, TUI, demos, docs

**Key Dependencies:**
- Backward ops already implemented in `src/ai/llm/ops/backward/`
- TrainableModel has weight/gradient storage
- SelfLearningSystem has RLHF/DPO infrastructure
- ViT and CLIPModel have forward implementations

**Risk Factors:**
- Memory pressure with large models - use gradient checkpointing
- Numerical stability - use proper initialization and normalization
- GPU memory - implement model parallelism if needed
