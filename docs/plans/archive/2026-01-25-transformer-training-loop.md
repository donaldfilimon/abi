# Transformer Training Loop Implementation Plan

> **Status: ✅ COMPLETED on January 25, 2026**

**Goal:** Implement actual forward/backward passes in `LlamaTrainer` so that `train new` can train transformer models from scratch with real gradient descent.

**Architecture:** Connect the existing `TrainableModel` weights, forward ops (`ops/*.zig`), and backward ops (`ops/backward/*.zig`) through the `LlamaTrainer.trainStepWithMetrics()` function. The training loop already handles optimizer logic, checkpointing, and logging - we just need to wire up the actual forward pass → loss → backward pass → weight updates.

**Tech Stack:** Zig 0.16, existing ops (matmul, attention, ffn, rmsnorm, rope), existing backward ops, `TrainableModel` with `TrainableWeights` and `ActivationCache`.

---

## Task 1: Add Forward Pass Method to TrainableModel

**Files:**
- Modify: `src/ai/training/trainable_model.zig:507-570`
- Test: `src/ai/training/trainable_model.zig` (add test at end)

**Step 1: Write the failing test**

Add this test at the end of `trainable_model.zig`:

```zig
test "trainable model forward pass" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 2,
        .num_kv_heads = 2,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 16,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    try model.prepareForTraining(8);

    // Input tokens
    var input_ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Output logits buffer
    const logits = try allocator.alloc(f32, 8 * config.vocab_size);
    defer allocator.free(logits);

    // Forward pass
    try model.forward(&input_ids, logits);

    // Check logits are non-zero (model produces output)
    var has_nonzero = false;
    for (logits) |v| {
        if (v != 0 and !std.math.isNan(v)) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/ai/training/trainable_model.zig --test-filter "trainable model forward pass"`
Expected: FAIL with "no member named 'forward'"

**Step 3: Write the forward pass implementation**

Add this method to `TrainableModel` struct after `prepareForTraining`:

```zig
/// Forward pass through the model.
/// input_ids: [seq_len] - token indices
/// logits: [seq_len, vocab_size] - output logits (pre-allocated)
pub fn forward(self: *TrainableModel, input_ids: []const u32, logits: []f32) !void {
    const seq_len: u32 = @intCast(input_ids.len);
    const hidden_dim = self.config.hidden_dim;
    const vocab_size = self.config.vocab_size;

    // Ensure activation cache is prepared
    if (self.activations == null) {
        try self.prepareForTraining(seq_len);
    }

    const act = self.activations.?;

    // Step 1: Token embeddings - lookup from embedding table
    // hidden: [seq_len, hidden_dim]
    const hidden = act.embeddings;
    for (0..seq_len) |i| {
        const token_id = input_ids[i];
        const emb_start = @as(usize, token_id) * hidden_dim;
        const hidden_start = i * hidden_dim;
        @memcpy(hidden[hidden_start..][0..hidden_dim], self.weights.token_embedding[emb_start..][0..hidden_dim]);
    }

    // Step 2: Process through each transformer layer
    for (self.weights.layers, 0..) |*layer, layer_idx| {
        const layer_cache = &act.layer_caches[layer_idx];
        try self.forwardLayer(hidden, layer, layer_cache, seq_len);
    }

    // Step 3: Final RMS normalization
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.rmsNormInPlace(
            hidden[h_start..][0..hidden_dim],
            self.weights.final_norm,
            self.config.norm_eps,
        );
    }

    // Step 4: Output projection (logits = hidden @ output_proj^T)
    const out_proj = self.weights.getOutputProj();
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        const logit_start = i * vocab_size;
        // Matrix-vector: logits[i] = out_proj @ hidden[i]
        for (0..vocab_size) |v| {
            var sum: f32 = 0;
            for (0..hidden_dim) |h| {
                sum += out_proj[v * hidden_dim + h] * hidden[h_start + h];
            }
            logits[logit_start + v] = sum;
        }
    }
}

/// Forward pass for a single transformer layer.
fn forwardLayer(
    self: *TrainableModel,
    hidden: []f32,
    layer: *const TrainableLayerWeights,
    cache: *ActivationCache.LayerActivationCache,
    seq_len: u32,
) !void {
    const hidden_dim = self.config.hidden_dim;
    const num_heads = self.config.num_heads;
    const head_dim = self.config.headDim();
    const intermediate_dim = self.config.intermediate_dim;

    // Cache pre-attention input for backward
    @memcpy(cache.pre_attn_norm[0..seq_len * hidden_dim], hidden[0..seq_len * hidden_dim]);

    // Attention sub-layer with residual
    // 1. RMS Norm
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.rmsNormInPlace(hidden[h_start..][0..hidden_dim], layer.attn_norm, self.config.norm_eps);
    }

    // 2. Self-attention (simplified single-head for now)
    // Q = hidden @ W_q, K = hidden @ W_k, V = hidden @ W_v
    const q = cache.q[0..seq_len * hidden_dim];
    const k = cache.k[0..seq_len * hidden_dim];
    const v = cache.v[0..seq_len * hidden_dim];

    // Compute Q, K, V projections
    ops.matmul.matrixMultiply(hidden[0..seq_len * hidden_dim], layer.w_q, q, seq_len, hidden_dim, hidden_dim);
    ops.matmul.matrixMultiply(hidden[0..seq_len * hidden_dim], layer.w_k, k, seq_len, hidden_dim, hidden_dim);
    ops.matmul.matrixMultiply(hidden[0..seq_len * hidden_dim], layer.w_v, v, seq_len, hidden_dim, hidden_dim);

    // Apply RoPE to Q and K
    if (self.rope_cache) |rc| {
        for (0..seq_len) |pos| {
            for (0..num_heads) |h| {
                const offset = pos * hidden_dim + h * head_dim;
                ops.rope.applyRope(q[offset..][0..head_dim], rc, @intCast(pos));
                ops.rope.applyRope(k[offset..][0..head_dim], rc, @intCast(pos));
            }
        }
    }

    // Attention: scores = Q @ K^T / sqrt(d), attn = softmax(scores), out = attn @ V
    const attn_out = cache.attn_output[0..seq_len * hidden_dim];
    try ops.attention.multiHeadAttention(
        self.allocator,
        q, k, v,
        attn_out,
        seq_len, seq_len, // seq_len == kv_len for training
        hidden_dim,
        num_heads,
        self.config.num_kv_heads,
        head_dim,
        true, // causal
    );

    // 3. Output projection: hidden += attn_out @ W_o
    const attn_proj = cache.attn_proj_output[0..seq_len * hidden_dim];
    ops.matmul.matrixMultiply(attn_out, layer.w_o, attn_proj, seq_len, hidden_dim, hidden_dim);

    // Residual connection
    for (0..seq_len * hidden_dim) |i| {
        hidden[i] = cache.pre_attn_norm[i] + attn_proj[i];
    }

    // FFN sub-layer with residual
    // Cache pre-FFN input
    @memcpy(cache.pre_ffn_norm[0..seq_len * hidden_dim], hidden[0..seq_len * hidden_dim]);

    // 1. RMS Norm
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.rmsNormInPlace(hidden[h_start..][0..hidden_dim], layer.ffn_norm, self.config.norm_eps);
    }

    // 2. FFN: SwiGLU
    const ffn_out = cache.ffn_output[0..seq_len * hidden_dim];
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        const ffn_start = i * hidden_dim;

        // gate = hidden @ W_gate, up = hidden @ W_up
        var gate: [4096]f32 = undefined;
        var up: [4096]f32 = undefined;
        const gate_slice = gate[0..intermediate_dim];
        const up_slice = up[0..intermediate_dim];

        for (0..intermediate_dim) |j| {
            var gate_sum: f32 = 0;
            var up_sum: f32 = 0;
            for (0..hidden_dim) |h| {
                gate_sum += layer.w_gate[j * hidden_dim + h] * hidden[h_start + h];
                up_sum += layer.w_up[j * hidden_dim + h] * hidden[h_start + h];
            }
            gate_slice[j] = gate_sum;
            up_slice[j] = up_sum;
        }

        // intermediate = silu(gate) * up
        for (0..intermediate_dim) |j| {
            gate_slice[j] = ops.activations.silu(gate_slice[j]) * up_slice[j];
        }

        // down = intermediate @ W_down
        for (0..hidden_dim) |h| {
            var sum: f32 = 0;
            for (0..intermediate_dim) |j| {
                sum += layer.w_down[h * intermediate_dim + j] * gate_slice[j];
            }
            ffn_out[ffn_start + h] = sum;
        }
    }

    // Residual connection
    for (0..seq_len * hidden_dim) |i| {
        hidden[i] = cache.pre_ffn_norm[i] + ffn_out[i];
    }
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/ai/training/trainable_model.zig --test-filter "trainable model forward pass"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ai/training/trainable_model.zig
git commit -m "feat(training): add forward pass to TrainableModel"
```

---

## Task 2: Ensure ActivationCache Has Required Fields

**Files:**
- Modify: `src/ai/training/trainable_model.zig:380-505` (ActivationCache and LayerActivationCache)

**Step 1: Verify LayerActivationCache has all required fields**

Check `LayerActivationCache` has these fields (add if missing):
- `pre_attn_norm: []f32` - input before attention norm
- `q: []f32` - Q projections
- `k: []f32` - K projections
- `v: []f32` - V projections
- `attn_output: []f32` - attention output before O projection
- `attn_proj_output: []f32` - output of O projection
- `pre_ffn_norm: []f32` - input before FFN norm
- `gate_out: []f32` - gate projection output (before SiLU)
- `up_out: []f32` - up projection output
- `ffn_output: []f32` - FFN output

**Step 2: Read current implementation**

```bash
zig test src/ai/training/trainable_model.zig --test-filter "trainable model forward pass"
```

If missing fields cause errors, add them to `LayerActivationCache.init()`.

**Step 3: Commit if changes made**

```bash
git add src/ai/training/trainable_model.zig
git commit -m "feat(training): add activation cache fields for forward pass"
```

---

## Task 3: Add Backward Pass Method to TrainableModel

**Files:**
- Modify: `src/ai/training/trainable_model.zig`
- Test: Add test after forward pass test

**Step 1: Write the failing test**

```zig
test "trainable model backward pass" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 2,
        .num_kv_heads = 2,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 16,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    try model.prepareForTraining(8);

    var input_ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const logits = try allocator.alloc(f32, 8 * config.vocab_size);
    defer allocator.free(logits);

    // Forward
    try model.forward(&input_ids, logits);

    // Create gradient of loss w.r.t. logits (simulated)
    const d_logits = try allocator.alloc(f32, 8 * config.vocab_size);
    defer allocator.free(d_logits);
    @memset(d_logits, 0.01); // Small gradient

    // Backward
    try model.backward(d_logits);

    // Check gradients are non-zero
    var has_grad = false;
    for (model.weights.d_token_embedding) |g| {
        if (g != 0) has_grad = true;
    }
    try std.testing.expect(has_grad);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/ai/training/trainable_model.zig --test-filter "trainable model backward pass"`
Expected: FAIL with "no member named 'backward'"

**Step 3: Write the backward pass implementation**

Add this method to `TrainableModel`:

```zig
/// Backward pass through the model.
/// d_logits: [seq_len, vocab_size] - gradient of loss w.r.t. logits
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
    // logits = hidden @ output_proj^T
    // d_hidden += d_logits @ output_proj
    // d_output_proj += d_logits^T @ hidden
    const out_proj = self.weights.getOutputProj();
    const d_out_proj = self.weights.getOutputProjGrad();

    for (0..seq_len) |i| {
        const logit_start = i * vocab_size;
        const h_start = i * hidden_dim;

        // d_hidden[i] += d_logits[i] @ output_proj
        for (0..hidden_dim) |h| {
            var sum: f32 = 0;
            for (0..vocab_size) |v| {
                sum += d_logits[logit_start + v] * out_proj[v * hidden_dim + h];
            }
            d_hidden[h_start + h] += sum;
        }

        // d_output_proj[v, h] += d_logits[i, v] * hidden[i, h]
        for (0..vocab_size) |v| {
            for (0..hidden_dim) |h| {
                d_out_proj[v * hidden_dim + h] += d_logits[logit_start + v] * act.final_hidden[h_start + h];
            }
        }
    }

    // Step 2: Backward through final norm
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.backward.rmsNormBackward(
            d_hidden[h_start..][0..hidden_dim],
            act.final_hidden[h_start..][0..hidden_dim],
            self.weights.final_norm,
            self.weights.d_final_norm,
            d_hidden[h_start..][0..hidden_dim], // d_input (accumulated)
            self.config.norm_eps,
        );
    }

    // Step 3: Backward through layers (reverse order)
    var layer_idx: usize = self.config.num_layers;
    while (layer_idx > 0) {
        layer_idx -= 1;
        const layer = &self.weights.layers[layer_idx];
        const layer_cache = &act.layer_caches[layer_idx];
        try self.backwardLayer(d_hidden, layer, layer_cache, seq_len);
    }

    // Step 4: Backward through embeddings
    // hidden[i] = token_embedding[input_ids[i]]
    // d_token_embedding[input_ids[i]] += d_hidden[i]
    for (0..seq_len) |i| {
        const token_id = layer_cache.input_ids[i]; // Need to cache input_ids
        const emb_start = @as(usize, token_id) * hidden_dim;
        const h_start = i * hidden_dim;
        for (0..hidden_dim) |h| {
            self.weights.d_token_embedding[emb_start + h] += d_hidden[h_start + h];
        }
    }
}

/// Backward pass for a single transformer layer.
fn backwardLayer(
    self: *TrainableModel,
    d_hidden: []f32,
    layer: *TrainableLayerWeights,
    cache: *const ActivationCache.LayerActivationCache,
    seq_len: u32,
) !void {
    const hidden_dim = self.config.hidden_dim;
    const intermediate_dim = self.config.intermediate_dim;

    // Backward through FFN residual
    // output = pre_ffn + ffn_out
    // d_pre_ffn = d_hidden, d_ffn_out = d_hidden

    // Backward through FFN
    // Use swigluBackward for each position
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;

        var swiglu_cache = ops.backward.ffn_backward.SwigluCache{
            .allocator = self.allocator,
            .x = cache.pre_ffn_norm[h_start..][0..hidden_dim],
            .gate_out = cache.gate_out[i * intermediate_dim..][0..intermediate_dim],
            .up_out = cache.up_out[i * intermediate_dim..][0..intermediate_dim],
            .intermediate = cache.ffn_intermediate[i * intermediate_dim..][0..intermediate_dim],
            .hidden_dim = hidden_dim,
            .intermediate_dim = intermediate_dim,
        };

        // Temporary d_x for this position
        var d_x_pos: [4096]f32 = undefined;
        @memset(d_x_pos[0..hidden_dim], 0);

        ops.backward.ffn_backward.swigluBackward(
            d_hidden[h_start..][0..hidden_dim],
            &swiglu_cache,
            layer.w_gate,
            layer.w_up,
            layer.w_down,
            layer.d_w_gate,
            layer.d_w_up,
            layer.d_w_down,
            d_x_pos[0..hidden_dim],
        );

        // Accumulate into d_hidden (for norm backward)
        for (0..hidden_dim) |h| {
            d_hidden[h_start + h] = d_x_pos[h];
        }
    }

    // Backward through FFN norm
    for (0..seq_len) |i| {
        const h_start = i * hidden_dim;
        ops.backward.rmsNormBackward(
            d_hidden[h_start..][0..hidden_dim],
            cache.pre_ffn_norm[h_start..][0..hidden_dim],
            layer.ffn_norm,
            layer.d_ffn_norm,
            d_hidden[h_start..][0..hidden_dim],
            self.config.norm_eps,
        );
    }

    // Add residual gradient: d_pre_ffn += d_hidden (already done above)

    // Similar pattern for attention backward (abbreviated)
    // ... attention backward ops ...
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/ai/training/trainable_model.zig --test-filter "trainable model backward pass"`
Expected: PASS (or identify missing pieces)

**Step 5: Commit**

```bash
git add src/ai/training/trainable_model.zig
git commit -m "feat(training): add backward pass to TrainableModel"
```

---

## Task 4: Connect Forward/Backward to LlamaTrainer.trainStepWithMetrics

**Files:**
- Modify: `src/ai/training/llm_trainer.zig:340-420`

**Step 1: Write the integration test**

```zig
test "llama trainer real training step" {
    const allocator = std.testing.allocator;

    const config = trainable_model.TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 2,
        .num_kv_heads = 2,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 16,
    };

    var model = try trainable_model.TrainableModel.init(allocator, config);
    defer model.deinit();

    var trainer = try LlamaTrainer.init(allocator, &model, .{
        .epochs = 1,
        .batch_size = 4,
        .max_seq_len = 8,
        .learning_rate = 0.01,
    });
    defer trainer.deinit();

    // Training tokens
    var tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    // Get initial loss
    const result1 = try trainer.trainStepWithMetrics(&tokens, tokens[1..]);
    const loss1 = result1.loss;

    // Train for several steps
    for (0..10) |_| {
        _ = try trainer.trainStepWithMetrics(&tokens, tokens[1..]);
    }

    const result2 = try trainer.trainStepWithMetrics(&tokens, tokens[1..]);
    const loss2 = result2.loss;

    // Loss should decrease (or at least change)
    try std.testing.expect(loss2 != loss1);
}
```

**Step 2: Replace placeholder forward pass with real forward pass**

Replace lines 358-362 in `trainStepWithMetrics`:

```zig
// OLD (placeholder):
// var rng = std.Random.DefaultPrng.init(self.stats.global_step);
// for (logits) |*l| {
//     l.* = rng.random().floatNorm(f32) * 0.1;
// }

// NEW (real forward pass):
try self.model.forward(input_ids, logits);
```

**Step 3: Add backward pass after loss computation**

After `self.loss_fn.backward(d_logits);`, add:

```zig
// Backpropagate through model
try self.model.backward(d_logits);
```

**Step 4: Implement real optimizer updates**

Replace the placeholder `applySgdUpdate` and `applyAdamUpdate`:

```zig
fn applySgdUpdate(self: *LlamaTrainer, lr: f32) void {
    // SGD: w = w - lr * grad
    const weights = &self.model.weights;

    // Update token embedding
    for (weights.d_token_embedding, 0..) |g, i| {
        weights.token_embedding[i] -= lr * g;
    }

    // Update each layer
    for (weights.layers) |*layer| {
        for (layer.d_w_q, 0..) |g, i| layer.w_q[i] -= lr * g;
        for (layer.d_w_k, 0..) |g, i| layer.w_k[i] -= lr * g;
        for (layer.d_w_v, 0..) |g, i| layer.w_v[i] -= lr * g;
        for (layer.d_w_o, 0..) |g, i| layer.w_o[i] -= lr * g;
        for (layer.d_attn_norm, 0..) |g, i| layer.attn_norm[i] -= lr * g;
        for (layer.d_w_gate, 0..) |g, i| layer.w_gate[i] -= lr * g;
        for (layer.d_w_up, 0..) |g, i| layer.w_up[i] -= lr * g;
        for (layer.d_w_down, 0..) |g, i| layer.w_down[i] -= lr * g;
        for (layer.d_ffn_norm, 0..) |g, i| layer.ffn_norm[i] -= lr * g;
    }

    // Update final norm
    for (weights.d_final_norm, 0..) |g, i| {
        weights.final_norm[i] -= lr * g;
    }

    // Update output projection if separate
    if (weights.output_proj) |out_proj| {
        if (weights.d_output_proj) |d_out| {
            for (d_out, 0..) |g, i| {
                out_proj[i] -= lr * g;
            }
        }
    }
}

fn applyAdamUpdate(self: *LlamaTrainer, lr: f32, use_weight_decay: bool) void {
    const beta1 = self.optimizer_state.beta1;
    const beta2 = self.optimizer_state.beta2;
    const eps = self.optimizer_state.epsilon;
    const t = @as(f32, @floatFromInt(self.stats.global_step + 1));

    // Bias correction
    const bc1 = 1.0 - std.math.pow(f32, beta1, t);
    const bc2 = 1.0 - std.math.pow(f32, beta2, t);

    // Get flat view of all parameters and gradients
    const params = self.flattenParams();
    const grads = self.flattenGrads();

    for (0..params.len) |i| {
        const g = grads[i];

        // Update first moment
        self.optimizer_state.m.?[i] = beta1 * self.optimizer_state.m.?[i] + (1 - beta1) * g;
        // Update second moment
        self.optimizer_state.v.?[i] = beta2 * self.optimizer_state.v.?[i] + (1 - beta2) * g * g;

        // Compute bias-corrected estimates
        const m_hat = self.optimizer_state.m.?[i] / bc1;
        const v_hat = self.optimizer_state.v.?[i] / bc2;

        // Update parameter
        var update = lr * m_hat / (@sqrt(v_hat) + eps);

        // Weight decay (AdamW style - applied directly to weights)
        if (use_weight_decay) {
            update += lr * self.config.weight_decay * params[i];
        }

        params[i] -= update;
    }
}
```

**Step 5: Run integration test**

Run: `zig test src/ai/training/llm_trainer.zig --test-filter "llama trainer real training step"`
Expected: PASS

**Step 6: Commit**

```bash
git add src/ai/training/llm_trainer.zig
git commit -m "feat(training): connect real forward/backward passes in LlamaTrainer"
```

---

## Task 5: End-to-End Training Test with train new CLI

**Files:**
- Test via CLI

**Step 1: Build the project**

```bash
zig build
```

**Step 2: Run training from scratch**

```bash
zig build run -- train new --hidden-dim 64 --num-layers 2 --num-heads 2 --intermediate-dim 128 --vocab-size 256 --max-seq-len 32 --batch-size 2 --epochs 5 --dataset-path training_data.txt
```

**Step 3: Verify loss decreases**

Expected: Loss should decrease over epochs (not stay at ~5.5)

**Step 4: Run full test suite**

```bash
zig build test --summary all
```

**Step 5: Commit all changes**

```bash
git add -A
git commit -m "feat(training): complete transformer training loop implementation"
```

---

## Summary

This plan implements the transformer training loop in 5 tasks:

1. **Forward pass** - Add `TrainableModel.forward()` that computes token embeddings → transformer layers → output logits
2. **Activation cache** - Ensure all intermediate activations are cached for backward pass
3. **Backward pass** - Add `TrainableModel.backward()` that backpropagates gradients through all layers
4. **Trainer integration** - Connect forward/backward to `LlamaTrainer.trainStepWithMetrics()` and implement real optimizer updates
5. **End-to-end test** - Verify `train new` CLI shows decreasing loss

**Dependencies:**
- All existing backward ops in `src/ai/llm/ops/backward/` are implemented and tested
- `CrossEntropyLoss` forward/backward are implemented
- `TrainableWeights` has gradient storage for all parameters

**Risk factors:**
- Stack overflow from large intermediate buffers (use heap allocation)
- Numerical instability (ensure proper scaling, use `norm_eps`)
- Memory leaks (ensure all allocations have corresponding frees)
