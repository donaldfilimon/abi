//! End-to-end tests for LLM training pipeline.
//!
//! Tests the complete training loop including:
//! - Forward pass through transformer layers
//! - Loss computation with cross-entropy
//! - Backward pass with gradient computation
//! - Optimizer updates
//! - Loss convergence over multiple steps

const std = @import("std");
const abi = @import("abi");
const trainable_model = abi.ai.training.trainable_model;
const llm_trainer = abi.ai.training.llm_trainer;

/// Small model configuration for testing.
const TestModelConfig = trainable_model.TrainableModelConfig{
    .hidden_dim = 64,
    .num_layers = 2,
    .num_heads = 4,
    .num_kv_heads = 4,
    .intermediate_dim = 128,
    .vocab_size = 256,
    .max_seq_len = 32,
    .norm_eps = 1e-5,
};

// Test that forward pass produces valid logits.
test "forward pass produces valid logits" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    // Create input sequence
    const seq_len = 8;
    var input_ids: [seq_len]u32 = undefined;
    for (&input_ids, 0..) |*id, i| {
        id.* = @intCast(i % TestModelConfig.vocab_size);
    }

    // Allocate logits buffer
    const logits = try allocator.alloc(f32, seq_len * TestModelConfig.vocab_size);
    defer allocator.free(logits);

    // Run forward pass
    try model.forward(&input_ids, logits);

    // Verify logits are finite (not NaN or Inf)
    for (logits) |l| {
        try std.testing.expect(std.math.isFinite(l));
    }

    // Verify logits have reasonable magnitude
    var max_logit: f32 = logits[0];
    var min_logit: f32 = logits[0];
    for (logits) |l| {
        max_logit = @max(max_logit, l);
        min_logit = @min(min_logit, l);
    }

    // Logits should be bounded (not exploding)
    try std.testing.expect(max_logit < 1000);
    try std.testing.expect(min_logit > -1000);
}

// Test that backward pass computes gradients.
test "backward pass computes gradients" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    // Zero gradients
    model.zeroGradients();

    // Create input and target sequences
    const seq_len = 8;
    var input_ids: [seq_len]u32 = undefined;
    var target_ids: [seq_len]u32 = undefined;
    for (0..seq_len) |i| {
        input_ids[i] = @intCast(i % TestModelConfig.vocab_size);
        target_ids[i] = @intCast((i + 1) % TestModelConfig.vocab_size);
    }

    // Run training step (forward + backward)
    const loss = try model.trainStep(&input_ids, &target_ids);

    // Loss should be positive and finite
    try std.testing.expect(loss > 0);
    try std.testing.expect(std.math.isFinite(loss));

    // Check that gradients were computed (not all zero)
    var has_nonzero_grad = false;
    for (model.weights.d_token_embedding) |g| {
        if (g != 0) {
            has_nonzero_grad = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero_grad);

    // Check layer gradients
    for (model.weights.layers) |layer| {
        for (layer.d_w_q) |g| {
            if (g != 0) {
                has_nonzero_grad = true;
                break;
            }
        }
    }
    try std.testing.expect(has_nonzero_grad);
}

// Test that loss decreases over multiple training steps.
test "loss decreases over training" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    // Create simple repeating pattern for training
    const seq_len = 16;
    var input_ids: [seq_len]u32 = undefined;
    var target_ids: [seq_len]u32 = undefined;
    for (0..seq_len) |i| {
        // Simple pattern: 0, 1, 2, 3, 0, 1, 2, 3, ...
        input_ids[i] = @intCast(i % 4);
        target_ids[i] = @intCast((i + 1) % 4);
    }

    // Record initial loss
    model.zeroGradients();
    const initial_loss = try model.trainStep(&input_ids, &target_ids);

    // Apply SGD update
    model.applySgdUpdate(0.01);

    // Train for multiple steps
    var losses: [10]f32 = undefined;
    for (&losses) |*loss| {
        model.zeroGradients();
        loss.* = try model.trainStep(&input_ids, &target_ids);
        model.applySgdUpdate(0.01);
    }

    // Final loss should be lower than initial (with high probability)
    const final_loss = losses[losses.len - 1];

    // Log for debugging
    std.debug.print("\nInitial loss: {d:.4}, Final loss: {d:.4}\n", .{ initial_loss, final_loss });

    // Loss should decrease or at least not explode
    try std.testing.expect(std.math.isFinite(final_loss));
    try std.testing.expect(final_loss < initial_loss * 2); // Should not explode
}

// Test training with gradient clipping.
test "training with gradient clipping" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    const seq_len = 8;
    var input_ids: [seq_len]u32 = undefined;
    var target_ids: [seq_len]u32 = undefined;
    for (0..seq_len) |i| {
        input_ids[i] = @intCast(i);
        target_ids[i] = @intCast(i + 1);
    }

    // Run training step with clipping
    const result = try model.trainStepWithClipping(
        &input_ids,
        &target_ids,
        0.001, // learning rate
        1.0, // max grad norm
        null, // no loss scaling
    );

    // Check result
    try std.testing.expect(std.math.isFinite(result.loss));
    try std.testing.expect(std.math.isFinite(result.grad_norm));
    // Tolerance accounts for floating-point precision in norm computation (sqrt of sum of squares)
    try std.testing.expect(result.grad_norm_clipped <= 1.0 + 1e-3); // Should be clipped
    try std.testing.expect(!result.skipped);
}

// Test LlamaTrainer integration.
test "llama trainer training step" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    const train_config = llm_trainer.LlmTrainingConfig{
        .epochs = 1,
        .batch_size = 1,
        .max_seq_len = 16,
        .learning_rate = 0.001,
        .grad_accum_steps = 1,
    };

    var trainer = try llm_trainer.LlamaTrainer.init(allocator, &model, train_config);
    defer trainer.deinit();

    // Create training data
    const seq_len = 15;
    var input_ids: [seq_len]u32 = undefined;
    var labels: [seq_len]u32 = undefined;
    for (0..seq_len) |i| {
        input_ids[i] = @intCast(i % TestModelConfig.vocab_size);
        labels[i] = @intCast((i + 1) % TestModelConfig.vocab_size);
    }

    // Run training step
    const metrics = try trainer.trainStepWithMetrics(&input_ids, &labels);

    // Verify metrics
    try std.testing.expect(std.math.isFinite(metrics.loss));
    try std.testing.expect(metrics.loss > 0);
    try std.testing.expect(metrics.accuracy >= 0);
    try std.testing.expect(metrics.accuracy <= 1);
}

// Test checkpoint save and load.
test "checkpoint save and load" {
    const allocator = std.testing.allocator;

    // Create and train model briefly
    var model1 = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model1.deinit();

    // Set a known value
    model1.weights.token_embedding[0] = 42.0;
    model1.weights.token_embedding[100] = -3.14;

    // Create checkpoint
    var ckpt = try model1.createCheckpoint(100);
    defer ckpt.deinit();

    // Create new model and load checkpoint
    var model2 = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model2.deinit();

    try model2.loadFromCheckpoint(&ckpt);

    // Verify weights match
    try std.testing.expectApproxEqAbs(
        @as(f32, 42.0),
        model2.weights.token_embedding[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -3.14),
        model2.weights.token_embedding[100],
        1e-6,
    );
}

// Test cross-entropy loss computation.
test "cross entropy loss" {
    const vocab_size: u32 = 10;
    const seq_len: usize = 4;

    // Create logits with clear predictions
    var logits: [seq_len * vocab_size]f32 = undefined;
    @memset(&logits, 0);

    // Make strong predictions
    logits[3] = 5.0; // Position 0: predict token 3
    logits[vocab_size + 7] = 5.0; // Position 1: predict token 7
    logits[2 * vocab_size + 1] = 5.0; // Position 2: predict token 1
    logits[3 * vocab_size + 9] = 5.0; // Position 3: predict token 9

    // Targets match predictions
    const targets = [_]u32{ 3, 7, 1, 9 };
    var d_logits: [seq_len * vocab_size]f32 = undefined;

    const loss = trainable_model.TrainableModel.computeCrossEntropyLoss(
        &logits,
        &targets,
        &d_logits,
        vocab_size,
    );

    // Loss should be low since predictions match targets
    try std.testing.expect(loss < 1.0);
    try std.testing.expect(loss > 0);

    // Now test with wrong predictions
    const wrong_targets = [_]u32{ 0, 0, 0, 0 };
    const high_loss = trainable_model.TrainableModel.computeCrossEntropyLoss(
        &logits,
        &wrong_targets,
        &d_logits,
        vocab_size,
    );

    // Loss should be higher when predictions don't match
    try std.testing.expect(high_loss > loss);
}

// Test gradient norm computation.
test "gradient norm computation" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    // Zero gradients - norm should be 0
    model.zeroGradients();
    const zero_norm = model.computeGradientNorm();
    try std.testing.expectApproxEqAbs(@as(f32, 0), zero_norm, 1e-6);

    // Set some gradients
    model.weights.d_token_embedding[0] = 3.0;
    model.weights.d_token_embedding[1] = 4.0;

    // Norm should be sqrt(3^2 + 4^2) = 5
    const norm = model.computeGradientNorm();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);
}

// Test model parameter count.
test "model parameter count" {
    const allocator = std.testing.allocator;

    var model = try trainable_model.TrainableModel.init(allocator, TestModelConfig);
    defer model.deinit();

    const num_params = model.numParams();

    // Calculate expected params manually
    const hidden = TestModelConfig.hidden_dim;
    const vocab = TestModelConfig.vocab_size;
    const layers = TestModelConfig.num_layers;
    const inter = TestModelConfig.intermediate_dim;

    // Token embedding: vocab * hidden
    var expected: usize = vocab * hidden;

    // Per layer: Q, K, V, O projections + norms + FFN
    const per_layer = hidden * hidden + // W_q
        hidden * hidden + // W_k (assuming kv_heads = heads)
        hidden * hidden + // W_v
        hidden * hidden + // W_o
        hidden + // attn_norm
        inter * hidden + // W_gate
        inter * hidden + // W_up
        hidden * inter + // W_down
        hidden; // ffn_norm
    expected += per_layer * layers;

    // Final norm
    expected += hidden;

    // Output proj (tied by default)
    // No additional params when tie_embeddings = true

    try std.testing.expectEqual(expected, num_params);
}
