//! Tests for trainable model components.

const std = @import("std");
const trainable_model = @import("trainable_model.zig");
const TrainableModelConfig = trainable_model.TrainableModelConfig;
const TrainableModel = trainable_model.TrainableModel;
const TrainableWeights = trainable_model.TrainableWeights;
const GradientCheckpointer = trainable_model.GradientCheckpointer;

test "trainable model config" {
    const config = TrainableModelConfig{
        .hidden_dim = 512,
        .num_layers = 4,
        .num_heads = 8,
        .num_kv_heads = 8,
        .intermediate_dim = 1024,
        .vocab_size = 1000,
    };

    try std.testing.expectEqual(@as(u32, 64), config.headDim());
    try std.testing.expect(config.numParams() > 0);
}

test "trainable weights init" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var weights = try TrainableWeights.init(allocator, config);
    defer weights.deinit();

    try std.testing.expectEqual(config.vocab_size * config.hidden_dim, weights.token_embedding.len);
    try std.testing.expectEqual(config.num_layers, weights.layers.len);

    // Test zero gradients
    weights.zeroGradients();
    for (weights.d_token_embedding) |g| {
        try std.testing.expectEqual(@as(f32, 0), g);
    }
}

test "trainable model init" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 64,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    try std.testing.expect(model.numParams() > 0);

    // Test prepare for training
    try model.prepareForTraining(32);
    try std.testing.expect(model.activations != null);
}

test "model checkpoint collect/distribute weights" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Collect weights
    const weights = try model.collectWeights();
    defer allocator.free(weights);

    try std.testing.expectEqual(config.numParams(), weights.len);

    // Modify a weight
    model.weights.token_embedding[0] = 42.0;

    // Distribute weights back (should restore original values)
    try model.distributeWeights(weights);

    // Weight should be restored to original value (from Xavier init)
    try std.testing.expect(model.weights.token_embedding[0] != 42.0);
}

test "model checkpoint create/load" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var model1 = try TrainableModel.init(allocator, config);
    defer model1.deinit();

    // Set a known value
    model1.weights.token_embedding[0] = 123.456;

    // Create checkpoint
    var ckpt = try model1.createCheckpoint(42);
    defer ckpt.deinit();

    try std.testing.expectEqual(@as(u64, 42), ckpt.step);
    try std.testing.expectEqual(@as(f32, 123.456), ckpt.weights[0]);

    // Create another model and load checkpoint
    var model2 = try TrainableModel.init(allocator, config);
    defer model2.deinit();

    try model2.loadFromCheckpoint(&ckpt);

    try std.testing.expectEqual(@as(f32, 123.456), model2.weights.token_embedding[0]);
}

test "gradient checkpointer every_n_layers" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 8,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .every_n_layers,
        .checkpoint_interval = 2,
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Layer 0 should be checkpointed (first layer)
    try std.testing.expect(checkpointer.shouldStoreActivations(0));
    // Layer 1 should not
    try std.testing.expect(!checkpointer.shouldStoreActivations(1));
    // Layer 2 should be (interval of 2)
    try std.testing.expect(checkpointer.shouldStoreActivations(2));
    // Layer 7 (last) should be
    try std.testing.expect(checkpointer.shouldStoreActivations(7));

    // Memory savings should be > 0
    const savings = checkpointer.estimateMemorySavings();
    try std.testing.expect(savings > 0);
}

test "gradient checkpointer full recomputation" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .full,
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Only first layer should be checkpointed
    try std.testing.expect(checkpointer.shouldStoreActivations(0));
    try std.testing.expect(!checkpointer.shouldStoreActivations(1));
    try std.testing.expect(!checkpointer.shouldStoreActivations(2));
    try std.testing.expect(!checkpointer.shouldStoreActivations(3));

    // High memory savings
    const savings = checkpointer.estimateMemorySavings();
    try std.testing.expect(savings >= 0.5);
}

test "gradient checkpointer store and retrieve" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .none, // Store all
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Store input for layer 0
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try checkpointer.storeLayerInput(0, &input);

    // Retrieve it
    const retrieved = checkpointer.getLayerInput(0);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualSlices(f32, &input, retrieved.?);

    // Clear
    checkpointer.clearStoredInputs();
    try std.testing.expect(checkpointer.getLayerInput(0) == null);
}

test "forward pass produces valid output" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 1,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 16,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Initialize with small random values (Xavier-like)
    const scale: f32 = 0.1;
    for (model.weights.token_embedding) |*w| {
        w.* = @as(f32, @floatFromInt(@as(i32, @truncate(@as(i64, @bitCast(@as(u64, @intFromPtr(w)))) % 1000)))) * scale * 0.001;
    }

    // Simple input
    const input_ids = [_]u32{ 1, 2, 3 };
    const logits = try allocator.alloc(f32, input_ids.len * config.vocab_size);
    defer allocator.free(logits);

    // Forward should not error
    try model.forward(&input_ids, logits);

    // Check logits are finite (not NaN or Inf)
    for (logits) |l| {
        try std.testing.expect(std.math.isFinite(l));
    }
}

test "cross entropy loss computation" {
    const vocab_size: u32 = 10;
    const seq_len: usize = 3;

    // Create simple logits (uniform)
    var logits: [seq_len * vocab_size]f32 = undefined;
    for (&logits) |*l| {
        l.* = 0.0;
    }

    // Set one logit higher to create non-uniform distribution
    logits[5] = 2.0; // First position, token 5
    logits[vocab_size + 3] = 2.0; // Second position, token 3
    logits[2 * vocab_size + 7] = 2.0; // Third position, token 7

    const targets = [_]u32{ 5, 3, 7 };
    var d_logits: [seq_len * vocab_size]f32 = undefined;

    const loss = TrainableModel.computeCrossEntropyLoss(&logits, &targets, &d_logits, vocab_size);

    // Loss should be positive
    try std.testing.expect(loss > 0);

    // Loss should be lower than log(vocab_size) since we boosted correct logits
    const max_loss = @log(@as(f32, vocab_size));
    try std.testing.expect(loss < max_loss);

    // Gradients should be finite
    for (d_logits) |g| {
        try std.testing.expect(std.math.isFinite(g));
    }

    // Gradient for correct class should be negative (prob - 1)
    try std.testing.expect(d_logits[5] < 0);
}

test "train step computes loss and gradients" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 16,
        .num_layers = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .intermediate_dim = 32,
        .vocab_size = 50,
        .max_seq_len = 8,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Initialize with small random-ish values
    for (model.weights.token_embedding, 0..) |*w, i| {
        w.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 100)))) * 0.01 - 0.5;
    }
    for (model.weights.layers) |*layer| {
        for (layer.w_q) |*w| {
            w.* = 0.01;
        }
        for (layer.w_k) |*w| {
            w.* = 0.01;
        }
        for (layer.w_v) |*w| {
            w.* = 0.01;
        }
        for (layer.w_o) |*w| {
            w.* = 0.01;
        }
        for (layer.w_gate) |*w| {
            w.* = 0.01;
        }
        for (layer.w_up) |*w| {
            w.* = 0.01;
        }
        for (layer.w_down) |*w| {
            w.* = 0.01;
        }
        for (layer.attn_norm) |*w| {
            w.* = 1.0;
        }
        for (layer.ffn_norm) |*w| {
            w.* = 1.0;
        }
    }
    for (model.weights.final_norm) |*w| {
        w.* = 1.0;
    }

    // Zero gradients before training step
    model.zeroGradients();

    const input_ids = [_]u32{ 1, 2, 3, 4 };
    const target_ids = [_]u32{ 2, 3, 4, 5 };

    const loss = try model.trainStep(&input_ids, &target_ids);

    // Loss should be positive and finite
    try std.testing.expect(loss > 0);
    try std.testing.expect(std.math.isFinite(loss));

    // Some gradients should be non-zero
    var has_nonzero_grad = false;
    for (model.weights.d_token_embedding) |g| {
        if (g != 0) {
            has_nonzero_grad = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero_grad);
}
