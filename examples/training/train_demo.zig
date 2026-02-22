//! LLM Training Demo
//!
//! Demonstrates the ABI LLM training pipeline:
//! - Creating a trainable model with configurable architecture
//! - Generating synthetic tokenized training data
//! - Configuring the training hyperparameters
//! - Running training epochs with loss tracking
//! - Reporting training statistics (loss, perplexity, accuracy)
//!
//! Usage: zig build run-train-demo
//!
//! This demo uses synthetic data so it runs without external files.

const std = @import("std");
const abi = @import("abi");

/// Configuration for the training demo
const DemoConfig = struct {
    /// Vocabulary size for the synthetic tokenizer
    vocab_size: u32 = 256,
    /// Hidden dimension of the model
    hidden_dim: u32 = 64,
    /// Number of transformer layers
    num_layers: u32 = 2,
    /// Number of attention heads
    num_heads: u32 = 4,
    /// Intermediate FFN dimension
    intermediate_dim: u32 = 128,
    /// Maximum sequence length
    max_seq_len: u32 = 32,
    /// Number of training epochs
    epochs: u32 = 3,
    /// Batch size
    batch_size: u32 = 2,
    /// Learning rate
    learning_rate: f32 = 1e-4,
    /// Number of synthetic training samples
    num_samples: usize = 16,
};

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("========================================\n", .{});
    std.debug.print("     ABI LLM Training Demo\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Check if AI feature is enabled
    if (!abi.ai.isEnabled()) {
        std.debug.print("Error: AI feature is disabled.\n", .{});
        std.debug.print("Build with: zig build -Denable-ai=true\n", .{});
        return;
    }

    const config = DemoConfig{};

    // === Step 1: Create the Model ===
    std.debug.print("Step 1: Creating Trainable Model\n", .{});
    std.debug.print("--------------------------------\n", .{});

    const model_config = abi.ai.training.TrainableModelConfig{
        .hidden_dim = config.hidden_dim,
        .num_layers = config.num_layers,
        .num_heads = config.num_heads,
        .num_kv_heads = config.num_heads,
        .intermediate_dim = config.intermediate_dim,
        .vocab_size = config.vocab_size,
        .max_seq_len = config.max_seq_len,
    };

    var model = abi.ai.training.TrainableModel.init(allocator, model_config) catch |err| {
        std.debug.print("Failed to create model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    const num_params = model.numParams();
    const param_mb = @as(f64, @floatFromInt(num_params * 4)) / (1024.0 * 1024.0);

    std.debug.print("  Architecture:\n", .{});
    std.debug.print("    - Hidden dim:       {d}\n", .{config.hidden_dim});
    std.debug.print("    - Layers:           {d}\n", .{config.num_layers});
    std.debug.print("    - Attention heads:  {d}\n", .{config.num_heads});
    std.debug.print("    - FFN dim:          {d}\n", .{config.intermediate_dim});
    std.debug.print("    - Vocab size:       {d}\n", .{config.vocab_size});
    std.debug.print("    - Max seq length:   {d}\n", .{config.max_seq_len});
    std.debug.print("  Parameters:           {d} ({d:.2} MB)\n\n", .{ num_params, param_mb });

    // === Step 2: Generate Synthetic Training Data ===
    std.debug.print("Step 2: Generating Synthetic Training Data\n", .{});
    std.debug.print("------------------------------------------\n", .{});

    const train_data = try generateSyntheticData(
        allocator,
        config.vocab_size,
        config.max_seq_len,
        config.num_samples,
    );
    defer allocator.free(train_data);

    std.debug.print("  Samples:              {d}\n", .{config.num_samples});
    std.debug.print("  Tokens per sample:    {d}\n", .{config.max_seq_len});
    std.debug.print("  Total tokens:         {d}\n\n", .{train_data.len});

    // Create a dataset from the synthetic data
    var dataset = abi.ai.training.TokenizedDataset.fromSlice(allocator, train_data);
    defer dataset.deinit();

    const num_batches = dataset.numBatches(config.batch_size, config.max_seq_len);
    std.debug.print("  Batches per epoch:    {d}\n", .{num_batches});
    std.debug.print("  Batch size:           {d}\n\n", .{config.batch_size});

    // === Step 3: Configure Training ===
    std.debug.print("Step 3: Configuring Training\n", .{});
    std.debug.print("----------------------------\n", .{});

    const train_config = abi.ai.training.LlmTrainingConfig{
        .epochs = config.epochs,
        .batch_size = config.batch_size,
        .max_seq_len = config.max_seq_len,
        .learning_rate = config.learning_rate,
        .lr_schedule = .warmup_cosine,
        .warmup_steps = 2,
        .decay_steps = 20,
        .min_learning_rate = 1e-6,
        .grad_accum_steps = 1,
        .max_grad_norm = 1.0,
        .weight_decay = 0.01,
        .optimizer = .adamw,
        .log_interval = 1,
    };

    std.debug.print("  Epochs:               {d}\n", .{train_config.epochs});
    std.debug.print("  Learning rate:        {e:.2}\n", .{train_config.learning_rate});
    std.debug.print("  LR schedule:          warmup_cosine\n", .{});
    std.debug.print("  Warmup steps:         {d}\n", .{train_config.warmup_steps});
    std.debug.print("  Optimizer:            AdamW\n", .{});
    std.debug.print("  Weight decay:         {d:.4}\n", .{train_config.weight_decay});
    std.debug.print("  Gradient clip norm:   {d:.2}\n\n", .{train_config.max_grad_norm});

    // === Step 4: Create Trainer and Run Training ===
    std.debug.print("Step 4: Running Training\n", .{});
    std.debug.print("------------------------\n\n", .{});

    var trainer = abi.ai.training.LlamaTrainer.init(allocator, &model, train_config) catch |err| {
        std.debug.print("Failed to create trainer: {t}\n", .{err});
        return;
    };
    defer trainer.deinit();

    var timer = abi.shared.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    // Training loop
    var epoch_losses: [10]f32 = undefined;
    var epoch_accuracies: [10]f32 = undefined;

    for (0..config.epochs) |epoch| {
        std.debug.print("Epoch {d}/{d}:\n", .{ epoch + 1, config.epochs });

        // Reset batch iterator for each epoch
        var batch_iter = try dataset.batches(allocator, config.batch_size, config.max_seq_len, true);
        defer batch_iter.deinit();

        var epoch_loss: f32 = 0;
        var epoch_acc: f32 = 0;
        var batch_count: u32 = 0;

        while (batch_iter.next()) |batch| {
            // Train on this batch
            const metrics = trainer.trainStepWithMetrics(batch.input_ids, batch.labels) catch |err| {
                std.debug.print("  Training step failed: {t}\n", .{err});
                continue;
            };

            epoch_loss += metrics.loss;
            epoch_acc += metrics.accuracy;
            batch_count += 1;

            // Print batch progress
            std.debug.print("  Batch {d}: loss={d:.4}, acc={d:.2}%\n", .{
                batch_count,
                metrics.loss,
                metrics.accuracy * 100,
            });
        }

        // Compute epoch averages
        if (batch_count > 0) {
            epoch_loss /= @as(f32, @floatFromInt(batch_count));
            epoch_acc /= @as(f32, @floatFromInt(batch_count));
        }

        if (epoch < 10) {
            epoch_losses[epoch] = epoch_loss;
            epoch_accuracies[epoch] = epoch_acc;
        }

        const perplexity = @exp(epoch_loss);
        std.debug.print("  -> Epoch avg: loss={d:.4}, ppl={d:.2}, acc={d:.2}%\n\n", .{
            epoch_loss,
            perplexity,
            epoch_acc * 100,
        });
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    // === Step 5: Training Report ===
    std.debug.print("Step 5: Training Report\n", .{});
    std.debug.print("-----------------------\n", .{});

    const stats = trainer.getStats();
    const final_ppl = @exp(stats.loss);

    std.debug.print("  Total steps:          {d}\n", .{stats.global_step});
    std.debug.print("  Tokens processed:     {d}\n", .{stats.tokens_processed});
    std.debug.print("  Final loss:           {d:.4}\n", .{stats.loss});
    std.debug.print("  Final perplexity:     {d:.2}\n", .{final_ppl});
    std.debug.print("  Final accuracy:       {d:.2}%\n", .{stats.accuracy * 100});
    std.debug.print("  Training time:        {d}ms\n", .{elapsed_ms});

    if (stats.tokens_processed > 0 and elapsed_ms > 0) {
        const tokens_per_sec = @as(f64, @floatFromInt(stats.tokens_processed)) /
            (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
        std.debug.print("  Throughput:           {d:.0} tokens/sec\n", .{tokens_per_sec});
    }

    // === Loss History Visualization ===
    std.debug.print("\n", .{});
    std.debug.print("Loss History:\n", .{});
    std.debug.print("-------------\n", .{});

    const epochs_to_show = @min(config.epochs, 10);
    for (0..epochs_to_show) |i| {
        const loss_val = epoch_losses[i];
        const bar_len = @min(@as(usize, @intFromFloat(loss_val * 20)), 40);
        var bar: [41]u8 = undefined;
        for (0..bar_len) |j| {
            bar[j] = '#';
        }
        for (bar_len..40) |j| {
            bar[j] = ' ';
        }
        bar[40] = 0;
        std.debug.print("  Epoch {d}: {d:.4} |{s}|\n", .{ i + 1, loss_val, bar[0..40] });
    }

    std.debug.print("\n========================================\n", .{});
    std.debug.print("     Training Demo Complete!\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Print next steps
    std.debug.print("Next Steps:\n", .{});
    std.debug.print("-----------\n", .{});
    std.debug.print("1. Try different hyperparameters:\n", .{});
    std.debug.print("   - Increase epochs for better convergence\n", .{});
    std.debug.print("   - Adjust learning rate (try 1e-3 to 1e-5)\n", .{});
    std.debug.print("   - Increase model size for more capacity\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("2. Use real training data:\n", .{});
    std.debug.print("   - Load tokenized data with TokenizedDataset.load()\n", .{});
    std.debug.print("   - Use parseInstructionDataset() for JSONL format\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("3. Enable checkpointing:\n", .{});
    std.debug.print("   - Set checkpoint_interval and checkpoint_path\n", .{});
    std.debug.print("   - Resume training with loadCheckpoint()\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("4. CLI training:\n", .{});
    std.debug.print("   zig build run -- train run --epochs 10 --batch-size 4\n", .{});
    std.debug.print("   zig build run -- train info\n", .{});
    std.debug.print("\n", .{});
}

/// Generate synthetic tokenized training data with learnable patterns.
/// Creates sequences where tokens follow simple arithmetic patterns
/// that a model can learn to predict.
fn generateSyntheticData(
    allocator: std.mem.Allocator,
    vocab_size: u32,
    seq_len: u32,
    num_samples: usize,
) ![]u32 {
    const total_tokens = num_samples * seq_len;
    const data = try allocator.alloc(u32, total_tokens);

    // Use a fixed seed for reproducibility
    var rng = std.Random.DefaultPrng.init(42);

    for (0..num_samples) |sample_idx| {
        const base_offset = sample_idx * seq_len;

        // Choose a pattern for this sample
        const pattern_type = sample_idx % 4;

        switch (pattern_type) {
            0 => {
                // Pattern 1: Counting sequence (token N is followed by token N+1)
                const start = rng.random().uintLessThan(u32, vocab_size / 2);
                for (0..seq_len) |i| {
                    data[base_offset + i] = (start + @as(u32, @intCast(i))) % vocab_size;
                }
            },
            1 => {
                // Pattern 2: Repeating sequence (ABCABC...)
                const period = 3 + rng.random().uintLessThan(u32, 3);
                var tokens: [6]u32 = undefined;
                for (0..period) |i| {
                    tokens[i] = rng.random().uintLessThan(u32, vocab_size);
                }
                for (0..seq_len) |i| {
                    data[base_offset + i] = tokens[i % period];
                }
            },
            2 => {
                // Pattern 3: Alternating tokens (ABABAB...)
                const a = rng.random().uintLessThan(u32, vocab_size);
                const b = rng.random().uintLessThan(u32, vocab_size);
                for (0..seq_len) |i| {
                    data[base_offset + i] = if (i % 2 == 0) a else b;
                }
            },
            3 => {
                // Pattern 4: Fibonacci-like (each token is sum of prev two mod vocab)
                var prev_prev: u32 = rng.random().uintLessThan(u32, vocab_size / 4);
                var prev: u32 = rng.random().uintLessThan(u32, vocab_size / 4);
                data[base_offset] = prev_prev;
                if (seq_len > 1) {
                    data[base_offset + 1] = prev;
                }
                for (2..seq_len) |i| {
                    const next = (prev + prev_prev) % vocab_size;
                    data[base_offset + i] = next;
                    prev_prev = prev;
                    prev = next;
                }
            },
            else => unreachable,
        }
    }

    return data;
}

test "synthetic data generation" {
    const allocator = std.testing.allocator;

    const data = try generateSyntheticData(allocator, 256, 32, 4);
    defer allocator.free(data);

    try std.testing.expectEqual(@as(usize, 4 * 32), data.len);

    // Check all tokens are within vocab range
    for (data) |token| {
        try std.testing.expect(token < 256);
    }
}

test "demo config defaults" {
    const config = DemoConfig{};
    try std.testing.expectEqual(@as(u32, 256), config.vocab_size);
    try std.testing.expectEqual(@as(u32, 3), config.epochs);
}
