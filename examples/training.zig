//! Training Example
//!
//! Demonstrates the training module for:
//! - Model training with various optimizers
//! - Checkpoint saving and resuming
//! - Training metrics and reporting
//! - Gradient checkpointing for memory efficiency

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI Training Example ===\n\n", .{});

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    // Initialize framework
    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
        .enable_gpu = false,
    }) catch |err| {
        std.debug.print("Framework initialization failed: {}\n", .{err});
        return err;
    };
    defer abi.shutdown(&framework);

    // === Training Configuration ===
    std.debug.print("--- Training Configuration ---\n", .{});

    const config = abi.ai.TrainingConfig{
        .epochs = 5,
        .batch_size = 32,
        .sample_count = 256,
        .model_size = 128,
        .learning_rate = 0.001,
        .optimizer = .adamw,
        .weight_decay = 0.01,
        .gradient_clip_norm = 1.0,
        .checkpoint_interval = 2,
        .checkpoint_path = "training_example.ckpt",
    };

    std.debug.print("Epochs: {d}\n", .{config.epochs});
    std.debug.print("Batch size: {d}\n", .{config.batch_size});
    std.debug.print("Sample count: {d}\n", .{config.sample_count});
    std.debug.print("Model size: {d}\n", .{config.model_size});
    std.debug.print("Learning rate: {d:.6}\n", .{config.learning_rate});
    std.debug.print("Optimizer: {t}\n", .{config.optimizer});

    // === Run Training ===
    std.debug.print("\n--- Starting Training ---\n", .{});

    var result = abi.ai.trainWithResult(allocator, config) catch |err| {
        std.debug.print("Training failed: {}\n", .{err});
        return err;
    };
    defer result.deinit();

    // === Training Report ===
    std.debug.print("\n--- Training Report ---\n", .{});

    const report = result.report;
    const initial_loss = if (result.loss_history.len > 0) result.loss_history[0] else report.final_loss;
    const improvement = if (initial_loss > 0) (1.0 - report.final_loss / initial_loss) * 100.0 else 0.0;
    const total_batches = @as(u64, report.batches) * @as(u64, report.epochs);
    std.debug.print("Initial loss: {d:.6}\n", .{initial_loss});
    std.debug.print("Final loss: {d:.6}\n", .{report.final_loss});
    std.debug.print("Improvement: {d:.2}%\n", .{improvement});
    std.debug.print("Total batches: {d}\n", .{total_batches});
    std.debug.print("Training time: {d:.2}ms\n", .{@as(f64, @floatFromInt(report.total_time_ms))});

    // === Loss History ===
    if (result.loss_history.len > 0) {
        std.debug.print("\n--- Loss History ---\n", .{});
        for (result.loss_history, 0..) |loss, epoch| {
            const bar_len = @as(usize, @intFromFloat(@min(loss * 50.0, 50.0)));
            var bar: [51]u8 = undefined;
            @memset(&bar, ' ');
            for (0..bar_len) |i| {
                bar[i] = '#';
            }
            bar[50] = 0;
            std.debug.print("Epoch {d:2}: {d:.6} |{s}|\n", .{ epoch + 1, loss, bar[0..50] });
        }
    }

    // === Checkpoint Info ===
    std.debug.print("\n--- Checkpoint ---\n", .{});
    if (result.checkpoints.latest()) |ckpt| {
        std.debug.print("Checkpoint saved at step {d}\n", .{ckpt.step});
        std.debug.print("Weights captured: {d}\n", .{ckpt.weights.len});
        if (config.checkpoint_path) |checkpoint_path| {
            std.debug.print("Path: {s}\n", .{checkpoint_path});
        }
    } else {
        std.debug.print("No checkpoint saved\n", .{});
    }

    // === Resume Training Demo ===
    std.debug.print("\n--- Resume Training Demo ---\n", .{});

    // Try to load checkpoint and continue training
    if (config.checkpoint_path) |checkpoint_path| {
        if (abi.ai.loadCheckpoint(allocator, checkpoint_path)) |checkpoint| {
            var checkpoint_mut = checkpoint;
            defer checkpoint_mut.deinit(allocator);
            std.debug.print("Loaded checkpoint from step {d}\n", .{checkpoint.step});
            std.debug.print("Checkpoint weights: {d}\n", .{checkpoint.weights.len});
        } else |_| {
            std.debug.print("No checkpoint to resume from\n", .{});
        }
    } else {
        std.debug.print("No checkpoint path configured\n", .{});
    }

    std.debug.print("\n=== Training Example Complete ===\n", .{});
}
