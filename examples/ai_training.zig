//! AI Training Example
//!
//! Demonstrates the training facade module: training pipeline
//! configuration, optimizer selection, and checkpoint management.
//!
//! Run with: `zig build run-ai-training`

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI AI Training Example ===\n\n", .{});

    if (!abi.training.isEnabled()) {
        std.debug.print("Training feature is disabled. Enable with -Denable-training=true\n", .{});
        return;
    }

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withAiDefaults()
        .build();
    defer framework.deinit();

    // --- Training Config ---
    std.debug.print("--- Training Pipeline ---\n", .{});
    const config = abi.ai.TrainingConfig{
        .epochs = 10,
        .batch_size = 64,
        .sample_count = 1024,
        .model_size = 256,
        .learning_rate = 0.0003,
        .optimizer = .adamw,
        .weight_decay = 0.01,
        .gradient_clip_norm = 1.0,
    };

    std.debug.print("Optimizer: adamw\n", .{});
    std.debug.print("Epochs: {d}, Batch size: {d}\n", .{ config.epochs, config.batch_size });
    std.debug.print("Learning rate: {d:.6}\n", .{config.learning_rate});
    std.debug.print("Weight decay: {d}\n", .{config.weight_decay});

    // --- Training Result Type ---
    std.debug.print("\n--- Result Types ---\n", .{});
    const TrainingResult = abi.ai.TrainingResult;
    _ = TrainingResult;
    std.debug.print("TrainingResult type available for tracking metrics\n", .{});

    std.debug.print("\nTraining example complete.\n", .{});
}
