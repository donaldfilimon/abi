//! Training info display handler.
//!
//! Handles the `abi train info` subcommand which shows default training
//! configuration values and available optimizers/schedules.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runInfo() void {
    const default_config = abi.features.ai.training.TrainingConfig{};

    utils.output.printHeader("Default Training Configuration");
    utils.output.println("", .{});
    utils.output.println("Recommended Model: GPT-2 Small (124M parameters)", .{});
    utils.output.println("  - Open source, no authentication required", .{});
    utils.output.println("  - Download: https://huggingface.co/TheBloke/gpt2-GGUF", .{});
    utils.output.println("  - Training: abi train llm models/gpt2.gguf --epochs 1", .{});
    utils.output.println("", .{});
    utils.output.println("Basic parameters:", .{});
    utils.output.printKeyValueFmt("epochs", "{d}", .{default_config.epochs});
    utils.output.printKeyValueFmt("batch_size", "{d}", .{default_config.batch_size});
    utils.output.printKeyValueFmt("sample_count", "{d}", .{default_config.sample_count});
    utils.output.printKeyValueFmt("model_size", "{d}", .{default_config.model_size});
    utils.output.println("", .{});
    utils.output.println("Optimizer settings:", .{});
    utils.output.printKeyValueFmt("learning_rate", "{d:.6}", .{default_config.learning_rate});
    utils.output.printKeyValueFmt("optimizer", "{t}", .{default_config.optimizer});
    utils.output.printKeyValueFmt("learning_rate_schedule", "{t}", .{default_config.learning_rate_schedule});
    utils.output.printKeyValueFmt("warmup_steps", "{d}", .{default_config.warmup_steps});
    utils.output.printKeyValueFmt("decay_steps", "{d}", .{default_config.decay_steps});
    utils.output.printKeyValueFmt("min_learning_rate", "{d:.6}", .{default_config.min_learning_rate});
    utils.output.printKeyValueFmt("weight_decay", "{d:.4}", .{default_config.weight_decay});
    utils.output.println("", .{});
    utils.output.println("Gradient settings:", .{});
    utils.output.printKeyValueFmt("gradient_accumulation_steps", "{d}", .{default_config.gradient_accumulation_steps});
    utils.output.printKeyValueFmt("gradient_clip_norm", "{d:.2}", .{default_config.gradient_clip_norm});
    utils.output.println("", .{});
    utils.output.println("Checkpointing:", .{});
    utils.output.printKeyValueFmt("checkpoint_interval", "{d}", .{default_config.checkpoint_interval});
    utils.output.printKeyValueFmt("max_checkpoints", "{d}", .{default_config.max_checkpoints});
    utils.output.println("", .{});
    utils.output.println("Early stopping:", .{});
    utils.output.printKeyValueFmt("early_stopping_patience", "{d}", .{default_config.early_stopping_patience});
    utils.output.printKeyValueFmt("early_stopping_threshold", "{d:.6}", .{default_config.early_stopping_threshold});
    utils.output.println("", .{});
    utils.output.println("Other:", .{});
    utils.output.printKeyValueFmt("mixed_precision", "{}", .{default_config.mixed_precision});
    utils.output.println("", .{});
    utils.output.println("Available optimizers: sgd, adam, adamw", .{});
    utils.output.println("Available LR schedules: constant, cosine, warmup_cosine, step, polynomial", .{});
}

test {
    std.testing.refAllDecls(@This());
}
