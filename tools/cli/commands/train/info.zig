//! Training info display handler.
//!
//! Handles the `abi train info` subcommand which shows default training
//! configuration values and available optimizers/schedules.

const std = @import("std");
const abi = @import("abi");

pub fn runInfo() void {
    const default_config = abi.ai.training.TrainingConfig{};

    std.debug.print("Default Training Configuration\n", .{});
    std.debug.print("==============================\n\n", .{});
    std.debug.print("Recommended Model: GPT-2 Small (124M parameters)\n", .{});
    std.debug.print("  - Open source, no authentication required\n", .{});
    std.debug.print("  - Download: https://huggingface.co/TheBloke/gpt2-GGUF\n", .{});
    std.debug.print("  - Training: abi train llm models/gpt2.gguf --epochs 1\n\n", .{});
    std.debug.print("Basic parameters:\n", .{});
    std.debug.print("  epochs:                    {d}\n", .{default_config.epochs});
    std.debug.print("  batch_size:                {d}\n", .{default_config.batch_size});
    std.debug.print("  sample_count:              {d}\n", .{default_config.sample_count});
    std.debug.print("  model_size:                {d}\n", .{default_config.model_size});
    std.debug.print("\nOptimizer settings:\n", .{});
    std.debug.print("  learning_rate:             {d:.6}\n", .{default_config.learning_rate});
    std.debug.print("  optimizer:                 {t}\n", .{default_config.optimizer});
    std.debug.print("  learning_rate_schedule:    {t}\n", .{default_config.learning_rate_schedule});
    std.debug.print("  warmup_steps:              {d}\n", .{default_config.warmup_steps});
    std.debug.print("  decay_steps:               {d}\n", .{default_config.decay_steps});
    std.debug.print("  min_learning_rate:         {d:.6}\n", .{default_config.min_learning_rate});
    std.debug.print("  weight_decay:              {d:.4}\n", .{default_config.weight_decay});
    std.debug.print("\nGradient settings:\n", .{});
    std.debug.print("  gradient_accumulation_steps:{d}\n", .{default_config.gradient_accumulation_steps});
    std.debug.print("  gradient_clip_norm:        {d:.2}\n", .{default_config.gradient_clip_norm});
    std.debug.print("\nCheckpointing:\n", .{});
    std.debug.print("  checkpoint_interval:       {d}\n", .{default_config.checkpoint_interval});
    std.debug.print("  max_checkpoints:           {d}\n", .{default_config.max_checkpoints});
    std.debug.print("\nEarly stopping:\n", .{});
    std.debug.print("  early_stopping_patience:   {d}\n", .{default_config.early_stopping_patience});
    std.debug.print("  early_stopping_threshold:  {d:.6}\n", .{default_config.early_stopping_threshold});
    std.debug.print("\nOther:\n", .{});
    std.debug.print("  mixed_precision:           {}\n", .{default_config.mixed_precision});
    std.debug.print("\nAvailable optimizers: sgd, adam, adamw\n", .{});
    std.debug.print("Available LR schedules: constant, cosine, warmup_cosine, step, polynomial\n", .{});
}
