//! Basic training pipeline handler.
//!
//! Handles the `abi train run` subcommand which runs a basic training pipeline
//! with configurable epochs, batch size, learning rate, optimizer, and more.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const common = @import("common.zig");
const mod = @import("mod.zig");

pub fn runTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (!abi.ai.training.isEnabled()) {
        utils.output.printError("training feature is not enabled. Build with -Denable-training=true", .{});
        return;
    }

    var config = abi.ai.training.TrainingConfig{};

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.batch_size = std.fmt.parseInt(u32, val, 10) catch 32;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--model-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.model_size = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--sample-count")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.sample_count = std.fmt.parseInt(usize, val, 10) catch 1024;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate = std.fmt.parseFloat(f32, val) catch 0.001;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--optimizer")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.optimizer = common.parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate_schedule = common.parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.warmup_steps = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--decay-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.decay_steps = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.gradient_clip_norm = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-accumulation")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.gradient_accumulation_steps = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.checkpoint_interval = std.fmt.parseInt(u32, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--checkpoint-path")) {
            if (i < args.len) {
                config.checkpoint_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-checkpoints")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_checkpoints = std.fmt.parseInt(u32, val, 10) catch 5;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--early-stopping-patience")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.early_stopping_patience = std.fmt.parseInt(u32, val, 10) catch 5;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mixed-precision")) {
            config.mixed_precision = true;
            continue;
        }
    }

    // Validate configuration
    config.validate() catch |err| {
        std.debug.print("Invalid configuration: {t}\n", .{err});
        return;
    };

    // Print configuration summary
    std.debug.print("Training Configuration\n", .{});
    std.debug.print("======================\n", .{});
    std.debug.print("Epochs:           {d}\n", .{config.epochs});
    std.debug.print("Batch size:       {d}\n", .{config.batch_size});
    std.debug.print("Sample count:     {d}\n", .{config.sample_count});
    std.debug.print("Model size:       {d}\n", .{config.model_size});
    std.debug.print("Learning rate:    {d:.6}\n", .{config.learning_rate});
    std.debug.print("Optimizer:        {t}\n", .{config.optimizer});
    std.debug.print("LR schedule:      {t}\n", .{config.learning_rate_schedule});
    std.debug.print("Warmup steps:     {d}\n", .{config.warmup_steps});
    std.debug.print("Decay steps:      {d}\n", .{config.decay_steps});
    std.debug.print("Weight decay:     {d:.6}\n", .{config.weight_decay});
    std.debug.print("Gradient clip:    {d:.2}\n", .{config.gradient_clip_norm});
    std.debug.print("Grad accumulation:{d}\n", .{config.gradient_accumulation_steps});
    if (config.checkpoint_interval > 0) {
        std.debug.print("Checkpoint interval: {d}\n", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        std.debug.print("Checkpoint path:  {s}\n", .{path});
    }
    std.debug.print("Mixed precision:  {}\n", .{config.mixed_precision});
    std.debug.print("\n", .{});

    // Run training
    std.debug.print("Starting training...\n\n", .{});

    var timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("Failed to start timer\n", .{});
        return;
    };

    var result = abi.ai.training.trainWithResult(allocator, config) catch |err| {
        std.debug.print("Training failed: {t}\n", .{err});
        return;
    };
    defer result.deinit();

    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    // Print results
    std.debug.print("Training Complete\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Epochs completed: {d}\n", .{result.report.epochs});
    std.debug.print("Batches/epoch:    {d}\n", .{result.report.batches});
    std.debug.print("Gradient updates: {d}\n", .{result.report.gradient_updates});
    std.debug.print("Final loss:       {d:.6}\n", .{result.report.final_loss});
    std.debug.print("Final accuracy:   {d:.2}%\n", .{result.report.final_accuracy * 100});
    std.debug.print("Best loss:        {d:.6}\n", .{result.report.best_loss});
    std.debug.print("Final LR:         {d:.8}\n", .{result.report.learning_rate});
    std.debug.print("Checkpoints saved:{d}\n", .{result.report.checkpoints_saved});
    if (result.report.early_stopped) {
        std.debug.print("Early stopped:    yes\n", .{});
    }
    std.debug.print("Total time:       {d}ms\n", .{elapsed_ms});

    // Print loss history (abbreviated)
    if (result.loss_history.len > 0) {
        std.debug.print("\nLoss history (sampled):\n", .{});
        const step_size = @max(1, result.loss_history.len / 5);
        var idx: usize = 0;
        while (idx < result.loss_history.len) : (idx += step_size) {
            std.debug.print("  Epoch {d}: loss={d:.6}, acc={d:.2}%\n", .{
                idx + 1,
                result.loss_history[idx],
                result.accuracy_history[idx] * 100,
            });
        }
    }
}
