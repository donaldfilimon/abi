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

    if (!abi.features.ai.training.isEnabled()) {
        utils.output.printError("training feature is not enabled. Build with -Dfeat-training=true (legacy: -Denable-training=true)", .{});
        return;
    }

    var config = abi.features.ai.training.TrainingConfig{};

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
        utils.output.printError("Invalid configuration: {t}", .{err});
        return;
    };

    // Print configuration summary
    utils.output.printHeader("Training Configuration");
    utils.output.printKeyValueFmt("Epochs", "{d}", .{config.epochs});
    utils.output.printKeyValueFmt("Batch size", "{d}", .{config.batch_size});
    utils.output.printKeyValueFmt("Sample count", "{d}", .{config.sample_count});
    utils.output.printKeyValueFmt("Model size", "{d}", .{config.model_size});
    utils.output.printKeyValueFmt("Learning rate", "{d:.6}", .{config.learning_rate});
    utils.output.printKeyValueFmt("Optimizer", "{t}", .{config.optimizer});
    utils.output.printKeyValueFmt("LR schedule", "{t}", .{config.learning_rate_schedule});
    utils.output.printKeyValueFmt("Warmup steps", "{d}", .{config.warmup_steps});
    utils.output.printKeyValueFmt("Decay steps", "{d}", .{config.decay_steps});
    utils.output.printKeyValueFmt("Weight decay", "{d:.6}", .{config.weight_decay});
    utils.output.printKeyValueFmt("Gradient clip", "{d:.2}", .{config.gradient_clip_norm});
    utils.output.printKeyValueFmt("Grad accumulation", "{d}", .{config.gradient_accumulation_steps});
    if (config.checkpoint_interval > 0) {
        utils.output.printKeyValueFmt("Checkpoint interval", "{d}", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        utils.output.printKeyValueFmt("Checkpoint path", "{s}", .{path});
    }
    utils.output.printKeyValueFmt("Mixed precision", "{}", .{config.mixed_precision});
    utils.output.println("", .{});

    // Run training
    utils.output.println("Starting training...", .{});
    utils.output.println("", .{});

    var timer = abi.services.shared.time.Timer.start() catch {
        utils.output.printError("Failed to start timer", .{});
        return;
    };

    var result = abi.features.ai.training.trainWithResult(allocator, config) catch |err| {
        utils.output.printError("Training failed: {t}", .{err});
        return;
    };
    defer result.deinit();

    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    // Print results
    utils.output.printHeader("Training Complete");
    utils.output.printKeyValueFmt("Epochs completed", "{d}", .{result.report.epochs});
    utils.output.printKeyValueFmt("Batches/epoch", "{d}", .{result.report.batches});
    utils.output.printKeyValueFmt("Gradient updates", "{d}", .{result.report.gradient_updates});
    utils.output.printKeyValueFmt("Final loss", "{d:.6}", .{result.report.final_loss});
    utils.output.printKeyValueFmt("Final accuracy", "{d:.2}%", .{result.report.final_accuracy * 100});
    utils.output.printKeyValueFmt("Best loss", "{d:.6}", .{result.report.best_loss});
    utils.output.printKeyValueFmt("Final LR", "{d:.8}", .{result.report.learning_rate});
    utils.output.printKeyValueFmt("Checkpoints saved", "{d}", .{result.report.checkpoints_saved});
    if (result.report.early_stopped) {
        utils.output.printKeyValue("Early stopped", "yes");
    }
    utils.output.printKeyValueFmt("Total time", "{d}ms", .{elapsed_ms});

    // Print loss history (abbreviated)
    if (result.loss_history.len > 0) {
        utils.output.println("", .{});
        utils.output.println("Loss history (sampled):", .{});
        const step_size = @max(1, result.loss_history.len / 5);
        var idx: usize = 0;
        while (idx < result.loss_history.len) : (idx += step_size) {
            utils.output.println("  Epoch {d}: loss={d:.6}, acc={d:.2}%", .{
                idx + 1,
                result.loss_history[idx],
                result.accuracy_history[idx] * 100,
            });
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
