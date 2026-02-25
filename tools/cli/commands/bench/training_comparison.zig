//! Training comparison benchmarks.
//!
//! Compares training optimizers (AdamW vs Adam vs SGD) side by side.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runTrainingComparisonBenchmarks(allocator: std.mem.Allocator, json_mode: bool) void {
    if (!abi.ai.training.isEnabled()) {
        if (!json_mode) {
            utils.output.printWarning("Training feature is disabled. Rebuild with -Denable-training=true", .{});
        }
        return;
    }

    const TrainingBenchEntry = struct {
        method: []const u8,
        optimizer: []const u8,
        final_loss: f32,
        total_time_ms: u64,
        epochs: u32,
        batches: u32,
    };

    const configs = [_]struct {
        method: []const u8,
        optimizer: abi.ai.training.OptimizerType,
    }{
        .{ .method = "Full fine-tune", .optimizer = .adamw },
        .{ .method = "Full fine-tune", .optimizer = .adam },
        .{ .method = "Full fine-tune", .optimizer = .sgd },
    };

    var results: [configs.len]TrainingBenchEntry = undefined;
    var result_count: usize = 0;

    if (!json_mode) {
        utils.output.println("\nTraining Benchmark Comparison", .{});
        utils.output.println("=============================", .{});
        utils.output.println("Config: epochs=5, batch_size=8, lr=0.001\n", .{});
    }

    for (configs) |cfg| {
        const train_config = abi.ai.training.TrainingConfig{
            .epochs = 5,
            .batch_size = 8,
            .learning_rate = 0.001,
            .optimizer = cfg.optimizer,
            .gradient_accumulation_steps = 1,
            .gradient_clip_norm = 1.0,
            .weight_decay = 0.01,
            .early_stopping_patience = 0,
            .checkpoint_interval = 0,
        };

        const report = abi.ai.training.trainAndReport(allocator, train_config) catch |err| {
            if (!json_mode) {
                utils.output.printError("  {s} ({t}): error - {t}", .{ cfg.method, cfg.optimizer, err });
            }
            continue;
        };

        results[result_count] = .{
            .method = cfg.method,
            .optimizer = std.mem.sliceTo(@tagName(cfg.optimizer), 0),
            .final_loss = report.final_loss,
            .total_time_ms = report.total_time_ms,
            .epochs = report.epochs,
            .batches = report.batches,
        };
        result_count += 1;
    }

    if (result_count == 0) {
        if (!json_mode) {
            utils.output.printWarning("No training benchmarks completed.", .{});
        }
        return;
    }

    if (json_mode) {
        utils.output.println("{{ \"training_benchmarks\": [", .{});
        for (results[0..result_count], 0..) |r, idx| {
            utils.output.print("  {{ \"method\": \"{s}\", \"optimizer\": \"{s}\", \"final_loss\": {d:.4}, \"time_ms\": {d}, \"epochs\": {d}, \"batches\": {d} }}", .{
                r.method,
                r.optimizer,
                r.final_loss,
                r.total_time_ms,
                r.epochs,
                r.batches,
            });
            if (idx < result_count - 1) utils.output.print(",", .{});
            utils.output.println("", .{});
        }
        utils.output.println("] }}", .{});
    } else {
        utils.output.println("  {s:<18} {s:<10} {s:>12} {s:>10} {s:>8}", .{ "Method", "Optimizer", "Final Loss", "Time (ms)", "Batches" });
        utils.output.println("  {s:<18} {s:<10} {s:>12} {s:>10} {s:>8}", .{ "-" ** 18, "-" ** 10, "-" ** 12, "-" ** 10, "-" ** 8 });
        for (results[0..result_count]) |r| {
            utils.output.println("  {s:<18} {s:<10} {d:>12.4} {d:>10} {d:>8}", .{
                r.method,
                r.optimizer,
                r.final_loss,
                r.total_time_ms,
                r.batches,
            });
        }

        // Speed comparison relative to first entry
        if (result_count >= 2) {
            const baseline_ms = results[0].total_time_ms;
            if (baseline_ms > 0) {
                utils.output.println("\n  Speed vs {s} ({s}):", .{ results[0].method, results[0].optimizer });
                for (results[1..result_count]) |r| {
                    if (r.total_time_ms > 0) {
                        const ratio = @as(f64, @floatFromInt(baseline_ms)) / @as(f64, @floatFromInt(r.total_time_ms));
                        utils.output.println("    {s} ({s}): {d:.2}x", .{ r.method, r.optimizer, ratio });
                    }
                }
            }
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
