//! Training comparison benchmarks.
//!
//! Compares training optimizers (AdamW vs Adam vs SGD) side by side.

const std = @import("std");
const abi = @import("abi");

pub fn runTrainingComparisonBenchmarks(allocator: std.mem.Allocator, json_mode: bool) void {
    if (!abi.ai.training.isEnabled()) {
        if (!json_mode) {
            std.debug.print("Training feature is disabled. Rebuild with -Denable-training=true\n", .{});
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
        std.debug.print("\nTraining Benchmark Comparison\n", .{});
        std.debug.print("=============================\n", .{});
        std.debug.print("Config: epochs=5, batch_size=8, lr=0.001\n\n", .{});
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
                std.debug.print("  {s} ({t}): error - {t}\n", .{ cfg.method, cfg.optimizer, err });
            }
            continue;
        };

        results[result_count] = .{
            .method = cfg.method,
            .optimizer = @tagName(cfg.optimizer),
            .final_loss = report.final_loss,
            .total_time_ms = report.total_time_ms,
            .epochs = report.epochs,
            .batches = report.batches,
        };
        result_count += 1;
    }

    if (result_count == 0) {
        if (!json_mode) {
            std.debug.print("  No training benchmarks completed.\n", .{});
        }
        return;
    }

    if (json_mode) {
        std.debug.print("{{ \"training_benchmarks\": [\n", .{});
        for (results[0..result_count], 0..) |r, idx| {
            std.debug.print("  {{ \"method\": \"{s}\", \"optimizer\": \"{s}\", \"final_loss\": {d:.4}, \"time_ms\": {d}, \"epochs\": {d}, \"batches\": {d} }}", .{
                r.method,
                r.optimizer,
                r.final_loss,
                r.total_time_ms,
                r.epochs,
                r.batches,
            });
            if (idx < result_count - 1) std.debug.print(",", .{});
            std.debug.print("\n", .{});
        }
        std.debug.print("] }}\n", .{});
    } else {
        std.debug.print("  {s:<18} {s:<10} {s:>12} {s:>10} {s:>8}\n", .{ "Method", "Optimizer", "Final Loss", "Time (ms)", "Batches" });
        std.debug.print("  {s:<18} {s:<10} {s:>12} {s:>10} {s:>8}\n", .{ "-" ** 18, "-" ** 10, "-" ** 12, "-" ** 10, "-" ** 8 });
        for (results[0..result_count]) |r| {
            std.debug.print("  {s:<18} {s:<10} {d:>12.4} {d:>10} {d:>8}\n", .{
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
                std.debug.print("\n  Speed vs {s} ({s}):\n", .{ results[0].method, results[0].optimizer });
                for (results[1..result_count]) |r| {
                    if (r.total_time_ms > 0) {
                        const ratio = @as(f64, @floatFromInt(baseline_ms)) / @as(f64, @floatFromInt(r.total_time_ms));
                        std.debug.print("    {s} ({s}): {d:.2}x\n", .{ r.method, r.optimizer, ratio });
                    }
                }
            }
        }
    }
}
