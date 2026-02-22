//! GPU + AI Training Cross-Module Integration Example
//!
//! Demonstrates how GPU acceleration and AI training subsystems work together:
//! - GPU device discovery and capability detection
//! - Training pipeline configuration with GPU-aware settings
//! - Distributed gradient synchronization across multiple GPUs
//! - Mixed precision training with GPU memory optimization
//!
//! Run with: `zig build run-gpu-training`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== GPU + AI Training Integration Example ===\n\n", .{});

    // ── Step 1: GPU Discovery ──────────────────────────────────────────────
    std.debug.print("--- Step 1: GPU Device Discovery ---\n", .{});

    const gpu_enabled = abi.gpu.backends.detect.moduleEnabled();
    std.debug.print("GPU module: {s}\n", .{if (gpu_enabled) "enabled" else "disabled (stub)"});

    if (gpu_enabled) {
        const backends = abi.gpu.backends.detect.availableBackends(allocator) catch |err| {
            std.debug.print("  Backend enumeration failed: {t}\n", .{err});
            return;
        };
        defer allocator.free(backends);
        std.debug.print("  Available backends: {d}\n", .{backends.len});
        for (backends) |b| {
            const avail = abi.gpu.backends.detect.backendAvailability(b);
            std.debug.print("    {t}: {d} device(s)\n", .{ b, avail.device_count });
        }
    } else {
        std.debug.print("  (GPU discovery skipped — feature disabled)\n", .{});
    }

    // ── Step 2: Training Configuration ─────────────────────────────────────
    std.debug.print("\n--- Step 2: Training Pipeline Config ---\n", .{});

    const training_config = abi.ai.training.TrainingConfig{
        .epochs = 20,
        .batch_size = 128,
        .sample_count = 50_000,
        .model_size = 768,
        .learning_rate = 0.0003,
        .optimizer = .adamw,
        .weight_decay = 0.01,
        .gradient_clip_norm = 1.0,
        .learning_rate_schedule = .warmup_cosine,
        .warmup_steps = 200,
        .decay_steps = 2000,
        .mixed_precision = gpu_enabled, // Enable mixed precision when GPU available
        .gradient_accumulation_steps = if (gpu_enabled) 1 else 4, // Accumulate on CPU
        .checkpoint_interval = 5,
    };

    training_config.validate() catch |err| {
        std.debug.print("  Config validation failed: {t}\n", .{err});
        return;
    };

    std.debug.print("  Optimizer:        adamw\n", .{});
    std.debug.print("  Epochs:           {d}\n", .{training_config.epochs});
    std.debug.print("  Batch size:       {d}\n", .{training_config.batch_size});
    std.debug.print("  Learning rate:    {d:.6}\n", .{training_config.learning_rate});
    std.debug.print("  Mixed precision:  {}\n", .{training_config.mixed_precision});
    std.debug.print("  Grad accumulate:  {d} step(s)\n", .{training_config.gradient_accumulation_steps});

    // ── Step 3: Distributed Training Setup ─────────────────────────────────
    std.debug.print("\n--- Step 3: Distributed Training Coordinator ---\n", .{});

    // Simulate a 4-GPU data-parallel training setup
    const dist_config = abi.ai.training.DistributedConfig{
        .world_size = 4,
        .rank = 0,
        .is_coordinator = true,
        .bucket_size_bytes = 25 * 1024 * 1024, // 25 MB gradient buckets
        .reduce_op = .average,
    };

    dist_config.validate() catch |err| {
        std.debug.print("  Distributed config invalid: {t}\n", .{err});
        return;
    };

    var dist_trainer = abi.ai.training.DistributedTrainer.init(allocator, dist_config);
    defer dist_trainer.deinit();

    std.debug.print("  World size:       {d} GPU(s)\n", .{dist_config.world_size});
    std.debug.print("  This rank:        {d}{s}\n", .{
        dist_config.rank,
        if (dist_config.is_coordinator) " (coordinator)" else "",
    });
    std.debug.print("  Reduce op:        {t}\n", .{dist_config.reduce_op});
    std.debug.print("  Bucket size:      {d} MB\n", .{dist_config.bucket_size_bytes / (1024 * 1024)});

    // ── Step 4: Simulate Gradient Synchronization ──────────────────────────
    std.debug.print("\n--- Step 4: Gradient Sync Simulation ---\n", .{});

    // Simulate a small gradient vector from a backward pass
    var gradients = [_]f32{ 0.12, -0.34, 0.56, -0.78, 0.91, -0.23, 0.45, -0.67 };

    std.debug.print("  Pre-sync gradients:  ", .{});
    for (gradients) |g| std.debug.print("{d: >7.4} ", .{g});
    std.debug.print("\n", .{});

    // AllReduce averages gradients across 4 workers
    dist_trainer.synchronizeGradients(&gradients);

    std.debug.print("  Post-sync gradients: ", .{});
    for (gradients) |g| std.debug.print("{d: >7.4} ", .{g});
    std.debug.print("\n", .{});

    // ── Step 5: Data Sharding ──────────────────────────────────────────────
    std.debug.print("\n--- Step 5: Data Sharding ---\n", .{});

    const dataset = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const shard = dist_trainer.shardData(u32, &dataset);

    std.debug.print("  Full dataset:     {d} samples\n", .{dataset.len});
    std.debug.print("  Rank {d} shard:     {d} samples (indices {d}..{d})\n", .{
        dist_config.rank,
        shard.len,
        shard[0],
        shard[shard.len - 1],
    });

    // Record a simulated epoch
    dist_trainer.recordEpoch();
    const stats = dist_trainer.getStats();
    std.debug.print("\n  Distributed stats:\n", .{});
    std.debug.print("    AllReduce calls:  {d}\n", .{stats.total_allreduce_calls});
    std.debug.print("    Bytes synced:     {d}\n", .{stats.total_bytes_synced});
    std.debug.print("    Epochs completed: {d}\n", .{stats.epochs_completed});
    std.debug.print("    Should log (coordinator): {}\n", .{dist_trainer.shouldLog()});

    std.debug.print("\n=== GPU + Training Integration Complete ===\n", .{});
}
