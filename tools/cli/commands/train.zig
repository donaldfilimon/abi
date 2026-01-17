//! Training pipeline CLI command.
//!
//! Commands:
//! - train run [options] - Run training pipeline
//! - train llm <model.gguf> [options] - Train LLM model
//! - train resume <checkpoint> - Resume training from checkpoint
//! - train info - Show default training configuration
//! - train help - Show help message

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the train command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "run")) {
        try runTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "llm")) {
        try runLlmTrain(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "resume")) {
        try runResume(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "info")) {
        runInfo();
        return;
    }

    std.debug.print("Unknown train command: {s}\n", .{command});
    printHelp();
}

fn runTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var config = abi.ai.TrainingConfig{};

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
                config.optimizer = parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate_schedule = parseLrSchedule(val);
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

    var timer = std.time.Timer.start() catch {
        std.debug.print("Error: Failed to start timer\n", .{});
        return;
    };

    var result = abi.ai.trainWithResult(allocator, config) catch |err| {
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

fn runLlmTrain(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    // Check if LLM feature is enabled
    if (!abi.ai.llm.isEnabled()) {
        std.debug.print("Error: LLM feature is not enabled. Build with -Denable-llm=true\n", .{});
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi train llm <model.gguf> [options]\n", .{});
        std.debug.print("\nUse 'abi train help' for full options list.\n", .{});
        return;
    }

    // First argument is the model path
    const model_path = std.mem.sliceTo(args[0], 0);

    // Parse LLM training options
    var config = abi.ai.LlmTrainingConfig{};
    var use_gpu: bool = false;
    var i: usize = 1;

    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.epochs = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.batch_size = std.fmt.parseInt(u32, val, 10) catch 4;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.learning_rate = std.fmt.parseFloat(f32, val) catch 1e-5;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--grad-accum", "--gradient-accumulation" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.grad_accum_steps = std.fmt.parseInt(u32, val, 10) catch 1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-seq-len")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_seq_len = std.fmt.parseInt(u32, val, 10) catch 512;
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
                config.max_grad_norm = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--label-smoothing")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.label_smoothing = std.fmt.parseFloat(f32, val) catch 0.0;
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

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--optimizer")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.optimizer = parseOptimizer(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lr-schedule")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.lr_schedule = parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--use-gpu")) {
            use_gpu = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mixed-precision")) {
            config.mixed_precision = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }
    }

    // Print configuration
    std.debug.print("LLM Training Configuration\n", .{});
    std.debug.print("==========================\n", .{});
    std.debug.print("Model:            {s}\n", .{model_path});
    std.debug.print("Epochs:           {d}\n", .{config.epochs});
    std.debug.print("Batch size:       {d}\n", .{config.batch_size});
    std.debug.print("Max seq len:      {d}\n", .{config.max_seq_len});
    std.debug.print("Learning rate:    {e:.2}\n", .{config.learning_rate});
    std.debug.print("Optimizer:        {t}\n", .{config.optimizer});
    std.debug.print("LR schedule:      {t}\n", .{config.lr_schedule});
    std.debug.print("Warmup steps:     {d}\n", .{config.warmup_steps});
    std.debug.print("Weight decay:     {d:.4}\n", .{config.weight_decay});
    std.debug.print("Gradient clip:    {d:.2}\n", .{config.max_grad_norm});
    std.debug.print("Grad accumulation:{d}\n", .{config.grad_accum_steps});
    if (config.label_smoothing > 0) {
        std.debug.print("Label smoothing:  {d:.2}\n", .{config.label_smoothing});
    }
    if (config.checkpoint_interval > 0) {
        std.debug.print("Checkpoint interval: {d}\n", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        std.debug.print("Checkpoint path:  {s}\n", .{path});
    }
    std.debug.print("Use GPU:          {}\n", .{use_gpu});
    std.debug.print("Mixed precision:  {}\n", .{config.mixed_precision});
    std.debug.print("\n", .{});

    // Load model
    std.debug.print("Loading model from {s}...\n", .{model_path});
    std.debug.print("Note: GGUF loading is a placeholder. Using demo configuration.\n\n", .{});

    // Create model with demo configuration
    // In full implementation, this would parse GGUF and extract config
    const model_config = abi.ai.trainable_model.TrainableModelConfig{
        .hidden_dim = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 512,
        .vocab_size = 32000,
        .max_seq_len = config.max_seq_len,
    };

    var model = abi.ai.TrainableModel.init(allocator, model_config) catch |err| {
        std.debug.print("Error initializing model: {t}\n", .{err});
        return;
    };
    defer model.deinit();

    const num_params = model.numParams();
    std.debug.print("Model initialized: {d} parameters ({d:.2} MB)\n\n", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });

    // Create trainer
    var trainer = abi.ai.LlamaTrainer.init(allocator, &model, config) catch |err| {
        std.debug.print("Error creating trainer: {t}\n", .{err});
        return;
    };
    defer trainer.deinit();

    std.debug.print("Starting LLM training...\n", .{});
    std.debug.print("Note: Training requires a data iterator with (input_ids, labels) pairs.\n", .{});
    std.debug.print("This command demonstrates the training setup. For actual training,\n", .{});
    std.debug.print("use the training API programmatically with your dataset.\n\n", .{});

    // Show training API usage
    std.debug.print("Training API Example:\n", .{});
    std.debug.print("  var trainer = try abi.ai.LlamaTrainer.init(allocator, &model, config);\n", .{});
    std.debug.print("  defer trainer.deinit();\n", .{});
    std.debug.print("  for (data_iterator) |batch| {{\n", .{});
    std.debug.print("      const loss = try trainer.trainStep(batch.input_ids, batch.labels);\n", .{});
    std.debug.print("  }}\n", .{});
    std.debug.print("\nTrainer initialized successfully. Ready for training.\n", .{});
}

fn runResume(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi train resume <checkpoint-path>\n", .{});
        return;
    }

    const checkpoint_path = std.mem.sliceTo(args[0], 0);
    std.debug.print("Loading checkpoint: {s}\n", .{checkpoint_path});

    // Load checkpoint
    var ckpt = abi.ai.loadCheckpoint(allocator, checkpoint_path) catch |err| {
        std.debug.print("Error loading checkpoint: {t}\n", .{err});
        std.debug.print("\nNote: Resume functionality loads model weights from a saved checkpoint.\n", .{});
        std.debug.print("Use 'abi train run --checkpoint-path <path>' to save checkpoints during training.\n", .{});
        return;
    };
    defer ckpt.deinit(allocator);

    std.debug.print("\nCheckpoint Info:\n", .{});
    std.debug.print("  Step:      {d}\n", .{ckpt.step});
    std.debug.print("  Timestamp: {d}\n", .{ckpt.timestamp});
    std.debug.print("  Weights:   {d} parameters\n", .{ckpt.weights.len});
    std.debug.print("\nNote: Full resume training not yet implemented.\n", .{});
    std.debug.print("Checkpoint loaded successfully.\n", .{});
}

fn runInfo() void {
    const default_config = abi.ai.TrainingConfig{};

    std.debug.print("Default Training Configuration\n", .{});
    std.debug.print("==============================\n\n", .{});
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

fn parseOptimizer(val: []const u8) abi.ai.OptimizerType {
    if (std.mem.eql(u8, val, "sgd")) return .sgd;
    if (std.mem.eql(u8, val, "adam")) return .adam;
    if (std.mem.eql(u8, val, "adamw")) return .adamw;
    return .adamw; // default
}

fn parseLrSchedule(val: []const u8) abi.ai.LearningRateSchedule {
    if (std.mem.eql(u8, val, "constant")) return .constant;
    if (std.mem.eql(u8, val, "cosine")) return .cosine;
    if (std.mem.eql(u8, val, "warmup_cosine")) return .warmup_cosine;
    if (std.mem.eql(u8, val, "step")) return .step;
    if (std.mem.eql(u8, val, "polynomial")) return .polynomial;
    return .warmup_cosine; // default
}

fn printHelp() void {
    const help_text =
        \\Usage: abi train <command> [options]
        \\
        \\Run the training pipeline for neural network models.
        \\
        \\Commands:
        \\  run [options]           Run basic training with specified configuration
        \\  llm <model> [options]   Train LLM from GGUF model file
        \\  resume <checkpoint>     Resume training from a checkpoint file
        \\  info                    Show default training configuration
        \\  help                    Show this help message
        \\
        \\Basic training options (for 'run'):
        \\  -e, --epochs <n>           Number of epochs (default: 10)
        \\  -b, --batch-size <n>       Batch size (default: 32)
        \\  --model-size <n>           Model parameters (default: 512)
        \\  --sample-count <n>         Training samples (default: 1024)
        \\  --lr, --learning-rate <f>  Learning rate (default: 0.001)
        \\  --optimizer <type>         sgd, adam, adamw (default: adamw)
        \\  --lr-schedule <type>       LR schedule (default: warmup_cosine)
        \\  --warmup-steps <n>         Warmup steps (default: 100)
        \\  --decay-steps <n>          Decay steps (default: 1000)
        \\  --weight-decay <f>         Weight decay (default: 0.01)
        \\  --gradient-clip <f>        Gradient clip norm (default: 1.0)
        \\  --gradient-accumulation <n> Gradient accumulation steps (default: 1)
        \\  --checkpoint-interval <n>  Steps between checkpoints (default: 0)
        \\  --checkpoint-path <path>   Path to save checkpoints
        \\  --max-checkpoints <n>      Max checkpoints to retain (default: 5)
        \\  --early-stopping-patience <n> Early stopping patience (default: 5)
        \\  --mixed-precision          Enable mixed precision training
        \\
        \\LLM training options (for 'llm'):
        \\  -e, --epochs <n>           Number of epochs (default: 1)
        \\  -b, --batch-size <n>       Batch size (default: 4)
        \\  --lr, --learning-rate <f>  Learning rate (default: 1e-5)
        \\  --grad-accum <n>           Gradient accumulation steps (default: 1)
        \\  --max-seq-len <n>          Maximum sequence length (default: 512)
        \\  --warmup-steps <n>         Warmup steps (default: 100)
        \\  --weight-decay <f>         Weight decay (default: 0.01)
        \\  --gradient-clip <f>        Gradient clip norm (default: 1.0)
        \\  --label-smoothing <f>      Label smoothing (default: 0.0)
        \\  --optimizer <type>         sgd, adam, adamw (default: adamw)
        \\  --lr-schedule <type>       LR schedule (default: warmup_cosine)
        \\  --checkpoint-interval <n>  Steps between checkpoints (default: 0)
        \\  --checkpoint-path <path>   Path to save checkpoints
        \\  --use-gpu                  Enable GPU acceleration (cuBLAS)
        \\  --mixed-precision          Enable mixed precision training
        \\  --log-interval <n>         Steps between log outputs (default: 10)
        \\
        \\LR schedules: constant, cosine, warmup_cosine, step, polynomial
        \\Optimizers: sgd, adam, adamw
        \\
        \\Examples:
        \\  abi train run --epochs 10 --batch-size 32
        \\  abi train run -e 5 -b 16 --optimizer adam --lr 0.0001
        \\  abi train llm model.gguf --epochs 1 --batch-size 4 --lr 1e-5
        \\  abi train llm model.gguf --grad-accum 8 --max-seq-len 512 --use-gpu
        \\  abi train llm model.gguf --checkpoint-interval 100 --checkpoint-path ./ckpt
        \\  abi train resume ./checkpoints/model.ckpt
        \\  abi train info
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
