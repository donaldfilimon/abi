//! LLM training handler.
//!
//! Handles the `abi train llm` subcommand which fine-tunes an existing
//! GGUF model with configurable training parameters, dataset loading,
//! and optional GGUF export.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const common = @import("common.zig");
const mod = @import("mod.zig");

pub fn runLlmTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    // Check if LLM feature is enabled
    if (!abi.ai.llm.isEnabled()) {
        utils.output.printError("LLM feature is not enabled. Build with -Denable-llm=true", .{});
        return;
    }

    if (args.len == 0) {
        utils.output.println("Usage: abi train llm <model.gguf> [options]", .{});
        utils.output.println("", .{});
        utils.output.println("Use 'abi train help' for full options list.", .{});
        return;
    }

    // First argument is the model path
    const model_path = std.mem.sliceTo(args[0], 0);

    // Parse LLM training options
    var config = abi.ai.training.LlmTrainingConfig{};
    var use_gpu: bool = true;
    var dataset_url: ?[]const u8 = null;
    var dataset_path: ?[]const u8 = null;
    var dataset_cache: ?[]const u8 = null;
    var dataset_wdbx: ?[]const u8 = null;
    var dataset_format: common.DatasetFormat = .tokenbin;
    var dataset_block_tokens: u32 = 2048;
    var dataset_max_tokens: usize = 0;
    var dataset_max_bytes: usize = 64 * 1024 * 1024;
    var export_gguf_path: ?[]const u8 = null;
    var export_gguf_name: ?[]const u8 = null;
    var log_dir: ?[]const u8 = null;
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

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--max-checkpoints")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.max_checkpoints = std.fmt.parseInt(u32, val, 10) catch 3;
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
                config.lr_schedule = common.parseLrSchedule(val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--use-gpu")) {
            use_gpu = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--cpu-only") or
            std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-gpu"))
        {
            use_gpu = false;
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

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-url")) {
            if (i < args.len) {
                dataset_url = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-path")) {
            if (i < args.len) {
                dataset_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-cache")) {
            if (i < args.len) {
                dataset_cache = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-wdbx")) {
            if (i < args.len) {
                dataset_wdbx = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-format")) {
            if (i < args.len) {
                dataset_format = common.parseDatasetFormat(std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-block-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_block_tokens = std.fmt.parseInt(u32, val, 10) catch 2048;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-max-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_max_tokens = std.fmt.parseInt(usize, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dataset-max-bytes")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dataset_max_bytes = std.fmt.parseInt(usize, val, 10) catch dataset_max_bytes;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-dir")) {
            if (i < args.len) {
                log_dir = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-gguf")) {
            if (i < args.len) {
                export_gguf_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--export-name")) {
            if (i < args.len) {
                export_gguf_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Print configuration
    utils.output.printHeader("LLM Training Configuration");
    utils.output.printKeyValueFmt("Model", "{s}", .{model_path});
    utils.output.printKeyValueFmt("Epochs", "{d}", .{config.epochs});
    utils.output.printKeyValueFmt("Batch size", "{d}", .{config.batch_size});
    utils.output.printKeyValueFmt("Max seq len", "{d}", .{config.max_seq_len});
    utils.output.printKeyValueFmt("Learning rate", "{e:.2}", .{config.learning_rate});
    utils.output.printKeyValueFmt("Optimizer", "{t}", .{config.optimizer});
    utils.output.printKeyValueFmt("LR schedule", "{t}", .{config.lr_schedule});
    utils.output.printKeyValueFmt("Warmup steps", "{d}", .{config.warmup_steps});
    utils.output.printKeyValueFmt("Weight decay", "{d:.4}", .{config.weight_decay});
    utils.output.printKeyValueFmt("Gradient clip", "{d:.2}", .{config.max_grad_norm});
    utils.output.printKeyValueFmt("Grad accumulation", "{d}", .{config.grad_accum_steps});
    if (config.label_smoothing > 0) {
        utils.output.printKeyValueFmt("Label smoothing", "{d:.2}", .{config.label_smoothing});
    }
    if (config.checkpoint_interval > 0) {
        utils.output.printKeyValueFmt("Checkpoint interval", "{d}", .{config.checkpoint_interval});
    }
    if (config.checkpoint_path) |path| {
        utils.output.printKeyValueFmt("Checkpoint path", "{s}", .{path});
    }
    config.use_gpu = use_gpu;
    utils.output.printKeyValueFmt("Use GPU", "{}", .{config.use_gpu});
    utils.output.printKeyValueFmt("Dataset format", "{t}", .{dataset_format});
    if (dataset_url) |url| {
        utils.output.printKeyValueFmt("Dataset URL", "{s}", .{url});
    }
    if (dataset_path) |path| {
        utils.output.printKeyValueFmt("Dataset path", "{s}", .{path});
    }
    if (dataset_wdbx) |path| {
        utils.output.printKeyValueFmt("Dataset WDBX", "{s}", .{path});
    }
    if (dataset_max_tokens > 0) {
        utils.output.printKeyValueFmt("Max tokens", "{d}", .{dataset_max_tokens});
    }
    utils.output.printKeyValueFmt("Mixed precision", "{}", .{config.mixed_precision});
    utils.output.println("", .{});

    // Load model
    utils.output.println("Loading model from {s}...", .{model_path});
    var model = abi.ai.training.TrainableModel.fromGguf(allocator, model_path) catch |err| {
        utils.output.printError("loading GGUF model: {t}", .{err});
        return;
    };
    defer model.deinit();

    const num_params = model.numParams();
    utils.output.println("Model initialized: {d} parameters ({d:.2} MB)", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    utils.output.println("", .{});

    if (export_gguf_path) |path| {
        config.export_gguf_path = path;
    }
    if (export_gguf_name) |name| {
        config.export_name = name;
    }

    // Default log directory and metrics stream for dashboards
    if (log_dir == null) {
        log_dir = "logs";
    }
    config.log_dir = log_dir;
    config.enable_metrics_stream = true;

    var tokenizer: ?abi.ai.llm.tokenizer.Tokenizer = null;
    defer if (tokenizer) |*tok| tok.deinit();

    if (dataset_format != .tokenbin) {
        var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, model_path) catch |err| {
            utils.output.printError("opening GGUF for tokenizer: {t}", .{err});
            return;
        };
        defer gguf_file.deinit();

        const tok = abi.ai.llm.tokenizer.loadFromGguf(allocator, &gguf_file) catch |err| {
            utils.output.printError("loading tokenizer from GGUF: {t}", .{err});
            return;
        };
        tokenizer = tok;
    }

    const dataset = try common.resolveDatasetPath(allocator, dataset_url, dataset_path, dataset_cache, dataset_max_bytes);
    defer if (dataset.owned and dataset.path.len > 0) allocator.free(dataset.path);

    var train_tokens: []u32 = &.{};
    const tokenizer_ptr: ?*abi.ai.llm.tokenizer.Tokenizer = if (tokenizer) |*tok| tok else null;
    if (dataset_wdbx) |db_path| {
        var wdbx_dataset = abi.ai.database.WdbxTokenDataset.init(allocator, db_path) catch |err| {
            utils.output.printError("opening WDBX dataset: {t}", .{err});
            return;
        };
        defer wdbx_dataset.deinit();

        if (dataset.path.len > 0) {
            const ingest_tokens = try common.loadTokensFromPath(
                allocator,
                dataset_format,
                dataset.path,
                tokenizer_ptr,
                dataset_max_tokens,
            );
            defer allocator.free(ingest_tokens);

            try wdbx_dataset.importTokenBin(ingest_tokens, dataset_block_tokens);
            try wdbx_dataset.save();
        }

        train_tokens = try wdbx_dataset.collectTokens(dataset_max_tokens);
    } else {
        if (dataset.path.len == 0) {
            utils.output.printError("dataset path or URL required when --dataset-wdbx is not provided.", .{});
            return;
        }
        train_tokens = try common.loadTokensFromPath(
            allocator,
            dataset_format,
            dataset.path,
            tokenizer_ptr,
            dataset_max_tokens,
        );
    }
    defer allocator.free(train_tokens);

    if (train_tokens.len == 0) {
        utils.output.printError("dataset yielded no tokens.", .{});
        return;
    }

    common.clampTokens(train_tokens, model.config.vocab_size);

    utils.output.println("Starting LLM training...", .{});

    var timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("failed to start timer", .{});
        return;
    };

    const report = abi.ai.training.llm_trainer.trainLlm(allocator, &model, config, train_tokens) catch |err| {
        utils.output.printError("Training failed: {t}", .{err});
        return;
    };

    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    utils.output.printHeader("Training Complete");
    utils.output.printKeyValueFmt("Final loss", "{d:.6}", .{report.final_loss});
    utils.output.printKeyValueFmt("Final accuracy", "{d:.2}%", .{report.final_accuracy * 100});
    utils.output.printKeyValueFmt("Total steps", "{d}", .{report.total_steps});
    utils.output.printKeyValueFmt("Tokens processed", "{d}", .{report.total_tokens});
    utils.output.printKeyValueFmt("Total time", "{d:.2}s", .{@as(f64, @floatFromInt(report.total_time_ns)) / 1e9});
    utils.output.printKeyValueFmt("Wall time", "{d}ms", .{elapsed_ms});
    utils.output.printKeyValueFmt("Checkpoints saved", "{d}", .{report.checkpoints_saved});
}

test {
    std.testing.refAllDecls(@This());
}
