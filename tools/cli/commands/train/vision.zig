//! Vision and CLIP training handlers.
//!
//! Handles the `abi train vision` and `abi train clip` subcommands for
//! training Vision Transformer (ViT) and CLIP multimodal models.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runVisionTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printVisionHelp();
        return;
    }

    // Check if Vision feature is enabled
    if (!abi.features.ai.vision.isEnabled()) {
        utils.output.printError("Vision feature is not enabled. Build with -Dfeat-vision=true (legacy: -Denable-vision=true)", .{});
        return;
    }

    // Default ViT configuration (tiny model for training from scratch)
    var image_size: u32 = 224;
    var patch_size: u32 = 16;
    var hidden_size: u32 = 384;
    var num_layers: u32 = 12;
    var num_heads: u32 = 6;
    var mlp_dim: u32 = 1536;
    var num_classes: u32 = 1000;
    var dropout: f32 = 0.1;

    // Training config
    var epochs: u32 = 10;
    var batch_size: u32 = 32;
    var learning_rate: f32 = 1e-4;
    var warmup_steps: u32 = 500;
    var weight_decay: f32 = 0.01;
    var gradient_clip: f32 = 1.0;
    var log_interval: u32 = 10;
    var dataset_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--image-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                image_size = std.fmt.parseInt(u32, val, 10) catch 224;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--patch-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                patch_size = std.fmt.parseInt(u32, val, 10) catch 16;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--hidden-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                hidden_size = std.fmt.parseInt(u32, val, 10) catch 384;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-heads")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_heads = std.fmt.parseInt(u32, val, 10) catch 6;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mlp-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mlp_dim = std.fmt.parseInt(u32, val, 10) catch 1536;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--num-classes")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                num_classes = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--dropout")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                dropout = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                batch_size = std.fmt.parseInt(u32, val, 10) catch 32;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                learning_rate = std.fmt.parseFloat(f32, val) catch 1e-4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                warmup_steps = std.fmt.parseInt(u32, val, 10) catch 500;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                weight_decay = std.fmt.parseFloat(f32, val) catch 0.01;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gradient_clip = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
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
    }

    // Create ViT config
    const vit_config = abi.features.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .mlp_dim = mlp_dim,
        .in_channels = 3,
        .use_class_token = true,
    };

    const trainable_vit_config = abi.features.ai.training.TrainableViTConfig{
        .vit_config = vit_config,
        .max_batch_size = batch_size,
        .num_classes = num_classes,
        .dropout = dropout,
    };

    const num_params = trainable_vit_config.numParams();

    // Print configuration
    utils.output.printHeader("Vision Transformer (ViT) Training Configuration");
    utils.output.println("Architecture:", .{});
    utils.output.printKeyValueFmt("Image size", "{d}x{d}", .{ image_size, image_size });
    utils.output.printKeyValueFmt("Patch size", "{d}x{d}", .{ patch_size, patch_size });
    utils.output.printKeyValueFmt("Hidden size", "{d}", .{hidden_size});
    utils.output.printKeyValueFmt("Num layers", "{d}", .{num_layers});
    utils.output.printKeyValueFmt("Num heads", "{d}", .{num_heads});
    utils.output.printKeyValueFmt("MLP dim", "{d}", .{mlp_dim});
    utils.output.printKeyValueFmt("Num classes", "{d}", .{num_classes});
    utils.output.printKeyValueFmt("Parameters", "{d} ({d:.2} MB)", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    utils.output.println("", .{});
    utils.output.println("Training:", .{});
    utils.output.printKeyValueFmt("Epochs", "{d}", .{epochs});
    utils.output.printKeyValueFmt("Batch size", "{d}", .{batch_size});
    utils.output.printKeyValueFmt("Learning rate", "{e:.2}", .{learning_rate});
    utils.output.printKeyValueFmt("Warmup steps", "{d}", .{warmup_steps});
    utils.output.printKeyValueFmt("Weight decay", "{d:.4}", .{weight_decay});
    utils.output.printKeyValueFmt("Gradient clip", "{d:.2}", .{gradient_clip});
    utils.output.printKeyValueFmt("Dropout", "{d:.2}", .{dropout});
    if (dataset_path) |path| {
        utils.output.printKeyValueFmt("Dataset", "{s}", .{path});
    } else {
        utils.output.printKeyValue("Dataset", "(synthetic)");
    }
    utils.output.println("", .{});

    // Initialize model
    utils.output.println("Initializing ViT model with random weights...", .{});
    var model = abi.features.ai.training.TrainableViTModel.init(allocator, trainable_vit_config) catch |err| {
        utils.output.printError("initializing model: {t}", .{err});
        return;
    };
    defer model.deinit();

    utils.output.println("Model initialized: {d} parameters", .{num_params});
    utils.output.println("", .{});

    // Generate synthetic training data (images)
    const image_dim = image_size * image_size * 3;
    const num_samples = batch_size * 10;
    var train_images = allocator.alloc(f32, num_samples * image_dim) catch |err| {
        utils.output.printError("allocating training data: {t}", .{err});
        return;
    };
    defer allocator.free(train_images);

    const train_labels = allocator.alloc(u32, num_samples) catch |err| {
        utils.output.printError("allocating labels: {t}", .{err});
        return;
    };
    defer allocator.free(train_labels);

    // Initialize with random data
    var rng = std.Random.DefaultPrng.init(42);
    for (train_images) |*p| {
        p.* = rng.random().float(f32);
    }
    for (train_labels) |*l| {
        l.* = rng.random().intRangeLessThan(u32, 0, num_classes);
    }

    utils.output.println("Generated {d} synthetic images for training", .{num_samples});
    utils.output.println("", .{});
    utils.output.println("Starting Vision training...", .{});

    var timer = abi.services.shared.time.Timer.start() catch {
        utils.output.printError("Failed to start timer", .{});
        return;
    };

    // Training loop
    const batches_per_epoch = num_samples / batch_size;
    var total_loss: f32 = 0;
    var step: u32 = 0;

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..batches_per_epoch) |batch_idx| {
            const batch_start = batch_idx * batch_size * image_dim;
            const batch_images = train_images[batch_start .. batch_start + batch_size * image_dim];

            // Forward pass
            const logits = allocator.alloc(f32, batch_size * num_classes) catch continue;
            defer allocator.free(logits);

            model.forward(batch_images, batch_size, logits) catch continue;

            // Compute cross-entropy loss (simplified)
            var batch_loss: f32 = 0;
            for (0..batch_size) |b| {
                const label = train_labels[batch_idx * batch_size + b];
                const logit_offset = b * num_classes;

                // Softmax + cross-entropy
                var max_logit: f32 = logits[logit_offset];
                for (0..num_classes) |c| {
                    if (logits[logit_offset + c] > max_logit) {
                        max_logit = logits[logit_offset + c];
                    }
                }

                var sum_exp: f32 = 0;
                for (0..num_classes) |c| {
                    sum_exp += @exp(logits[logit_offset + c] - max_logit);
                }

                const log_prob = logits[logit_offset + label] - max_logit - @log(sum_exp);
                batch_loss -= log_prob;
            }
            batch_loss /= @as(f32, @floatFromInt(batch_size));
            epoch_loss += batch_loss;

            // Apply gradient update (SGD)
            _ = model.clipGradients(gradient_clip);
            model.applySgdUpdate(learning_rate);
            model.zeroGradients();

            step += 1;

            if (step % log_interval == 0) {
                utils.output.println("  Step {d}: loss={d:.4}", .{ step, batch_loss });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(batches_per_epoch));
        total_loss = epoch_loss;

        utils.output.println("Epoch {d}/{d}: avg_loss={d:.4}", .{ epoch + 1, epochs, epoch_loss });
    }

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    utils.output.printHeader("Vision Training Complete");
    utils.output.printKeyValueFmt("Final loss", "{d:.6}", .{total_loss});
    utils.output.printKeyValueFmt("Total steps", "{d}", .{step});
    utils.output.printKeyValueFmt("Wall time", "{d:.2}s", .{elapsed_s});
}

pub fn runClipTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printClipHelp();
        return;
    }

    // Check if Vision feature is enabled
    if (!abi.features.ai.vision.isEnabled()) {
        utils.output.printError("Vision feature is not enabled. Build with -Dfeat-vision=true (legacy: -Denable-vision=true)", .{});
        return;
    }

    // Default CLIP configuration
    var image_size: u32 = 224;
    var patch_size: u32 = 16;
    var vision_hidden: u32 = 768;
    var vision_layers: u32 = 12;
    const vision_heads: u32 = 12;
    var text_hidden: u32 = 512;
    var text_layers: u32 = 12;
    const text_heads: u32 = 8;
    var projection_dim: u32 = 512;
    var temperature: f32 = 0.07;

    // Training config
    var epochs: u32 = 10;
    var batch_size: u32 = 64;
    var learning_rate: f32 = 1e-4;
    var warmup_steps: u32 = 2000;
    var weight_decay: f32 = 0.1;
    var gradient_clip: f32 = 1.0;
    var log_interval: u32 = 10;
    var dataset_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--image-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                image_size = std.fmt.parseInt(u32, val, 10) catch 224;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--patch-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                patch_size = std.fmt.parseInt(u32, val, 10) catch 16;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vision-hidden")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                vision_hidden = std.fmt.parseInt(u32, val, 10) catch 768;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vision-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                vision_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--text-hidden")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                text_hidden = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--text-layers")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                text_layers = std.fmt.parseInt(u32, val, 10) catch 12;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--projection-dim")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                projection_dim = std.fmt.parseInt(u32, val, 10) catch 512;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--temperature")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                temperature = std.fmt.parseFloat(f32, val) catch 0.07;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--epochs", "-e" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                epochs = std.fmt.parseInt(u32, val, 10) catch 10;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--batch-size", "-b" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                batch_size = std.fmt.parseInt(u32, val, 10) catch 64;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--learning-rate", "--lr" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                learning_rate = std.fmt.parseFloat(f32, val) catch 1e-4;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--warmup-steps")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                warmup_steps = std.fmt.parseInt(u32, val, 10) catch 2000;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--weight-decay")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                weight_decay = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gradient-clip")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gradient_clip = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--log-interval")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                log_interval = std.fmt.parseInt(u32, val, 10) catch 10;
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
    }

    // Create CLIP config
    const vit_config = abi.features.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = vision_hidden,
        .num_layers = vision_layers,
        .num_heads = vision_heads,
        .mlp_dim = vision_hidden * 4,
        .in_channels = 3,
        .use_class_token = true,
    };

    const vision_config = abi.features.ai.training.TrainableViTConfig{
        .vit_config = vit_config,
        .max_batch_size = batch_size,
        .num_classes = 0, // CLIP uses projection
        .projection_dim = projection_dim,
    };

    const clip_config = abi.features.ai.training.CLIPTrainingConfig{
        .vision_config = vision_config,
        .text_hidden_size = text_hidden,
        .text_num_layers = text_layers,
        .text_num_heads = text_heads,
        .projection_dim = projection_dim,
        .temperature = temperature,
        .learnable_temperature = true,
    };

    const num_params = clip_config.numParams();

    // Print configuration
    utils.output.printHeader("CLIP (Contrastive Language-Image Pretraining) Configuration");
    utils.output.println("Vision Encoder:", .{});
    utils.output.printKeyValueFmt("Image size", "{d}x{d}", .{ image_size, image_size });
    utils.output.printKeyValueFmt("Patch size", "{d}x{d}", .{ patch_size, patch_size });
    utils.output.printKeyValueFmt("Hidden size", "{d}", .{vision_hidden});
    utils.output.printKeyValueFmt("Num layers", "{d}", .{vision_layers});
    utils.output.printKeyValueFmt("Num heads", "{d}", .{vision_heads});
    utils.output.println("", .{});
    utils.output.println("Text Encoder:", .{});
    utils.output.printKeyValueFmt("Hidden size", "{d}", .{text_hidden});
    utils.output.printKeyValueFmt("Num layers", "{d}", .{text_layers});
    utils.output.printKeyValueFmt("Num heads", "{d}", .{text_heads});
    utils.output.println("", .{});
    utils.output.println("Contrastive:", .{});
    utils.output.printKeyValueFmt("Projection dim", "{d}", .{projection_dim});
    utils.output.printKeyValueFmt("Temperature", "{d:.4}", .{temperature});
    utils.output.printKeyValueFmt("Parameters", "{d} ({d:.2} MB)", .{
        num_params,
        @as(f64, @floatFromInt(num_params * 4)) / (1024 * 1024),
    });
    utils.output.println("", .{});
    utils.output.println("Training:", .{});
    utils.output.printKeyValueFmt("Epochs", "{d}", .{epochs});
    utils.output.printKeyValueFmt("Batch size", "{d}", .{batch_size});
    utils.output.printKeyValueFmt("Learning rate", "{e:.2}", .{learning_rate});
    utils.output.printKeyValueFmt("Warmup steps", "{d}", .{warmup_steps});
    utils.output.printKeyValueFmt("Weight decay", "{d:.4}", .{weight_decay});
    utils.output.printKeyValueFmt("Gradient clip", "{d:.2}", .{gradient_clip});
    if (dataset_path) |path| {
        utils.output.printKeyValueFmt("Dataset", "{s}", .{path});
    } else {
        utils.output.printKeyValue("Dataset", "(synthetic)");
    }
    utils.output.println("", .{});

    // Initialize model
    utils.output.println("Initializing CLIP model with random weights...", .{});
    var model = abi.features.ai.training.TrainableCLIPModel.init(allocator, clip_config) catch |err| {
        utils.output.printError("initializing model: {t}", .{err});
        return;
    };
    defer model.deinit();

    utils.output.println("Model initialized: {d} parameters", .{num_params});
    utils.output.println("", .{});

    // Generate synthetic training data (image-text pairs)
    const image_dim = image_size * image_size * 3;
    const text_max_len: u32 = 77;
    const num_samples = batch_size * 10;

    var train_images = allocator.alloc(f32, num_samples * image_dim) catch |err| {
        utils.output.printError("allocating training images: {t}", .{err});
        return;
    };
    defer allocator.free(train_images);

    var train_tokens = allocator.alloc(u32, num_samples * text_max_len) catch |err| {
        utils.output.printError("allocating training tokens: {t}", .{err});
        return;
    };
    defer allocator.free(train_tokens);

    // Initialize with random data
    var rng = std.Random.DefaultPrng.init(42);
    for (train_images) |*p| {
        p.* = rng.random().float(f32);
    }
    for (train_tokens) |*t| {
        t.* = rng.random().intRangeLessThan(u32, 0, clip_config.text_vocab_size);
    }

    utils.output.println("Generated {d} synthetic image-text pairs for training", .{num_samples});
    utils.output.println("", .{});
    utils.output.println("Starting CLIP contrastive training...", .{});

    var timer = abi.services.shared.time.Timer.start() catch {
        utils.output.printError("Failed to start timer", .{});
        return;
    };

    // Training loop
    const batches_per_epoch = num_samples / batch_size;
    var total_loss: f32 = 0;
    var step: u32 = 0;

    // Allocate embedding buffers
    const image_embeddings = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        utils.output.printError("allocating embeddings: {t}", .{err});
        return;
    };
    defer allocator.free(image_embeddings);

    const text_embeddings = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        utils.output.printError("allocating embeddings: {t}", .{err});
        return;
    };
    defer allocator.free(text_embeddings);

    const d_image_emb = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        utils.output.printError("allocating gradients: {t}", .{err});
        return;
    };
    defer allocator.free(d_image_emb);

    const d_text_emb = allocator.alloc(f32, batch_size * projection_dim) catch |err| {
        utils.output.printError("allocating gradients: {t}", .{err});
        return;
    };
    defer allocator.free(d_text_emb);

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..batches_per_epoch) |batch_idx| {
            const img_start = batch_idx * batch_size * image_dim;
            const txt_start = batch_idx * batch_size * text_max_len;

            const batch_images = train_images[img_start .. img_start + batch_size * image_dim];
            const batch_tokens = train_tokens[txt_start .. txt_start + batch_size * text_max_len];

            // Encode images and text
            model.encodeImages(batch_images, batch_size, image_embeddings) catch continue;
            model.encodeText(batch_tokens, batch_size, text_embeddings) catch continue;

            // Compute contrastive loss
            const loss = model.computeContrastiveLoss(
                image_embeddings,
                text_embeddings,
                batch_size,
                d_image_emb,
                d_text_emb,
            );
            epoch_loss += loss;

            // Apply gradient update (SGD)
            model.applySgdUpdate(learning_rate);
            model.zeroGradients();

            step += 1;

            if (step % log_interval == 0) {
                utils.output.println("  Step {d}: loss={d:.4}", .{ step, loss });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(batches_per_epoch));
        total_loss = epoch_loss;

        utils.output.println("Epoch {d}/{d}: avg_loss={d:.4}", .{ epoch + 1, epochs, epoch_loss });
    }

    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    utils.output.printHeader("CLIP Training Complete");
    utils.output.printKeyValueFmt("Final loss", "{d:.6}", .{total_loss});
    utils.output.printKeyValueFmt("Total steps", "{d}", .{step});
    utils.output.printKeyValueFmt("Wall time", "{d:.2}s", .{elapsed_s});
}

pub fn printVisionHelp() void {
    const help_text =
        \\Usage: abi train vision [options]
        \\
        \\Train a Vision Transformer (ViT) model for image classification.
        \\
        \\Architecture options:
        \\  --image-size <n>     Input image size (default: 224)
        \\  --patch-size <n>     Patch size (default: 16)
        \\  --hidden-size <n>    Hidden dimension (default: 384)
        \\  --num-layers <n>     Number of transformer layers (default: 12)
        \\  --num-heads <n>      Number of attention heads (default: 6)
        \\  --mlp-dim <n>        MLP hidden dimension (default: 1536)
        \\  --num-classes <n>    Number of output classes (default: 1000)
        \\  --dropout <f>        Dropout rate (default: 0.1)
        \\
        \\Training options:
        \\  -e, --epochs <n>     Number of epochs (default: 10)
        \\  -b, --batch-size <n> Batch size (default: 32)
        \\  --lr, --learning-rate <f> Learning rate (default: 1e-4)
        \\  --warmup-steps <n>   Warmup steps (default: 500)
        \\  --weight-decay <f>   Weight decay (default: 0.01)
        \\  --gradient-clip <f>  Gradient clip norm (default: 1.0)
        \\  --log-interval <n>   Log every N steps (default: 10)
        \\  --dataset-path <path> Dataset directory
        \\
        \\Examples:
        \\  abi train vision --epochs 10 --batch-size 64
        \\  abi train vision --hidden-size 768 --num-layers 12 --num-heads 12
        \\  abi train vision --dataset-path ./imagenet --epochs 90
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

pub fn printClipHelp() void {
    const help_text =
        \\Usage: abi train clip [options]
        \\
        \\Train a CLIP (Contrastive Language-Image Pretraining) model.
        \\
        \\Vision encoder options:
        \\  --image-size <n>     Input image size (default: 224)
        \\  --patch-size <n>     Patch size (default: 16)
        \\  --vision-hidden <n>  Vision hidden dimension (default: 768)
        \\  --vision-layers <n>  Vision transformer layers (default: 12)
        \\
        \\Text encoder options:
        \\  --text-hidden <n>    Text hidden dimension (default: 512)
        \\  --text-layers <n>    Text transformer layers (default: 12)
        \\
        \\Contrastive options:
        \\  --projection-dim <n> Shared projection dimension (default: 512)
        \\  --temperature <f>    Temperature for InfoNCE loss (default: 0.07)
        \\
        \\Training options:
        \\  -e, --epochs <n>     Number of epochs (default: 10)
        \\  -b, --batch-size <n> Batch size (default: 64)
        \\  --lr, --learning-rate <f> Learning rate (default: 1e-4)
        \\  --warmup-steps <n>   Warmup steps (default: 2000)
        \\  --weight-decay <f>   Weight decay (default: 0.1)
        \\  --gradient-clip <f>  Gradient clip norm (default: 1.0)
        \\  --log-interval <n>   Log every N steps (default: 10)
        \\  --dataset-path <path> Image-text pairs dataset
        \\
        \\Examples:
        \\  abi train clip --epochs 10 --batch-size 256
        \\  abi train clip --vision-hidden 768 --text-hidden 512 --projection-dim 512
        \\  abi train clip --dataset-path ./laion --epochs 32
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

test {
    std.testing.refAllDecls(@This());
}
