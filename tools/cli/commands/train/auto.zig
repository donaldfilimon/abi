//! Auto-training handler.
//!
//! Handles the `abi train auto` subcommand which runs automated training
//! with seed data for Abbey, Aviva, and Abi personas, with optional
//! vision/multimodal micro-training steps.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runAutoTrain(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printAutoHelp();
        return;
    }

    var run_multimodal = false;
    for (args) |arg| {
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--multimodal")) {
            run_multimodal = true;
            break;
        }
    }

    std.debug.print("Auto-train: Abbey, Aviva, Abi (basic data + vision/multimodal)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    const vision_enabled = abi.ai.vision.isEnabled();
    const config = abi.ai.training.SelfLearningConfig{
        .enable_vision = vision_enabled,
        .enable_video = true,
        .enable_audio = true,
        .enable_all_modalities = true,
        .enable_documents = true,
        .replay_buffer_size = 1000,
        .batch_size = 16,
        .min_buffer_size = 16,
    };

    var system = abi.ai.training.SelfLearningSystem.init(allocator, config) catch |err| {
        utils.output.printError("Self-learning init failed: {t}\n", .{err});
        return;
    };
    defer system.deinit();

    // Seed text experiences for Abbey, Aviva, Abi (dummy token IDs per persona)
    const abbey_in: []const u32 = &[_]u32{ 1, 2, 3, 4, 5 };
    const abbey_out: []const u32 = &[_]u32{ 1, 2, 3, 4, 5, 6 };
    const aviva_in: []const u32 = &[_]u32{ 10, 11, 12, 13, 14 };
    const aviva_out: []const u32 = &[_]u32{ 10, 11, 12, 13, 14, 15 };
    const abi_in: []const u32 = &[_]u32{ 20, 21, 22, 23, 24 };
    const abi_out: []const u32 = &[_]u32{ 20, 21, 22, 23, 24, 25 };

    var i: usize = 0;
    while (i < 6) : (i += 1) {
        system.recordExperience(
            abbey_in,
            abbey_out,
            .positive,
            0.9,
            .text_conversation,
        ) catch |e| {
            std.debug.print("Error recording Abbey experience: {t}\n", .{e});
            return;
        };
    }
    i = 0;
    while (i < 6) : (i += 1) {
        system.recordExperience(
            aviva_in,
            aviva_out,
            .positive,
            0.9,
            .text_conversation,
        ) catch |e| {
            std.debug.print("Error recording Aviva experience: {t}\n", .{e});
            return;
        };
    }
    i = 0;
    while (i < 6) : (i += 1) {
        system.recordExperience(
            abi_in,
            abi_out,
            .positive,
            0.9,
            .text_conversation,
        ) catch |e| {
            std.debug.print("Error recording Abi experience: {t}\n", .{e});
            return;
        };
    }

    if (vision_enabled) {
        // Synthetic 224x224x3 image bytes for vision experience
        const synth_img = allocator.alloc(u8, 224 * 224 * 3) catch return;
        defer allocator.free(synth_img);
        for (synth_img, 0..) |*p, j| {
            p.* = @intCast((j % 256));
        }
        const vision_in: []const u32 = &[_]u32{ 100, 101 };
        const vision_out: []const u32 = &[_]u32{ 100, 101, 102 };
        system.recordVisionExperience(vision_in, vision_out, synth_img, .positive, 0.85) catch |e| {
            std.debug.print("Error recording vision experience: {t}\n", .{e});
            return;
        };
        system.recordVisionExperience(vision_in, vision_out, synth_img, .positive, 0.8) catch |e| {
            std.debug.print("Error recording vision experience 2: {t}\n", .{e});
            return;
        };
    }

    system.update() catch |e| {
        std.debug.print("Error during self-learning update: {t}\n", .{e});
        return;
    };

    const stats = system.getStats();
    std.debug.print("Self-learning complete\n", .{});
    std.debug.print("=====================\n", .{});
    std.debug.print("Total experiences: {d}\n", .{stats.total_experiences});
    std.debug.print("Total updates:      {d}\n", .{stats.total_updates});
    std.debug.print("Vision samples:     {d}\n", .{stats.vision_samples});
    std.debug.print("Document samples:   {d}\n", .{stats.document_samples});
    std.debug.print("Avg reward:          {d:.4}\n", .{stats.avg_reward});
    std.debug.print("Improvement rate:    {d:.4}\n\n", .{stats.improvement_rate});

    if (vision_enabled and run_multimodal) {
        std.debug.print("Multimodal micro-steps (vision + CLIP)...\n", .{});
        runAutoTrainMicroVision(allocator) catch |e| {
            std.debug.print("Warning: vision micro-step failed: {t}\n", .{e});
        };
        runAutoTrainMicroClip(allocator) catch |e| {
            std.debug.print("Warning: CLIP micro-step failed: {t}\n", .{e});
        };
        std.debug.print("Multimodal micro-steps done.\n\n", .{});
    }

    std.debug.print("Auto-train complete.\n", .{});
}

pub fn runAutoTrainMicroVision(allocator: std.mem.Allocator) !void {
    const image_size: u32 = 224;
    const patch_size: u32 = 16;
    const batch_size: u32 = 4;
    const num_classes: u32 = 5;
    const num_batches: u32 = 2;
    const learning_rate: f32 = 1e-4;
    const gradient_clip: f32 = 1.0;

    const vit_config = abi.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = 96,
        .num_layers = 2,
        .num_heads = 2,
        .mlp_dim = 384,
        .in_channels = 3,
        .use_class_token = true,
    };
    const trainable_vit_config = abi.ai.training.TrainableViTConfig{
        .vit_config = vit_config,
        .max_batch_size = batch_size,
        .num_classes = num_classes,
        .dropout = 0.1,
    };

    var model = abi.ai.training.TrainableViTModel.init(allocator, trainable_vit_config) catch return;
    defer model.deinit();

    const image_dim = image_size * image_size * 3;
    const num_samples = batch_size * num_batches;
    var train_images = try allocator.alloc(f32, num_samples * image_dim);
    defer allocator.free(train_images);
    const train_labels = try allocator.alloc(u32, num_samples);
    defer allocator.free(train_labels);

    var rng = std.Random.DefaultPrng.init(123);
    for (train_images) |*p| p.* = rng.random().float(f32);
    for (train_labels) |*l| l.* = rng.random().intRangeLessThan(u32, 0, num_classes);

    var step: u32 = 0;
    while (step < num_batches) : (step += 1) {
        const batch_start = step * batch_size * image_dim;
        const batch_images = train_images[batch_start .. batch_start + batch_size * image_dim];
        const logits = try allocator.alloc(f32, batch_size * num_classes);
        defer allocator.free(logits);
        model.forward(batch_images, batch_size, logits) catch return;
        _ = model.clipGradients(gradient_clip);
        model.applySgdUpdate(learning_rate);
        model.zeroGradients();
    }
    std.debug.print("  ViT micro: {d} batches\n", .{num_batches});
}

pub fn runAutoTrainMicroClip(allocator: std.mem.Allocator) !void {
    const image_size: u32 = 224;
    const patch_size: u32 = 16;
    const batch_size: u32 = 4;
    const num_batches: u32 = 2;
    const projection_dim: u32 = 64;
    const text_max_len: u32 = 16;
    const text_vocab: u32 = 256;

    const vit_config = abi.ai.vision.ViTConfig{
        .image_size = image_size,
        .patch_size = patch_size,
        .hidden_size = 96,
        .num_layers = 2,
        .num_heads = 2,
        .mlp_dim = 384,
        .in_channels = 3,
        .use_class_token = true,
    };
    const vision_config = abi.ai.training.TrainableViTConfig{
        .vit_config = vit_config,
        .max_batch_size = batch_size,
        .num_classes = 0,
        .projection_dim = projection_dim,
    };
    const clip_config = abi.ai.training.CLIPTrainingConfig{
        .vision_config = vision_config,
        .text_hidden_size = 64,
        .text_vocab_size = text_vocab,
        .text_max_len = text_max_len,
        .text_num_layers = 2,
        .text_num_heads = 2,
        .projection_dim = projection_dim,
        .temperature = 0.07,
        .learnable_temperature = false,
    };

    var model = abi.ai.training.TrainableCLIPModel.init(allocator, clip_config) catch return;
    defer model.deinit();

    const image_dim = image_size * image_size * 3;
    const num_samples = batch_size * num_batches;
    var train_images = try allocator.alloc(f32, num_samples * image_dim);
    defer allocator.free(train_images);
    var train_tokens = try allocator.alloc(u32, num_samples * text_max_len);
    defer allocator.free(train_tokens);

    var rng = std.Random.DefaultPrng.init(456);
    for (train_images) |*p| p.* = rng.random().float(f32);
    for (train_tokens) |*t| t.* = rng.random().intRangeLessThan(u32, 0, text_vocab);

    const image_emb = try allocator.alloc(f32, batch_size * projection_dim);
    defer allocator.free(image_emb);
    const text_emb = try allocator.alloc(f32, batch_size * projection_dim);
    defer allocator.free(text_emb);
    const d_img = try allocator.alloc(f32, batch_size * projection_dim);
    defer allocator.free(d_img);
    const d_txt = try allocator.alloc(f32, batch_size * projection_dim);
    defer allocator.free(d_txt);

    var step: u32 = 0;
    while (step < num_batches) : (step += 1) {
        const img_start = step * batch_size * image_dim;
        const txt_start = step * batch_size * text_max_len;
        const batch_images = train_images[img_start .. img_start + batch_size * image_dim];
        const batch_tokens = train_tokens[txt_start .. txt_start + batch_size * text_max_len];
        model.encodeImages(batch_images, batch_size, image_emb) catch return;
        model.encodeText(batch_tokens, batch_size, text_emb) catch return;
        _ = model.computeContrastiveLoss(image_emb, text_emb, batch_size, d_img, d_txt);
        model.zeroGradients();
        model.applySgdUpdate(1e-4);
    }
    std.debug.print("  CLIP micro: {d} batches\n", .{num_batches});
}

pub fn printAutoHelp() void {
    std.debug.print(
        \\Usage: train auto [options]
        \\Auto-train Abbey, Aviva, and Abi with basic seed data and optional vision/multimodal.
        \\
        \\  Seeds the self-learning system with:
        \\  - Text experiences for Abbey (empathetic), Aviva (direct), Abi (adaptive)
        \\  - Vision experiences if -Denable-vision=true
        \\  Runs one update step and prints stats.
        \\
        \\  With --multimodal (and -Denable-vision=true): also runs minimal ViT and CLIP
        \\  micro-training steps to exercise vision and multimodal pipelines.
        \\
        \\Options:
        \\  --help, -h       Show this help
        \\  --multimodal     Run vision + CLIP micro-steps when vision is enabled
        \\
    , .{});
}
