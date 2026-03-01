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

    utils.output.printHeader("Auto-train: Abbey, Aviva, Abi (basic data + vision/multimodal)");
    utils.output.println("", .{});

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
        utils.output.printError("Self-learning init failed: {t}", .{err});
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
            utils.output.printError("recording Abbey experience: {t}", .{e});
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
            utils.output.printError("recording Aviva experience: {t}", .{e});
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
            utils.output.printError("recording Abi experience: {t}", .{e});
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
            utils.output.printError("recording vision experience: {t}", .{e});
            return;
        };
        system.recordVisionExperience(vision_in, vision_out, synth_img, .positive, 0.8) catch |e| {
            utils.output.printError("recording vision experience 2: {t}", .{e});
            return;
        };
    }

    system.update() catch |e| {
        utils.output.printError("during self-learning update: {t}", .{e});
        return;
    };

    const stats = system.getStats();
    utils.output.printHeader("Self-learning complete");
    utils.output.printKeyValueFmt("Total experiences", "{d}", .{stats.total_experiences});
    utils.output.printKeyValueFmt("Total updates", "{d}", .{stats.total_updates});
    utils.output.printKeyValueFmt("Vision samples", "{d}", .{stats.vision_samples});
    utils.output.printKeyValueFmt("Document samples", "{d}", .{stats.document_samples});
    utils.output.printKeyValueFmt("Avg reward", "{d:.4}", .{stats.avg_reward});
    utils.output.printKeyValueFmt("Improvement rate", "{d:.4}", .{stats.improvement_rate});
    utils.output.println("", .{});

    if (vision_enabled and run_multimodal) {
        utils.output.println("Multimodal micro-steps (vision + CLIP)...", .{});
        runAutoTrainMicroVision(allocator) catch |e| {
            utils.output.printWarning("vision micro-step failed: {t}", .{e});
        };
        runAutoTrainMicroClip(allocator) catch |e| {
            utils.output.printWarning("CLIP micro-step failed: {t}", .{e});
        };
        utils.output.println("Multimodal micro-steps done.", .{});
        utils.output.println("", .{});
    }

    utils.output.printSuccess("Auto-train complete.", .{});
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
    utils.output.println("  ViT micro: {d} batches", .{num_batches});
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
    utils.output.println("  CLIP micro: {d} batches", .{num_batches});
}

pub fn printAutoHelp() void {
    utils.output.print(
        \\Usage: train auto [options]
        \\Auto-train Abbey, Aviva, and Abi with basic seed data and optional vision/multimodal.
        \\
        \\  Seeds the self-learning system with:
        \\  - Text experiences for Abbey (empathetic), Aviva (direct), Abi (adaptive)
        \\  - Vision experiences if -Dfeat-vision=true (legacy: -Denable-vision=true)
        \\  Runs one update step and prints stats.
        \\
        \\  With --multimodal (and -Dfeat-vision=true (legacy: -Denable-vision=true)): also runs minimal ViT and CLIP
        \\  micro-training steps to exercise vision and multimodal pipelines.
        \\
        \\Options:
        \\  --help, -h       Show this help
        \\  --multimodal     Run vision + CLIP micro-steps when vision is enabled
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
