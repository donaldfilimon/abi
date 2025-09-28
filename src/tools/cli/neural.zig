const std = @import("std");
const abi = @import("abi");
const common = @import("common.zig");
const ml = @import("ml_support.zig");

pub const command = common.Command{
    .id = .neural,
    .name = "neural",
    .summary = "Train and inspect neural network models",
    .usage = "abi neural <train|predict|info|benchmark> [options]",
    .details = "  train      Train a neural model from CSV data\n" ++
        "  predict    Run inference with a saved model\n" ++
        "  info       Inspect saved model metadata\n" ++
        "  benchmark  Execute neural benchmark\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "train")) {
        var data_path: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var epochs: usize = 100;
        var learning_rate: f32 = 0.001;
        var batch_size: usize = 32;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--data") and i + 1 < args.len) {
                data_path = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--output") and i + 1 < args.len) {
                output_path = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--epochs") and i + 1 < args.len) {
                epochs = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--lr") and i + 1 < args.len) {
                learning_rate = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--batch-size") and i + 1 < args.len) {
                batch_size = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            }
        }

        if (data_path == null) {
            std.debug.print("neural train requires --data <path>\n", .{});
            return;
        }

        const final_output = output_path orelse "neural_model.bin";
        std.debug.print("Training neural network on {s}...\n", .{data_path.?});

        var training_data = try ml.loadTrainingData(allocator, data_path.?);
        defer training_data.deinit();

        try ml.trainNeuralNetwork(allocator, training_data, final_output, epochs, learning_rate, batch_size, false);
        std.debug.print("Training completed. Model saved to: {s}\n", .{final_output});
    } else if (std.mem.eql(u8, sub, "predict")) {
        var model_path: ?[]const u8 = null;
        var input_str: ?[]const u8 = null;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
                model_path = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--input") and i + 1 < args.len) {
                input_str = args[i + 1];
                i += 1;
            }
        }

        if (model_path == null or input_str == null) {
            std.debug.print("neural predict requires --model and --input\n", .{});
            return;
        }

        const input = try common.parseCsvFloats(allocator, input_str.?);
        defer allocator.free(input);

        var network = try abi.ai.NeuralNetwork.loadFromFile(allocator, model_path.?);
        defer network.deinit();

        const output = try allocator.alloc(f32, network.output_shape[0]);
        defer allocator.free(output);

        try network.predict(input, output);

        std.debug.print("Prediction: ", .{});
        for (output, 0..) |val, idx| {
            if (idx > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.6}", .{val});
        }
        std.debug.print("\n", .{});
    } else if (std.mem.eql(u8, sub, "info")) {
        var model_path: ?[]const u8 = null;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
                model_path = args[i + 1];
                i += 1;
            }
        }

        if (model_path == null) {
            std.debug.print("neural info requires --model <path>\n", .{});
            return;
        }

        var network = try abi.ai.NeuralNetwork.loadFromFile(allocator, model_path.?);
        defer network.deinit();

        const info = network.getParameterCount();
        std.debug.print("Neural Network Info:\n", .{});
        std.debug.print("  Input size: {}\n", .{network.input_shape[0]});
        std.debug.print("  Output size: {}\n", .{network.output_shape[0]});
        std.debug.print("  Layers: {}\n", .{network.layers.items.len});
        std.debug.print("  Parameters: {}\n", .{info});
    } else if (std.mem.eql(u8, sub, "benchmark")) {
        var size: usize = 1000;
        var iterations: usize = 1000;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--size") and i + 1 < args.len) {
                size = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--iterations") and i + 1 < args.len) {
                iterations = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            }
        }

        std.debug.print("Running neural network benchmark...\n", .{});
        try runNeuralBenchmark(allocator, size, iterations);
    } else {
        std.debug.print("Unknown neural subcommand: {s}\n", .{sub});
    }
}

fn runNeuralBenchmark(allocator: std.mem.Allocator, size: usize, iterations: usize) !void {
    std.debug.print("Neural benchmark: size={}, iterations={}\n", .{ size, iterations });

    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{size}, &[_]usize{1});
    defer network.deinit();

    const layer1 = try abi.ai.Layer.init(allocator, .dense, &[_]usize{size}, &[_]usize{64});
    layer1.activation = .relu;
    try network.addLayer(layer1);

    const layer2 = try abi.ai.Layer.init(allocator, .dense, &[_]usize{64}, &[_]usize{1});
    try network.addLayer(layer2);

    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const target = try allocator.alloc(f32, 1);
    defer allocator.free(target);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (input) |*val| {
        val.* = random.float(f32) * 2.0 - 1.0;
    }
    target[0] = f32(0.0);

    var timer = try std.time.Timer.start();
    var i: usize = 0;
    var total_loss: f32 = 0;
    while (i < iterations) : (i += 1) {
        const loss = try network.trainStep(input, target);
        total_loss += loss;
    }
    const total = timer.read();
    const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iterations));

    std.debug.print(
        "Neural benchmark completed. Avg loss: {d:.6}, Time: {d:.2}ms\n",
        .{ @as(f64, total_loss) / @as(f64, @floatFromInt(iterations)), avg / 1_000_000.0 },
    );
}
