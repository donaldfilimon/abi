const std = @import("std");
const abi = @import("abi");

pub const TrainingData = struct {
    inputs: []const []const f32,
    targets: []const []const f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TrainingData) void {
        for (self.inputs) |input| {
            self.allocator.free(input);
        }
        for (self.targets) |target| {
            self.allocator.free(target);
        }
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
    }
};

pub fn loadTrainingData(allocator: std.mem.Allocator, path: []const u8) !TrainingData {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var inputs = try std.ArrayList([]f32).initCapacity(allocator, 0);
    var targets = try std.ArrayList([]f32).initCapacity(allocator, 0);
    defer {
        for (inputs.items) |input| allocator.free(input);
        for (targets.items) |target| allocator.free(target);
        inputs.deinit(allocator);
        targets.deinit(allocator);
    }

    var buf: [1024]u8 = undefined;
    var content_list = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer content_list.deinit(allocator);

    while (true) {
        const n = try file.read(&buf);
        if (n == 0) break;
        try content_list.appendSlice(allocator, buf[0..n]);
    }

    const file_content = content_list.items;
    var lines = std.mem.splitScalar(u8, file_content, '\n');
    while (lines.next()) |line| {
        const trimmed_line = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed_line.len == 0) continue;

        var parts = std.mem.splitScalar(u8, trimmed_line, ',');
        var values = try std.ArrayList(f32).initCapacity(allocator, 0);
        defer values.deinit(allocator);

        while (parts.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\r\n");
            if (trimmed.len > 0) {
                const val = try std.fmt.parseFloat(f32, trimmed);
                try values.append(allocator, val);
            }
        }

        if (values.items.len >= 2) {
            const input = try allocator.dupe(f32, values.items[0 .. values.items.len - 1]);
            const target = try allocator.dupe(f32, values.items[values.items.len - 1 ..]);

            try inputs.append(allocator, input);
            try targets.append(allocator, target);
        }
    }

    return TrainingData{
        .inputs = try inputs.toOwnedSlice(allocator),
        .targets = try targets.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

pub fn trainNeuralNetwork(
    allocator: std.mem.Allocator,
    data: TrainingData,
    output_path: []const u8,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    use_gpu: bool,
) !void {
    _ = use_gpu;

    const input_size = if (data.inputs.len > 0) data.inputs[0].len else 1;
    const output_size = if (data.targets.len > 0) data.targets[0].len else 1;

    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{input_size}, &[_]usize{output_size});
    defer network.deinit();

    const hidden_size = @max(32, input_size * 2);
    const layer1 = try abi.ai.Layer.init(allocator, .dense, &[_]usize{input_size}, &[_]usize{hidden_size});
    layer1.activation = .relu;
    try network.addLayer(layer1);

    const layer2 = try abi.ai.Layer.init(allocator, .dense, &[_]usize{hidden_size}, &[_]usize{output_size});
    try network.addLayer(layer2);

    var prng = std.Random.DefaultPrng.init(42);
    var random = prng.random();
    for (network.layers.items) |*layer| {
        try layer.*.initializeWeights(allocator, &random);
    }

    try network.compile();

    const config = abi.ai.TrainingConfig{
        .learning_rate = learning_rate,
        .batch_size = batch_size,
        .epochs = epochs,
        .validation_split = 0.2,
        .early_stopping_patience = 10,
        .log_frequency = 10,
    };

    const trainer = try abi.ai.ModelTrainer.init(
        allocator,
        network,
        config,
        .adam,
        .mean_squared_error,
    );
    defer trainer.deinit();

    var metrics = try trainer.train(data.inputs, data.targets);
    defer {
        for (metrics.items) |_| {}
        metrics.deinit(allocator);
    }

    if (output_path.len > 0) {
        std.debug.print("Saving model to {s}...\n", .{output_path});
        const file = try std.fs.cwd().createFile(output_path, .{});
        defer file.close();

        try file.writeAll("ABI Neural Network Model\n");
        try file.writeAll("Note: Full model serialization requires additional implementation\n");
        std.debug.print("Model metadata saved successfully\n", .{});
    }

    std.debug.print("Neural network training completed. Final loss: {d:.6}\n", .{metrics.items[metrics.items.len - 1].loss});
}

pub fn trainLinearModel(
    allocator: std.mem.Allocator,
    data: TrainingData,
    output_path: []const u8,
    epochs: usize,
    learning_rate: f32,
) !void {
    _ = output_path;

    const num_features = if (data.inputs.len > 0) data.inputs[0].len else 1;
    const num_samples = data.inputs.len;

    var weights = try allocator.alloc(f32, num_features);
    defer allocator.free(weights);
    var bias: f32 = 0.0;

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (weights) |*w| {
        w.* = (random.float(f32) - 0.5) * 0.1;
    }

    var epoch: usize = 0;
    while (epoch < epochs) : (epoch += 1) {
        var total_loss: f32 = 0.0;

        for (data.inputs, data.targets) |input, target_slice| {
            const target = target_slice[0];

            var prediction: f32 = bias;
            for (input, weights) |x, w| {
                prediction += x * w;
            }

            const err = prediction - target;
            total_loss += err * err;

            const lr = learning_rate / @as(f32, @floatFromInt(num_samples));
            bias -= lr * err;

            for (0..num_features) |i| {
                weights[i] -= lr * err * input[i];
            }
        }

        if (epoch % 10 == 0) {
            std.debug.print("Epoch {d}: Loss = {d:.6}\n", .{ epoch, total_loss / @as(f32, @floatFromInt(num_samples)) });
        }
    }

    std.debug.print("Linear model training completed!\n", .{});
    std.debug.print("Final weights: ", .{});
    for (weights, 0..) |w, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.4}", .{w});
    }
    std.debug.print("\nBias: {d:.4}\n", .{bias});
}
