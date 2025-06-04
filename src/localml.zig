const std = @import("std");

pub const MLError = error{
    EmptyDataset,
    InvalidData,
    InvalidUsage,
    FileReadError,
    FileWriteError,
    ModelNotInitialized,
    InvalidParameters,
    TrainingFailed,
};

pub const DataRow = struct {
    x1: f64,
    x2: f64,
    label: f64,

    pub fn validate(self: DataRow) MLError!void {
        if (std.math.isNan(self.x1) or std.math.isNan(self.x2) or std.math.isNan(self.label)) {
            return MLError.InvalidData;
        }
        if (std.math.isInf(self.x1) or std.math.isInf(self.x2) or std.math.isInf(self.label)) {
            return MLError.InvalidData;
        }
    }

    pub fn fromArray(values: []const f64) MLError!DataRow {
        if (values.len != 3) return MLError.InvalidData;
        const row = DataRow{
            .x1 = values[0],
            .x2 = values[1],
            .label = values[2],
        };
        try row.validate();
        return row;
    }
};

pub const Model = struct {
    weights: [2]f64,
    bias: f64,
    is_trained: bool,

    pub fn init() Model {
        return Model{
            .weights = .{ 0, 0 },
            .bias = 0,
            .is_trained = false,
        };
    }

    pub fn predict(self: Model, row: DataRow) MLError!f64 {
        if (!self.is_trained) return MLError.ModelNotInitialized;
        try row.validate();
        return row.x1 * self.weights[0] + row.x2 * self.weights[1] + self.bias;
    }
    pub fn train(self: *Model, data: []const DataRow, learning_rate: f64, epochs: usize) MLError!void {
        if (data.len == 0) return MLError.EmptyDataset;
        if (learning_rate <= 0 or learning_rate >= 1) return MLError.InvalidParameters;
        if (epochs == 0) return MLError.InvalidParameters;

        // Validate all data first
        for (data) |row| {
            try row.validate();
        }

        // Simple gradient descent
        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            var total_err: f64 = 0;

            for (data) |row| {
                const prediction = row.x1 * self.weights[0] + row.x2 * self.weights[1] + self.bias;
                const err_val = prediction - row.label;

                // Update weights and bias
                self.weights[0] -= learning_rate * err_val * row.x1;
                self.weights[1] -= learning_rate * err_val * row.x2;
                self.bias -= learning_rate * err_val;

                total_err += err_val * err_val;
            }

            // Early stopping if error is small enough
            if (total_err < 0.0001) break;
        }

        self.is_trained = true;
    }
};

fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]DataRow {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var reader = file.reader();
    var rows = std.ArrayList(DataRow).init(allocator);
    var buf: [256]u8 = undefined;
    while (true) {
        const line = (try reader.readUntilDelimiterOrEof(&buf, '\n')) orelse break;
        var it = std.mem.splitScalar(u8, line, ',');
        const p1 = it.next() orelse continue;
        const p2 = it.next() orelse continue;
        const p3 = it.next() orelse continue;
        const x1 = try std.fmt.parseFloat(f64, std.mem.trim(u8, p1, " \t\r\n"));
        const x2 = try std.fmt.parseFloat(f64, std.mem.trim(u8, p2, " \t\r\n"));
        const label = try std.fmt.parseFloat(f64, std.mem.trim(u8, p3, " \t\r\n"));
        try rows.append(.{ .x1 = x1, .x2 = x2, .label = label });
    }
    return rows.toOwnedSlice();
}

fn logistic(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn train(data: []const DataRow, iterations: u32, lr: f64) !struct { w: [2]f64, b: f64 } {
    if (data.len == 0) return error.EmptyDataset;

    var w = [2]f64{ 0.0, 0.0 };
    var b: f64 = 0.0;
    var i: u32 = 0;
    var prev_loss: f64 = std.math.inf(f64);

    // Training loop
    while (i < iterations) : (i += 1) {
        var loss: f64 = 0.0;
        var grad_w0: f64 = 0.0;
        var grad_w1: f64 = 0.0;
        var grad_b: f64 = 0.0;
        const n = @as(f64, @floatFromInt(data.len));

        // Compute gradients and loss
        for (data) |row| {
            const x1 = row.x1;
            const x2 = row.x2;
            const y = row.label;

            const z = w[0] * x1 + w[1] * x2 + b;
            const sigmoid = 1.0 / (1.0 + std.math.exp(-z));
            const diff = sigmoid - y;

            loss += -y * std.math.log(sigmoid) - (1 - y) * std.math.log(1 - sigmoid);
            grad_w0 += diff * x1;
            grad_w1 += diff * x2;
            grad_b += diff;
        }

        loss /= n;
        if (@mod(i, 100) == 0) {
            std.log.info("iteration {d}: loss = {d:.6}", .{ i, loss });
        }
        if (@abs(loss - prev_loss) < 1e-7) {
            std.log.info("converged at iteration {d}", .{i});
            break;
        }
        prev_loss = loss;

        w[0] -= lr * grad_w0 / n;
        w[1] -= lr * grad_w1 / n;
        b -= lr * grad_b / n;
    }
    return .{ .w = .{ w[0], w[1] }, .b = b };
}

fn saveModel(path: []const u8, w: [2]f64, b: f64) !void {
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writer().print("{d} {d} {d}\n", .{ w[0], w[1], b });
}

fn loadModel(path: []const u8) !struct { w: [2]f64, b: f64 } {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var buf: [128]u8 = undefined;
    const line = (try file.reader().readUntilDelimiterOrEof(&buf, '\n')) orelse "";
    var it = std.mem.splitScalar(u8, line, ' ');
    const w0 = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    const w1 = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    const b = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    return .{ .w = .{ w0, w1 }, .b = b };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var args = std.process.args();
    _ = args.next(); // skip executable name

    const cmd = args.next() orelse {
        std.log.err("Usage: localml [train|predict] [args...]", .{});
        return error.InvalidUsage;
    };

    if (std.mem.eql(u8, cmd, "train")) {
        const data_path = args.next() orelse {
            std.log.err("Usage: localml train <data.csv> <model.txt>", .{});
            return error.InvalidUsage;
        };
        const model_path = args.next() orelse {
            std.log.err("Usage: localml train <data.csv> <model.txt>", .{});
            return error.InvalidUsage;
        };

        var data = std.ArrayList(DataRow).init(alloc);
        defer data.deinit();

        // Load training data
        const data_contents = try std.fs.cwd().readFileAlloc(alloc, data_path, 1024 * 1024);
        defer alloc.free(data_contents);

        var lines = std.mem.tokenize(u8, data_contents, "\n");
        while (lines.next()) |line| {
            var cols = std.mem.tokenize(u8, line, ",");
            const x1 = try std.fmt.parseFloat(f64, cols.next() orelse continue);
            const x2 = try std.fmt.parseFloat(f64, cols.next() orelse continue);
            const label = try std.fmt.parseFloat(f64, cols.next() orelse continue);
            try data.append(.{ .x1 = x1, .x2 = x2, .label = label });
        }

        // Train model
        const model = try train(data.items, 1000, 0.1);
        try saveModel(model_path, model.w, model.b);
        std.log.info("Model saved to {s}", .{model_path});
    } else if (std.mem.eql(u8, cmd, "predict")) {
        const model_path = args.next() orelse {
            std.log.err("Usage: localml predict <model.txt> <x1> <x2>", .{});
            return error.InvalidUsage;
        };
        const x1_str = args.next() orelse {
            std.log.err("Usage: localml predict <model.txt> <x1> <x2>", .{});
            return error.InvalidUsage;
        };
        const x2_str = args.next() orelse {
            std.log.err("Usage: localml predict <model.txt> <x1> <x2>", .{});
            return error.InvalidUsage;
        };

        const model = try loadModel(model_path);
        const x1 = try std.fmt.parseFloat(f64, x1_str);
        const x2 = try std.fmt.parseFloat(f64, x2_str);

        const z = model.w[0] * x1 + model.w[1] * x2 + model.b;
        const prob = 1.0 / (1.0 + std.math.exp(-z));
        std.log.info("Probability: {d:.6}", .{prob});
    } else {
        std.log.err("Unknown command: {s}", .{cmd});
        return error.InvalidCommand;
    }
}

test "DataRow validation" {
    // Valid data
    const valid_row = DataRow{ .x1 = 1.0, .x2 = 2.0, .label = 0.0 };
    try valid_row.validate();

    // Test array constructor
    const array_row = try DataRow.fromArray(&[_]f64{ 1.0, 2.0, 0.0 });
    try std.testing.expectEqual(valid_row.x1, array_row.x1);
    try std.testing.expectEqual(valid_row.x2, array_row.x2);
    try std.testing.expectEqual(valid_row.label, array_row.label);

    // Invalid data
    const invalid_row = DataRow{
        .x1 = std.math.nan(f64),
        .x2 = 2.0,
        .label = 0.0,
    };
    try std.testing.expectError(MLError.InvalidData, invalid_row.validate());
}

test "Model training and prediction" {
    var model = Model.init();

    // Test uninitialized model
    const test_row = DataRow{ .x1 = 1.0, .x2 = 1.0, .label = 0.0 };
    try std.testing.expectError(MLError.ModelNotInitialized, model.predict(test_row));

    // Training data for simple binary classification
    const training_data = [_]DataRow{
        .{ .x1 = 0.0, .x2 = 0.0, .label = 0.0 },
        .{ .x1 = 1.0, .x2 = 1.0, .label = 1.0 },
        .{ .x1 = 0.0, .x2 = 0.2, .label = 0.0 },
        .{ .x1 = 0.8, .x2 = 0.9, .label = 1.0 },
    };

    // Train model
    try model.train(&training_data, 0.1, 5000);
    try std.testing.expect(model.is_trained);

    // Test predictions (allowing for some error margin)
    const pred1 = try model.predict(training_data[0]);
    try std.testing.expect(pred1 < 0.3); // Should be closer to 0

    const pred2 = try model.predict(training_data[1]);
    try std.testing.expect(pred2 > 0.7); // Should be closer to 1
}

test "Error handling" {
    var model = Model.init();

    // Empty dataset
    const empty_data = [_]DataRow{};
    try std.testing.expectError(MLError.EmptyDataset, model.train(&empty_data, 0.1, 100));

    // Invalid learning rate
    const data = [_]DataRow{.{ .x1 = 0.0, .x2 = 0.0, .label = 0.0 }};
    try std.testing.expectError(MLError.InvalidParameters, model.train(&data, 1.5, 100));
    try std.testing.expectError(MLError.InvalidParameters, model.train(&data, -0.1, 100));
    try std.testing.expectError(MLError.InvalidParameters, model.train(&data, 0.1, 0));
}
