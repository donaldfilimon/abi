//! LocalML Module - On-Device Machine Learning
//!
//! This module provides lightweight machine learning capabilities that run entirely
//! on the local device without requiring external services:
//! - Simple linear regression and classification models
//! - Basic neural network implementations
//! - Data preprocessing and feature engineering
//! - Model serialization and persistence
//! - Cross-validation and evaluation metrics
//! - Memory-efficient streaming data processing

const std = @import("std");

/// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// LocalML-specific error types
pub const MLError = error{
    EmptyDataset,
    InvalidData,
    InvalidUsage,
    FileReadError,
    FileWriteError,
    ModelNotInitialized,
    InvalidParameters,
    TrainingFailed,
    InsufficientData,
    ConvergenceFailed,
    InvalidModelState,
};

/// Represents a single data point with two features and a label
pub const DataRow = struct {
    /// First feature value
    x1: f64,
    /// Second feature value
    x2: f64,
    /// Target label/output value
    label: f64,

    /// Validates that all values in the data row are finite numbers
    pub fn validate(self: DataRow) MLError!void {
        if (std.math.isNan(self.x1) or std.math.isNan(self.x2) or std.math.isNan(self.label)) {
            return MLError.InvalidData;
        }
        if (std.math.isInf(self.x1) or std.math.isInf(self.x2) or std.math.isInf(self.label)) {
            return MLError.InvalidData;
        }
    }

    /// Creates a DataRow from an array of values
    /// Expects exactly 3 values: [x1, x2, label]
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

    /// Converts the DataRow to an array representation
    pub fn toArray(self: DataRow) [3]f64 {
        return [3]f64{ self.x1, self.x2, self.label };
    }

    /// Creates a copy of the DataRow with normalized features
    pub fn normalize(self: DataRow, x1_min: f64, x1_max: f64, x2_min: f64, x2_max: f64) DataRow {
        const x1_range = x1_max - x1_min;
        const x2_range = x2_max - x2_min;
        return DataRow{
            .x1 = if (x1_range > 0) (self.x1 - x1_min) / x1_range else 0,
            .x2 = if (x2_range > 0) (self.x2 - x2_min) / x2_range else 0,
            .label = self.label,
        };
    }

    /// Calculates the Euclidean distance between two data points
    pub fn distance(self: DataRow, other: DataRow) f64 {
        const dx = self.x1 - other.x1;
        const dy = self.x2 - other.x2;
        return std.math.sqrt(dx * dx + dy * dy);
    }
};

/// A simple linear/logistic regression model
pub const Model = struct {
    /// Model weights for the two features
    weights: [2]f64,
    /// Model bias term
    bias: f64,
    /// Whether the model has been trained
    is_trained: bool,
    /// Training metadata
    training_loss: f64,
    training_epochs: usize,

    /// Creates a new untrained model with zero-initialized parameters
    pub fn init() Model {
        return Model{
            .weights = .{ 0, 0 },
            .bias = 0,
            .is_trained = false,
            .training_loss = 0,
            .training_epochs = 0,
        };
    }

    /// Creates a model with pre-initialized parameters
    pub fn initWithParams(w1: f64, w2: f64, bias: f64) Model {
        return Model{
            .weights = .{ w1, w2 },
            .bias = bias,
            .is_trained = true,
            .training_loss = 0,
            .training_epochs = 0,
        };
    }

    /// Makes a prediction for a given input
    /// Returns the raw linear combination for regression
    pub fn predict(self: Model, row: DataRow) MLError!f64 {
        if (!self.is_trained) return MLError.ModelNotInitialized;
        try row.validate();
        return row.x1 * self.weights[0] + row.x2 * self.weights[1] + self.bias;
    }

    /// Makes a classification prediction using logistic function
    /// Returns a probability between 0 and 1
    pub fn predictProba(self: Model, row: DataRow) MLError!f64 {
        const linear_output = try self.predict(row);
        return 1.0 / (1.0 + std.math.exp(-linear_output));
    }

    /// Trains the model using gradient descent
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
        var prev_loss: f64 = std.math.inf(f64);

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

            // Calculate mean squared error
            const mse = total_err / @as(f64, @floatFromInt(data.len));

            // Early stopping if error is small enough or not improving
            if (mse < 0.0001 or @abs(mse - prev_loss) < 1e-8) break;
            prev_loss = mse;
        }

        self.is_trained = true;
        self.training_loss = prev_loss;
        self.training_epochs = epoch;
    }

    /// Evaluates the model on test data and returns mean squared error
    pub fn evaluate(self: Model, test_data: []const DataRow) MLError!f64 {
        if (!self.is_trained) return MLError.ModelNotInitialized;
        if (test_data.len == 0) return MLError.EmptyDataset;

        var total_error: f64 = 0;
        for (test_data) |row| {
            const prediction = try self.predict(row);
            const diff = prediction - row.label;
            total_error += diff * diff;
        }
        return total_error / @as(f64, @floatFromInt(test_data.len));
    }

    /// Calculates classification accuracy on binary classification data
    pub fn accuracy(self: Model, test_data: []const DataRow, threshold: f64) MLError!f64 {
        if (!self.is_trained) return MLError.ModelNotInitialized;
        if (test_data.len == 0) return MLError.EmptyDataset;

        var correct: usize = 0;
        for (test_data) |row| {
            const prob = try self.predictProba(row);
            const predicted_class: f64 = if (prob >= threshold) 1.0 else 0.0;
            if (predicted_class == row.label) {
                correct += 1;
            }
        }
        return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(test_data.len));
    }

    /// Resets the model to untrained state
    pub fn reset(self: *Model) void {
        self.weights = .{ 0, 0 };
        self.bias = 0;
        self.is_trained = false;
        self.training_loss = 0;
        self.training_epochs = 0;
    }

    /// Serializes the model to JSON format for persistence
    pub fn toJson(self: Model, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        try std.json.stringify(.{
            .weights = self.weights,
            .bias = self.bias,
            .is_trained = self.is_trained,
            .training_loss = self.training_loss,
            .training_epochs = self.training_epochs,
        }, .{}, buffer.writer());
        return buffer.toOwnedSlice();
    }

    /// Deserializes a model from JSON format
    pub fn fromJson(allocator: Allocator, json_data: []const u8) !Model {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
        defer parsed.deinit();

        const obj = parsed.value.object;
        const weights_array = obj.get("weights").?.array;

        return Model{
            .weights = .{
                weights_array.items[0].float,
                weights_array.items[1].float,
            },
            .bias = obj.get("bias").?.float,
            .is_trained = obj.get("is_trained").?.bool,
            .training_loss = obj.get("training_loss").?.float,
            .training_epochs = @intFromFloat(obj.get("training_epochs").?.float),
        };
    }
};

/// Data preprocessing utilities
pub const DataProcessor = struct {
    /// Normalizes a dataset by scaling features to [0, 1] range
    pub fn normalizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
        if (data.len == 0) return MLError.EmptyDataset;

        // Find min/max values
        var x1_min = data[0].x1;
        var x1_max = data[0].x1;
        var x2_min = data[0].x2;
        var x2_max = data[0].x2;

        for (data) |row| {
            x1_min = @min(x1_min, row.x1);
            x1_max = @max(x1_max, row.x1);
            x2_min = @min(x2_min, row.x2);
            x2_max = @max(x2_max, row.x2);
        }

        // Normalize data
        var normalized = try allocator.alloc(DataRow, data.len);
        for (data, 0..) |row, i| {
            normalized[i] = row.normalize(x1_min, x1_max, x2_min, x2_max);
        }

        return normalized;
    }

    /// Splits dataset into training and testing sets
    pub fn trainTestSplit(allocator: Allocator, data: []const DataRow, train_ratio: f64, random_seed: u64) !struct { train: []DataRow, @"test": []DataRow } {
        if (data.len == 0) return MLError.EmptyDataset;
        if (train_ratio <= 0 or train_ratio >= 1) return MLError.InvalidParameters;

        var prng = std.Random.DefaultPrng.init(random_seed);
        const random = prng.random();
        // Create shuffled indices
        const indices = try allocator.alloc(usize, data.len);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| {
            idx.* = i;
        }
        random.shuffle(usize, indices);

        const train_size = @as(usize, @intFromFloat(@as(f64, @floatFromInt(data.len)) * train_ratio));
        const test_size = data.len - train_size;

        var train_data = try allocator.alloc(DataRow, train_size);
        var test_data = try allocator.alloc(DataRow, test_size);

        for (0..train_size) |i| {
            train_data[i] = data[indices[i]];
        }
        for (0..test_size) |i| {
            test_data[i] = data[indices[train_size + i]];
        }

        return .{ .train = train_data, .@"test" = test_data };
    }

    /// Standardizes a dataset using z-score normalization (mean=0, std=1)
    pub fn standardizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
        if (data.len == 0) return MLError.EmptyDataset;

        // Calculate means
        var x1_sum: f64 = 0;
        var x2_sum: f64 = 0;
        for (data) |row| {
            x1_sum += row.x1;
            x2_sum += row.x2;
        }
        const x1_mean = x1_sum / @as(f64, @floatFromInt(data.len));
        const x2_mean = x2_sum / @as(f64, @floatFromInt(data.len));

        // Calculate standard deviations
        var x1_var_sum: f64 = 0;
        var x2_var_sum: f64 = 0;
        for (data) |row| {
            const x1_diff = row.x1 - x1_mean;
            const x2_diff = row.x2 - x2_mean;
            x1_var_sum += x1_diff * x1_diff;
            x2_var_sum += x2_diff * x2_diff;
        }
        const x1_std = std.math.sqrt(x1_var_sum / @as(f64, @floatFromInt(data.len)));
        const x2_std = std.math.sqrt(x2_var_sum / @as(f64, @floatFromInt(data.len)));

        // Standardize data
        var standardized = try allocator.alloc(DataRow, data.len);
        for (data, 0..) |row, i| {
            standardized[i] = DataRow{
                .x1 = if (x1_std > 0) (row.x1 - x1_mean) / x1_std else 0,
                .x2 = if (x2_std > 0) (row.x2 - x2_mean) / x2_std else 0,
                .label = row.label,
            };
        }

        return standardized;
    }
};

/// Cross-validation utilities
pub const CrossValidator = struct {
    /// Performs k-fold cross-validation on a model
    pub fn kFoldValidation(allocator: Allocator, data: []const DataRow, k: usize, learning_rate: f64, epochs: usize) !struct { mean_accuracy: f64, std_accuracy: f64 } {
        if (data.len == 0) return MLError.EmptyDataset;
        if (k == 0 or k > data.len) return MLError.InvalidParameters;

        const fold_size = data.len / k;
        var accuracies = try allocator.alloc(f64, k);
        defer allocator.free(accuracies);

        for (0..k) |fold| {
            const test_start = fold * fold_size;
            const test_end = if (fold == k - 1) data.len else (fold + 1) * fold_size;

            // Create training set (everything except current fold)
            var train_data = try allocator.alloc(DataRow, data.len - (test_end - test_start));
            defer allocator.free(train_data);

            var train_idx: usize = 0;
            for (data, 0..) |row, i| {
                if (i < test_start or i >= test_end) {
                    train_data[train_idx] = row;
                    train_idx += 1;
                }
            }

            // Train model on fold
            var model = Model.init();
            try model.train(train_data, learning_rate, epochs);

            // Test on validation fold
            const test_data = data[test_start..test_end];
            accuracies[fold] = try model.accuracy(test_data, 0.5);
        }

        // Calculate mean and standard deviation
        var mean: f64 = 0;
        for (accuracies) |acc| {
            mean += acc;
        }
        mean /= @as(f64, @floatFromInt(k));

        var variance: f64 = 0;
        for (accuracies) |acc| {
            const diff = acc - mean;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(k));
        const std_dev = std.math.sqrt(variance);

        return .{ .mean_accuracy = mean, .std_accuracy = std_dev };
    }
};

/// K-Nearest Neighbors classifier for non-parametric classification
pub const KNNClassifier = struct {
    training_data: []const DataRow,
    k: usize,

    pub fn init(training_data: []const DataRow, k: usize) MLError!KNNClassifier {
        if (training_data.len == 0) return MLError.EmptyDataset;
        if (k == 0 or k > training_data.len) return MLError.InvalidParameters;

        return KNNClassifier{
            .training_data = training_data,
            .k = k,
        };
    }

    pub fn predict(self: KNNClassifier, allocator: Allocator, query_point: DataRow) !f64 {
        // Calculate distances to all training points
        var distances = try allocator.alloc(struct { distance: f64, label: f64 }, self.training_data.len);
        defer allocator.free(distances);

        for (self.training_data, 0..) |train_point, i| {
            distances[i] = .{
                .distance = query_point.distance(train_point),
                .label = train_point.label,
            };
        }

        // Sort by distance
        std.sort.heap(struct { distance: f64, label: f64 }, distances, {}, struct {
            fn lessThan(_: void, a: struct { distance: f64, label: f64 }, b: struct { distance: f64, label: f64 }) bool {
                return a.distance < b.distance;
            }
        }.lessThan);

        // Majority vote among k nearest neighbors
        var class_0_count: usize = 0;
        var class_1_count: usize = 0;

        for (0..self.k) |i| {
            if (distances[i].label == 0.0) {
                class_0_count += 1;
            } else {
                class_1_count += 1;
            }
        }

        return if (class_1_count > class_0_count) 1.0 else 0.0;
    }
};

/// Reads a dataset from a CSV file
/// Expected format: x1,x2,label (one row per line)
pub fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]DataRow {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => return MLError.FileReadError,
        else => return err,
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);
    _ = try file.readAll(contents);

    var rows = try std.ArrayList(DataRow).initCapacity(allocator, 0);
    defer rows.deinit(allocator);
    var lines = std.mem.splitScalar(u8, contents, '\n');

    while (lines.next()) |line| {
        const trimmed_line = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed_line.len == 0) continue;

        var parts = std.mem.splitScalar(u8, trimmed_line, ',');
        const x1_str = parts.next() orelse continue;
        const x2_str = parts.next() orelse continue;
        const label_str = parts.next() orelse continue;

        const x1 = std.fmt.parseFloat(f64, std.mem.trim(u8, x1_str, " \t")) catch continue;
        const x2 = std.fmt.parseFloat(f64, std.mem.trim(u8, x2_str, " \t")) catch continue;
        const label = std.fmt.parseFloat(f64, std.mem.trim(u8, label_str, " \t")) catch continue;

        const row = DataRow{ .x1 = x1, .x2 = x2, .label = label };
        try row.validate();
        try rows.append(allocator, row);
    }

    return rows.toOwnedSlice(allocator);
}

/// Saves a dataset to a CSV file
pub fn saveDataset(path: []const u8, data: []const DataRow) !void {
    const file = std.fs.cwd().createFile(path, .{}) catch |err| switch (err) {
        else => return MLError.FileWriteError,
    };
    defer file.close();

    var writer = file.writer();
    for (data) |row| {
        try writer.print("{d},{d},{d}\n", .{ row.x1, row.x2, row.label });
    }
}

/// Saves a trained model to file in a simple text format
pub fn saveModel(path: []const u8, model: Model) !void {
    const file = std.fs.cwd().createFile(path, .{}) catch |err| switch (err) {
        else => return MLError.FileWriteError,
    };
    defer file.close();

    var writer = file.writer();
    try writer.print("{d} {d} {d} {} {d} {d}\n", .{ model.weights[0], model.weights[1], model.bias, model.is_trained, model.training_loss, model.training_epochs });
}

/// Loads a trained model from file
pub fn loadModel(path: []const u8) !Model {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => return MLError.FileReadError,
        else => return err,
    };
    defer file.close();

    var buf: [256]u8 = undefined;
    const line = (try file.reader().readUntilDelimiterOrEof(&buf, '\n')) orelse return MLError.InvalidData;
    var parts = std.mem.splitScalar(u8, line, ' ');

    const w0 = std.fmt.parseFloat(f64, parts.next() orelse return MLError.InvalidData) catch return MLError.InvalidData;
    const w1 = std.fmt.parseFloat(f64, parts.next() orelse return MLError.InvalidData) catch return MLError.InvalidData;
    const bias = std.fmt.parseFloat(f64, parts.next() orelse return MLError.InvalidData) catch return MLError.InvalidData;
    const is_trained = std.mem.eql(u8, parts.next() orelse return MLError.InvalidData, "true");
    const training_loss = std.fmt.parseFloat(f64, parts.next() orelse return MLError.InvalidData) catch return MLError.InvalidData;
    const training_epochs = std.fmt.parseInt(usize, parts.next() orelse return MLError.InvalidData, 10) catch return MLError.InvalidData;

    return Model{
        .weights = .{ w0, w1 },
        .bias = bias,
        .is_trained = is_trained,
        .training_loss = training_loss,
        .training_epochs = training_epochs,
    };
}

// Note: This module provides machine learning functionality but is not a standalone executable.
// Use the main CLI application for command-line usage of ML features.

test "DataRow validation" {
    // Valid data
    const valid_row = DataRow{ .x1 = 1.0, .x2 = 2.0, .label = 0.0 };
    try valid_row.validate();

    // Test array constructor
    const array_row = try DataRow.fromArray(&[_]f64{ 1.0, 2.0, 0.0 });
    try std.testing.expectEqual(valid_row.x1, array_row.x1);
    try std.testing.expectEqual(valid_row.x2, array_row.x2);
    try std.testing.expectEqual(valid_row.label, array_row.label);

    // Test toArray
    const array_result = valid_row.toArray();
    try std.testing.expectEqual([3]f64{ 1.0, 2.0, 0.0 }, array_result);

    // Test distance calculation
    const other_row = DataRow{ .x1 = 4.0, .x2 = 6.0, .label = 1.0 };
    const dist = valid_row.distance(other_row);
    try std.testing.expectApproxEqRel(5.0, dist, 0.001);

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

    // Test probability predictions
    const prob1 = try model.predictProba(training_data[0]);
    try std.testing.expect(prob1 >= 0.0 and prob1 <= 1.0);

    // Test evaluation
    const mse = try model.evaluate(&training_data);
    try std.testing.expect(mse >= 0.0);

    // Test accuracy
    const acc = try model.accuracy(&training_data, 0.5);
    try std.testing.expect(acc >= 0.0 and acc <= 1.0);
}

test "Model serialization" {
    const allocator = std.testing.allocator;

    var model = Model.initWithParams(0.5, -0.3, 0.1);
    model.training_loss = 0.123;
    model.training_epochs = 100;

    // Test JSON serialization
    const json_data = try model.toJson(allocator);
    defer allocator.free(json_data);

    const loaded_model = try Model.fromJson(allocator, json_data);

    try std.testing.expectApproxEqRel(model.weights[0], loaded_model.weights[0], 0.001);
    try std.testing.expectApproxEqRel(model.weights[1], loaded_model.weights[1], 0.001);
    try std.testing.expectApproxEqRel(model.bias, loaded_model.bias, 0.001);
    try std.testing.expectEqual(model.is_trained, loaded_model.is_trained);
    try std.testing.expectEqual(model.training_epochs, loaded_model.training_epochs);
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

test "Data preprocessing" {
    const allocator = std.testing.allocator;

    const data = [_]DataRow{
        .{ .x1 = 0.0, .x2 = 0.0, .label = 0.0 },
        .{ .x1 = 10.0, .x2 = 20.0, .label = 1.0 },
        .{ .x1 = 5.0, .x2 = 10.0, .label = 0.5 },
    };

    // Test normalization
    const normalized = try DataProcessor.normalizeDataset(allocator, &data);
    defer allocator.free(normalized);

    try std.testing.expectApproxEqRel(0.0, normalized[0].x1, 0.001);
    try std.testing.expectApproxEqRel(1.0, normalized[1].x1, 0.001);
    try std.testing.expectApproxEqRel(0.5, normalized[2].x1, 0.001);

    // Test standardization
    const standardized = try DataProcessor.standardizeDataset(allocator, &data);
    defer allocator.free(standardized);

    // Check that standardized data has approximately zero mean
    var mean_x1: f64 = 0;
    for (standardized) |row| {
        mean_x1 += row.x1;
    }
    mean_x1 /= @as(f64, @floatFromInt(standardized.len));
    try std.testing.expectApproxEqRel(0.0, mean_x1, 0.001);

    // Test train-test split
    const split = try DataProcessor.trainTestSplit(allocator, &data, 0.67, 42);
    defer allocator.free(split.train);
    defer allocator.free(split.@"test");

    try std.testing.expect(split.train.len >= 1);
    try std.testing.expect(split.@"test".len >= 1);
    try std.testing.expectEqual(data.len, split.train.len + split.@"test".len);
}

test "KNN Classifier" {
    const allocator = std.testing.allocator;

    const training_data = [_]DataRow{
        .{ .x1 = 0.0, .x2 = 0.0, .label = 0.0 },
        .{ .x1 = 1.0, .x2 = 1.0, .label = 1.0 },
        .{ .x1 = 0.1, .x2 = 0.1, .label = 0.0 },
        .{ .x1 = 0.9, .x2 = 0.9, .label = 1.0 },
    };

    const knn = try KNNClassifier.init(&training_data, 3);

    // Test prediction
    const query_point = DataRow{ .x1 = 0.05, .x2 = 0.05, .label = 0.0 };
    const prediction = try knn.predict(allocator, query_point);

    try std.testing.expectEqual(@as(f64, 0.0), prediction);
}
