//! Trainable point neural network — a real (if tiny) MLP with genuine
//! backprop/SGD, topology optimization, and weight pruning.
//!
//! Design reference: docs/spec/wdbx-rust-capability-extract.mdx §4.
//! Pure Zig implementation — no Rust dependency, no external ML framework.
//! The network is a fully-connected MLP over a 3-D point derived from text
//! features, trained with stochastic gradient descent on MSE loss.
//!
//! Honest scope: demo-grade tiny neural net (default [3,8,1]), not a
//! production LLM or distributed trainer.

const std = @import("std");

pub const ActivationFunction = enum {
    relu,
    tanh,
    sigmoid,
    linear,

    pub fn apply(self: ActivationFunction, x: f32) f32 {
        return switch (self) {
            .relu => if (x > 0) x else 0,
            .tanh => std.math.tanh(x),
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .linear => x,
        };
    }

    pub fn derivative(self: ActivationFunction, x: f32) f32 {
        return switch (self) {
            .relu => if (x > 0) 1.0 else 0.0,
            .tanh => {
                const t = std.math.tanh(x);
                return 1.0 - t * t;
            },
            .sigmoid => {
                const s = 1.0 / (1.0 + @exp(-x));
                return s * (1.0 - s);
            },
            .linear => 1.0,
        };
    }
};

pub const Point = struct {
    x: f32,
    y: f32,
    z: f32,
    value: f32 = 1.0,

    pub fn fromText(text: []const u8) Point {
        const len: f32 = @floatFromInt(text.len);
        var vowels: f32 = 0;
        var punct: f32 = 0;
        for (text) |c| {
            switch (c) {
                'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' => vowels += 1,
                '.', ',', '!', '?', ';', ':', '-', '"', '\'', '(', ')' => punct += 1,
                else => {},
            }
        }
        return .{
            .x = len / 32.0,
            .y = vowels / (len + 1.0),
            .z = punct / (len + 1.0),
            .value = 1.0,
        };
    }

    pub fn toArray(self: Point) [3]f32 {
        return .{ self.x, self.y, self.z };
    }
};

pub const DenseLayer = struct {
    weights: [][]f32,
    bias: []f32,
    activation: ActivationFunction,
    input_size: usize,
    output_size: usize,
};

pub const ForwardTrace = struct {
    activations: [][]f32,
    pre_activations: [][]f32,

    pub fn deinit(self: *ForwardTrace, allocator: std.mem.Allocator) void {
        for (self.activations) |a| allocator.free(a);
        for (self.pre_activations) |a| allocator.free(a);
        allocator.free(self.activations);
        allocator.free(self.pre_activations);
    }
};

pub const TopologyOptimizationReport = struct {
    centroid: [3]f32,
    spread: f32,
    density_gain: f32,
    bias_shift: f32,
    pruned_count: usize,
    regularization_factor: f32,
};

pub const PointNeuralNetwork = struct {
    allocator: std.mem.Allocator,
    layers: []DenseLayer,
    dims: []const usize,
    learning_rate: f32,
    default_activation: ActivationFunction,

    pub fn init(allocator: std.mem.Allocator, dims: []const usize, lr: f32) !PointNeuralNetwork {
        if (dims.len < 2) return error.InvalidDimensions;
        for (dims) |d| if (d == 0) return error.InvalidDimensions;

        const layers = try allocator.alloc(DenseLayer, dims.len - 1);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            const in_size = dims[i];
            const out_size = dims[i + 1];

            layer.input_size = in_size;
            layer.output_size = out_size;
            layer.activation = if (i == dims.len - 2) .linear else .relu;

            layer.weights = try allocator.alloc([]f32, out_size);
            layer.bias = try allocator.alloc(f32, out_size);

            const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_size + out_size)));
            var prng = std.Random.DefaultPrng.init(@intCast(i + 1));
            const rng = prng.random();
            for (layer.weights) |*w_row| {
                w_row.* = try allocator.alloc(f32, in_size);
                for (w_row.*) |*w| w.* = rng.floatNorm(f32) * scale;
            }
            for (layer.bias) |*b| b.* = 0;
        }

        return .{
            .allocator = allocator,
            .layers = layers,
            .dims = dims,
            .learning_rate = lr,
            .default_activation = .relu,
        };
    }

    pub fn deinit(self: *PointNeuralNetwork) void {
        for (self.layers) |*layer| {
            for (layer.weights) |w| self.allocator.free(w);
            self.allocator.free(layer.weights);
            self.allocator.free(layer.bias);
        }
        self.allocator.free(self.layers);
    }

    pub fn forwardWithTrace(self: *PointNeuralNetwork, input: []const f32) !ForwardTrace {
        const activations = try self.allocator.alloc([]f32, self.layers.len + 1);
        const pre_activations = try self.allocator.alloc([]f32, self.layers.len);

        activations[0] = try self.allocator.dupe(f32, input);

        for (self.layers, 0..) |layer, i| {
            pre_activations[i] = try self.allocator.alloc(f32, layer.output_size);
            activations[i + 1] = try self.allocator.alloc(f32, layer.output_size);

            const is_output = i == self.layers.len - 1;
            const act_fn: ActivationFunction = if (is_output) .linear else layer.activation;

            for (0..layer.output_size) |j| {
                var sum: f32 = layer.bias[j];
                for (layer.weights[j], 0..) |w, k| {
                    sum += w * activations[i][k];
                }
                pre_activations[i][j] = sum;
                activations[i + 1][j] = act_fn.apply(sum);
            }
        }

        return .{ .activations = activations, .pre_activations = pre_activations };
    }

    pub fn forward(self: *PointNeuralNetwork, input: []const f32) ![]f32 {
        var trace = try self.forwardWithTrace(input);
        defer trace.deinit(self.allocator);
        return try self.allocator.dupe(f32, trace.activations[self.layers.len]);
    }

    pub fn backtraceTeach(self: *PointNeuralNetwork, point: Point, target: []const f32) !f32 {
        const input = point.toArray();
        if (target.len != self.layers[self.layers.len - 1].output_size) return error.DimensionMismatch;

        var trace = try self.forwardWithTrace(&input);
        defer trace.deinit(self.allocator);

        const output = trace.activations[self.layers.len];
        var loss: f32 = 0;
        for (output, target) |o, t| {
            const err = o - t;
            loss += 0.5 * err * err;
        }

        const num_layers = self.layers.len;
        var deltas = try self.allocator.alloc([]f32, num_layers);
        defer {
            for (deltas) |d| self.allocator.free(d);
            self.allocator.free(deltas);
        }

        const output_size = self.layers[num_layers - 1].output_size;
        deltas[num_layers - 1] = try self.allocator.alloc(f32, output_size);
        for (output, target, 0..) |o, t, j| {
            deltas[num_layers - 1][j] = o - t;
        }

        var li: usize = num_layers - 1;
        while (li > 0) : (li -= 1) {
            const layer = self.layers[li];
            const prev_layer = self.layers[li - 1];
            const prev_size = prev_layer.output_size;
            deltas[li - 1] = try self.allocator.alloc(f32, prev_size);

            for (0..prev_size) |k| {
                var sum: f32 = 0;
                for (0..layer.output_size) |j| {
                    sum += layer.weights[j][k] * deltas[li][j];
                }
                const pre_act = trace.pre_activations[li - 1][k];
                const deriv = prev_layer.activation.derivative(pre_act);
                deltas[li - 1][k] = sum * deriv;
            }
        }

        for (self.layers, 0..) |*layer, i| {
            const act_input = trace.activations[i];
            for (0..layer.output_size) |j| {
                for (0..layer.input_size) |k| {
                    layer.weights[j][k] -= self.learning_rate * deltas[i][j] * act_input[k];
                }
                layer.bias[j] -= self.learning_rate * deltas[i][j];
            }
        }

        return loss;
    }

    pub fn train(self: *PointNeuralNetwork, points: []const Point, targets: []const []const f32, epochs: usize) !f32 {
        if (points.len != targets.len) return error.DimensionMismatch;
        var last_loss: f32 = 0;
        for (0..epochs) |_| {
            for (points, targets) |p, t| {
                last_loss = try self.backtraceTeach(p, t);
            }
        }
        return last_loss;
    }

    pub fn optimizeTopology(self: *PointNeuralNetwork, points: []const Point) TopologyOptimizationReport {
        var cx: f32 = 0;
        var cy: f32 = 0;
        var cz: f32 = 0;
        var total_value: f32 = 0;
        for (points) |p| {
            cx += p.x * p.value;
            cy += p.y * p.value;
            cz += p.z * p.value;
            total_value += p.value;
        }
        if (total_value > 0) {
            cx /= total_value;
            cy /= total_value;
            cz /= total_value;
        }
        const centroid = [3]f32{ cx, cy, cz };

        var spread: f32 = 0;
        for (points) |p| {
            const dx = p.x - cx;
            const dy = p.y - cy;
            const dz = p.z - cz;
            spread += @sqrt(dx * dx + dy * dy + dz * dz);
        }
        if (points.len > 0) spread /= @as(f32, @floatFromInt(points.len));

        const density_gain = 1.0 / (1.0 + spread);
        const regularization_factor: f32 = 0.95 + 0.05 * density_gain;
        const prune_threshold = 0.001 * (1.0 + density_gain);

        var pruned_count: usize = 0;
        for (self.layers) |*layer| {
            for (layer.weights) |w_row| {
                for (w_row) |*w| {
                    w.* *= regularization_factor;
                    if (@abs(w.*) < prune_threshold) {
                        w.* = 0;
                        pruned_count += 1;
                    }
                }
            }
        }

        var bias_shift: f32 = 0;
        if (self.layers.len > 0) {
            const first = &self.layers[0];
            if (first.bias.len >= 3) {
                const shift_x = cx * 0.1;
                const shift_y = cy * 0.1;
                const shift_z = cz * 0.1;
                first.bias[0] += shift_x;
                first.bias[1] += shift_y;
                first.bias[2] += shift_z;
                bias_shift = @abs(shift_x) + @abs(shift_y) + @abs(shift_z);
            }
        }

        return .{
            .centroid = centroid,
            .spread = spread,
            .density_gain = density_gain,
            .bias_shift = bias_shift,
            .pruned_count = pruned_count,
            .regularization_factor = regularization_factor,
        };
    }

    pub fn save(self: *PointNeuralNetwork) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        var jw = std.json.Stringify{ .writer = &out.writer, .options = .{} };
        try jw.beginObject();
        try jw.objectField("dims");
        try jw.beginArray();
        for (self.dims) |d| try jw.write(@as(u64, @intCast(d)));
        try jw.endArray();
        try jw.objectField("learning_rate");
        try jw.write(self.learning_rate);
        try jw.objectField("layers");
        try jw.beginArray();
        for (self.layers) |layer| {
            try jw.beginObject();
            try jw.objectField("weights");
            try jw.beginArray();
            for (layer.weights) |w_row| {
                try jw.beginArray();
                for (w_row) |w| try jw.write(w);
                try jw.endArray();
            }
            try jw.endArray();
            try jw.objectField("bias");
            try jw.beginArray();
            for (layer.bias) |b| try jw.write(b);
            try jw.endArray();
            try jw.endObject();
        }
        try jw.endArray();
        try jw.endObject();
        return try self.allocator.dupe(u8, out.written());
    }

    pub fn telemetry(self: *PointNeuralNetwork) NeuralTelemetry {
        var total_weights: usize = 0;
        var nonzero_weights: usize = 0;
        const layer_count = self.layers.len;
        var l1_norms = self.allocator.alloc(f32, layer_count) catch return .{
            .layer_count = layer_count,
            .total_weights = 0,
            .nonzero_weights = 0,
            .l1_norms = &.{},
        };

        for (self.layers, 0..) |layer, i| {
            var l1: f32 = 0;
            for (layer.weights) |w_row| {
                for (w_row) |w| {
                    total_weights += 1;
                    if (w != 0) nonzero_weights += 1;
                    l1 += @abs(w);
                }
            }
            l1_norms[i] = l1;
        }

        return .{
            .layer_count = layer_count,
            .total_weights = total_weights,
            .nonzero_weights = nonzero_weights,
            .l1_norms = l1_norms,
        };
    }
};

pub const NeuralTelemetry = struct {
    layer_count: usize,
    total_weights: usize,
    nonzero_weights: usize,
    l1_norms: []f32,

    pub fn deinit(self: NeuralTelemetry, allocator: std.mem.Allocator) void {
        if (self.l1_norms.len > 0) allocator.free(self.l1_norms);
    }
};

pub const SoulRecord = struct {
    label: []const u8,
    point: Point,
};

pub const SoulLayout = struct {
    records: []SoulRecord,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SoulLayout) void {
        for (self.records) |r| self.allocator.free(r.label);
        self.allocator.free(self.records);
    }

    pub fn fromJson(allocator: std.mem.Allocator, json_text: []const u8) !SoulLayout {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
        defer parsed.deinit();

        const arr = switch (parsed.value) {
            .array => |a| a,
            else => return error.InvalidSoulFormat,
        };
        if (arr.len == 0) return error.EmptySoulPrompt;

        const records = try allocator.alloc(SoulRecord, arr.len);
        errdefer allocator.free(records);

        for (arr.items, 0..) |item, i| {
            const obj = switch (item) {
                .object => |o| o,
                else => return error.InvalidSoulRecord,
            };
            const label_val = obj.get("label") orelse return error.MissingLabel;
            const label_str = switch (label_val) {
                .string => |s| s,
                else => return error.InvalidLabel,
            };
            records[i].label = try allocator.dupe(u8, label_str);

            const x_val = obj.get("x");
            const y_val = obj.get("y");
            const z_val = obj.get("z");
            const v_val = obj.get("value");

            if (x_val != null and y_val != null and z_val != null) {
                records[i].point = .{
                    .x = switch (x_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .y = switch (y_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .z = switch (z_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .value = if (v_val) |v| (switch (v) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 1.0,
                    }) else 1.0,
                };
            } else {
                records[i].point = Point.fromText(label_str);
            }
        }

        return .{ .records = records, .allocator = allocator };
    }

    pub fn bootstrap(self: *SoulLayout, net: *PointNeuralNetwork) !TopologyOptimizationReport {
        const targets = try self.allocator.alloc([]const f32, self.records.len);
        defer self.allocator.free(targets);
        for (self.records, 0..) |r, i| {
            const target = try self.allocator.alloc(f32, 1);
            target[0] = r.point.value;
            targets[i] = target;
        }
        defer for (targets) |t| self.allocator.free(t);

        const points = try self.allocator.alloc(Point, self.records.len);
        defer self.allocator.free(points);
        for (self.records, 0..) |r, i| points[i] = r.point;

        _ = try net.train(points, targets, 100);
        return net.optimizeTopology(points);
    }
};

test "Point.fromText is deterministic" {
    const p1 = Point.fromText("hello world");
    const p2 = Point.fromText("hello world");
    try std.testing.expectEqual(p1.x, p2.x);
    try std.testing.expectEqual(p1.y, p2.y);
    try std.testing.expectEqual(p1.z, p2.z);
    try std.testing.expectEqual(@as(f32, 1.0), p1.value);
}

test "Point.fromText extracts vowels and punctuation" {
    const p = Point.fromText("Hello, world!");
    try std.testing.expect(p.x > 0);
    try std.testing.expect(p.y > 0);
    try std.testing.expect(p.z > 0);
}

test "activation functions apply and derive correctly" {
    try std.testing.expectEqual(@as(f32, 0), ActivationFunction.relu.apply(-1));
    try std.testing.expectEqual(@as(f32, 5), ActivationFunction.relu.apply(5));
    try std.testing.expectEqual(@as(f32, 1), ActivationFunction.relu.derivative(5));
    try std.testing.expectEqual(@as(f32, 0), ActivationFunction.relu.derivative(-1));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ActivationFunction.sigmoid.apply(100), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ActivationFunction.sigmoid.apply(-100), 1e-3);
    try std.testing.expectEqual(@as(f32, 42), ActivationFunction.linear.apply(42));
    try std.testing.expectEqual(@as(f32, 1), ActivationFunction.linear.derivative(0));
}

test "PointNeuralNetwork init rejects invalid dims" {
    try std.testing.expectError(error.InvalidDimensions, PointNeuralNetwork.init(std.testing.allocator, &.{}, 0.01));
    try std.testing.expectError(error.InvalidDimensions, PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 0, 1 }, 0.01));
}

test "PointNeuralNetwork forward pass produces output" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 8, 1 }, 0.01);
    defer net.deinit();
    const input = [_]f32{ 0.5, 0.3, 0.2 };
    const output = try net.forward(&input);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 1), output.len);
}

test "backtraceTeach reduces loss over training" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 8, 1 }, 0.1);
    defer net.deinit();
    const point = Point.fromText("hello world");
    const target = [_]f32{0.9};
    const initial_loss = try net.backtraceTeach(point, &target);
    var last_loss = initial_loss;
    for (0..200) |_| {
        last_loss = try net.backtraceTeach(point, &target);
    }
    try std.testing.expect(last_loss < initial_loss);
}

test "train runs multiple epochs" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 4, 1 }, 0.1);
    defer net.deinit();
    const points = [_]Point{
        Point.fromText("hello"),
        Point.fromText("world"),
        Point.fromText("test"),
    };
    const targets = [_][]const f32{
        &.{0.9},
        &.{0.1},
        &.{0.5},
    };
    const loss = try net.train(&points, &targets, 50);
    try std.testing.expect(loss >= 0);
}

test "optimizeTopology prunes near-zero weights" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 8, 1 }, 0.01);
    defer net.deinit();
    const points = [_]Point{
        Point.fromText("hello"),
        Point.fromText("world"),
        Point.fromText("test"),
    };
    const report = net.optimizeTopology(&points);
    try std.testing.expect(report.pruned_count > 0);
    try std.testing.expect(report.regularization_factor >= 0.95);
    try std.testing.expect(report.regularization_factor <= 1.0);
    try std.testing.expect(report.spread >= 0);
    try std.testing.expect(report.density_gain > 0);
}

test "save produces valid JSON" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 4, 1 }, 0.01);
    defer net.deinit();
    const json = try net.save();
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.ExpectedObject,
    };
    try std.testing.expect(obj.get("dims") != null);
    try std.testing.expect(obj.get("layers") != null);
    try std.testing.expect(obj.get("learning_rate") != null);
}

test "telemetry reports weight statistics" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 4, 1 }, 0.01);
    defer net.deinit();
    const tel = net.telemetry();
    defer tel.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), tel.layer_count);
    try std.testing.expect(tel.total_weights > 0);
    try std.testing.expect(tel.nonzero_weights > 0);
    try std.testing.expectEqual(tel.l1_norms.len, 2);
}

test "SoulLayout fromJson parses labeled records" {
    const json =
        \\[
        \\  {"label": "helpful", "x": 0.5, "y": 0.3, "z": 0.2},
        \\  {"label": "honest"},
        \\  {"label": "safe", "value": 1.0}
        \\]
    ;
    var layout = try SoulLayout.fromJson(std.testing.allocator, json);
    defer layout.deinit();
    try std.testing.expectEqual(@as(usize, 3), layout.records.len);
    try std.testing.expectEqualStrings("helpful", layout.records[0].label);
    try std.testing.expectEqual(@as(f32, 0.5), layout.records[0].point.x);
    try std.testing.expect(layout.records[1].point.x > 0);
    try std.testing.expectEqual(@as(f32, 1.0), layout.records[2].point.value);
}

test "SoulLayout rejects empty and invalid input" {
    try std.testing.expectError(error.EmptySoulPrompt, SoulLayout.fromJson(std.testing.allocator, "[]"));
    try std.testing.expectError(error.InvalidSoulFormat, SoulLayout.fromJson(std.testing.allocator, "{}"));
}

test "SoulLayout bootstrap trains and optimizes topology" {
    const json =
        \\[
        \\  {"label": "helpful"},
        \\  {"label": "honest"},
        \\  {"label": "safe"},
        \\  {"label": "creative"},
        \\  {"label": "analytical"}
        \\]
    ;
    var layout = try SoulLayout.fromJson(std.testing.allocator, json);
    defer layout.deinit();
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 8, 1 }, 0.1);
    defer net.deinit();
    const report = try layout.bootstrap(&net);
    try std.testing.expect(report.spread >= 0);
    try std.testing.expect(report.density_gain > 0);
}

test "PointNeuralNetwork dimension mismatch on wrong target size" {
    var net = try PointNeuralNetwork.init(std.testing.allocator, &.{ 3, 8, 1 }, 0.01);
    defer net.deinit();
    const point = Point.fromText("test");
    const bad_target = [_]f32{ 0.5, 0.5 };
    try std.testing.expectError(error.DimensionMismatch, net.backtraceTeach(point, &bad_target));
}

test {
    std.testing.refAllDecls(@This());
}
