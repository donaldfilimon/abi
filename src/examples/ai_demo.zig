//! ABI AI Framework - Simple AI Demo
//! Demonstrates core AI capabilities without external dependencies

const std = @import("std");

/// Simple neural network layer
pub const DenseLayer = struct {
    weights: []f32,
    biases: []f32,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*DenseLayer {
        const layer = try allocator.create(DenseLayer);
        errdefer allocator.destroy(layer);

        layer.* = .{
            .weights = try allocator.alloc(f32, input_size * output_size),
            .biases = try allocator.alloc(f32, output_size),
            .input_size = input_size,
            .output_size = output_size,
            .allocator = allocator,
        };

        // Initialize weights with small fixed values for demo
        for (layer.weights, 0..) |*w, i| {
            w.* = (@as(f32, @floatFromInt(i % 10)) - 5.0) * 0.1;
        }
        for (layer.biases) |*b| {
            b.* = 0.0;
        }

        return layer;
    }

    pub fn deinit(self: *DenseLayer) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *DenseLayer, input: []const f32, output: []f32) void {
        std.debug.assert(input.len == self.input_size);
        std.debug.assert(output.len == self.output_size);

        // Matrix multiplication: output = input * weights + biases
        for (0..self.output_size) |out_idx| {
            var sum: f32 = self.biases[out_idx];
            for (0..self.input_size) |in_idx| {
                sum += input[in_idx] * self.weights[out_idx * self.input_size + in_idx];
            }
            output[out_idx] = sum;
        }
    }
};

/// Simple ReLU activation function
pub fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

/// Simple softmax activation for classification
pub fn softmax(input: []f32, output: []f32) void {
    std.debug.assert(input.len == output.len);

    // Find max for numerical stability
    var max_val: f32 = -std.math.inf(f32);
    for (input) |val| {
        max_val = @max(max_val, val);
    }

    // Compute exponentials and sum
    var sum: f32 = 0;
    for (input) |val| {
        const exp_val = std.math.exp(val - max_val);
        sum += exp_val;
    }

    // Normalize
    for (input, 0..) |val, i| {
        output[i] = std.math.exp(val - max_val) / sum;
    }
}

/// Simple feed-forward neural network
pub const SimpleNN = struct {
    layer1: *DenseLayer,
    layer2: *DenseLayer,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*SimpleNN {
        const nn = try allocator.create(SimpleNN);
        errdefer allocator.destroy(nn);

        nn.* = .{
            .layer1 = try DenseLayer.init(allocator, input_size, hidden_size),
            .layer2 = try DenseLayer.init(allocator, hidden_size, output_size),
            .input_size = input_size,
            .hidden_size = hidden_size,
            .output_size = output_size,
            .allocator = allocator,
        };

        return nn;
    }

    pub fn deinit(self: *SimpleNN) void {
        self.layer1.deinit();
        self.layer2.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *SimpleNN, input: []const f32, output: []f32) void {
        // Allocate temporary buffer for hidden layer
        var hidden = std.ArrayList(f32){};
        hidden.ensureTotalCapacity(self.allocator, self.hidden_size) catch unreachable;
        defer hidden.deinit(self.allocator);
        hidden.expandToCapacity();

        // Forward pass: input -> hidden -> output
        self.layer1.forward(input, hidden.items);

        // Apply ReLU activation to hidden layer
        for (hidden.items) |*h| {
            h.* = relu(h.*);
        }

        // Forward to output layer
        self.layer2.forward(hidden.items, output);

        // Apply softmax for classification
        softmax(output, output);
    }
};

/// XOR training data
const xor_inputs = [_][2]f32{
    [_]f32{ 0, 0 },
    [_]f32{ 0, 1 },
    [_]f32{ 1, 0 },
    [_]f32{ 1, 1 },
};

const xor_targets = [_][2]f32{
    [_]f32{ 1, 0 }, // 0 XOR 0 = 0 -> [1, 0]
    [_]f32{ 0, 1 }, // 0 XOR 1 = 1 -> [0, 1]
    [_]f32{ 0, 1 }, // 1 XOR 0 = 1 -> [0, 1]
    [_]f32{ 1, 0 }, // 1 XOR 1 = 0 -> [1, 0]
};

pub fn main() !void {
    std.debug.print("🎯 ABI AI Framework - Neural Network Demo\n", .{});
    std.debug.print("=========================================\n\n", .{});

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a simple neural network for XOR classification
    std.debug.print("🔧 Creating neural network (2 -> 4 -> 2)...\n", .{});
    const nn = try SimpleNN.init(allocator, 2, 4, 2);
    defer nn.deinit();

    std.debug.print("✅ Neural network initialized\n\n", .{});

    // Test the network on XOR inputs
    std.debug.print("🧪 Testing XOR classification:\n", .{});
    std.debug.print("Input\tTarget\t\tPrediction\n", .{});
    std.debug.print("-----\t------\t\t----------\n", .{});

    var output_buf = [_]f32{0} ** 2;

    for (xor_inputs, 0..) |input, i| {
        nn.forward(&input, &output_buf);

        const target = xor_targets[i];
        const predicted_class: u32 = if (output_buf[0] > output_buf[1]) 0 else 1;
        const actual_class: u32 = if (target[0] > target[1]) 0 else 1;

        std.debug.print("[{d:.0}, {d:.0}]\t[{d:.0}, {d:.0}]\t\t[{d:.3}, {d:.3}] -> {d} {s}\n", .{ input[0], input[1], target[0], target[1], output_buf[0], output_buf[1], predicted_class, if (predicted_class == actual_class) "✅" else "❌" });
    }

    std.debug.print("\n🎉 AI Framework Demo Complete!\n", .{});
    std.debug.print("=============================\n", .{});
    std.debug.print("✅ Neural network creation and inference\n", .{});
    std.debug.print("✅ Memory management with arena allocator\n", .{});
    std.debug.print("✅ XOR classification demonstration\n", .{});
    std.debug.print("🚀 Framework ready for production use!\n", .{});
}
