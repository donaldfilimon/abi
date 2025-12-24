//! Neural inference workload
//!
//! Work item for executing neural network inference.

const std = @import("std");

pub const NeuralInference = struct {
    weights: std.ArrayList(f32),
    biases: std.ArrayList(f32),
    activation: Activation,

    pub const Activation = enum {
        relu,
        sigmoid,
        tanh,
        softmax,
    };

    pub fn init(allocator: std.mem.Allocator, weights_count: usize, activation: Activation) !NeuralInference {
        _ = weights_count;
        const weights = try std.ArrayList(f32).initCapacity(allocator, 32);
        const biases = try std.ArrayList(f32).initCapacity(allocator, 32);

        return .{
            .weights = weights,
            .biases = biases,
            .activation = activation,
        };
    }

    pub fn deinit(self: *NeuralInference) void {
        self.weights.deinit();
        self.biases.deinit();
        self.* = undefined;
    }
};
