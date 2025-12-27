//! Workload execution primitives for the compute runtime.
//!
//! Defines the core types and functions for workload execution, including hints,
//! execution contexts, result handles, vtables, and common compute operations
//! like matrix multiplication, dense layers, and ReLU activation.

const std = @import("std");

pub const WorkloadHints = struct {
    cpu_affinity: ?u32 = null,
    estimated_duration_us: ?u64 = null,
    prefers_gpu: bool = false,
    requires_gpu: bool = false,
};

pub const ExecutionContext = struct {
    allocator: std.mem.Allocator,
    worker_index: u32 = 0,
    start_ns: u64 = 0,
};

pub const ResultHandle = struct {
    bytes: []const u8,
    owned: bool = false,
    allocator: ?std.mem.Allocator = null,

    pub fn fromSlice(bytes: []const u8) ResultHandle {
        return .{ .bytes = bytes, .owned = false, .allocator = null };
    }

    pub fn fromOwned(allocator: std.mem.Allocator, bytes: []u8) ResultHandle {
        return .{ .bytes = bytes, .owned = true, .allocator = allocator };
    }

    pub fn deinit(self: *ResultHandle) void {
        if (self.owned) {
            if (self.allocator) |allocator| {
                allocator.free(self.bytes);
            }
        }
        self.* = undefined;
    }
};

pub const WorkloadVTable = struct {
    execute: *const fn (ctx: *ExecutionContext, user: *anyopaque) anyerror!ResultHandle,
};

pub const ResultVTable = struct {
    release: *const fn (user: *anyopaque, allocator: std.mem.Allocator) void,
};

pub const GPUWorkloadVTable = struct {
    execute: *const fn (ctx: *ExecutionContext, user: *anyopaque) anyerror!ResultHandle,
};

pub const WorkItem = struct {
    id: u64,
    user: *anyopaque,
    vtable: *const WorkloadVTable,
    priority: i32 = 0,
    hints: WorkloadHints = .{},
    gpu_vtable: ?*const GPUWorkloadVTable = null,
};

pub fn runWorkItem(ctx: *ExecutionContext, item: *const WorkItem) !ResultHandle {
    return item.vtable.execute(ctx, item.user);
}

pub fn matMul(
    a: []const f32,
    b: []const f32,
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
    out: []f32,
) void {
    std.debug.assert(a.len >= rows_a * cols_a);
    std.debug.assert(b.len >= cols_a * cols_b);
    std.debug.assert(out.len >= rows_a * cols_b);

    var row: usize = 0;
    while (row < rows_a) : (row += 1) {
        var col: usize = 0;
        while (col < cols_b) : (col += 1) {
            var sum: f32 = 0;
            var k: usize = 0;
            while (k < cols_a) : (k += 1) {
                sum += a[row * cols_a + k] * b[k * cols_b + col];
            }
            out[row * cols_b + col] = sum;
        }
    }
}

pub fn dense(
    input: []const f32,
    weights: []const f32,
    biases: []const f32,
    in_features: usize,
    out_features: usize,
    out: []f32,
) void {
    std.debug.assert(input.len >= in_features);
    std.debug.assert(weights.len >= in_features * out_features);
    std.debug.assert(biases.len >= out_features);
    std.debug.assert(out.len >= out_features);

    var o: usize = 0;
    while (o < out_features) : (o += 1) {
        var sum: f32 = biases[o];
        var i: usize = 0;
        while (i < in_features) : (i += 1) {
            sum += weights[o * in_features + i] * input[i];
        }
        out[o] = sum;
    }
}

pub fn relu(values: []f32) void {
    for (values) |*value| {
        if (value.* < 0) value.* = 0;
    }
}

pub const MatrixMultiplyTask = struct {
    a: []const f32,
    b: []const f32,
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,

    pub fn execute(self: MatrixMultiplyTask, allocator: std.mem.Allocator) ![]f32 {
        const out = try allocator.alloc(f32, self.rows_a * self.cols_b);
        matMul(self.a, self.b, self.rows_a, self.cols_a, self.cols_b, out);
        return out;
    }
};

pub const MlpTask = struct {
    input: []const f32,
    hidden_weights: []const f32,
    hidden_biases: []const f32,
    output_weights: []const f32,
    output_biases: []const f32,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    pub fn execute(self: MlpTask, allocator: std.mem.Allocator) ![]f32 {
        const hidden = try allocator.alloc(f32, self.hidden_size);
        defer allocator.free(hidden);

        dense(
            self.input,
            self.hidden_weights,
            self.hidden_biases,
            self.input_size,
            self.hidden_size,
            hidden,
        );
        relu(hidden);

        const output = try allocator.alloc(f32, self.output_size);
        dense(
            hidden,
            self.output_weights,
            self.output_biases,
            self.hidden_size,
            self.output_size,
            output,
        );
        return output;
    }
};

fn runSample(_: *ExecutionContext, user: *anyopaque) !ResultHandle {
    const ptr: *u32 = @ptrCast(@alignCast(user));
    ptr.* = 7;
    return ResultHandle.fromSlice(&.{});
}

test "work item executes vtable" {
    const allocator = std.testing.allocator;
    var value: u32 = 0;
    const ctx = ExecutionContext{ .allocator = allocator };
    const vtable = WorkloadVTable{ .execute = runSample };
    const item = WorkItem{
        .id = 1,
        .user = &value,
        .vtable = &vtable,
        .priority = 0,
        .hints = .{},
    };

    const result = try runWorkItem(&ctx, &item);
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 7), value);
    try std.testing.expectEqual(@as(usize, 0), result.bytes.len);
}

test "matMul multiplies small matrices" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [4]f32 = undefined;

    matMul(&a, &b, 2, 2, 2, out[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 19, 22, 43, 50 }, &out);
}

test "matrix multiply task allocates output" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 };
    const task = MatrixMultiplyTask{
        .a = &a,
        .b = &b,
        .rows_a = 2,
        .cols_a = 2,
        .cols_b = 2,
    };

    const output = try task.execute(std.testing.allocator);
    defer std.testing.allocator.free(output);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, output);
}

test "mlp task runs forward pass" {
    const input = [_]f32{ 1, 2 };
    const hidden_weights = [_]f32{ 1, 0, 0, 1 };
    const hidden_biases = [_]f32{ 0, 0 };
    const output_weights = [_]f32{ 1, 1, 1, 1 };
    const output_biases = [_]f32{ 0, 0 };

    const task = MlpTask{
        .input = &input,
        .hidden_weights = &hidden_weights,
        .hidden_biases = &hidden_biases,
        .output_weights = &output_weights,
        .output_biases = &output_biases,
        .input_size = 2,
        .hidden_size = 2,
        .output_size = 2,
    };

    const output = try task.execute(std.testing.allocator);
    defer std.testing.allocator.free(output);

    try std.testing.expectEqual(@as(usize, 2), output.len);
    try std.testing.expectEqual(@as(f32, 3), output[0]);
    try std.testing.expectEqual(@as(f32, 3), output[1]);
}
