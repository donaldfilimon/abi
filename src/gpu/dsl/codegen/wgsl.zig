//! WGSL Code Generator
//!
//! Generates WGSL (WebGPU Shading Language) compute shader source code from kernel IR.
//! This module uses the generic code generator with WGSL configuration.

const std = @import("std");
const generic = @import("generic.zig");
const common = @import("common.zig");
const backend = @import("backend.zig");
const kernel = @import("../kernel.zig");

// ============================================================================
// Generic WGSL Generator
// ============================================================================

/// WGSL code generator using the generic template with WGSL configuration.
/// This provides the standard KernelIR-based code generation.
pub const WgslGenerator = generic.WgslGenerator;

// ============================================================================
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for WGSL.
/// These functions generate optimized WGSL compute shaders for vision operations.
pub const VisionKernels = struct {
    /// Generate a Conv2D WGSL compute shader.
    pub fn generateConv2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL Conv2D compute shader");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, in_channels: u32, out_channels: u32,");
        try writer.writeLine("    in_height: u32, in_width: u32, out_height: u32, out_width: u32,");
        try writer.writeLine("    kernel_h: u32, kernel_w: u32, stride_h: u32, stride_w: u32,");
        try writer.writeLine("    pad_h: u32, pad_w: u32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> input_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read> weights: array<f32>;");
        try writer.writeLine("@group(0) @binding(2) var<storage, read> bias: array<f32>;");
        try writer.writeLine("@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(4) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(16, 16, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {");
        try writer.writeLine("    let out_x = gid.x;");
        try writer.writeLine("    let out_y = gid.y;");
        try writer.writeLine("    let batch_oc = wid.z;");
        try writer.writeLine("    let batch_idx = batch_oc / params.out_channels;");
        try writer.writeLine("    let oc = batch_oc % params.out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) { return; }");
        try writer.newline();
        try writer.writeLine("    var sum: f32 = 0.0;");
        try writer.writeLine("    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {");
        try writer.writeLine("        for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {");
        try writer.writeLine("            for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {");
        try writer.writeLine("                let ih = i32(out_y * params.stride_h + ky) - i32(params.pad_h);");
        try writer.writeLine("                let iw = i32(out_x * params.stride_w + kx) - i32(params.pad_w);");
        try writer.writeLine("                if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {");
        try writer.writeLine("                    let input_idx = ((batch_idx * params.in_channels + ic) * params.in_height + u32(ih)) * params.in_width + u32(iw);");
        try writer.writeLine("                    let weight_idx = ((oc * params.in_channels + ic) * params.kernel_h + ky) * params.kernel_w + kx;");
        try writer.writeLine("                    sum = sum + input_data[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum = sum + bias[oc];");
        try writer.writeLine("    let output_idx = ((batch_idx * params.out_channels + oc) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D WGSL compute shader.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL MaxPool2D compute shader");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, channels: u32,");
        try writer.writeLine("    in_height: u32, in_width: u32, out_height: u32, out_width: u32,");
        try writer.writeLine("    kernel_size: u32, stride: u32, padding: u32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> input_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(2) var<storage, read_write> indices: array<u32>;");
        try writer.writeLine("@group(0) @binding(3) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(16, 16, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {");
        try writer.writeLine("    let out_x = gid.x;");
        try writer.writeLine("    let out_y = gid.y;");
        try writer.writeLine("    let bc = wid.z;");
        try writer.writeLine("    let batch_idx = bc / params.channels;");
        try writer.writeLine("    let channel = bc % params.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) { return; }");
        try writer.newline();
        try writer.writeLine("    var max_val: f32 = -3.4028235e+38;");
        try writer.writeLine("    var max_idx: u32 = 0u;");
        try writer.writeLine("    for (var ky: u32 = 0u; ky < params.kernel_size; ky = ky + 1u) {");
        try writer.writeLine("        for (var kx: u32 = 0u; kx < params.kernel_size; kx = kx + 1u) {");
        try writer.writeLine("            let ih = i32(out_y * params.stride + ky) - i32(params.padding);");
        try writer.writeLine("            let iw = i32(out_x * params.stride + kx) - i32(params.padding);");
        try writer.writeLine("            if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {");
        try writer.writeLine("                let input_idx = ((batch_idx * params.channels + channel) * params.in_height + u32(ih)) * params.in_width + u32(iw);");
        try writer.writeLine("                let val = input_data[input_idx];");
        try writer.writeLine("                if (val > max_val) { max_val = val; max_idx = input_idx; }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    let output_idx = ((batch_idx * params.channels + channel) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D WGSL compute shader.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL AvgPool2D compute shader");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, channels: u32,");
        try writer.writeLine("    in_height: u32, in_width: u32, out_height: u32, out_width: u32,");
        try writer.writeLine("    kernel_size: u32, stride: u32, padding: u32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> input_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(2) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(16, 16, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {");
        try writer.writeLine("    let out_x = gid.x;");
        try writer.writeLine("    let out_y = gid.y;");
        try writer.writeLine("    let bc = wid.z;");
        try writer.writeLine("    let batch_idx = bc / params.channels;");
        try writer.writeLine("    let channel = bc % params.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) { return; }");
        try writer.newline();
        try writer.writeLine("    var sum: f32 = 0.0;");
        try writer.writeLine("    var count: u32 = 0u;");
        try writer.writeLine("    for (var ky: u32 = 0u; ky < params.kernel_size; ky = ky + 1u) {");
        try writer.writeLine("        for (var kx: u32 = 0u; kx < params.kernel_size; kx = kx + 1u) {");
        try writer.writeLine("            let ih = i32(out_y * params.stride + ky) - i32(params.padding);");
        try writer.writeLine("            let iw = i32(out_x * params.stride + kx) - i32(params.padding);");
        try writer.writeLine("            if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {");
        try writer.writeLine("                let input_idx = ((batch_idx * params.channels + channel) * params.in_height + u32(ih)) * params.in_width + u32(iw);");
        try writer.writeLine("                sum = sum + input_data[input_idx];");
        try writer.writeLine("                count = count + 1u;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    let output_idx = ((batch_idx * params.channels + channel) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = select(0.0, sum / f32(count), count > 0u);");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D WGSL compute shader (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL BatchNorm2D compute shader (inference)");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, channels: u32, height: u32, width: u32,");
        try writer.writeLine("    epsilon: f32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> input_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read> gamma: array<f32>;");
        try writer.writeLine("@group(0) @binding(2) var<storage, read> beta: array<f32>;");
        try writer.writeLine("@group(0) @binding(3) var<storage, read> running_mean: array<f32>;");
        try writer.writeLine("@group(0) @binding(4) var<storage, read> running_var: array<f32>;");
        try writer.writeLine("@group(0) @binding(5) var<storage, read_write> output_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(6) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(256, 1, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {");
        try writer.writeLine("    let idx = gid.x;");
        try writer.writeLine("    let total = params.batch_size * params.channels * params.height * params.width;");
        try writer.writeLine("    if (idx >= total) { return; }");
        try writer.newline();
        try writer.writeLine("    let hw = params.height * params.width;");
        try writer.writeLine("    let c = (idx / hw) % params.channels;");
        try writer.newline();
        try writer.writeLine("    let x = input_data[idx];");
        try writer.writeLine("    let mean = running_mean[c];");
        try writer.writeLine("    let var_val = running_var[c];");
        try writer.writeLine("    let g = gamma[c];");
        try writer.writeLine("    let b = beta[c];");
        try writer.newline();
        try writer.writeLine("    let normalized = (x - mean) * inverseSqrt(var_val + params.epsilon);");
        try writer.writeLine("    output_data[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col WGSL compute shader.
    pub fn generateIm2col(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL im2col compute shader");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, channels: u32,");
        try writer.writeLine("    in_height: u32, in_width: u32, out_height: u32, out_width: u32,");
        try writer.writeLine("    kernel_h: u32, kernel_w: u32, stride_h: u32, stride_w: u32,");
        try writer.writeLine("    pad_h: u32, pad_w: u32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> input_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;");
        try writer.writeLine("@group(0) @binding(2) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(256, 1, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {");
        try writer.writeLine("    let idx = gid.x;");
        try writer.writeLine("    let kernel_hw = params.kernel_h * params.kernel_w;");
        try writer.writeLine("    let col_h = params.channels * kernel_hw;");
        try writer.writeLine("    let col_w = params.out_height * params.out_width;");
        try writer.writeLine("    let col_size = col_h * col_w;");
        try writer.writeLine("    let total = params.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) { return; }");
        try writer.newline();
        try writer.writeLine("    let batch_idx = idx / col_size;");
        try writer.writeLine("    let idx_in_batch = idx % col_size;");
        try writer.writeLine("    let row = idx_in_batch / col_w;");
        try writer.writeLine("    let col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    let c = row / kernel_hw;");
        try writer.writeLine("    let row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    let ky = row_in_kernel / params.kernel_w;");
        try writer.writeLine("    let kx = row_in_kernel % params.kernel_w;");
        try writer.newline();
        try writer.writeLine("    let oh = col / params.out_width;");
        try writer.writeLine("    let ow = col % params.out_width;");
        try writer.newline();
        try writer.writeLine("    let ih = i32(oh * params.stride_h + ky) - i32(params.pad_h);");
        try writer.writeLine("    let iw = i32(ow * params.stride_w + kx) - i32(params.pad_w);");
        try writer.newline();
        try writer.writeLine("    var val: f32 = 0.0;");
        try writer.writeLine("    if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {");
        try writer.writeLine("        let input_idx = ((batch_idx * params.channels + c) * params.in_height + u32(ih)) * params.in_width + u32(iw);");
        try writer.writeLine("        val = input_data[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output_data[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im WGSL compute shader.
    pub fn generateCol2im(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated WGSL col2im compute shader");
        try writer.newline();

        try writer.writeLine("struct Params {");
        try writer.writeLine("    batch_size: u32, channels: u32,");
        try writer.writeLine("    in_height: u32, in_width: u32, out_height: u32, out_width: u32,");
        try writer.writeLine("    kernel_h: u32, kernel_w: u32, stride_h: u32, stride_w: u32,");
        try writer.writeLine("    pad_h: u32, pad_w: u32,");
        try writer.writeLine("}");
        try writer.newline();

        try writer.writeLine("@group(0) @binding(0) var<storage, read> col_input: array<f32>;");
        try writer.writeLine("@group(0) @binding(1) var<storage, read_write> output_data: array<atomic<u32>>;");
        try writer.writeLine("@group(0) @binding(2) var<uniform> params: Params;");
        try writer.newline();

        try writer.writeLine("@compute @workgroup_size(256, 1, 1)");
        try writer.writeLine("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {");
        try writer.writeLine("    let idx = gid.x;");
        try writer.writeLine("    let kernel_hw = params.kernel_h * params.kernel_w;");
        try writer.writeLine("    let col_h = params.channels * kernel_hw;");
        try writer.writeLine("    let col_w = params.out_height * params.out_width;");
        try writer.writeLine("    let col_size = col_h * col_w;");
        try writer.writeLine("    let total = params.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) { return; }");
        try writer.newline();
        try writer.writeLine("    let batch_idx = idx / col_size;");
        try writer.writeLine("    let idx_in_batch = idx % col_size;");
        try writer.writeLine("    let row = idx_in_batch / col_w;");
        try writer.writeLine("    let col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    let c = row / kernel_hw;");
        try writer.writeLine("    let row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    let ky = row_in_kernel / params.kernel_w;");
        try writer.writeLine("    let kx = row_in_kernel % params.kernel_w;");
        try writer.newline();
        try writer.writeLine("    let oh = col / params.out_width;");
        try writer.writeLine("    let ow = col % params.out_width;");
        try writer.newline();
        try writer.writeLine("    let ih = i32(oh * params.stride_h + ky) - i32(params.pad_h);");
        try writer.writeLine("    let iw = i32(ow * params.stride_w + kx) - i32(params.pad_w);");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < i32(params.in_height) && iw >= 0 && iw < i32(params.in_width)) {");
        try writer.writeLine("        let output_idx = ((batch_idx * params.channels + c) * params.in_height + u32(ih)) * params.in_width + u32(iw);");
        try writer.writeLine("        // Note: WGSL atomicAdd works on u32/i32, need bitcast for f32");
        try writer.writeLine("        let val_bits = bitcast<u32>(col_input[idx]);");
        try writer.writeLine("        atomicAdd(&output_data[output_idx], val_bits);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WgslGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = WgslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "@compute") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "@workgroup_size") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "fn test_kernel") != null);
}
