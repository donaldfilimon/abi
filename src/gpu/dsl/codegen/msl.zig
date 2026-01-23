//! Metal Shading Language (MSL) Code Generator
//!
//! Generates Metal compute shader source code from kernel IR.
//! This module uses the generic code generator with MSL configuration.

const std = @import("std");
const generic = @import("generic.zig");
const common = @import("common.zig");
const backend = @import("backend.zig");
const kernel = @import("../kernel.zig");

// ============================================================================
// Generic MSL Generator
// ============================================================================

/// MSL code generator using the generic template with MSL configuration.
/// This provides the standard KernelIR-based code generation.
pub const MslGenerator = generic.MslGenerator;

// ============================================================================
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for MSL.
/// These functions generate optimized Metal compute shaders for vision operations.
pub const VisionKernels = struct {
    /// Generate a Conv2D MSL compute shader.
    pub fn generateConv2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal Conv2D compute shader");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct Conv2dParams {");
        try writer.writeLine("    uint batch_size, in_channels, out_channels;");
        try writer.writeLine("    uint in_height, in_width, out_height, out_width;");
        try writer.writeLine("    uint kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    uint pad_h, pad_w;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void conv2d(");
        try writer.writeLine("    const device float* input [[buffer(0)]],");
        try writer.writeLine("    const device float* weights [[buffer(1)]],");
        try writer.writeLine("    const device float* bias [[buffer(2)]],");
        try writer.writeLine("    device float* output [[buffer(3)]],");
        try writer.writeLine("    constant Conv2dParams& params [[buffer(4)]],");
        try writer.writeLine("    uint3 gid [[thread_position_in_grid]],");
        try writer.writeLine("    uint3 wid [[threadgroup_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint out_x = gid.x;");
        try writer.writeLine("    uint out_y = gid.y;");
        try writer.writeLine("    uint batch_oc = wid.z;");
        try writer.writeLine("    uint batch_idx = batch_oc / params.out_channels;");
        try writer.writeLine("    uint oc = batch_oc % params.out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    for (uint ic = 0; ic < params.in_channels; ++ic) {");
        try writer.writeLine("        for (uint ky = 0; ky < params.kernel_h; ++ky) {");
        try writer.writeLine("            for (uint kx = 0; kx < params.kernel_w; ++kx) {");
        try writer.writeLine("                int ih = int(out_y * params.stride_h + ky) - int(params.pad_h);");
        try writer.writeLine("                int iw = int(out_x * params.stride_w + kx) - int(params.pad_w);");
        try writer.writeLine("                if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {");
        try writer.writeLine("                    uint input_idx = ((batch_idx * params.in_channels + ic) * params.in_height + uint(ih)) * params.in_width + uint(iw);");
        try writer.writeLine("                    uint weight_idx = ((oc * params.in_channels + ic) * params.kernel_h + ky) * params.kernel_w + kx;");
        try writer.writeLine("                    sum += input[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum += bias[oc];");
        try writer.writeLine("    uint output_idx = ((batch_idx * params.out_channels + oc) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D MSL compute shader.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal MaxPool2D compute shader");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct Pool2dParams {");
        try writer.writeLine("    uint batch_size, channels;");
        try writer.writeLine("    uint in_height, in_width, out_height, out_width;");
        try writer.writeLine("    uint kernel_size, stride, padding;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void max_pool2d(");
        try writer.writeLine("    const device float* input [[buffer(0)]],");
        try writer.writeLine("    device float* output [[buffer(1)]],");
        try writer.writeLine("    device uint* indices [[buffer(2)]],");
        try writer.writeLine("    constant Pool2dParams& params [[buffer(3)]],");
        try writer.writeLine("    uint3 gid [[thread_position_in_grid]],");
        try writer.writeLine("    uint3 wid [[threadgroup_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint out_x = gid.x;");
        try writer.writeLine("    uint out_y = gid.y;");
        try writer.writeLine("    uint bc = wid.z;");
        try writer.writeLine("    uint batch_idx = bc / params.channels;");
        try writer.writeLine("    uint channel = bc % params.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float max_val = -FLT_MAX;");
        try writer.writeLine("    uint max_idx = 0;");
        try writer.writeLine("    for (uint ky = 0; ky < params.kernel_size; ++ky) {");
        try writer.writeLine("        for (uint kx = 0; kx < params.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = int(out_y * params.stride + ky) - int(params.padding);");
        try writer.writeLine("            int iw = int(out_x * params.stride + kx) - int(params.padding);");
        try writer.writeLine("            if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {");
        try writer.writeLine("                uint input_idx = ((batch_idx * params.channels + channel) * params.in_height + uint(ih)) * params.in_width + uint(iw);");
        try writer.writeLine("                float val = input[input_idx];");
        try writer.writeLine("                if (val > max_val) { max_val = val; max_idx = input_idx; }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    uint output_idx = ((batch_idx * params.channels + channel) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D MSL compute shader.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal AvgPool2D compute shader");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct Pool2dParams {");
        try writer.writeLine("    uint batch_size, channels;");
        try writer.writeLine("    uint in_height, in_width, out_height, out_width;");
        try writer.writeLine("    uint kernel_size, stride, padding;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void avg_pool2d(");
        try writer.writeLine("    const device float* input [[buffer(0)]],");
        try writer.writeLine("    device float* output [[buffer(1)]],");
        try writer.writeLine("    constant Pool2dParams& params [[buffer(2)]],");
        try writer.writeLine("    uint3 gid [[thread_position_in_grid]],");
        try writer.writeLine("    uint3 wid [[threadgroup_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint out_x = gid.x;");
        try writer.writeLine("    uint out_y = gid.y;");
        try writer.writeLine("    uint bc = wid.z;");
        try writer.writeLine("    uint batch_idx = bc / params.channels;");
        try writer.writeLine("    uint channel = bc % params.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= params.out_width || out_y >= params.out_height || batch_idx >= params.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    uint count = 0;");
        try writer.writeLine("    for (uint ky = 0; ky < params.kernel_size; ++ky) {");
        try writer.writeLine("        for (uint kx = 0; kx < params.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = int(out_y * params.stride + ky) - int(params.padding);");
        try writer.writeLine("            int iw = int(out_x * params.stride + kx) - int(params.padding);");
        try writer.writeLine("            if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {");
        try writer.writeLine("                uint input_idx = ((batch_idx * params.channels + channel) * params.in_height + uint(ih)) * params.in_width + uint(iw);");
        try writer.writeLine("                sum += input[input_idx];");
        try writer.writeLine("                count++;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    uint output_idx = ((batch_idx * params.channels + channel) * params.out_height + out_y) * params.out_width + out_x;");
        try writer.writeLine("    output[output_idx] = count > 0 ? sum / float(count) : 0.0f;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D MSL compute shader (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal BatchNorm2D compute shader (inference)");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct BatchNorm2dParams {");
        try writer.writeLine("    uint batch_size, channels, height, width;");
        try writer.writeLine("    float epsilon;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void batch_norm2d(");
        try writer.writeLine("    const device float* input [[buffer(0)]],");
        try writer.writeLine("    const device float* gamma [[buffer(1)]],");
        try writer.writeLine("    const device float* beta [[buffer(2)]],");
        try writer.writeLine("    const device float* running_mean [[buffer(3)]],");
        try writer.writeLine("    const device float* running_var [[buffer(4)]],");
        try writer.writeLine("    device float* output [[buffer(5)]],");
        try writer.writeLine("    constant BatchNorm2dParams& params [[buffer(6)]],");
        try writer.writeLine("    uint idx [[thread_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint total = params.batch_size * params.channels * params.height * params.width;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    uint hw = params.height * params.width;");
        try writer.writeLine("    uint c = (idx / hw) % params.channels;");
        try writer.newline();
        try writer.writeLine("    float x = input[idx];");
        try writer.writeLine("    float mean = running_mean[c];");
        try writer.writeLine("    float var_val = running_var[c];");
        try writer.writeLine("    float g = gamma[c];");
        try writer.writeLine("    float b = beta[c];");
        try writer.newline();
        try writer.writeLine("    float normalized = (x - mean) * rsqrt(var_val + params.epsilon);");
        try writer.writeLine("    output[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col MSL compute shader.
    pub fn generateIm2col(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal im2col compute shader");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct Im2colParams {");
        try writer.writeLine("    uint batch_size, channels;");
        try writer.writeLine("    uint in_height, in_width, out_height, out_width;");
        try writer.writeLine("    uint kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    uint pad_h, pad_w;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void im2col(");
        try writer.writeLine("    const device float* input [[buffer(0)]],");
        try writer.writeLine("    device float* output [[buffer(1)]],");
        try writer.writeLine("    constant Im2colParams& params [[buffer(2)]],");
        try writer.writeLine("    uint idx [[thread_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint kernel_hw = params.kernel_h * params.kernel_w;");
        try writer.writeLine("    uint col_h = params.channels * kernel_hw;");
        try writer.writeLine("    uint col_w = params.out_height * params.out_width;");
        try writer.writeLine("    uint col_size = col_h * col_w;");
        try writer.writeLine("    uint total = params.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    uint batch_idx = idx / col_size;");
        try writer.writeLine("    uint idx_in_batch = idx % col_size;");
        try writer.writeLine("    uint row = idx_in_batch / col_w;");
        try writer.writeLine("    uint col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    uint c = row / kernel_hw;");
        try writer.writeLine("    uint row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    uint ky = row_in_kernel / params.kernel_w;");
        try writer.writeLine("    uint kx = row_in_kernel % params.kernel_w;");
        try writer.newline();
        try writer.writeLine("    uint oh = col / params.out_width;");
        try writer.writeLine("    uint ow = col % params.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = int(oh * params.stride_h + ky) - int(params.pad_h);");
        try writer.writeLine("    int iw = int(ow * params.stride_w + kx) - int(params.pad_w);");
        try writer.newline();
        try writer.writeLine("    float val = 0.0f;");
        try writer.writeLine("    if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {");
        try writer.writeLine("        uint input_idx = ((batch_idx * params.channels + c) * params.in_height + uint(ih)) * params.in_width + uint(iw);");
        try writer.writeLine("        val = input[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im MSL compute shader.
    pub fn generateCol2im(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated Metal col2im compute shader");
        try writer.newline();
        try writer.writeLine("#include <metal_stdlib>");
        try writer.writeLine("using namespace metal;");
        try writer.newline();

        try writer.writeLine("struct Col2imParams {");
        try writer.writeLine("    uint batch_size, channels;");
        try writer.writeLine("    uint in_height, in_width, out_height, out_width;");
        try writer.writeLine("    uint kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    uint pad_h, pad_w;");
        try writer.writeLine("};");
        try writer.newline();

        try writer.writeLine("kernel void col2im(");
        try writer.writeLine("    const device float* col_input [[buffer(0)]],");
        try writer.writeLine("    device atomic_float* output [[buffer(1)]],");
        try writer.writeLine("    constant Col2imParams& params [[buffer(2)]],");
        try writer.writeLine("    uint idx [[thread_position_in_grid]]");
        try writer.writeLine(") {");
        try writer.writeLine("    uint kernel_hw = params.kernel_h * params.kernel_w;");
        try writer.writeLine("    uint col_h = params.channels * kernel_hw;");
        try writer.writeLine("    uint col_w = params.out_height * params.out_width;");
        try writer.writeLine("    uint col_size = col_h * col_w;");
        try writer.writeLine("    uint total = params.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    uint batch_idx = idx / col_size;");
        try writer.writeLine("    uint idx_in_batch = idx % col_size;");
        try writer.writeLine("    uint row = idx_in_batch / col_w;");
        try writer.writeLine("    uint col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    uint c = row / kernel_hw;");
        try writer.writeLine("    uint row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    uint ky = row_in_kernel / params.kernel_w;");
        try writer.writeLine("    uint kx = row_in_kernel % params.kernel_w;");
        try writer.newline();
        try writer.writeLine("    uint oh = col / params.out_width;");
        try writer.writeLine("    uint ow = col % params.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = int(oh * params.stride_h + ky) - int(params.pad_h);");
        try writer.writeLine("    int iw = int(ow * params.stride_w + kx) - int(params.pad_w);");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {");
        try writer.writeLine("        uint output_idx = ((batch_idx * params.channels + c) * params.in_height + uint(ih)) * params.in_width + uint(iw);");
        try writer.writeLine("        atomic_fetch_add_explicit(&output[output_idx], col_input[idx], memory_order_relaxed);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MslGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = MslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#include <metal_stdlib>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "kernel void") != null);
}
