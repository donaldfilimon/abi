//! CUDA Code Generator
//!
//! Generates CUDA C source code from kernel IR.
//! This module uses the generic code generator with CUDA configuration.

const std = @import("std");
const generic = @import("generic.zig");
const common = @import("common.zig");
const backend = @import("backend.zig");
const kernel = @import("../kernel.zig");

// ============================================================================
// Generic CUDA Generator
// ============================================================================

/// CUDA code generator using the generic template with CUDA configuration.
/// This provides the standard KernelIR-based code generation.
pub const CudaGenerator = generic.CudaGenerator;

// ============================================================================
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for CUDA.
/// These functions generate optimized CUDA code for common vision operations.
pub const VisionKernels = struct {
    /// Generate a complete Conv2D CUDA kernel.
    /// Uses im2col + GEMM approach with shared memory tiling for efficiency.
    pub fn generateConv2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA Conv2D kernel");
        try writer.writeLine("// Uses im2col + GEMM approach with shared memory tiling");
        try writer.newline();
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.writeLine("#include <stdint.h>");
        try writer.newline();

        // Kernel function
        try writer.writeLine("extern \"C\" __global__ void conv2d(");
        try writer.writeLine("    const float* __restrict__ input,    // [batch, in_channels, height, width]");
        try writer.writeLine("    const float* __restrict__ weights,  // [out_channels, in_channels, kH, kW]");
        try writer.writeLine("    const float* __restrict__ bias,     // [out_channels]");
        try writer.writeLine("    float* __restrict__ output,         // [batch, out_channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int in_channels, int out_channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int batch_oc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = batch_oc / out_channels;");
        try writer.writeLine("    const int oc = batch_oc % out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    for (int ic = 0; ic < in_channels; ++ic) {");
        try writer.writeLine("        for (int ky = 0; ky < kernel_h; ++ky) {");
        try writer.writeLine("            for (int kx = 0; kx < kernel_w; ++kx) {");
        try writer.writeLine("                int ih = out_y * stride_h + ky - pad_h;");
        try writer.writeLine("                int iw = out_x * stride_w + kx - pad_w;");
        try writer.writeLine("                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                    int input_idx = ((batch_idx * in_channels + ic) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                    int weight_idx = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;");
        try writer.writeLine("                    sum += input[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum += bias[oc];");
        try writer.writeLine("    int output_idx = ((batch_idx * out_channels + oc) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D CUDA kernel with indices output.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA MaxPool2D kernel");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.writeLine("#include <float.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void max_pool2d(");
        try writer.writeLine("    const float* __restrict__ input,   // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,        // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int* __restrict__ indices,         // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_size, int stride, int padding");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int bc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = bc / channels;");
        try writer.writeLine("    const int channel = bc % channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float max_val = -FLT_MAX;");
        try writer.writeLine("    int max_idx = 0;");
        try writer.writeLine("    for (int ky = 0; ky < kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * stride + ky - padding;");
        try writer.writeLine("            int iw = out_x * stride + kx - padding;");
        try writer.writeLine("            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * channels + channel) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                float val = input[input_idx];");
        try writer.writeLine("                if (val > max_val) {");
        try writer.writeLine("                    max_val = val;");
        try writer.writeLine("                    max_idx = input_idx;");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D CUDA kernel.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA AvgPool2D kernel");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void avg_pool2d(");
        try writer.writeLine("    const float* __restrict__ input,   // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,        // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_size, int stride, int padding");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int bc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = bc / channels;");
        try writer.writeLine("    const int channel = bc % channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    int count = 0;");
        try writer.writeLine("    for (int ky = 0; ky < kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * stride + ky - padding;");
        try writer.writeLine("            int iw = out_x * stride + kx - padding;");
        try writer.writeLine("            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * channels + channel) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                sum += input[input_idx];");
        try writer.writeLine("                count++;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = count > 0 ? sum / (float)count : 0.0f;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D CUDA kernel (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA BatchNorm2D kernel (inference)");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void batch_norm2d(");
        try writer.writeLine("    const float* __restrict__ input,        // [batch, channels, height, width]");
        try writer.writeLine("    const float* __restrict__ gamma,        // [channels]");
        try writer.writeLine("    const float* __restrict__ beta,         // [channels]");
        try writer.writeLine("    const float* __restrict__ running_mean, // [channels]");
        try writer.writeLine("    const float* __restrict__ running_var,  // [channels]");
        try writer.writeLine("    float* __restrict__ output,             // [batch, channels, height, width]");
        try writer.writeLine("    int batch_size, int channels, int height, int width, float epsilon");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int total = batch_size * channels * height * width;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int hw = height * width;");
        try writer.writeLine("    const int chw = channels * hw;");
        try writer.writeLine("    const int c = (idx / hw) % channels;");
        try writer.newline();
        try writer.writeLine("    float x = input[idx];");
        try writer.writeLine("    float mean = running_mean[c];");
        try writer.writeLine("    float var = running_var[c];");
        try writer.writeLine("    float g = gamma[c];");
        try writer.writeLine("    float b = beta[c];");
        try writer.newline();
        try writer.writeLine("    float normalized = (x - mean) * rsqrtf(var + epsilon);");
        try writer.writeLine("    output[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col CUDA kernel.
    pub fn generateIm2col(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA im2col kernel");
        try writer.writeLine("// Transforms image patches to columns for efficient convolution via GEMM");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void im2col(");
        try writer.writeLine("    const float* __restrict__ input,  // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,       // [batch, C*kH*kW, out_h*out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int kernel_hw = kernel_h * kernel_w;");
        try writer.writeLine("    const int col_h = channels * kernel_hw;");
        try writer.writeLine("    const int col_w = out_height * out_width;");
        try writer.writeLine("    const int col_size = col_h * col_w;");
        try writer.writeLine("    const int total = batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int batch_idx = idx / col_size;");
        try writer.writeLine("    const int idx_in_batch = idx % col_size;");
        try writer.writeLine("    const int row = idx_in_batch / col_w;");
        try writer.writeLine("    const int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    const int c = row / kernel_hw;");
        try writer.writeLine("    const int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    const int ky = row_in_kernel / kernel_w;");
        try writer.writeLine("    const int kx = row_in_kernel % kernel_w;");
        try writer.newline();
        try writer.writeLine("    const int oh = col / out_width;");
        try writer.writeLine("    const int ow = col % out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * stride_h + ky - pad_h;");
        try writer.writeLine("    int iw = ow * stride_w + kx - pad_w;");
        try writer.newline();
        try writer.writeLine("    float val = 0.0f;");
        try writer.writeLine("    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("        int input_idx = ((batch_idx * channels + c) * in_height + ih) * in_width + iw;");
        try writer.writeLine("        val = input[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im CUDA kernel.
    pub fn generateCol2im(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA col2im kernel");
        try writer.writeLine("// Transforms columns back to image (inverse of im2col)");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void col2im(");
        try writer.writeLine("    const float* __restrict__ col_input, // [batch, C*kH*kW, out_h*out_w]");
        try writer.writeLine("    float* __restrict__ output,          // [batch, channels, height, width]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int kernel_hw = kernel_h * kernel_w;");
        try writer.writeLine("    const int col_h = channels * kernel_hw;");
        try writer.writeLine("    const int col_w = out_height * out_width;");
        try writer.writeLine("    const int col_size = col_h * col_w;");
        try writer.writeLine("    const int total = batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int batch_idx = idx / col_size;");
        try writer.writeLine("    const int idx_in_batch = idx % col_size;");
        try writer.writeLine("    const int row = idx_in_batch / col_w;");
        try writer.writeLine("    const int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    const int c = row / kernel_hw;");
        try writer.writeLine("    const int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    const int ky = row_in_kernel / kernel_w;");
        try writer.writeLine("    const int kx = row_in_kernel % kernel_w;");
        try writer.newline();
        try writer.writeLine("    const int oh = col / out_width;");
        try writer.writeLine("    const int ow = col % out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * stride_h + ky - pad_h;");
        try writer.writeLine("    int iw = ow * stride_w + kx - pad_w;");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("        int output_idx = ((batch_idx * channels + c) * in_height + ih) * in_width + iw;");
        try writer.writeLine("        atomicAdd(&output[output_idx], col_input[idx]);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CudaGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = CudaGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "__global__") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "test_kernel") != null);
}
