//! GLSL Code Generator
//!
//! Generates GLSL compute shader source code from kernel IR.
//! This module uses the generic code generator with GLSL configuration.
//!
//! ## Target Versioning
//!
//! The generic `GlslGenerator` always produces Vulkan-style output (`#version 450`).
//! For target-specific version strings (OpenGL 430, OpenGL ES 310), use the
//! `VisionKernels` functions which accept a `GlslTarget` parameter.
//!
//! | Target    | Version Directive    | Use Case                    |
//! |-----------|---------------------|-----------------------------|
//! | vulkan    | #version 450        | Vulkan compute shaders      |
//! | opengl    | #version 430        | Desktop OpenGL compute      |
//! | opengles  | #version 310 es     | Mobile/embedded compute     |

const std = @import("std");
const generic = @import("generic.zig");
const common = @import("common.zig");
const backend = @import("backend.zig");
const kernel = @import("../kernel.zig");

// ============================================================================
// Generic GLSL Generator
// ============================================================================

/// Target GLSL variant.
pub const GlslTarget = enum {
    vulkan, // GLSL 450 with Vulkan extensions
    opengl, // GLSL 430 compute shaders
    opengles, // GLSL ES 310+ compute
};

/// GLSL code generator using the generic template with GLSL configuration.
/// This provides the standard KernelIR-based code generation.
///
/// **Note**: Always generates Vulkan-style output (`#version 450` with extensions).
/// For OpenGL/OpenGL ES targets, the generated code may need version directive adjustment,
/// or use `VisionKernels` functions which support target-specific versioning.
pub const GlslGenerator = generic.GlslGenerator;

// ============================================================================
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for GLSL.
/// These functions generate optimized GLSL compute shaders for vision operations.
pub const VisionKernels = struct {
    /// Generate a Conv2D GLSL compute shader.
    pub fn generateConv2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        // Version header
        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL Conv2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        // Buffer bindings
        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) readonly buffer WeightsBuffer { float weights[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) readonly buffer BiasBuffer { float bias[]; };");
        try writer.writeLine("layout(set = 0, binding = 3) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        // Push constants
        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, in_channels, out_channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int batch_oc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = batch_oc / pc.out_channels;");
        try writer.writeLine("    int oc = batch_oc % pc.out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0;");
        try writer.writeLine("    for (int ic = 0; ic < pc.in_channels; ++ic) {");
        try writer.writeLine("        for (int ky = 0; ky < pc.kernel_h; ++ky) {");
        try writer.writeLine("            for (int kx = 0; kx < pc.kernel_w; ++kx) {");
        try writer.writeLine("                int ih = out_y * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("                int iw = out_x * pc.stride_w + kx - pc.pad_w;");
        try writer.writeLine("                if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                    int input_idx = ((batch_idx * pc.in_channels + ic) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                    int weight_idx = ((oc * pc.in_channels + ic) * pc.kernel_h + ky) * pc.kernel_w + kx;");
        try writer.writeLine("                    sum += input_data[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum += bias[oc];");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.out_channels + oc) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D GLSL compute shader.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL MaxPool2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) writeonly buffer IndicesBuffer { int indices[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_size, stride, padding;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int bc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = bc / pc.channels;");
        try writer.writeLine("    int channel = bc % pc.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float max_val = -3.4028235e+38;");
        try writer.writeLine("    int max_idx = 0;");
        try writer.writeLine("    for (int ky = 0; ky < pc.kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < pc.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * pc.stride + ky - pc.padding;");
        try writer.writeLine("            int iw = out_x * pc.stride + kx - pc.padding;");
        try writer.writeLine("            if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * pc.channels + channel) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                float val = input_data[input_idx];");
        try writer.writeLine("                if (val > max_val) { max_val = val; max_idx = input_idx; }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.channels + channel) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D GLSL compute shader.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL AvgPool2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_size, stride, padding;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int bc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = bc / pc.channels;");
        try writer.writeLine("    int channel = bc % pc.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0;");
        try writer.writeLine("    int count = 0;");
        try writer.writeLine("    for (int ky = 0; ky < pc.kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < pc.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * pc.stride + ky - pc.padding;");
        try writer.writeLine("            int iw = out_x * pc.stride + kx - pc.padding;");
        try writer.writeLine("            if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * pc.channels + channel) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                sum += input_data[input_idx];");
        try writer.writeLine("                count++;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.channels + channel) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = count > 0 ? sum / float(count) : 0.0;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D GLSL compute shader (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL BatchNorm2D compute shader (inference)");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) readonly buffer GammaBuffer { float gamma[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) readonly buffer BetaBuffer { float beta[]; };");
        try writer.writeLine("layout(set = 0, binding = 3) readonly buffer MeanBuffer { float running_mean[]; };");
        try writer.writeLine("layout(set = 0, binding = 4) readonly buffer VarBuffer { float running_var[]; };");
        try writer.writeLine("layout(set = 0, binding = 5) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels, height, width;");
        try writer.writeLine("    float epsilon;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int total = pc.batch_size * pc.channels * pc.height * pc.width;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int hw = pc.height * pc.width;");
        try writer.writeLine("    int c = (idx / hw) % pc.channels;");
        try writer.newline();
        try writer.writeLine("    float x = input_data[idx];");
        try writer.writeLine("    float mean = running_mean[c];");
        try writer.writeLine("    float var_val = running_var[c];");
        try writer.writeLine("    float g = gamma[c];");
        try writer.writeLine("    float b = beta[c];");
        try writer.newline();
        try writer.writeLine("    float normalized = (x - mean) * inversesqrt(var_val + pc.epsilon);");
        try writer.writeLine("    output_data[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col GLSL compute shader.
    pub fn generateIm2col(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL im2col compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int kernel_hw = pc.kernel_h * pc.kernel_w;");
        try writer.writeLine("    int col_h = pc.channels * kernel_hw;");
        try writer.writeLine("    int col_w = pc.out_height * pc.out_width;");
        try writer.writeLine("    int col_size = col_h * col_w;");
        try writer.writeLine("    int total = pc.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int batch_idx = idx / col_size;");
        try writer.writeLine("    int idx_in_batch = idx % col_size;");
        try writer.writeLine("    int row = idx_in_batch / col_w;");
        try writer.writeLine("    int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    int c = row / kernel_hw;");
        try writer.writeLine("    int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    int ky = row_in_kernel / pc.kernel_w;");
        try writer.writeLine("    int kx = row_in_kernel % pc.kernel_w;");
        try writer.newline();
        try writer.writeLine("    int oh = col / pc.out_width;");
        try writer.writeLine("    int ow = col % pc.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("    int iw = ow * pc.stride_w + kx - pc.pad_w;");
        try writer.newline();
        try writer.writeLine("    float val = 0.0;");
        try writer.writeLine("    if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("        int input_idx = ((batch_idx * pc.channels + c) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("        val = input_data[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output_data[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im GLSL compute shader.
    pub fn generateCol2im(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL col2im compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer ColInputBuffer { float col_input[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int kernel_hw = pc.kernel_h * pc.kernel_w;");
        try writer.writeLine("    int col_h = pc.channels * kernel_hw;");
        try writer.writeLine("    int col_w = pc.out_height * pc.out_width;");
        try writer.writeLine("    int col_size = col_h * col_w;");
        try writer.writeLine("    int total = pc.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int batch_idx = idx / col_size;");
        try writer.writeLine("    int idx_in_batch = idx % col_size;");
        try writer.writeLine("    int row = idx_in_batch / col_w;");
        try writer.writeLine("    int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    int c = row / kernel_hw;");
        try writer.writeLine("    int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    int ky = row_in_kernel / pc.kernel_w;");
        try writer.writeLine("    int kx = row_in_kernel % pc.kernel_w;");
        try writer.newline();
        try writer.writeLine("    int oh = col / pc.out_width;");
        try writer.writeLine("    int ow = col % pc.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("    int iw = ow * pc.stride_w + kx - pc.pad_w;");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("        int output_idx = ((batch_idx * pc.channels + c) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("        atomicAdd(output_data[output_idx], col_input[idx]);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "GlslGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = GlslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 450") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "void main()") != null);
}
